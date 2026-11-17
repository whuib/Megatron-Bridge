# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Qwen3-VL mock-data case for native, DistTrain, and multiplexing runs.

Every setting except the selected execution mode and its parallel layout is shared, so
the three modes run the same model, data, batch shape, precision, recompute policy and
optimizer. With ``--warmup-iters`` a mode-agnostic throughput meter measures a steady
window identically in every mode (see ``throughput.py``).
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import MockVLMSFTDatasetConfig
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import DistTrainConfig
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.multiplexing.config import MultiplexingConfig
from megatron.bridge.training.pretrain import pretrain


sys.path.insert(0, str(Path(__file__).resolve().parent))
from throughput import ThroughputMeter, Workload  # noqa: E402


logger = logging.getLogger(__name__)

RECOMPUTE_POLICIES = ("full", "selective", "none")
# "inherit" reuses the language policy (legacy Qwen3-VL behavior); the others set the vision tower alone.
VISION_RECOMPUTE_POLICIES = ("inherit", *RECOMPUTE_POLICIES)
MUX_ENCODER_SCHEDULES = ("deferred", "interleaved")


def parallel_layout(args: argparse.Namespace, world_size: int) -> dict[str, int | str]:
    """Validate and describe the vision and language layouts of the selected mode.

    Besides the GPU split, the layout records each mode's own tuning knobs: the vision recompute
    policy (memory feasibility differs per mode) and the multiplexing encoder schedule.
    """
    language_model_parallel = args.language_tp * args.language_pp
    if args.mode == "disttrain":
        vision_world = args.vision_gpus
        language_world = world_size - vision_world
        if vision_world <= 0 or language_world <= 0:
            raise ValueError("DistTrain requires 0 < --vision-gpus < WORLD_SIZE")
        if vision_world % args.encoder_tp or language_world % language_model_parallel:
            raise ValueError("DistTrain module GPU counts must be divisible by their TP/PP products")
        layout = {
            "vision_gpus": vision_world,
            "language_gpus": language_world,
            "vision_dp": vision_world // args.encoder_tp,
            "language_dp": language_world // language_model_parallel,
        }
    else:
        if world_size % language_model_parallel:
            raise ValueError("Language TP * PP must divide WORLD_SIZE")
        language_dp = world_size // language_model_parallel
        # Native runs the encoder inside the first language stage, with the language DP.
        vision_dp = world_size // args.encoder_tp if args.mode == "multiplexing" else language_dp
        if args.mode == "multiplexing" and world_size % args.encoder_tp:
            raise ValueError("Encoder TP must divide WORLD_SIZE")
        layout = {"vision_gpus": world_size, "language_gpus": world_size, "vision_dp": vision_dp}
        layout["language_dp"] = language_dp
    if layout["vision_dp"] % layout["language_dp"]:
        raise ValueError("vision DP must be a multiple of language DP")
    if args.sequence_parallel and args.language_tp < 2:
        raise ValueError("--sequence-parallel requires --language-tp > 1")
    if args.mode == "multiplexing" and args.mux_encoder_schedule == "interleaved" and args.language_pp != 1:
        raise ValueError("--mux-encoder-schedule interleaved requires --language-pp 1")
    layout.update(
        language_tp=args.language_tp,
        language_pp=args.language_pp,
        encoder_tp=args.encoder_tp,
        sequence_parallel=int(args.sequence_parallel),
        vision_recompute=args.vision_recompute,
        mux_encoder_schedule=args.mux_encoder_schedule if args.mode == "multiplexing" else "n/a",
    )
    return layout


def validate_workload(args: argparse.Namespace, layout: dict[str, int]) -> None:
    """Reject workloads that a mode would split unevenly or incorrectly.

    DistTrain splits a micro-batch's patch rows into ``vision_dp / language_dp`` equal chunks
    and its image grids by count; the two only describe the same images when every chunk holds
    whole, equally sized images. The mock dataset uses one image size, so the image count per
    micro-batch must be divisible by the ratio. The same rule keeps multiplexing encoder shards
    equally loaded, so both split modes see balanced per-rank vision work.
    """
    if args.global_batch_size % (args.micro_batch_size * layout["language_dp"]):
        raise ValueError("global batch size must divide evenly into microbatch size * language DP")
    ratio = layout["vision_dp"] // layout["language_dp"]
    images_per_microbatch = args.micro_batch_size * args.num_images
    if args.mode in ("disttrain", "multiplexing") and images_per_microbatch % ratio:
        raise ValueError(
            f"{args.mode} needs micro_batch_size * num_images ({images_per_microbatch}) divisible by "
            f"vision DP / language DP ({ratio}) for whole, equal image shards"
        )


def apply_recompute(model: Any, policy: str) -> None:
    """Apply one language activation-recompute policy (the vision tower inherits it by default)."""
    if policy == "full":
        model.recompute_granularity = "full"
        model.recompute_method = "uniform"
        model.recompute_num_layers = 1
    elif policy == "selective":
        model.recompute_granularity = "selective"
        model.recompute_method = None
        model.recompute_num_layers = None
        model.recompute_modules = ["core_attn"]
    elif policy == "none":
        model.recompute_granularity = None
        model.recompute_method = None
        model.recompute_num_layers = None
    else:
        raise ValueError(f"recompute policy must be one of {RECOMPUTE_POLICIES}, got {policy!r}")


def apply_vision_recompute(model: Any, policy: str) -> None:
    """Set the vision-tower recompute policy; native, DistTrain and multiplexing all honor it."""
    model.vision_recompute_method = None
    model.vision_recompute_num_layers = None
    model.vision_recompute_modules = None
    if policy == "inherit":
        model.vision_recompute_granularity = "inherit"
    elif policy == "full":
        model.vision_recompute_granularity = "full"
        model.vision_recompute_method = "uniform"
        model.vision_recompute_num_layers = 1
    elif policy == "selective":
        model.vision_recompute_granularity = "selective"
        model.vision_recompute_modules = ["core_attn"]
    elif policy == "none":
        model.vision_recompute_granularity = None
    else:
        raise ValueError(f"vision recompute policy must be one of {VISION_RECOMPUTE_POLICIES}, got {policy!r}")


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build matched workload settings; vary only the selected execution mode."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    layout = parallel_layout(args, world_size)
    validate_workload(args, layout)
    model = AutoBridge.from_hf_pretrained(args.hf_model).to_megatron_provider(load_weights=False)
    model.tensor_model_parallel_size = args.language_tp
    model.pipeline_model_parallel_size = args.language_pp
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.sequence_parallel = args.sequence_parallel
    model.seq_length = args.seq_length
    if args.num_layers is not None:
        model.num_layers = args.num_layers
    model.freeze_language_model = False
    model.freeze_vision_model = False
    model.freeze_vision_projection = False
    apply_recompute(model, args.recompute)
    apply_vision_recompute(model, args.vision_recompute)
    model.cuda_graph_impl = "none"
    model.apply_rope_fusion = False
    # Applied identically to every mode. Disable only where APEX's fused wgrad extension is absent.
    model.gradient_accumulation_fusion = args.gradient_accumulation_fusion
    model.variable_seq_lengths = True
    model.deallocate_pipeline_outputs = False
    if args.mode == "disttrain":
        model.dist_train = DistTrainConfig(
            use_dist_train=True,
            vision_to_llm_dp_ratio=layout["vision_dp"] // layout["language_dp"],
            vision_world_size=layout["vision_gpus"],
            language_world_size=layout["language_gpus"],
            vision_tensor_model_parallel_size=args.encoder_tp,
            vision_pipeline_model_parallel_size=1,
            vision_context_parallel_size=1,
            vision_expert_tensor_parallel_size=1,
            vision_expert_model_parallel_size=1,
        )
    if args.mode == "multiplexing":
        model.multiplexing = MultiplexingConfig(
            use_multiplexing=True,
            encoder_tensor_model_parallel_size=args.encoder_tp,
            encoder_schedule=args.mux_encoder_schedule,
        )
        model.multiplexing.layouts(world_size=world_size, language_tp=args.language_tp, language_pp=args.language_pp)
    optimizer, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=args.train_iters,
        clip_grad=0.0,
    )
    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(
            train_iters=args.train_iters,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            eval_iters=0,
        ),
        validation=ValidationConfig(eval_iters=0),
        optimizer=optimizer,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=MockVLMSFTDatasetConfig(
            seq_length=args.seq_length,
            hf_processor_path=args.hf_model,
            prompt="Describe this image.",
            num_workers=args.num_workers,
            dataloader_type="cyclic",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            pad_to_max_length=True,
            num_images=args.num_images,
            image_size=(args.image_size, args.image_size),
        ),
        logger=LoggerConfig(log_interval=args.log_interval, tensorboard_dir=None, log_progress=False),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(save_interval=0, save=None, load=None, async_save=False),
        rng=RNGConfig(seed=args.seed),
        comm_overlap=CommOverlapConfig(tp_comm_overlap=False, overlap_grad_reduce=False, overlap_param_gather=False),
        mixed_precision="bf16_mixed",
        dist=DistributedInitConfig(use_decentralized_pg=True, use_gloo_process_groups=False),
    )
    logger.info(
        "mode=%s world_size=%d layout=%s global_batch=%d", args.mode, world_size, layout, args.global_batch_size
    )
    return cfg


def patches_per_image(dataset_config: MockVLMSFTDatasetConfig, hf_model: str) -> int:
    """Patch count per image after the training data pipeline's own preprocessing.

    One mock example goes through the dataset's collate function, so model-specific resizing
    (for example Qwen's ``min_pixels``, which upsamples 256x256 inputs to 448x448) is counted
    exactly as in training.
    """
    import numpy
    from transformers import AutoProcessor

    from megatron.bridge.data.builders.mock_vlm_sft import build_mock_vlm_sft_split, make_mock_vlm_example

    processor = AutoProcessor.from_pretrained(hf_model)
    example = make_mock_vlm_example(dataset_config, numpy.random.default_rng(0), "probe")
    split = build_mock_vlm_sft_split(dataset_config, [example], target_length=1, processor=processor)
    grid = split.collate_fn([split[0]])["visual_inputs"].normalized_for_model()["image_grid_thw"]
    counts = grid.prod(dim=-1)
    if counts.numel() != dataset_config.num_images or not torch.all(counts == counts[0]):
        raise ValueError(f"expected {dataset_config.num_images} equally sized mock images, got grids {grid.tolist()}")
    return int(counts[0])


def run_metadata(args: argparse.Namespace, layout: dict[str, int]) -> dict[str, Any]:
    """Record what must stay fixed across compared runs (the measurement contract)."""
    import megatron.core
    import transformer_engine
    import transformers

    import megatron.bridge

    def git(*command: str) -> str:
        try:
            return subprocess.run(
                ["git", *command], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    env_prefixes = ("CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_", "NVTE_", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")
    return {
        "mode": args.mode,
        "label": "benchmark-only: mock image data, random initialization",
        "layout": layout,
        "args": {key: value for key, value in vars(args).items() if key != "throughput_json"},
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "transformer_engine": transformer_engine.__version__,
            "transformers": transformers.__version__,
            "megatron_core": getattr(megatron.core, "__version__", "unknown"),
            "megatron_bridge": getattr(megatron.bridge, "__version__", "unknown"),
        },
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "gpu": torch.cuda.get_device_name(int(os.environ.get("LOCAL_RANK", "0"))),
        "env": {key: value for key, value in os.environ.items() if key.startswith(env_prefixes)},
    }


def main() -> None:
    """Launch the selected backend with the same model/data configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "disttrain", "multiplexing"), required=True)
    parser.add_argument("--hf-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--language-tp", type=int, default=1)
    parser.add_argument("--language-pp", type=int, default=2)
    parser.add_argument("--encoder-tp", type=int, default=1)
    parser.add_argument(
        "--sequence-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="language sequence parallelism (requires --language-tp > 1); part of the layout",
    )
    parser.add_argument("--vision-gpus", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--train-iters", type=int, default=20, help="total steps, including the warmup steps")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--recompute", choices=RECOMPUTE_POLICIES, default="full")
    parser.add_argument(
        "--vision-recompute",
        choices=VISION_RECOMPUTE_POLICIES,
        default="inherit",
        help="vision-tower recompute (every mode); 'selective' checkpoints core_attn, 'inherit' follows --recompute",
    )
    parser.add_argument(
        "--mux-encoder-schedule",
        choices=MUX_ENCODER_SCHEDULES,
        default="deferred",
        help="multiplexing only: 'interleaved' (language PP 1) keeps one micro-batch of encoder activations alive",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="data-loader workers per rank (same for every mode). Multiplexing fetches all micro-batches of "
        "a step before its encoder phase, so one worker stalls it (~0.7 s/step measured) but not native",
    )
    parser.add_argument("--log-interval", type=int, default=1, help="use the same value for every compared mode")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        help="enable the throughput meter: steps excluded before the steady window (window = the rest)",
    )
    parser.add_argument("--throughput-json", help="where global rank 0 writes the throughput record")
    parser.add_argument(
        "--gradient-accumulation-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fused wgrad accumulation; requires APEX fused_weight_gradient_mlp_cuda for MCore linear layers",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = build_config(args)
    callbacks = None
    if args.warmup_iters is not None:
        if not 1 <= args.warmup_iters < args.train_iters:
            raise ValueError("--warmup-iters must be at least 1 and smaller than --train-iters")
        layout = parallel_layout(args, int(os.environ.get("WORLD_SIZE", "1")))
        workload = Workload(
            global_batch_size=args.global_batch_size,
            seq_length=args.seq_length,
            images_per_sample=args.num_images,
            patches_per_image=patches_per_image(cfg.dataset, args.hf_model),
        )
        callbacks = [
            ThroughputMeter(
                warmup_iters=args.warmup_iters,
                measure_iters=args.train_iters - args.warmup_iters,
                workload=workload,
                results_path=args.throughput_json,
                metadata=run_metadata(args, layout),
            )
        ]
    elif args.throughput_json:
        raise ValueError("--throughput-json requires --warmup-iters")
    pretrain(cfg, forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()

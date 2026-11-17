# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One Qwen3-Omni (thinker) mock-data case for native, DistTrain and multiplexing runs.

The three modes run the same model (``--model-size``, defined in ``omni_benchmark.py``), mock data,
batch shape, precision and optimizer, with activation recompute disabled everywhere.
Only the encoder/language placement differs:

* native: vision and audio towers inside the first language pipeline stage (language DP);
* disttrain: towers on ``--vision-gpus`` ranks, the language model on the rest;
* multiplexing: towers on every GPU with encoder DP = WORLD_SIZE, the language model on the same GPUs.
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
from megatron.bridge.models.qwen_omni.qwen3_omni_step import forward_step
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import DistTrainConfig
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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qwen3_vl"))
from omni_benchmark import (  # noqa: E402
    MODEL_SIZES,
    benchmark_hf_config,
    make_dense,
    omni_flops_per_iteration,
    parameter_counts,
    peak_tflops,
)
from throughput import ThroughputMeter, Workload  # noqa: E402


logger = logging.getLogger(__name__)

# Media tokens per item at the fixed media shapes: 448x448 images (784 patches) and 15 s clips (1500 frames).
IMAGE_SIZE = 448
AUDIO_SECONDS = 15.0


def parallel_layout(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    """Validate and describe the encoder and language layouts of the selected mode.

    ``encoder_tp`` is the TP size the towers are sharded over: ``--encoder-tp`` for DistTrain and
    multiplexing; for native training the language TP with Megatron towers, else 1 (Hugging Face towers
    are replicated on the language TP ranks).
    """
    language_model_parallel = args.language_tp * args.language_pp
    megatron_towers = args.encoder_implementation == "megatron"
    if args.mode == "native":
        if args.encoder_tp != 1:
            raise ValueError("native training shards the towers over the language TP; leave --encoder-tp at 1")
        encoder_tp = args.language_tp if megatron_towers else 1
    else:
        encoder_tp = args.encoder_tp
        if encoder_tp > 1 and not megatron_towers:
            raise ValueError("--encoder-tp > 1 needs --encoder-implementation megatron")
    size = MODEL_SIZES[args.model_size]
    for tower, heads in (("vision", size["vision"]["num_heads"]), ("audio", size["audio"]["encoder_attention_heads"])):
        if heads % encoder_tp:
            raise ValueError(f"encoder TP {encoder_tp} does not divide the {heads} {tower} attention heads")
    if args.mode == "disttrain":
        encoder_world = args.vision_gpus
        language_world = world_size - encoder_world
        if encoder_world <= 0 or language_world <= 0:
            raise ValueError("DistTrain requires 0 < --vision-gpus < WORLD_SIZE")
        if language_world % language_model_parallel:
            raise ValueError("DistTrain language GPUs must be divisible by language TP * PP")
        if encoder_world % encoder_tp:
            raise ValueError("DistTrain encoder GPUs must be divisible by --encoder-tp")
        layout = {
            "encoder_gpus": encoder_world,
            "language_gpus": language_world,
            "encoder_dp": encoder_world // encoder_tp,
        }
        layout["language_dp"] = language_world // language_model_parallel
    else:
        if world_size % language_model_parallel:
            raise ValueError("Language TP * PP must divide WORLD_SIZE")
        if world_size % encoder_tp:
            raise ValueError("--encoder-tp must divide WORLD_SIZE")
        language_dp = world_size // language_model_parallel
        # Native runs the towers on the first language stage, with the language DP.
        encoder_dp = world_size // encoder_tp if args.mode == "multiplexing" else language_dp
        layout = {"encoder_gpus": world_size, "language_gpus": world_size, "encoder_dp": encoder_dp}
        layout["language_dp"] = language_dp
    layout.update(encoder_implementation=args.encoder_implementation, encoder_tp=encoder_tp)
    if layout["encoder_dp"] % layout["language_dp"]:
        raise ValueError("encoder DP must be a multiple of language DP")
    if args.sequence_parallel and args.language_tp < 2:
        raise ValueError("--sequence-parallel requires --language-tp > 1")
    first_stage_layers = args.language_first_stage_layers
    if first_stage_layers is not None:
        num_layers = MODEL_SIZES[args.model_size]["llm"]["num_hidden_layers"]
        rest_stages = args.language_pp - 1
        if (
            rest_stages < 1
            or not 1 <= first_stage_layers < num_layers
            or (num_layers - first_stage_layers) % rest_stages
        ):
            raise ValueError(
                f"--language-first-stage-layers {first_stage_layers} needs --language-pp > 1 and the remaining "
                f"{num_layers} - {first_stage_layers} layers divisible by the other {rest_stages} stage(s)"
            )
    if args.mode == "multiplexing" and args.mux_encoder_schedule == "interleaved" and args.language_pp != 1:
        raise ValueError("--mux-encoder-schedule interleaved requires --language-pp 1")
    layout.update(
        language_tp=args.language_tp,
        language_pp=args.language_pp,
        language_first_stage_layers=first_stage_layers,
        sequence_parallel=int(args.sequence_parallel),
        mux_encoder_schedule=args.mux_encoder_schedule if args.mode == "multiplexing" else "n/a",
    )
    return layout


def media_per_sample(args: argparse.Namespace) -> int:
    """Images and audio clips per sample (the same count of each): one of each per 1024 tokens by default."""
    count = args.media_per_sample if args.media_per_sample is not None else args.seq_length // 1024
    if count < 1:
        raise ValueError("each sample needs at least one image and one audio clip")
    return count


def validate_workload(args: argparse.Namespace, layout: dict[str, Any]) -> None:
    """Reject workloads that a mode would split into unequal encoder shards."""
    if args.global_batch_size % (args.micro_batch_size * layout["language_dp"]):
        raise ValueError("global batch size must divide evenly into micro-batch size * language DP")
    ratio = layout["encoder_dp"] // layout["language_dp"]
    items = args.micro_batch_size * media_per_sample(args)
    if args.mode in ("disttrain", "multiplexing") and items % ratio:
        raise ValueError(f"{args.mode} needs images (and clips) per micro-batch ({items}) divisible by {ratio}")


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build matched workload settings; vary only the selected execution mode."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    layout = parallel_layout(args, world_size)
    validate_workload(args, layout)
    hf_config = benchmark_hf_config(args.hf_model, args.model_size)
    model = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    make_dense(model, args.model_size)
    model.tensor_model_parallel_size = args.language_tp
    model.pipeline_model_parallel_size = args.language_pp
    # Uneven split: the first stage also runs the towers in native mode (None keeps the even split).
    model.num_layers_in_first_pipeline_stage = args.language_first_stage_layers
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.expert_model_parallel_size = 1
    model.sequence_parallel = args.sequence_parallel
    model.seq_length = args.seq_length
    model.freeze_language_model = False
    model.freeze_vision_model = False
    model.freeze_audio_model = False
    model.encoder_implementation = args.encoder_implementation
    # No activation recompute anywhere: the language model and both Hugging Face towers.
    model.recompute_granularity = None
    model.recompute_method = None
    model.recompute_num_layers = None
    model.vit_gradient_checkpointing = False
    model.cuda_graph_impl = "none"
    model.apply_rope_fusion = False
    model.gradient_accumulation_fusion = args.gradient_accumulation_fusion
    model.variable_seq_lengths = True
    model.deallocate_pipeline_outputs = False
    model.bf16 = True
    model.fp16 = False
    model.params_dtype = torch.bfloat16
    if args.mode == "disttrain":
        model.dist_train = DistTrainConfig(
            use_dist_train=True,
            vision_to_llm_dp_ratio=layout["encoder_dp"] // layout["language_dp"],
            vision_world_size=layout["encoder_gpus"],
            language_world_size=layout["language_gpus"],
            vision_tensor_model_parallel_size=layout["encoder_tp"],
            vision_pipeline_model_parallel_size=1,
            vision_context_parallel_size=1,
            vision_expert_tensor_parallel_size=1,
            vision_expert_model_parallel_size=1,
        )
    if args.mode == "multiplexing":
        model.multiplexing = MultiplexingConfig(
            use_multiplexing=True,
            encoder_tensor_model_parallel_size=layout["encoder_tp"],
            encoder_schedule=args.mux_encoder_schedule,
        )
        model.multiplexing.layouts(world_size=world_size, language_tp=args.language_tp, language_pp=args.language_pp)
    optimizer, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=args.train_iters,
        clip_grad=0.0,
    )
    media = media_per_sample(args)
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
            prompt="Describe the images and the audio.",
            num_workers=args.num_workers,
            dataloader_type="cyclic",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            pad_to_max_length=True,
            num_images=media,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_audios=media,
            audio_seconds=AUDIO_SECONDS,
            num_base_examples=args.num_base_examples,
        ),
        logger=LoggerConfig(log_interval=args.log_interval, tensorboard_dir=None, log_progress=False),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(save_interval=0, save=None, load=None, async_save=False),
        rng=RNGConfig(seed=args.seed),
        comm_overlap=CommOverlapConfig(tp_comm_overlap=False, overlap_grad_reduce=False, overlap_param_gather=False),
        mixed_precision="bf16_mixed",
        dist=DistributedInitConfig(use_decentralized_pg=True, use_gloo_process_groups=False),
    )
    logger.info("mode=%s world_size=%d layout=%s", args.mode, world_size, layout)
    return cfg


def media_shapes(dataset_config: MockVLMSFTDatasetConfig, hf_model: str) -> tuple[int, int]:
    """Patches per image and mel frames per clip, measured through the training collate path."""
    from transformers import AutoProcessor

    from megatron.bridge.data.builders.mock_vlm_sft import build_mock_vlm_sft_split, make_mock_vlm_examples

    probe = MockVLMSFTDatasetConfig(
        seq_length=dataset_config.seq_length,
        hf_processor_path=hf_model,
        num_images=1,
        image_size=dataset_config.image_size,
        num_audios=1,
        audio_seconds=dataset_config.audio_seconds,
        num_base_examples=1,
        pad_to_max_length=True,
    )
    processor = AutoProcessor.from_pretrained(hf_model)
    split = build_mock_vlm_sft_split(probe, make_mock_vlm_examples(probe), 1, processor)
    batch = split.collate_fn([split[0]])
    patches = int(batch["image_grid_thw"][0].prod())
    frames = int(batch["audio_feature_lengths"][0])
    return patches, frames


def run_metadata(args: argparse.Namespace, layout: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    """Record what must stay fixed across compared runs (the measurement contract)."""
    import megatron.core
    import transformer_engine
    import transformers

    import megatron.bridge

    def git(*command: str) -> str:
        try:
            return subprocess.run(
                ["git", *command], capture_output=True, text=True, check=True, cwd=HERE
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    env_prefixes = ("CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_", "NVTE_", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")
    return {
        "mode": args.mode,
        "model": f"qwen3_omni_benchmark_{args.model_size}",
        "label": "benchmark-only: mock image+audio data, random initialization, no activation recompute",
        "parameters": counts,
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


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of one benchmark case."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("native", "disttrain", "multiplexing"), required=True)
    parser.add_argument("--hf-model", required=True, help="local Qwen3-Omni directory: config, tokenizer, processor")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), required=True, help="vision-audio-LLM size")
    parser.add_argument("--language-tp", type=int, default=1)
    parser.add_argument("--language-pp", type=int, default=1)
    parser.add_argument(
        "--language-first-stage-layers",
        type=int,
        help="LLM layers on the first pipeline stage (uneven split; the rest are divided evenly)",
    )
    parser.add_argument("--sequence-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vision-gpus", type=int, default=4, help="DistTrain encoder (vision+audio) GPUs")
    parser.add_argument(
        "--encoder-implementation",
        choices=("huggingface", "megatron"),
        default="huggingface",
        help="vision/audio towers: Hugging Face modules (replicated over TP) or tensor-parallel Megatron towers",
    )
    parser.add_argument(
        "--encoder-tp",
        type=int,
        default=1,
        help="DistTrain/multiplexing encoder TP (needs megatron towers when > 1); native towers use the language TP",
    )
    parser.add_argument(
        "--mux-encoder-schedule",
        choices=("deferred", "interleaved"),
        default="interleaved",
        help="multiplexing only; interleaved (language PP 1) keeps one micro-batch of encoder activations alive",
    )
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--media-per-sample", type=int, help="images and clips per sample (default seq/1024)")
    parser.add_argument("--train-iters", type=int, default=13, help="total steps, including the warmup steps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-base-examples", type=int, default=32, help="distinct synthetic samples, cycled")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--throughput-json")
    parser.add_argument("--gradient-accumulation-fusion", action=argparse.BooleanOptionalAction, default=True)
    return parser


def unsupported_reason(argv: list[str], world_size: int) -> str | None:
    """Why a case cannot run at all (layout or workload rejected before launch), or None."""
    args = build_parser().parse_args(argv)
    try:
        validate_workload(args, parallel_layout(args, world_size))
    except ValueError as error:
        return str(error)
    return None


def main() -> None:
    """Launch the selected backend with the same model/data configuration."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = build_config(args)
    callbacks = None
    if args.warmup_iters is not None:
        if not 1 <= args.warmup_iters < args.train_iters:
            raise ValueError("--warmup-iters must be at least 1 and smaller than --train-iters")
        layout = parallel_layout(args, int(os.environ.get("WORLD_SIZE", "1")))
        patches, frames = media_shapes(cfg.dataset, args.hf_model)
        media = media_per_sample(args)
        workload = Workload(
            global_batch_size=args.global_batch_size,
            seq_length=args.seq_length,
            images_per_sample=media,
            patches_per_image=patches,
            audios_per_sample=media,
            audio_frames_per_clip=frames,
        )
        gpu = torch.cuda.get_device_name(int(os.environ.get("LOCAL_RANK", "0")))
        callbacks = [
            ThroughputMeter(
                warmup_iters=args.warmup_iters,
                measure_iters=args.train_iters - args.warmup_iters,
                workload=workload,
                results_path=args.throughput_json,
                metadata=run_metadata(args, layout, parameter_counts(cfg.model)),
                flops_fn=omni_flops_per_iteration,
                peak_tflops_per_gpu=peak_tflops(gpu),
            )
        ]
    elif args.throughput_json:
        raise ValueError("--throughput-json requires --warmup-iters")
    pretrain(cfg, forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()

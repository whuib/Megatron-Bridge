# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and validation for the opt-in Qwen3-VL multiplexing runtime."""

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from .layout import ModuleParallelLayout


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def pg_timeout() -> timedelta:
    """Return the timeout used by boundary communication groups."""
    return timedelta(minutes=10)


@dataclass
class MultiplexingConfig:
    """Independent encoder and language layouts sharing the same GPUs.

    Language TP/PP come from the model provider. Encoder PP is one in the
    Qwen3-VL implementation, as in the source project's two-module path.

    ``encoder_schedule`` places the encoder work of a step:

    * ``"deferred"``: every encoder forward, then the language schedule, then every encoder
      backward. Works with any language PP, but keeps the encoder activations of all
      micro-batches alive across the language phase.
    * ``"interleaved"``: encoder forward of micro-batch ``i`` right before its language forward
      and its encoder backward right after its language backward, so at most one micro-batch of
      encoder activations is alive. Requires language PP == 1 (all ranks share one schedule).

    The encoder's activation-recompute policy is the provider's ``vision_recompute_*`` setting,
    which native and DistTrain Qwen3-VL honor in the same way.
    """

    use_multiplexing: bool = False
    encoder_tensor_model_parallel_size: int | None = None
    encoder_schedule: Literal["deferred", "interleaved"] = "deferred"

    def layouts(
        self, *, world_size: int, language_tp: int, language_pp: int
    ) -> tuple[ModuleParallelLayout, ModuleParallelLayout]:
        """Validate divisibility and return encoder and language rank layouts."""
        encoder_tp = self.encoder_tensor_model_parallel_size
        for name, size in (("encoder TP", encoder_tp), ("language TP", language_tp), ("language PP", language_pp)):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"multiplexing {name} must be an explicit positive integer")
        if world_size % encoder_tp or world_size % (language_tp * language_pp):
            raise ValueError("multiplexing TP/PP products must divide world_size")
        encoder = ModuleParallelLayout(
            "encoder",
            world_size,
            tp_size=encoder_tp,
            dp_size=world_size // encoder_tp,
            dimension_order="tp-cp-dp-pp",
        )
        language = ModuleParallelLayout(
            "llm",
            world_size,
            tp_size=language_tp,
            pp_size=language_pp,
            dp_size=world_size // (language_tp * language_pp),
            dimension_order="tp-cp-dp-pp",
        )
        if encoder.dp_size % language.dp_size:
            raise ValueError("Qwen3-VL multiplexing requires encoder DP to be a multiple of language DP")
        return encoder, language

    def validate(self, cfg: "ConfigContainer") -> None:
        """Reject incompatible modes before creating models or process groups."""
        if not self.use_multiplexing:
            return
        model = cfg.model
        if getattr(getattr(model, "dist_train", None), "use_dist_train", False):
            raise ValueError("multiplexing and DistTrain are mutually exclusive")
        if self.encoder_tensor_model_parallel_size is None:
            raise ValueError("Set multiplexing.encoder_tensor_model_parallel_size explicitly")
        if self.encoder_schedule not in ("deferred", "interleaved"):
            raise ValueError(
                f"multiplexing.encoder_schedule must be 'deferred' or 'interleaved', got {self.encoder_schedule!r}"
            )
        if self.encoder_schedule == "interleaved" and model.pipeline_model_parallel_size != 1:
            raise ValueError("multiplexing.encoder_schedule='interleaved' requires language pipeline parallel size 1")
        if not cfg.dist.use_decentralized_pg or cfg.dist.use_gloo_process_groups:
            raise ValueError("multiplexing requires decentralized process groups and use_gloo_process_groups=False")
        unsupported = {
            "context parallelism": model.context_parallel_size != 1,
            "virtual pipeline parallelism": model.virtual_pipeline_model_parallel_size is not None,
            "MoE": bool(model.num_moe_experts),
            "MTP": bool(getattr(model, "mtp_num_layers", None)),
            "CUDA graphs": model.cuda_graph_impl != "none",
            "vision CUDA graphs": getattr(model, "vision_cuda_graph_impl", "none") != "none",
            "FP16/FP8": bool(model.fp16 or model.fp8),
            "frozen encoder/language": bool(
                model.freeze_vision_model or model.freeze_language_model or getattr(model, "freeze_audio_model", False)
            ),
            "TE RNG tracker": bool(model.use_te_rng_tracker),
            "GTP rematerialization": getattr(model, "gtp_weight_remat_size", 1) > 1,
            "per-token schedule normalization": bool(model.calculate_per_token_loss),
            "gradient clipping": cfg.optimizer.clip_grad != 0,
            "overlap": bool(cfg.ddp.overlap_grad_reduce or cfg.ddp.overlap_param_gather),
            "FSDP": bool(cfg.ddp.use_megatron_fsdp or cfg.dist.use_torch_fsdp2),
            "PEFT": cfg.peft is not None,
            "HF weight loading/custom model hooks": not model.perform_initialization
            or bool(getattr(model, "_pre_wrap_hooks", [])),
            "checkpoint save/load": bool(
                cfg.checkpoint.save or cfg.checkpoint.load or cfg.checkpoint.pretrained_checkpoint
            ),
            "evaluation": bool(cfg.validation.eval_iters or cfg.validation.skip_train),
            "restart": bool(cfg.inprocess_restart and cfg.inprocess_restart.enabled),
            "rerun validation": cfg.rerun_state_machine.rerun_mode != "disabled",
            "epoch-based training": cfg.train.num_epochs is not None,
            "sequence packing": bool(getattr(cfg.dataset, "enable_in_batch_packing", False)),
            "rampup batch size": bool(cfg.train.rampup_batch_size),
        }
        rejected = [name for name, enabled in unsupported.items() if enabled]
        if rejected:
            raise ValueError("Qwen3-VL multiplexing does not yet support: " + ", ".join(rejected))

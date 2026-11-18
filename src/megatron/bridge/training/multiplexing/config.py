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

    ``module_state_offload`` keeps every module's parameters, FP32 main gradients and Adam state
    (one module per transformer layer or remaining leaf module, see ``state_offload``) in pinned
    host memory and copies a module's state to the GPU only around its forward, backward and
    optimizer step, in line with the computation (one module GPU resident); each module steps right
    after its last backward of the step. ``module_state_offload_overlap`` additionally prefetches the
    next module and writes the previous one back on a copy stream while the current module computes
    (the current and next modules). ``module_state_offload_resident_param_mib``
    allows some modules' parameters to stay on the GPU throughout training with this overlap schedule.
    The cap is per rank and applies to whole module weight buffers.
    ``module_state_offload_resident_optim_mib`` keeps selected FP32 Adam buffers on the GPU.
    ``module_state_offload_prefetch_mib`` bounds the extra state staged ahead of the previous
    module's write-back and separates readiness for backward and Adam updates.
    ``module_state_offload_writeback_chunk_mib`` defers the tail of an Adam write-back when its
    buffer fits that allowance, allowing the next uploads to go first.
    ``module_state_offload_scoped_handoff`` leaves state copies in flight across local
    encoder/language handoffs. It requires identical TP layouts and non-budgeted overlap.

    ``module_state_offload_budget`` keeps the predicted peak 2.8 GiB under the memory PyTorch can
    use, which leaves a few hundred MiB free after transient workspaces. It offloads a mix of Adam
    state, saved activations and parameters — whole modules for Adam state and parameters, and only
    as many saved tensors as the margin requires (see ``state_budget``). Hidden copies can still
    slow the collectives they overlap on systems where host-device DMA contends with NVLink.
    """

    use_multiplexing: bool = False
    encoder_tensor_model_parallel_size: int | None = None
    encoder_schedule: Literal["deferred", "interleaved"] = "deferred"
    module_state_offload: bool = False
    module_state_offload_overlap: bool = False
    module_state_offload_budget: bool = False
    module_state_offload_resident_param_mib: int = 0
    module_state_offload_resident_optim_mib: int = 0
    module_state_offload_prefetch_mib: int = 0
    module_state_offload_writeback_chunk_mib: int = 0
    module_state_offload_scoped_handoff: bool = False

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
        for name in (
            "module_state_offload_resident_param_mib",
            "module_state_offload_resident_optim_mib",
            "module_state_offload_prefetch_mib",
            "module_state_offload_writeback_chunk_mib",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"multiplexing.{name} must be nonnegative")
            if value and (not self.module_state_offload_overlap or self.module_state_offload_budget):
                raise ValueError(f"multiplexing.{name} requires non-budgeted overlap")
        if (
            self.module_state_offload_writeback_chunk_mib
            and self.module_state_offload_writeback_chunk_mib >= self.module_state_offload_prefetch_mib
        ):
            raise ValueError("multiplexing write-back chunk must be smaller than module_state_offload_prefetch_mib")
        if self.module_state_offload_budget and not self.module_state_offload_overlap:
            raise ValueError("multiplexing.module_state_offload_budget requires module_state_offload_overlap")
        if self.module_state_offload_scoped_handoff and (
            not self.module_state_offload_overlap or self.module_state_offload_budget
        ):
            raise ValueError("multiplexing.module_state_offload_scoped_handoff requires non-budgeted overlap")
        if self.module_state_offload_overlap and not self.module_state_offload:
            raise ValueError("multiplexing.module_state_offload_overlap requires module_state_offload")
        if not self.use_multiplexing:
            if self.module_state_offload:
                raise ValueError("multiplexing.module_state_offload requires use_multiplexing")
            return
        model = cfg.model
        if getattr(getattr(model, "dist_train", None), "use_dist_train", False):
            raise ValueError("multiplexing and DistTrain are mutually exclusive")
        if self.encoder_tensor_model_parallel_size is None:
            raise ValueError("Set multiplexing.encoder_tensor_model_parallel_size explicitly")
        if (
            self.module_state_offload_scoped_handoff
            and self.encoder_tensor_model_parallel_size != model.tensor_model_parallel_size
        ):
            raise ValueError("multiplexing.module_state_offload_scoped_handoff requires identical encoder/language TP")
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
        if self.module_state_offload:
            self.validate_module_state_offload(cfg)

    def validate_module_state_offload(self, cfg: "ConfigContainer") -> None:
        """Reject settings the module state offload does not handle."""
        model, optimizer = cfg.model, cfg.optimizer
        unsupported = {
            "language pipeline parallelism": model.pipeline_model_parallel_size != 1,
            "activation recompute": model.recompute_granularity is not None
            or bool(getattr(model, "vit_gradient_checkpointing", False)),
            "activation CPU offloading": bool(getattr(model, "cpu_offloading", False)),
            "tied embeddings": bool(model.share_embeddings_and_output_weights),
            "non-BF16 parameters": not model.bf16,
            "optimizers other than Adam": optimizer.optimizer != "adam",
            "the non-distributed optimizer": not cfg.ddp.use_distributed_optimizer,
            "the precision-aware optimizer": bool(optimizer.use_precision_aware_optimizer),
            "optimizer CPU offload": bool(optimizer.optimizer_cpu_offload),
            "optimizer CUDA graphs": bool(optimizer.optimizer_cuda_graph),
        }
        rejected = [name for name, enabled in unsupported.items() if enabled]
        if rejected:
            raise ValueError("multiplexing module state offload does not support: " + ", ".join(rejected))

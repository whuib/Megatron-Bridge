# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiplexing training: encoder and language share every GPU with independent layouts.

The ``deferred`` schedule runs three phases (all encoder forwards, language 1F1B, all encoder
backwards); the ``interleaved`` schedule (language PP == 1) wraps each micro-batch's language
forward/backward with its encoder forward/backward.

This runtime owns its two optimizers and module boundaries. No MCore global
parallel state is switched, and no process groups are created inside a step.
"""

import copy
import logging
import os
import time
from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import get_model_config, unwrap_model

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.callbacks import (
    Callback,
    CallbackContext,
    CallbackManager,
    normalize_callbacks,
    should_fire,
)
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.initialize import _create_pg_collection
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.pretrain import _cleanup_after_pretrain_failure, _maybe_destroy_process_group
from megatron.bridge.training.setup import _build_distributed_model, setup
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import _finish_train

from .adapters import adapter_for
from .bridge.input_bridge import InputPartitioner, split_vision_by_images
from .bridge.layout_bridge import LayoutBridge
from .bridge.payload import BoundaryPayload
from .microbatches import collate_microbatches, scale_microbatch_loss
from .rng import EncoderRNG
from .state_budget import BudgetedStateOffload
from .state_offload import ModuleStateOffload


logger = logging.getLogger(__name__)


def _handoff(state_offload: ModuleStateOffload | None = None, *, skip: bool = False) -> None:
    # Drain NCCL streams before changing grids, including under
    # CUDA_DEVICE_MAX_CONNECTIONS=1. This does not create a world barrier.
    # ``skip`` is only set when every boundary is local (see ``skip_local_handoff``): the module
    # collectives are then ordered by the compute stream, so the host may run ahead.
    if skip:
        return
    if state_offload is None:
        torch.cuda.synchronize()
    else:
        state_offload.synchronize_handoff()


def _new_transfer(template: LayoutBridge) -> LayoutBridge:
    # Group/plan objects are immutable after startup. Each microbatch owns
    # independent shape metadata and saved tensors while sharing communicators.
    transfer = copy.copy(template)
    transfer._saved_payload = None
    transfer._recv_payload = None
    transfer._saved_grads = None
    transfer._recv_grads = []
    transfer._split_sizes = []
    transfer._src_metas = []
    transfer._forward_done = False
    transfer._backward_done = False
    transfer.phase_log = []
    return transfer


class _LocalDPSplit:
    """Keep one DP shard of a tensor every rank already holds in full."""

    needs_communication = False

    def __init__(self, layout) -> None:
        self.layout = layout

    def scatter_vision_inputs(self, data: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout.dp_size == 1:
            return data, grid
        chunk = self.layout.local_rank("dp", dist.get_rank())
        return split_vision_by_images(data, grid, self.layout.dp_size, chunk)


def _boundaries_are_local(templates: list[LayoutBridge], partitioner: list) -> bool:
    """Whether no boundary transfer or input redistribution communicates between ranks."""
    return not any(template.needs_cross_grid_sync() for template in templates) and not any(
        getattr(part, "needs_communication", True) for part in partitioner
    )


def _average_audio_data_parallel(encoder: list, group) -> None:
    """Average audio-tower gradients across the audio DP group.

    The encoder DDP group follows the vision layout. An audio tower with its own DP replicas
    computes different clips, so its parameter gradients are averaged here after that DDP step.
    """
    if group is None or dist.get_world_size(group) == 1:
        return
    audio = unwrap_model(encoder[0]).thinker.audio_model
    grads = [param.main_grad for param in audio.parameters() if getattr(param, "main_grad", None) is not None]
    if not grads:
        return
    flat = torch.cat([grad.detach().reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=group)
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad))
        offset += count


def _all_finite(models: list) -> bool:
    finite = torch.ones((), device="cuda", dtype=torch.int32)
    for model in models:
        for parameter in model.parameters():
            grad = getattr(parameter, "main_grad", parameter.grad)
            if grad is not None:
                finite.mul_(torch.isfinite(grad).all().to(torch.int32))
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _encode(
    batch: dict,
    *,
    adapter,
    encoder,
    encoder_config,
    templates,
    partitioner,
    encoder_rng,
    language_module,
    state_offload: ModuleStateOffload | None,
    local_handoff: bool = False,
) -> tuple[list[LayoutBridge], list[torch.Tensor]]:
    """Run one micro-batch's encoder forward and hand every boundary stream to the language ranks."""
    inputs = adapter.encoder_inputs(batch, encoder_config.params_dtype)
    stream_ratios = [max(template.src_layout.dp_size // template.dest_layout.dp_size, 1) for template in templates]
    # The shapes are host metadata, so this needs no device sync.
    valid = inputs is not None and all(
        grid.shape[0] >= ratio for (_, grid), ratio in zip(inputs, stream_ratios, strict=True)
    )
    if templates[0].dest_layout.dp_size > 1:
        # Language DP ranks read different batches and must agree on whether to stop. With language
        # DP 1 every rank reads the same batch, so each reaches the same verdict without a collective.
        valid_tensor = torch.tensor(int(valid), device="cuda", dtype=torch.int32)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
        valid = bool(valid_tensor.item())
    if not valid:
        raise ValueError(
            "multiplexing requires every encoder stream in every micro-batch, with at least one item per encoder shard"
        )
    scattered = []
    for stream_partitioner, (data, grid) in zip(partitioner, inputs, strict=True):
        _handoff(state_offload, skip=local_handoff)
        scattered.append(stream_partitioner.scatter_vision_inputs(data, grid))
    _handoff(state_offload, skip=local_handoff)
    with encoder_rng.activate():
        outputs = adapter.run_encoder(encoder, scattered)
    transfers, payloads = [], []
    for template, tensors in zip(templates, outputs, strict=True):
        payload = BoundaryPayload.from_tensor_list([tensor.detach() for tensor in tensors], -1, tensors[0].shape[0])
        transfer = _new_transfer(template)
        _handoff(state_offload, skip=local_handoff)
        transfer.forward_send(payload)
        received = transfer.forward_recv()
        # Even an identity transfer must cut the autograd edge and expose leaves.
        if received is not None:
            leaves = [tensor.detach().requires_grad_(True) for tensor in received.tensors()]
            received = BoundaryPayload.from_tensor_list(leaves, -1, received.n_merged_tokens)
            transfer._recv_payload = received
        transfers.append(transfer)
        payloads.append(received)
    if language_module is not None:
        if any(payload is None for payload in payloads):
            raise RuntimeError("a language first-stage rank did not receive every encoder stream")
        adapter.deliver(language_module, payloads)
    return transfers, [tensor for tensors in outputs for tensor in tensors]


def _encoder_backward(
    transfers: list[LayoutBridge],
    tensors: list[torch.Tensor],
    *,
    encoder_rng,
    state_offload: ModuleStateOffload | None,
    local_handoff: bool = False,
) -> None:
    """Return one micro-batch's boundary gradients to the encoder ranks and run its backward."""
    encoder_grads = []
    grad_scales = []
    for transfer in transfers:
        received = transfer._recv_payload
        grads = (
            None
            if received is None
            else [torch.zeros_like(tensor) if tensor.grad is None else tensor.grad for tensor in received.tensors()]
        )
        transfer.backward_send(grads)
        returned = transfer.backward_recv()
        # Undo an encoder-DP average. A smaller encoder DP gathers language replicas and does not
        # average those gradients, so the scale stays 1.
        src_dp, dest_dp = transfer.src_layout.dp_size, transfer.dest_layout.dp_size
        scale = src_dp / dest_dp if src_dp % dest_dp == 0 else 1.0
        encoder_grads.extend(returned)
        grad_scales.extend([scale] * len(returned))
    _handoff(state_offload, skip=local_handoff)
    # Compensate the different encoder DP averaging denominator. Under full vision recompute the
    # native encoder returns DeepStack features without autograd history, so (as in the native and
    # DistTrain paths) they contribute no encoder gradient.
    differentiable = [
        (tensor, grad * scale)
        for tensor, grad, scale in zip(tensors, encoder_grads, grad_scales, strict=True)
        if tensor.requires_grad
    ]
    if not differentiable:
        raise RuntimeError("multiplexing encoder produced no differentiable boundary tensor")
    with encoder_rng.activate():
        torch.autograd.backward(*zip(*differentiable))
    _handoff(state_offload, skip=local_handoff)


class _FetchedBatches:
    """Iterate a list that may still be growing: item ``i`` must exist when it is requested.

    ``transform`` maps the batch the encoder consumed to the one the language step reads.
    """

    def __init__(self, batches: list, transform: Callable[[dict], dict] | None = None) -> None:
        self.batches = batches
        self.transform = transform
        self.index = 0

    def __iter__(self) -> "_FetchedBatches":
        return self

    def __next__(self) -> dict:
        if self.index >= len(self.batches):
            raise RuntimeError("the language schedule requested a micro-batch the encoder has not consumed")
        batch = self.batches[self.index]
        self.index += 1
        return batch if self.transform is None else self.transform(batch)


def _step(
    state,
    output,
    adapter,
    encoder,
    encoder_optimizer,
    encoder_pgs,
    templates,
    partitioner,
    encoder_rng,
    state_offload: ModuleStateOffload | None,
) -> tuple[float, int, float]:
    cfg = state.cfg
    language = output.model
    language_config = get_model_config(language[0])
    encoder_config = get_model_config(encoder[0])
    multiplexing = cfg.model.multiplexing
    interleaved = multiplexing.encoder_schedule == "interleaved"
    if state_offload is None:
        for model in (*encoder, *language):
            model.zero_grad_buffer()
        for optimizer in (encoder_optimizer, output.optimizer):
            optimizer.zero_grad()
    else:
        state_offload.begin_step()
    step_tokens = None
    if multiplexing.micro_batch_sizes is not None:
        batches = next(output.train_data_iterator)
        count = len(multiplexing.micro_batch_sizes)
        step_tokens = sum(int(batch["loss_mask"].sum().item()) for batch in batches)
        if step_tokens <= 0:
            raise ValueError("Unequal microbatches require at least one valid loss token per step")
    else:
        count = get_num_microbatches()
        # The deferred schedule needs every batch up front. The interleaved schedule fetches each
        # batch right before encoding it.
        batches = [] if interleaved else [next(output.train_data_iterator) for _ in range(count)]

    raw_language = unwrap_model(language[0])
    language_module = raw_language if raw_language.pre_process else None
    if language_module is not None and adapter.pending(language_module):
        raise RuntimeError("unconsumed multiplexing boundary from the previous iteration")

    # With every boundary local (identical encoder/language layouts), no collective crosses grids:
    # the encoder and language collectives are ordered on the compute stream, so the host may run
    # ahead of the device and overlap its Python work with the previous phase's GPU tail.
    local_handoff = (
        multiplexing.skip_local_handoff
        and state_offload is None
        # CUDA_DEVICE_MAX_CONNECTIONS=1 serializes streams in launch order, where the drain is required.
        and os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1"
        and _boundaries_are_local(templates, partitioner)
    )

    def encode(batch: dict) -> tuple[list[LayoutBridge], list[torch.Tensor]]:
        return _encode(
            batch,
            adapter=adapter,
            encoder=encoder,
            encoder_config=encoder_config,
            templates=templates,
            partitioner=partitioner,
            encoder_rng=encoder_rng,
            language_module=language_module,
            state_offload=state_offload,
            local_handoff=local_handoff,
        )

    def backward(transfers: list[LayoutBridge], tensors: list[torch.Tensor]) -> None:
        _encoder_backward(
            transfers, tensors, encoder_rng=encoder_rng, state_offload=state_offload, local_handoff=local_handoff
        )

    def mark_final_microbatch(iterator: _FetchedBatches) -> None:
        if state_offload is not None and iterator.index == count - 1:
            state_offload.final_microbatch()

    # (per-stream transfers, encoder outputs) per micro-batch whose encoder backward has not run yet.
    encoded: list[tuple[list[LayoutBridge], list[torch.Tensor]]] = []
    try:
        if interleaved:

            def language_step(iterator, model):
                # With PP == 1 every rank runs the same no-pipelining schedule, which finishes the
                # language backward of micro-batch i - 1 before the forward of micro-batch i. Finish
                # i - 1's encoder backward first, so at most one micro-batch of encoder activations
                # is alive, then fetch and encode micro-batch i right before the language consumes it.
                if encoded:
                    backward(*encoded.pop())
                mark_final_microbatch(iterator)
                if step_tokens is None:
                    batches.append(next(output.train_data_iterator))
                encoded.append(encode(batches[iterator.index]))
                _handoff(state_offload, skip=local_handoff)
                result, loss_func = adapter.forward_step(state, iterator, model)
                if step_tokens is not None:
                    loss_func = partial(
                        scale_microbatch_loss, loss_func=loss_func, count=count, step_tokens=step_tokens
                    )
                return result, loss_func

        else:
            for batch in batches:
                encoded.append(encode(batch))
            _handoff(state_offload, skip=local_handoff)

            def language_step(iterator, model):
                mark_final_microbatch(iterator)
                return adapter.forward_step(state, iterator, model)

        # Deferred encoder backward and DP synchronization belong to this runtime.
        language_config.finalize_model_grads_func = None
        schedule = get_forward_backward_func(
            pp_size=output.pg_collection.pp.size(),
            vp_size=None,
        )
        metrics = schedule(
            forward_step_func=language_step,
            # Yields batches[i] when the language step for micro-batch i reads it (interleaved:
            # appended just before); the language consumes exactly what the encoder consumed.
            data_iterator=_FetchedBatches(batches, transform=adapter.language_batch),
            model=language,
            num_microbatches=count,
            seq_length=cfg.model.seq_length,
            micro_batch_size=cfg.train.micro_batch_size,
            forward_only=False,
            p2p_communicator=P2PCommunicator(output.pg_collection.pp, language_config),
            pg_collection=output.pg_collection,
        )
        _handoff(state_offload, skip=local_handoff)
        if language_module is not None and adapter.pending(language_module):
            raise RuntimeError("language schedule did not consume every multiplexing boundary")
        if interleaved and (len(encoded) != 1 or len(batches) != count):
            raise RuntimeError("interleaved multiplexing schedule did not encode every micro-batch")
        # Deferred: reverse micro-batch order. Interleaved: only the last micro-batch remains.
        while encoded:
            backward(*encoded.pop())
        if state_offload is None:
            finalize_model_grads(encoder, pg_collection=encoder_pgs)
            _average_audio_data_parallel(encoder, getattr(encoder_pgs, "audio_dp", None))
            _handoff()
            finalize_model_grads(language, pg_collection=output.pg_collection)
            _handoff()
            # With check_for_nan_in_grad, both DDPs already rejected NaN/Inf grads (fatally) while
            # finalizing above, before either optimizer stepped; avoid a second full-gradient scan.
            if not cfg.ddp.check_for_nan_in_grad and not _all_finite([*encoder, *language]):
                raise FloatingPointError("non-finite multiplexing gradients; neither optimizer was updated")
            squared_norm = 0.0
            for optimizer in (encoder_optimizer, output.optimizer):
                success, grad_norm, _ = optimizer.step()
                if not success:
                    raise RuntimeError("multiplexing optimizer unexpectedly skipped a finite-gradient step")
                if grad_norm is not None:
                    squared_norm += grad_norm * grad_norm
        else:
            squared_norm = state_offload.end_step() ** 2
            _handoff()
        # Telemetry only: the native forward step/schedule owns loss normalization.
        total_tokens = sum(batch["loss_mask"].sum().to(device="cuda", dtype=torch.float32) for batch in batches)
        dist.all_reduce(total_tokens, group=output.pg_collection.dp)
        report = torch.zeros(2, device="cuda")
        if output.pg_collection.pp.rank() == output.pg_collection.pp.size() - 1:
            for item in metrics:
                report += item["lm loss"].to(report)
        dist.all_reduce(report, group=output.pg_collection.dp)
        dist.all_reduce(report, group=output.pg_collection.pp)
        # Both module norms are global; their root-sum-square is the native whole-model norm.
        return (report[0] / report[1]).item(), int(total_tokens.item()), squared_norm**0.5
    finally:
        if language_module is not None:
            adapter.clear(language_module)
        encoded.clear()


def _fire(callback_manager: CallbackManager | None, event: str, **context_fields) -> None:
    if should_fire(callback_manager, event):
        callback_manager.fire(event, CallbackContext(user_state=callback_manager.user_state, **context_fields))


def pretrain_multiplexing(
    config: ConfigContainer,
    forward_step_func: Callable,
    callbacks: list[Callback] | CallbackManager | None = None,
) -> None:
    """Run isolated multiplexing training (Qwen3-VL or Qwen3-Omni) using Bridge setup and data.

    Training callbacks fire with native semantics (``on_train_start``, ``on_train_step_start``,
    ``on_train_step_end`` before the step counter advances, ``on_train_end``). ``model`` holds the
    encoder chunks followed by the language chunks; ``optimizer`` and ``scheduler`` are the
    language module's. Evaluation, test and checkpoint events are not fired (unsupported).
    """
    callback_manager = normalize_callbacks(callbacks)
    adapter = adapter_for(config.model)
    if forward_step_func is not adapter.forward_step:
        raise ValueError(f"multiplexing for {type(config.model).__name__} requires {adapter.forward_step.__module__}")
    cfg = copy.deepcopy(config)
    cfg.model.multiplexing.validate(cfg)
    _, language_layout = cfg.model.multiplexing.layouts(
        world_size=dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1")),
        language_tp=cfg.model.tensor_model_parallel_size,
        language_pp=cfg.model.pipeline_model_parallel_size,
    )
    micro_batch_sizes = cfg.model.multiplexing.micro_batch_sizes
    if micro_batch_sizes is not None:
        if language_layout.dp_size != 1:
            raise ValueError("multiplexing.micro_batch_sizes requires language DP1")
        # Ask the sampler for the raw samples of one optimizer step. Its collate function
        # will split them before creating the multimodal tensors of each microbatch.
        cfg.train.micro_batch_size = cfg.train.global_batch_size
    runtime_config_update(cfg)
    cfg.model.multiplexing.validate(cfg)
    encoder_cfg = copy.deepcopy(cfg)
    encoder_cfg.model = adapter.encoder_provider(cfg.model)
    encoder_cfg.model.add_encoder = True
    encoder_cfg.model.add_decoder = False
    encoder_cfg.model.tensor_model_parallel_size = cfg.model.multiplexing.encoder_tensor_model_parallel_size
    encoder_cfg.model.pipeline_model_parallel_size = 1
    encoder_cfg.model.virtual_pipeline_model_parallel_size = None
    encoder_cfg.model.first_pipeline_num_layers = None
    encoder_cfg.model.last_pipeline_num_layers = None
    encoder_cfg.model.sequence_parallel = False
    encoder_cfg.model.multiplexing.use_multiplexing = False
    adapter.prepare_language(cfg.model)
    state = GlobalState()
    state.cfg = cfg
    owns_world = not dist.is_initialized()
    output = None
    state_offload = None
    dataset_provider = get_dataset_provider(cfg.dataset)
    if micro_batch_sizes is not None:
        original_provider = dataset_provider

        def dataset_provider(*args, **kwargs):
            train, valid, test = original_provider(*args, **kwargs)
            collate = getattr(train, "collate_fn", None)
            if not callable(collate):
                raise ValueError("Unequal microbatches require a train dataset collate function")
            train.collate_fn = partial(collate_microbatches, sizes=micro_batch_sizes, collate=collate)
            return train, valid, test

    try:
        output = setup(state, dataset_provider)
        encoder_layout, language_layout = cfg.model.multiplexing.layouts(
            world_size=dist.get_world_size(),
            language_tp=cfg.model.tensor_model_parallel_size,
            language_pp=cfg.model.pipeline_model_parallel_size,
        )
        encoder_pgs = _create_pg_collection(
            encoder_cfg.model,
            num_distributed_optimizer_instances=1,
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
        )
        # Vision has no language embedding shared across pipeline boundaries.
        encoder_pgs.embd = None
        encoder_pgs.pos_embd = None
        multiplexing = cfg.model.multiplexing
        audio_tp = multiplexing.audio_tensor_model_parallel_size
        separate_audio = audio_tp not in (None, multiplexing.encoder_tensor_model_parallel_size)
        if separate_audio:
            if len(adapter.streams) < 2:
                raise ValueError("audio encoder TP is only used by Qwen3-Omni multiplexing")
            saved_tp = encoder_cfg.model.tensor_model_parallel_size
            encoder_cfg.model.tensor_model_parallel_size = audio_tp
            audio_pgs = _create_pg_collection(
                encoder_cfg.model,
                num_distributed_optimizer_instances=1,
                get_embedding_ranks=None,
                get_position_embedding_ranks=None,
            )
            encoder_cfg.model.tensor_model_parallel_size = saved_tp
            encoder_cfg.model.audio_pg_collection = audio_pgs
            encoder_pgs.audio_dp = audio_pgs.dp
            logger.info(
                "multiplexing audio encoder TP=%d DP=%d (vision encoder TP=%d)",
                audio_tp,
                dist.get_world_size(audio_pgs.dp),
                multiplexing.encoder_tensor_model_parallel_size,
            )
        # Encoder initialization must not perturb the language dropout streams.
        encoder_rng = EncoderRNG(seed=cfg.rng.seed, tp_rank=encoder_pgs.tp.rank())
        with encoder_rng.activate():
            encoder = _build_distributed_model(encoder_cfg, encoder_pgs)
        # Hugging Face towers are created in FP32 and cast to BF16 after the language model and its
        # optimizer already hold memory. Release the cached FP32 copies: NCCL allocates outside
        # PyTorch's caching allocator and could otherwise not create the encoder communicators.
        reserved = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        logger.info(
            "multiplexing encoder built: %.1f GiB allocated, %.1f GiB reserved (%.1f GiB before releasing the cache)",
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
            reserved / 2**30,
        )
        # Encoder DP replicas live on every language PP stage. Make them bit-identical
        # before the optimizer snapshots its main parameters.
        for chunk in encoder:
            chunk.broadcast_params()
        encoder_optimizer, encoder_scheduler = setup_optimizer(
            encoder_cfg.optimizer,
            encoder_cfg.scheduler,
            encoder,
            use_gloo_process_groups=False,
            pg_collection=encoder_pgs,
        )
        multiplexing = cfg.model.multiplexing
        if multiplexing.module_state_offload:
            offloaded = [
                ("encoder", encoder, encoder_optimizer, encoder_pgs),
                ("language", output.model, output.optimizer, output.pg_collection),
            ]
            if multiplexing.module_state_offload_budget:
                # 2.8 GiB under the allocator limit. The output projection allocates about 2.3 GiB that
                # the per-hook profile misses, so the step still has a few hundred MiB free after it.
                state_offload = BudgetedStateOffload.from_megatron(offloaded, headroom_bytes=2800 * 2**20)
            else:
                state_offload = ModuleStateOffload.from_megatron(
                    offloaded,
                    recompute_mlp_layers=multiplexing.module_state_offload_recompute_mlp_layers,
                    overlap=multiplexing.module_state_offload_overlap,
                    resident_param_bytes=multiplexing.module_state_offload_resident_param_mib * 2**20,
                    resident_optim_bytes=multiplexing.module_state_offload_resident_optim_mib * 2**20,
                    prefetch_bytes=multiplexing.module_state_offload_prefetch_mib * 2**20,
                    writeback_chunk_bytes=multiplexing.module_state_offload_writeback_chunk_mib * 2**20,
                    scoped_handoff=multiplexing.module_state_offload_scoped_handoff,
                )
            logger.info(
                "multiplexing module state offloaded: %.1f GiB allocated, %.1f GiB reserved",
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
            )
        # One boundary transfer per encoder output stream (e.g. vision+DeepStack, audio).
        audio_layout = multiplexing.audio_layout(world_size=dist.get_world_size(), language_dp=language_layout.dp_size)
        templates = []
        for index, n_deepstack in enumerate(adapter.streams):
            source = audio_layout if separate_audio and index == len(adapter.streams) - 1 else encoder_layout
            templates.append(
                LayoutBridge(
                    source,
                    language_layout,
                    hidden_size=cfg.model.hidden_size,
                    n_deepstack=n_deepstack,
                    dtype=get_model_config(encoder[0]).params_dtype,
                    replicated_destination_grads=True,
                )
            )
        if multiplexing.module_state_offload_scoped_handoff and any(
            template.needs_cross_grid_sync() for template in templates
        ):
            raise ValueError("Scoped module offload handoffs require local encoder/language transport on every rank")
        partitioner = []
        for template in templates:
            # Language DP 1 replicates the batch on every rank, so an audio DP shard is a local split.
            if language_layout.dp_size == 1 and template.src_layout.dp_size > 1:
                partitioner.append(_LocalDPSplit(template.src_layout))
            else:
                partitioner.append(InputPartitioner(language_layout, template.src_layout))
        models = [*encoder, *output.model]
        context = {"state": state, "model": models, "optimizer": output.optimizer, "scheduler": output.scheduler}
        log_interval = cfg.logger.log_interval
        _fire(callback_manager, "on_train_start", **context)
        for step in range(cfg.train.train_iters):
            # Per-step timing needs device syncs and a collective; only pay them on logged steps.
            log_this_step = (step + 1) % log_interval == 0
            _handoff()
            start = time.perf_counter()
            _fire(callback_manager, "on_train_step_start", **context)
            loss, tokens, grad_norm = _step(
                state,
                output,
                adapter,
                encoder,
                encoder_optimizer,
                encoder_pgs,
                templates,
                partitioner,
                encoder_rng,
                state_offload,
            )
            for scheduler in (output.scheduler, encoder_scheduler):
                scheduler.step(increment=cfg.train.global_batch_size)
            _fire(
                callback_manager,
                "on_train_step_end",
                loss_dict={"lm loss": torch.tensor(loss, device="cuda")},
                grad_norm=grad_norm,
                skipped_iter=False,
                **context,
            )
            state.train_state.step = step + 1
            state.train_state.consumed_train_samples += cfg.train.global_batch_size
            if not log_this_step:
                continue
            _handoff()
            # Slowest rank's wall time and the largest per-rank peak allocation.
            stats = torch.tensor(
                [time.perf_counter() - start, float(torch.cuda.max_memory_allocated())],
                device="cuda",
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                logger.info(
                    "multiplexing iteration=%d loss=%.6f grad_norm=%.4f step_seconds=%.6f valid_tokens=%d "
                    "peak_memory_bytes=%d",
                    step + 1,
                    loss,
                    grad_norm,
                    stats[0].item(),
                    tokens,
                    int(stats[1].item()),
                )
                if state_offload is not None:
                    logger.info(
                        "multiplexing module state offload iteration=%d %s",
                        step + 1,
                        " ".join(f"{name}={value:.2f}" for name, value in state_offload.step_summary().items()),
                    )
        _fire(callback_manager, "on_train_end", **context)
        _finish_train(state, output.checkpoint_manager)
        if state_offload is not None:
            state_offload.close()
    except BaseException as error:
        # Same teardown as native pretrain: peers may be blocked in a collective, so abort.
        logger.exception(
            "Multiplexing pretrain failed on rank=%s with %s: %s; aborting framework-owned process groups",
            dist.get_rank() if dist.is_initialized() else os.environ.get("RANK", "unknown"),
            type(error).__name__,
            error,
        )
        try:
            _cleanup_after_pretrain_failure(state, owns_world)
        except BaseException:
            logger.exception("Unexpected failure while cleaning up after multiplexing pretrain failure")
        raise
    # Release DataLoader workers, then let every rank finish before the world is destroyed: a rank
    # that re-initializes (several trainings in one process) must not race a peer's teardown.
    del output
    _maybe_destroy_process_group(owns_world)

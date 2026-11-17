# Qwen3-VL multiplexing migration

The opt-in implementation lives in `src/megatron/bridge/training/multiplexing/`.
Its layout, input partitioning, activation transport, and reverse gradient transport
are adapted from `megatron_patch/multiplexing` in Pai-Megatron-Patch. There is no
runtime dependency on that checkout. Core source and the DistTrain runtime are
unchanged. Module-local RNG streams and explicit process groups keep encoder
parallelism independent of language parallelism on the same GPUs.

| Source component | Bridge integration |
|---|---|
| `layout.py`, `bridge/input_bridge.py`, `bridge/layout_bridge.py` | Migrated into the independent `training/multiplexing` package |
| Module process groups and RNG | Bridge decentralized process groups and an encoder-local RNG scope |
| Qwen3-VL encoder/language wrappers | Native Bridge Qwen3-VL modules with a parameter-free activation boundary |
| Two-module training loop | Separate three-phase runtime with its own optimizer updates |

This is the Qwen3-VL two-module port. The source project's Qwen3-Omni five-module
extension is outside this case.

## Switches

```python
cfg.model.multiplexing.use_multiplexing = True
cfg.model.multiplexing.encoder_tensor_model_parallel_size = 1
cfg.model.multiplexing.encoder_schedule = "interleaved"  # or "deferred" (default; any language PP)
cfg.dist.use_decentralized_pg = True
cfg.dist.use_gloo_process_groups = False
# Optional, honored by native, DistTrain and multiplexing alike:
cfg.model.vision_recompute_granularity = None  # "inherit" (default), "full", "selective" or None
```

`encoder_schedule` decides where a step's encoder work runs:

| Schedule | Order within a step | Encoder activations alive | Language PP |
|---|---|---|---|
| `deferred` | all encoder forwards → language 1F1B → all encoder backwards | every micro-batch | any |
| `interleaved` | per micro-batch: encoder fwd → language fwd/bwd → encoder bwd | one micro-batch | 1 |

The interleaved schedule hooks the language forward-step function of MCore's
no-pipelining schedule. That schedule finishes micro-batch `i - 1`'s backward before
micro-batch `i`'s forward, and with PP = 1 every rank runs it in lockstep. So the
encoder backward of `i - 1` and the encoder forward of `i` can run, as collectives
on all ranks, right before the language forward of `i`. Bounding encoder activations
to one micro-batch is what lets the encoder drop activation recompute.

`vision_recompute_*` on the Qwen3-VL providers (the same contract as Qwen3.5-VL)
sets the vision tower's recompute policy independently of the language. `"inherit"`
keeps the legacy behavior of following the language policy. The multiplexing encoder
copies the provider, so all three modes read the same knob.

The switch defaults to `False`. With both it and `dist_train.use_dist_train`
disabled, the existing Bridge/MCore training path runs. DistTrain and multiplexing
are mutually exclusive. The multiplexing branch requires
`qwen3_vl_step.forward_step` and owns a separate training loop and two optimizers.
There is no monkey patch of MCore's parallel state or native training schedule.

## Matched workload entry point

In a supported Python 3.12 Bridge environment, run each mode separately:

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/qwen/qwen3_vl/compare_training.py \
  --mode native --num-layers 12 --train-iters 20

uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/qwen/qwen3_vl/compare_training.py \
  --mode disttrain --vision-gpus 4 --num-layers 12 --train-iters 20

uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/qwen/qwen3_vl/compare_training.py \
  --mode multiplexing --encoder-tp 1 --num-layers 12 --train-iters 20
```

The common case uses the same HF architecture, mock image data configuration,
sequence length, global batch, seed, precision, optimizer and recomputation.
`--num-layers 12` explicitly shrinks the language model in **all three** runs;
omit it for the complete HF architecture. `--hf-model` also accepts a local model
directory. This is random-initialized training, not HF weight loading.

`--gradient-accumulation-fusion` (default) needs APEX's `fused_weight_gradient_mlp_cuda`
extension for MCore's non-TE linear layers such as the language output layer. In an
environment without it, pass `--no-gradient-accumulation-fusion`; the flag applies to
all three modes, so keep it identical across runs being compared.

| Default 8-GPU case | Encoder | Language |
|---|---|---|
| Native | On the first language PP stage | TP1 / PP2 / DP4 |
| DistTrain | 4 GPUs, TP1 / PP1 / DP4 | 4 GPUs, TP1 / PP2 / DP2 |
| Multiplexing | All 8 GPUs, TP1 / PP1 / DP8 | Same 8 GPUs, TP1 / PP2 / DP4 |

The global batch stays 32; microbatch counts follow each language DP size.
These are equal-total-GPU comparisons, not equal-language-GPU comparisons.
Keep this distinction when reporting results. For throughput, use the protocol
below rather than per-iteration log lines: the three modes log step time
differently, and no throughput improvement is claimed by this migration.

## Throughput comparison protocol

`compare_training.py --warmup-iters N --throughput-json PATH` attaches one
mode-agnostic callback (`throughput.py`) to all three modes. Multiplexing fires
Bridge's training callbacks with native semantics, so the same code measures
every mode:

- The steady window opens after `N` completed steps and closes at
  `--train-iters`. Both boundaries synchronize the device and then all ranks
  with a global barrier, so the window covers every rank's work, including
  DistTrain's vision ranks and both multiplexing optimizers. Nothing extra
  is synchronized inside the window.
- `step_seconds` = window wall time / measured steps. Samples/s, padded
  tokens/s and images/s follow from the shared workload.
- Model TFLOPS/GPU uses one FLOPs-per-iteration value: Bridge's language
  estimator at the padded length plus its per-image ViT estimator, with patch
  counts from the model's own image processor. Native training logs the same
  estimators. Ratios between modes therefore equal throughput ratios.
- Peak allocated and reserved memory is gathered from every rank (DistTrain's
  vision and language ranks differ widely), together with loss/gradient
  finiteness and the environment: versions, GPU, commit, `NCCL_*`/`NVTE_*`.

`run_throughput_comparison.sh` repeats every mode `REPS` times and rotates the
mode order each repetition, so drift on a shared machine is not attributed to
one mode. `summarize_throughput.py` refuses to compare records whose contract
differs: workload, training features, versions, GPU, commit and window must
match, and only the mode and its parallel layout may differ.

```bash
# Inside the Bridge container; set LAUNCH/SUMMARIZE to another Python launcher elsewhere.
REPS=3 WARMUP_ITERS=10 MEASURE_ITERS=40 LOG_INTERVAL=10 \
WORKLOAD_ARGS="--num-layers 12 --seq-length 8192 --global-batch-size 32 --micro-batch-size 2 \
  --num-images 8 --image-size 256 --recompute full" \
NATIVE_ARGS="--language-tp 1 --language-pp 2" \
DISTTRAIN_ARGS="--language-tp 1 --language-pp 2 --vision-gpus 4 --encoder-tp 1" \
MULTIPLEXING_ARGS="--language-tp 1 --language-pp 2 --encoder-tp 1" \
bash examples/models/qwen/qwen3_vl/run_throughput_comparison.sh
```

What the protocol holds fixed, and why:

| Setting | Value in every mode | Reason |
|---|---|---|
| Model, data, sequence, GBS/MBS, seed, optimizer, BF16 | identical (`WORKLOAD_ARGS`) | same work per step |
| Language recompute (`--recompute full/selective/none`) | identical | recompute changes compute by up to ~1/3 |
| Vision recompute (`--vision-recompute inherit/full/selective/none`) | per-mode tuning knob, recorded in the layout | memory feasibility differs by mode; every mode can use every policy |
| Multiplexing encoder schedule (`--mux-encoder-schedule`) | multiplexing-only knob, recorded in the layout | `interleaved` needs language PP 1 |
| Foreign GPU work | gated: wait for quiet GPUs, monitor during the run, repeat contaminated runs | shared GPUs (`gpu_activity.py`, `GPU_GATE`, `MAX_FOREIGN_SM`) |
| DP grad-reduce / param-gather overlap, TP overlap, CUDA graphs | off | multiplexing does not support them yet |
| Gradient accumulation fusion | identical (needs APEX; `--no-gradient-accumulation-fusion` otherwise) | changes wgrad kernels |
| `--log-interval` | identical; use ≥ 10 | Bridge's log timer adds a global barrier and multiplexing a sync on every logged step |
| Window | ≥ 10 warmup + ≥ 40 measured steps | first steps include allocator and NCCL warmup |

Fairness limits to report alongside the numbers:

- **Layout is part of the result.** Each mode's best layout differs, and
  DistTrain's GPU split is its main tuning knob. Sweep the legal layouts per
  mode (`*_ARGS`) and report both the best-vs-best and the matched-layout
  points. With language PP2 on 8 GPUs and GBS 32 / MBS 2, DistTrain's legal
  splits are 4 vision + 4 language GPUs and 6 vision + 2 language GPUs. The
  latter needs `micro_batch_size * num_images` divisible by 6. Vision DP must
  be a multiple of language DP, and 6 language GPUs give DP3, which does not
  divide 16 micro-batches. Other splits need another language TP/PP.
- **DistTrain needs uniform image shards.** It splits patch rows into equal
  chunks and concatenates sources along the batch dimension, so every chunk
  must hold whole, equally sized images. The script rejects workloads where
  `micro_batch_size * num_images` is not divisible by vision DP / language DP,
  and applies the same rule to multiplexing so both get balanced shards. The
  mock data uses one image size. Variable-resolution data cannot be
  benchmarked with DistTrain as implemented.
- **Mock data, random init.** This is benchmark-only evidence. The VLM
  guidance applies: image-heavy real data can change the encoder/language
  balance that decides between the modes. Vary `--num-images` and
  `--image-size` to cover vision-heavy and language-heavy points.
- **Shared GPUs.** Each run records `nvidia-smi` utilization before it starts.
  Discard runs that overlapped other GPU work.
- **Memory feasibility differs by mode.** On 8x A100 80GB with the default
  workload (12 language layers, sequence 8192, MBS 2, eight 448x448
  images per sample after Qwen's `min_pixels` resize):

  | Recompute | Native | DistTrain 4+4 | Multiplexing |
  |---|---|---|---|
  | `none` | OOM (first stage) | fits, 47.5 GiB | OOM (language 1F1B) |
  | `selective` (`core_attn`) | OOM | fits | OOM |
  | `full` | fits, 30.2 GiB | fits, 35.2 GiB | fits, 36.4 GiB |

  Report the matched point (`full` here) and, separately, each mode's
  least-recompute feasible point. Disaggregation lets DistTrain drop
  recompute. Multiplexing holds every micro-batch's encoder activations until
  its deferred encoder backward. Both effects are properties of the designs,
  not of the harness. Qwen3-VL's vision tower inherits the language recompute
  policy (the Qwen3-VL provider has no separate vision recompute fields).

The native Qwen loss and MCore microbatch normalization are reused. Encoder
gradient seeds are multiplied by DP_encoder / DP_language before encoder DP
averaging. Native Qwen TP input gradients are replicated, so the boundary averages
those replicas rather than summing them twice. The generic migrated transport
retains the original partial-gradient SUM contract as its default.

Encoder DP replicas span every language pipeline stage, whose global seeds differ.
The encoder therefore owns its CPU and CUDA random streams, and its parameters are
broadcast across encoder DP before the optimizer copies them. Without this, the
vision tower's plain PyTorch layers (`patch_embed.proj`, `pos_embed`) would be
initialized differently on each language PP stage.

With full vision recompute (the default `vision_recompute_granularity="inherit"`
under `recompute_granularity="full"`), the native Qwen3-VL vision block returns
DeepStack features without autograd history, so no gradient reaches the DeepStack
mergers in native or DistTrain training. Multiplexing reproduces this: only
differentiable boundary tensors are back-propagated into the encoder. With vision
recompute `None` or `"selective"`, the DeepStack features keep their history and the
mergers train in every mode.

Same seed across different module placements does **not** give identical initial
weights: native builds the vision tower before the first language stage on one RNG
stream. Loss curves from independently initialized runs therefore differ by
initialization, not only by execution mode. Compare convergence only from identical
initial weights and identical logical batches (see Verification).

## Current supported scope

Dense Qwen3-VL, FP32/BF16, image-only batches, encoder PP=1, language TP/PP,
and encoder DP an integer multiple of language DP. Each encoder shard must
contain at least one image. With `encoder_schedule="deferred"`, modules execute in
three phases: all encoder micro-batch forwards, language non-interleaved 1F1B, then
the deferred encoder backwards. With `"interleaved"` (language PP 1), each
micro-batch's encoder forward and backward wrap its language forward and backward.
Activations remain on GPU until their backward completes.

This port does not yet support Omni's five-module path, arbitrary rank mappings,
encoder PP, CP, VPP, MoE, CUDA graphs, FP16/FP8, gradient clipping, communication
overlap, frozen modules, PEFT, HF weight-loading hooks, checkpoint save/resume,
evaluation, or evaluation/test/checkpoint callbacks. Training callbacks
(`on_train_start`, `on_train_step_start`, `on_train_step_end`, `on_train_end`) fire
with native semantics; `model` lists the encoder then the language chunks, and
`optimizer`/`scheduler` are the language module's.
Use `to_megatron_provider(load_weights=False)` for this case.
Rerun validation and epoch-based training are also rejected.
Incompatible configurations fail before training; there is no silent fallback.
The shared example disables unsupported options for all three modes.

## Verification

```bash
uv run python -m pytest tests/unit_tests/training/multiplexing -q \
  --confcutdir=tests/unit_tests/training/multiplexing

uv run python -m pytest tests/unit_tests/examples/test_qwen3_vl_throughput_comparison.py -q

CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest tests/unit_tests/training/multiplexing/test_transport.py -q \
  --confcutdir=tests/unit_tests/training/multiplexing

CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest tests/functional_tests/test_groups/training/test_pretrain_multiplexing.py -q
```

Transport tests load only the portable package, independently of Bridge's model
registration imports. They compare activations, gradients and SGD updates to an
unpartitioned reference, exercise TP and PP boundaries, and test variable image
patch counts and unused DeepStack features. RNG tests check that CPU-initialized
encoder modules are identical across pipeline stages. The functional test runs two
real Qwen3-VL multiplexing steps and requires the complete Bridge dependency stack.

Validation on 2026-11-14 (8x A100 80GB, PyTorch 2.14.1+cu130, Transformer Engine
2.20.2, the pinned Megatron-Core submodule):

- Unit tests: 13 passed; the 4 distributed transport cases passed under 2-GPU NCCL.
- Two-GPU functional smoke (language TP2, encoder TP1/DP2): both cases, deferred
  with inherited vision recompute and interleaved without vision recompute, passed,
  each in its own process. On this host a second training in the same process fails
  in the NCCL re-initialization barrier for native Qwen3-VL as well, so that is an
  environment limit rather than a multiplexing one. The runtime tears down exactly
  like native `pretrain` (`_finish_train`, release the loader, barrier, destroy; abort
  on failure).
- Interleaved vs deferred, same weights and data (4 GPUs, 4 layers, 4 images,
  vision recompute off): iteration-1 loss identical (12.551568), gradient norm
  140.5967 vs 140.5966 (encoder gradient accumulation order); iterations 2–4 within
  0.2%. Peak allocated memory 31.0 GiB vs 52.6 GiB. With 12 images, deferred runs out
  of memory and interleaved peaks at 45.5 GiB.
- Eight-GPU runs of all three modes with the default case above
  (`--num-layers 12 --train-iters 10 --no-gradient-accumulation-fusion`): passed.

Cross-mode parity from identical initial weights: every parameter was re-initialized
from its canonical name and language pipeline stage before the first step (TP=1, so
all modes hold full tensors), and the optimizers reloaded their main parameters.
Native and multiplexing see the same logical batches (both use language DP=4).

Four runs per mode (ranges over runs):

| Iteration | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| Native loss | 13.02582 | 11.2826–11.2833 | 10.680–10.833 | 8.998–9.046 |
| Multiplexing loss | 13.02582 | 11.2823–11.2829 | 10.890–10.960 | 9.064–9.083 |
| Native, `average_in_collective=True` (2 runs) | 13.02582 | 11.2823–11.2838 | 10.785–10.940 | 9.035–9.075 |
| Native, DDP bucket 10M (2 runs) | 13.02582 | 11.2829–11.2836 | 10.829–10.876 | 9.044–9.061 |

Iteration 1 matches exactly in loss, whole-model gradient norm (57.832 vs
57.833) and the first optimizer update of all 450 parameter tensors. Neither
mode is deterministic from run to run. From iteration 5, multiplexing sits
consistently above the baseline native runs (+0.05 at iteration 10). Native
runs whose DDP reduction rounds differently but is mathematically equivalent
land in the same range, so the offset matches what a different reduction
order produces after the iteration-3 loss spike. It is not evidence of a
correctness difference. Data order is identical across modes (same language
DP).

DistTrain from the same weights starts at 12.9267. Its correctness was not
analyzed here (see the throughput protocol for its workload constraints).

Ten-iteration step times from per-iteration logs varied by more than 20% between
repeated runs of the same mode, so they do not support a throughput comparison.
Use the throughput protocol above.

## Layout sweep and multiplexing bottlenecks (2026-11-14)

Screening runs (1 run each, 4–6 measured steps; differences under ~3% are not
resolved) on 8x A100 80GB with 12 images per sample (`layouts_8gpu.txt` makes
every split legal), sequence 8192, GBS 32 / MBS 2, full recompute, no APEX:

| Language depth | Native best | Multiplexing best | DistTrain best |
|---|---|---|---|
| 12 layers | TP1/PP1/DP8 9.61 s | TP1/PP1, encoder DP8 9.73 s | 4+4, TP1/PP1 11.90 s |
| 36 layers (full) | TP2/PP1+SP 19.45 s | TP2/PP1+SP, encoder DP8 19.46 s* | 4+4, TP1/PP2 28.12 s |

\* with 4 loader workers; 20.00 s with 1 worker. DistTrain's 6+2 and 2+6 splits
were 50–80% slower than 4+4, and pure DP8 runs out of memory at full depth in all
modes. Multiplexing beats native at the same language layout whenever native
needs pipeline parallelism (36 layers TP1/PP2: +4.4%; 12 layers PP2 +9%, PP4
+24%), because native places the whole vision tower on the first pipeline stage.

Phase breakdown of multiplexing at full depth, TP2/PP1, on the slowest rank:
language 1F1B 69%, encoder backward 16% (about half is the full-recompute
replay), encoder forward 8%, data-loader stall 4–6% with 1 worker, boundary
transfers 0.2%, DP gradient sync 0.5%, optimizers 0.3%, other host work about
2%. Measured levers:

- Prefetch depth: multiplexing reads every micro-batch of a step before its
  encoder phase. Four workers remove the stall (−5.4% step time). Native is
  unchanged, so `--num-workers` now defaults to 4 for every mode.
- Encoder-only selective recompute (`core_attn`): 12 layers TP2/PP1 10.31 s vs
  native 10.71 s (+3.7%). It does not fit at full depth: the three-phase schedule
  keeps the encoder activations of every micro-batch alive until the deferred
  encoder backward. Both limits are addressed below, and native gets the same knob.

## Vision recompute and the interleaved encoder schedule (2026-11-14)

Two changes followed from the sweep:

1. `vision_recompute_*` on the Qwen3-VL providers sets the vision tower's recompute
   policy independently of the language. Every mode reads it, so it is a fair
   per-mode tuning knob rather than a multiplexing-only advantage.
2. `multiplexing.encoder_schedule="interleaved"` bounds encoder activations to one
   micro-batch (language PP 1), so the multiplexing encoder can drop recompute.

Full-depth screen (1 run, 2 warmup + 4 measured steps; same workload as above, 4
loader workers; foreign GPU work gated and monitored):

| Mode, layout | Vision recompute | Step s | Model TFLOPS/GPU | Peak alloc GiB |
|---|---|---|---|---|
| Native TP2/PP1+SP | full (inherit; earlier sweep) | 19.45 | 94.5 | 48.0 |
| Native TP2/PP1+SP | selective | 19.04 | 96.6 | 59.4 |
| Native TP2/PP1+SP | none | 18.01 | 102.1 | 70.6 |
| Multiplexing TP2/PP1+SP, deferred | full (inherit; earlier sweep) | 19.46 | 94.5 | 52.4 |
| Multiplexing TP2/PP1+SP, deferred | none / selective | OOM | | |
| Multiplexing TP2/PP1+SP, interleaved | full (inherit) | 20.09 | 91.5 | 49.3 |
| Multiplexing TP2/PP1+SP, interleaved | selective | 19.11 | 96.2 | 58.8 |
| Multiplexing TP2/PP1+SP, interleaved | none | 17.97 | 102.3 | 69.8 |
| DistTrain 4+4, TP1/PP2 | none | 28.03 | 65.6 | 68.8 |

Dropping vision recompute saves about 1.45 s per step in both native and
multiplexing. Multiplexing needs the interleaved schedule to fit, and native fits
because it already runs the vision tower per micro-batch. At TP2/PP1 the two modes do
the same work per GPU and stay within noise of each other.

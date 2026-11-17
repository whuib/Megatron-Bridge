# Qwen3-Omni: native vs DistTrain vs multiplexing throughput

This benchmark compares three ways of training the Qwen3-Omni thinker on the same 8 GPUs.

- **Native**: Bridge's Qwen3-Omni model. The vision and audio towers run inside the first language
  pipeline stage with the language DP. These Hugging Face towers cannot be tensor-parallel, so with
  language TP > 1 every TP rank runs them on the same inputs.
- **DistTrain**: the towers run on `--vision-gpus` dedicated GPUs and the language model on the rest.
  Each encoder rank encodes an equal shard of a language micro-batch and sends it through the
  multi-module pipeline communicator.
- **Multiplexing**: the towers and the language model share every GPU with independent layouts. The
  encoders use encoder DP = 8 (TP 1). Each micro-batch's vision (with DeepStack) and audio features
  travel to the language layout over their own boundary streams.

All three run the same model, data, batch, precision and optimizer, with activation recompute
disabled everywhere. Only the encoder/language placement and each framework's parallel layout differ.

Two model sizes (vision + audio + LLM parameters) were measured:

- [`1b-1b-3b`](#model-1b-1b-3b): every framework fits from seq 2048 to 8192, so they can be compared
  point by point.
- [`3b-3b-9b`](#model-3b-3b-9b): memory decides which framework can run at all. No Seq x GBS point
  fits two frameworks.

## Workload and measurement contract (both sizes)

| Setting | Value (identical for every framework) |
|---|---|
| Hardware | 8x A100-SXM4-80GB, one node, NVLink |
| Software | PyTorch 2.14.1+cu130, Transformer Engine 2.20.2, NCCL 2.30.7, pinned Megatron-Core |
| Data | mock conversations: per 1024 tokens of sequence, one 448x448 image (784 patches, 196 LLM tokens) and one 15 s audio clip (1500 mel frames, 195 LLM tokens); padded to the sequence length |
| Batch | micro-batch 1; GBS 16/32/64; sequence 2048–16384 (`1b-1b-3b`) or 1024–16384 (`3b-3b-9b`) |
| Precision / optimizer | BF16 mixed, distributed Adam, FP32 grad reduce, no gradient clipping |
| Recompute | none: language recompute off, `vit_gradient_checkpointing` off |
| Overlap / fusion | no DP/TP overlap, no CUDA graphs, no gradient-accumulation fusion (no APEX) |
| Data loader | 4 workers per rank, 32 distinct samples cycled |
| Window | 3 warmup + 5 measured steps, device sync and global barrier at both ends; one run per point |
| Model FLOPs | Bridge's LLM estimator at the padded length + Bridge's ViT estimator + an audio-encoder count (conv front end, windowed attention, projections); multiply-add = 2, training = 3x forward |
| MFU | model TFLOPS/GPU / 312 (A100 dense BF16) |
| Allocator | PyTorch's default caching allocator (no `expandable_segments`) |
| Shared GPUs | before each run, wait for 30 s without foreign work above 30% SM; during the run, measure foreign SM time per GPU and repeat a run if any GPU averaged more than 1%. `3b-3b-9b` also repeats an OOM when other processes held more than 2 GiB on the failing GPU at the failed allocation |

## Model `1b-1b-3b`

### Model

The architecture (`--model-size 1b-1b-3b`) is built in `omni_benchmark.py` from the local
Qwen3-Omni-30B-A3B-Instruct config.
It keeps that model's tokenizer, processor, tower types and LLM attention shape, and resizes the widths
and depths.

| Component | Configuration | Parameters |
|---|---|---|
| Vision encoder (HF `Qwen3OmniMoeVisionEncoder`) | depth 27, hidden 1536, 16 heads, MLP 6144, DeepStack at layers 8/16/24, merger output 2048 | 0.972B |
| Audio encoder (HF `Qwen3OmniMoeAudioEncoder`) | 34 layers, d_model 1536, 24 heads, FFN 6144, output 2048 | 0.985B |
| LLM (Megatron, **dense**) | 48 layers, hidden 2048, 32 heads / 4 KV groups (head dim 128), SwiGLU 5120, untied 152k vocab | 3.039B |

The released Qwen3-Omni thinker LLM is MoE (30B total / 3B active). Here "3B LLM" is taken as
3B total parameters, so the LLM is dense. The multiplexing runtime does not support MoE yet.

### Layouts

Each framework ran two fixed layouts at every point: its fastest layout at seq 4096 / GBS 32
(from a 9-layout screen) and its fast layout with lower peak memory. If a configuration ran out of
memory it is reported as OOM. No layout, recompute or batch setting was changed to make it fit.

| Framework | Layouts (language TP/PP; DistTrain encoder+language GPUs) |
|---|---|
| Native | TP1/PP1/DP8; TP2/PP1/DP4 + SP |
| Multiplexing | TP1/PP1/DP8, encoder DP8; TP2/PP1/DP4 + SP, encoder DP8 (interleaved encoder schedule) |
| DistTrain | 4+4 with language TP1/PP1/DP4; 2+6 with language TP2/PP3/DP1 + SP |

Layout screen at seq 4096 / GBS 32 (step seconds): native TP1 4.83, TP2 7.28, PP2 6.75;
multiplexing TP1 4.84, TP2 7.36, PP2 6.46; DistTrain 4+4 TP1 7.29, 4+4 TP2 9.19, 2+6 TP2/PP3 8.43.
(The screen used the earlier multiplexing build, before the lazy-fetch fix described below.)

### Results: best layout per framework

Step time, throughput and MFU of each framework's faster surviving layout. "Mux vs" columns are
throughput ratios (> 1: multiplexing is faster).

| Seq | GBS | Native | Multiplexing | DistTrain | Mux vs native | Mux vs DistTrain |
|---|---|---|---|---|---|---|
| 2048 | 16 | 1.40 s, 11.42 samples/s, 22.6% MFU (TP1) | 1.43 s, 11.17 samples/s, 22.1% (TP1) | 1.95 s, 8.22 samples/s, 16.3% (4+4) | 0.98x | 1.36x |
| 2048 | 32 | 2.65 s, 12.08 samples/s, 23.9% (TP1) | 2.77 s, 11.57 samples/s, 22.9% (TP1) | 3.70 s, 8.64 samples/s, 17.1% (4+4) | 0.96x | 1.34x |
| 2048 | 64 | 5.39 s, 11.87 samples/s, 23.5% (TP1) | 6.19 s, 10.34 samples/s, 20.5% (TP1)* | 7.09 s, 9.03 samples/s, 17.9% (4+4) | 0.87x* | 1.15x |
| 4096 | 16 | 2.50 s, 6.40 samples/s, 27.9% (TP1) | 2.54 s, 6.30 samples/s, 27.4% (TP1) | 3.77 s, 4.24 samples/s, 18.5% (4+4) | 0.98x | 1.48x |
| 4096 | 32 | 4.83 s, 6.63 samples/s, 28.9% (TP1) | 4.80 s, 6.67 samples/s, 29.0% (TP1) | 7.28 s, 4.40 samples/s, 19.2% (4+4) | 1.01x | 1.52x |
| 4096 | 64 | 9.62 s, 6.65 samples/s, 29.0% (TP1) | 9.85 s, 6.50 samples/s, 28.3% (TP1) | 14.00 s, 4.57 samples/s, 19.9% (4+4) | 0.98x | 1.42x |
| 8192 | 16 | 7.48 s, 2.14 samples/s, 22.0% (TP2; TP1 OOM) | 6.35 s, 2.52 samples/s, 26.0% (TP2; TP1 OOM) | 8.36 s, 1.91 samples/s, 19.7% (2+6) | **1.18x** | 1.32x |
| 8192 | 32 | 14.19 s, 2.25 samples/s, 23.2% (TP2; TP1 OOM) | 13.39 s, 2.39 samples/s, 24.6% (TP2; TP1 OOM) | 16.35 s, 1.96 samples/s, 20.2% (2+6) | **1.06x** | 1.22x |
| 8192 | 64 | 27.40 s, 2.34 samples/s, 24.1% (TP2; TP1 OOM) | 25.32 s, 2.53 samples/s, 26.0% (TP2; TP1 OOM) | 29.13 s, 2.20 samples/s, 22.6% (2+6) | **1.08x** | 1.15x |
| 16384 | 16, 32, 64 | OOM (both layouts) | OOM (both layouts) | OOM (both layouts) | | |

\* See "Measurement notes": the 5-step mean includes two slow steps. With 20 measured steps:
native 5.23 s, multiplexing 5.60 s (0.93x).

### Results: every run

Step time / MFU / peak allocated memory (max over ranks). "Eager" columns are the multiplexing build
before the lazy-fetch fix and are kept for reference.

| Seq | GBS | nat TP1 | nat TP2 | mux TP1 | mux TP2 | DT 4+4 TP1 | DT 2+6 TP2/PP3 | mux TP1 (eager) | mux TP2 (eager) |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 16 | 1.40 s / 22.6% / 49 GiB | 2.56 s / 12.4% / 38 GiB | 1.43 s / 22.1% / 49 GiB | 2.31 s / 13.7% / 34 GiB | 1.95 s / 16.3% / 36 GiB | 2.98 s / 10.6% / 28 GiB | 1.43 s / 22.2% | 2.32 s / 13.7% |
| 2048 | 32 | 2.65 s / 23.9% / 49 GiB | 5.39 s / 11.8% / 38 GiB | 2.77 s / 22.9% / 49 GiB | 4.86 s / 13.0% / 34 GiB | 3.70 s / 17.1% / 36 GiB | 6.45 s / 9.8% / 29 GiB | 2.79 s / 22.8% | 4.85 s / 13.1% |
| 2048 | 64 | 5.39 s / 23.5% / 49 GiB | 9.22 s / 13.7% / 38 GiB | 6.19 s / 20.5% / 49 GiB | 9.05 s / 14.0% / 34 GiB | 7.09 s / 17.9% / 36 GiB | 12.33 s / 10.3% / 29 GiB | 6.08 s / 20.9% | 9.64 s / 13.1% |
| 4096 | 16 | 2.50 s / 27.9% / 63 GiB | 3.54 s / 19.7% / 47 GiB | 2.54 s / 27.4% / 63 GiB | 3.16 s / 22.1% / 41 GiB | 3.77 s / 18.5% / 47 GiB | 4.07 s / 17.1% / 35 GiB | 2.55 s / 27.4% | 3.21 s / 21.7% |
| 4096 | 32 | 4.83 s / 28.9% / 63 GiB | 7.85 s / 17.8% / 47 GiB | 4.80 s / 29.0% / 63 GiB | 6.87 s / 20.3% / 41 GiB | 7.28 s / 19.2% / 47 GiB | 8.19 s / 17.0% / 35 GiB | 4.93 s / 28.3% | 7.36 s / 18.9% |
| 4096 | 64 | 9.62 s / 29.0% / 63 GiB | 13.98 s / 20.0% / 47 GiB | 9.85 s / 28.3% / 63 GiB | 12.92 s / 21.6% / 41 GiB | 14.00 s / 19.9% / 47 GiB | 15.87 s / 17.6% / 35 GiB | 9.91 s / 28.1% | 13.52 s / 20.6% |
| 8192 | 16 | OOM | 7.48 s / 22.0% / 64 GiB | OOM | 6.35 s / 26.0% / 55 GiB | 9.10 s / 18.1% / 68 GiB | 8.36 s / 19.7% / 48 GiB | OOM | 6.41 s / 25.7% |
| 8192 | 32 | OOM | 14.19 s / 23.2% / 64 GiB | OOM | 13.39 s / 24.6% / 55 GiB | 17.70 s / 18.6% / 68 GiB | 16.35 s / 20.2% / 48 GiB | OOM | 13.14 s / 25.1% |
| 8192 | 64 | OOM | 27.40 s / 24.1% / 64 GiB | OOM | 25.32 s / 26.0% / 55 GiB | 34.66 s / 19.0% / 68 GiB | 29.13 s / 22.6% / 48 GiB | OOM | 27.82 s / 23.7% |
| 16384 | 16 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 32 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 64 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |

### Findings

1. **When TP1/PP1 fits (seq ≤ 4096), native is equal or slightly faster.** With language DP = encoder
   DP = 8, multiplexing has the same per-GPU work as native, so it can at best match native. It is
   1–4% behind at seq 4096 (ahead by 0.6% at 4096/32) and 2–7% behind at seq 2048 (20-step check
   below). The gap is the runtime's per-micro-batch host work, such as device synchronizations,
   validity checks and shape negotiation for two boundary streams. Short steps expose it most.
2. **When the LLM needs TP2 (seq 8192 without recompute), multiplexing wins by 6–18%.** Native TP2
   runs both Hugging Face towers on both ranks of each TP pair. Multiplexing spreads them over all 8
   GPUs (encoder DP 8). The same structure appears at seq ≤ 4096: multiplexing TP2 beats native TP2 by
   2–14%. It also needs less memory: 55 GiB vs 64 GiB at seq 8192 (41 vs 47 GiB at 4096).
3. **DistTrain is the slowest framework at every point**: 1.15–1.52x behind multiplexing's best and
   1.06–1.51x behind native's best (the gap to native narrows at seq 8192, where native needs TP2). Dedicating GPUs to the encoders splits the 8 GPUs into an encoder
   pool and a language pool. A pool idles whenever the other one is the bottleneck. Its 4+4 split
   starves the 3B LLM, and the 2+6 split needs TP2/PP3 (pipeline bubbles at DP 1). At seq 8192
   DistTrain 4+4 TP1 still fits, because its language GPUs hold no encoder activations. It is still
   slower than both TP2 layouts of the other frameworks.
4. **MFU peaks at 29% (seq 4096, TP1)** for native and multiplexing, versus 20% for DistTrain. Seq 2048
   is less efficient (24%), and seq 8192 with TP2 reaches 24–26%.
5. **Without recompute, seq 16384 does not fit in any layout of any framework**, including the
   memory-lighter TP2, 2+6 and multiplexing layouts.

### Measurement notes

- **Lazy fetch fix (multiplexing).** The first grid ran a multiplexing build that fetched every
  micro-batch of a step before the step started. The data loader buffers only 4 workers × 2 batches
  per rank, and Omni's audio and image preprocessing is expensive. At GBS 64 / seq 8192 (16
  micro-batches per rank) the step therefore stalled on data: 27.82 s eager vs 25.17–25.32 s lazy.
  The interleaved schedule now fetches each micro-batch right before encoding it, as native does.
  All multiplexing results above use the fixed build. The deferred schedule (PP > 1) must still fetch
  up front. GPU memory is unaffected.
- **Seq 2048 / GBS 64, TP1.** The multiplexing 5-step mean (6.19 s) includes two slow steps of about
  7.3–7.9 s. Re-measured with 3 warmup + 20 measured steps:

  | Seq / GBS (TP1) | Native | Multiplexing | Median step, native / mux |
  |---|---|---|---|
  | 2048 / 64 | 5.23 s, 24.2% MFU | 5.60 s, 22.6% MFU (−6.7%) | 5.11 s / 5.28 s |
  | 4096 / 64 | 9.45 s, 29.5% MFU | 9.54 s, 29.2% MFU (−1.0%) | 9.33 s / 9.37 s |

  The intermittent slow steps are systematic for multiplexing at short sequences: a few steps of up
  to 9.1 s in 20. Together with about 3% more per-step time, they come from the runtime's
  per-micro-batch host work, which weighs most when steps are short. At TP1 multiplexing therefore
  trails native by 1–7%.
- **Foreign GPU work.** Another user's benchmark processes ran on GPUs 0–3 throughout. Two runs
  exceeded 1% mean foreign SM on some GPU and were repeated (native TP2 and DistTrain 2+6, both at
  8192/64). Native TP2 at 8192/64 measured 32.1 s
  with 4.2% foreign SM, versus 27.40 s on a clean repeat. Its first repeat failed with a transient
  NCCL error and was run again cleanly. Every reported run had less than 1% mean foreign SM on every GPU.
- **One run per point, 5 measured steps.** Differences below about 3% are not resolved.
- **Benchmark-only data**: mock image and audio inputs with random initialization. DistTrain's
  numerical correctness is not evaluated here.

## Model `3b-3b-9b`

### Model

`--model-size 3b-3b-9b`, built from the same Qwen3-Omni config. Only the widths and depths change.

| Component | Configuration | Parameters |
|---|---|---|
| Vision encoder (HF `Qwen3OmniMoeVisionEncoder`) | depth 27, hidden 2688, 21 heads (head dim 128), MLP 10752, DeepStack at layers 8/16/24, merger output 4096 | 2.991B |
| Audio encoder (HF `Qwen3OmniMoeAudioEncoder`) | 38 layers, d_model 2560, 20 heads, FFN 10240, output 4096 | 3.031B |
| LLM (Megatron, **dense**) | 48 layers, hidden 4096, 32 heads / 8 KV groups (head dim 128), SwiGLU 9728, untied 152k vocab | 8.997B |

The LLM uses 8 KV groups so that it can also run with TP 8. The workload, precision, optimizer and
measurement are the contract above. Activation recompute is off everywhere.

### Method: candidate layouts at every point

With 15B parameters on 8 GPUs, which layouts fit depends on the sequence length. Fixed layouts per
framework would therefore report OOM where another layout of the same framework runs. Instead, every
framework ran a set of candidate layouts at every point, and the fastest measured layout is reported:

| Framework | Candidates (language TP/PP; DistTrain encoder+language GPUs) |
|---|---|
| Native | TP1; TP2+SP; TP4+SP; PP2; PP2 with 4 or 6 LLM layers on the first stage (`nat_p2f4`, `nat_p2f6`); TP2/PP2+SP with 4 on the first stage |
| Multiplexing (encoder DP 8) | TP1; TP2+SP; TP4+SP; TP8+SP (interleaved schedule); TP2/PP2+SP (deferred schedule) |
| DistTrain | 4+4 with TP1, TP2+SP, TP4+SP or TP2/PP2+SP; 2+6 with TP1/PP3 or TP2/PP3+SP |

The uneven native splits put the towers on a first stage with few LLM layers. They are ordinary
Megatron Core settings (`num_layers_in_first_pipeline_stage`). Each case has one of four outcomes:

- **measured**: step time, throughput and MFU (3 warmup + 5 measured steps);
- **OOM**: ran out of memory. The table shows the GiB allocated by PyTorch and the GiB held outside
  PyTorch's allocator (CUDA context, NCCL buffers) at the failed allocation. Two native layouts ran
  out of memory inside NCCL's own allocation (`Cuda failure 2 'out of memory'`);
- **OOM\***: not launched, because the case ran out of memory at a point whose sequence length and
  global batch are both no larger. With micro-batch 1, per-rank memory does not decrease with either;
- **unsupported**: rejected before launch. DistTrain needs an equal number of images (and clips) per
  encoder rank. Multiplexing needs at least one per encoder rank: language DP 2 (TP4) spreads each
  micro-batch over 4 encoder ranks, and language DP 1 (TP8) over 8. With one image and one clip per
  1024 tokens, TP4 needs seq ≥ 4096 and TP8 needs seq 8192.

### Results: best layout per framework

| Seq | GBS | Native | Multiplexing | DistTrain |
|---|---|---|---|---|
| 1024 | 16 | 2.19 s, 7.30 samples/s, 20.1% MFU (`nat_p2f6`) | OOM | OOM |
| 1024 | 32 | 3.79 s, 8.45 samples/s, 23.3% (`nat_p2f6`) | OOM | OOM |
| 1024 | 64 | 7.13 s, 8.98 samples/s, 24.7% (`nat_p2f6`) | OOM | OOM |
| 2048 | 16 | OOM | OOM | 4.76 s, 3.36 samples/s, 18.8% (4+4 TP2) |
| 2048 | 32 | OOM | OOM | 8.73 s, 3.67 samples/s, 20.6% (4+4 TP2) |
| 2048 | 64 | OOM | OOM | 16.35 s, 3.91 samples/s, 21.9% (4+4 TP2) |
| 4096 | 16 | OOM | OOM | 8.26 s, 1.94 samples/s, 22.5% (4+4 TP2/PP2) |
| 4096 | 32 | OOM | OOM | 15.02 s, 2.13 samples/s, 24.7% (4+4 TP2) |
| 4096 | 64 | OOM | OOM | 29.24 s, 2.19 samples/s, 25.4% (4+4 TP2) |
| 8192 | 16 | OOM | OOM | 16.86 s, 0.95 samples/s, 23.5% (4+4 TP2/PP2) |
| 8192 | 32 | OOM | OOM | 32.40 s, 0.99 samples/s, 24.5% (4+4 TP2/PP2) |
| 8192 | 64 | OOM | OOM | 62.98 s, 1.02 samples/s, 25.2% (4+4 TP2/PP2) |
| 16384 | 16, 32, 64 | OOM | OOM | OOM |

"OOM" in this table means every candidate of the framework was OOM, OOM\* or unsupported there.

### Results: every candidate

GBS 16 / 32 / 64 step time and the MFU range; OOM cells show GiB allocated by PyTorch + GiB outside
PyTorch at the failed allocation.

| Case | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| nat_t1 | **OOM** (56.0 + 1.1 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| nat_t2 | **OOM** (39.2 + 1.1 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| nat_t4 | **OOM** (NCCL) | OOM\* | OOM\* | OOM\* | OOM\* |
| nat_p2 | **OOM** (39.2 + 1.1 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| nat_p2f4 | **OOM** (NCCL) | OOM\* | OOM\* | OOM\* | OOM\* |
| nat_p2f6 | 2.19 / 3.79 / 7.13 s (20.1–24.7%) | **OOM** (75.4 + 3.1 GiB) | OOM\* | OOM\* | OOM\* |
| nat_t2p2f4 | **OOM** (75.3 + 3.6 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| mux_t1 | **OOM** (75.7 + 1.1 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| mux_t2 | unsupported | **OOM** (51.8 + 1.1 GiB) | OOM\* | OOM\* | OOM\* |
| mux_t4 | unsupported | unsupported | **OOM** (69.5 + 8.6 GiB) | OOM\* | OOM\* |
| mux_t8 | unsupported | unsupported | unsupported | **OOM** (65.7 + 13.1 GiB) | OOM\* |
| mux_t2p2 (deferred) | unsupported | unsupported | **OOM** (70.6 + 7.7 GiB) | OOM\* | OOM\* |
| dt44_t1 | **OOM** (73.1 + 3.0 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| dt44_t2 | unsupported | 4.76 / 8.73 / 16.35 s (18.8–21.9%) | 8.30 / 15.02 / 29.24 s (22.4–25.4%) | **OOM** (72.4 + 4.5 GiB) | OOM\* |
| dt44_t4 | unsupported | unsupported | 10.29 / 19.20 / 37.50 s (18.0–19.8%) | 19.61 / 37.88 / 75.02 s (20.2–21.2%) | **OOM** (69.9 + 4.7 GiB) |
| dt44_t2p2 | unsupported | unsupported | 8.26 / 15.56 / 29.79 s (22.5–24.9%) | 16.86 / 32.40 / 62.98 s (23.5–25.2%) | **OOM** (74.8 + 3.4 GiB) |
| dt26_t1p3 | **OOM** (74.7 + 3.4 GiB) | OOM\* | OOM\* | OOM\* | OOM\* |
| dt26_t2p3 | unsupported | **OOM** (74.7 + 3.4 GiB) | OOM\* | OOM\* | OOM\* |

Every measured OOM was at GBS 16. An A100-80GB exposes 79.2 GiB. OOMs at about 1 GiB outside
PyTorch failed during setup or the first allocation of a large buffer, before most communicators
existed.

Peak allocated memory of the measured runs: native `nat_p2f6` 71.6 GiB; DistTrain 4+4 TP2 58.9 GiB
(seq 2048) and 67.3 GiB (4096); TP4 56.3 / 62.0 GiB and TP2/PP2 59.1 / 68.4 GiB (seq 4096 / 8192).

### Why the layouts run out of memory

Per-rank model state is BF16 parameters (2 B) + FP32 gradient buffer (4 B) + distributed Adam (12 B
over DP), on the most loaded rank:

| Framework | Layout | Model state (GiB) |
|---|---|---|
| Native | TP1 / TP2 / TP4 / PP2 (even) | 105 / 88 / 92 / 88 |
| Native | PP2 with 6 (4) LLM layers on the towers' stage | 64 and 62 (61 and 65) |
| Native | TP2/PP2 with 4 layers on the towers' stage | 74 |
| Multiplexing | TP1 / TP2 / TP4 / TP8 / TP2+PP2 | 105 / 80 / 67 / 61 / 67 |
| DistTrain 4+4 | encoder GPUs; language GPUs at TP1 / TP2 / TP4 / TP2+PP2 | 51; 75 / 50 / 38 / 38 |
| DistTrain 2+6 | encoder GPUs; language GPUs at TP1+PP3 / TP2+PP3 | 67; 36 / 27 |

The 6.0B tower parameters are the deciding term. Native keeps them on every TP rank of the first
stage, with their optimizer state sharded only over the language DP. Multiplexing shards their
optimizer state over all 8 GPUs (45 GiB per rank), and DistTrain over its encoder GPUs.

Memory outside PyTorch's allocator differs by framework. In the in-training OOMs, native and
DistTrain held 3.1–4.7 GiB, and multiplexing 7.7 GiB (TP2/PP2), 8.6 GiB (TP4) and 13.1 GiB (TP8).
In the measured native and DistTrain runs, device memory used minus PyTorch's reserved memory was
3.2–5.7 GiB per rank (the upper end includes another user's 0.5 GiB process on GPUs 4–7).

### Findings

1. **No Seq x GBS point fits two frameworks.** Native fits only at seq 1024 and DistTrain only from
   seq 2048 to 8192. Multiplexing fits nowhere, and nothing fits at seq 16384. Without activation
   recompute, this model does not fit on 8 A100-80GB GPUs in a way that allows a matched comparison.
2. **Native fits only by moving the towers to their own pipeline stage.** Every even layout needs
   at least 88 GiB of model state, because the towers are replicated on TP ranks. PP2 with only 6 LLM
   layers next to the towers balances the two stages (64 / 62 GiB). It peaks at 71.6 GiB at seq 1024
   (20.1–24.7% MFU) and runs out of memory at seq 2048.
3. **Multiplexing has the smallest model state (61 GiB at TP8, 67 GiB at TP4), but no point that
   runs.** Two limits combine:
   - Its encoder sharding is per image or clip, and every encoder rank needs one. This rules out TP4
     below seq 4096 and TP8 below seq 8192.
   - Where those layouts are valid, they run out of memory with 7.7–13.1 GiB outside PyTorch, against
     3.1–4.7 GiB for the other frameworks. With native's overhead, TP4 at seq 4096 (69.5 GiB allocated
     when it failed) would have had 4–5 GiB more, and TP8 at seq 8192 (65.7 GiB) 8–10 GiB more.
     Whether that is enough was not tested.

   A likely source of the extra memory is that the runtime creates its own NCCL communicators for
   each boundary: the input partitioner, the vision stream and the audio stream, plus a second set of
   process groups for the encoder. This was not confirmed with an allocation trace.
4. **DistTrain runs from seq 2048 to 8192 at 18.8–25.4% MFU.** Its language GPUs hold no tower state,
   so language TP2 (DP 2) is its fastest layout up to seq 4096 (tied with TP2/PP2 at 4096/16). At
   seq 8192 TP2 runs out of memory and TP2/PP2 is fastest. MFU grows with GBS. Below seq 2048 its equal-shard rule leaves only 2+6 TP1/PP3,
   whose two encoder GPUs (towers at DP 2) run out of memory.

### Measurement notes

- **Multiplexing setup fix.** The first multiplexing smoke run failed before its first step with an
  NCCL allocation error (`Failed to CUDA calloc async 136 bytes`). The runtime builds the encoder
  after the language model and its optimizer. The Hugging Face towers are created in FP32 and cast to
  BF16, and the freed FP32 copies stayed in PyTorch's cache: 77.6 GiB reserved, 50.4 GiB allocated.
  NCCL allocates outside that cache, so it found the GPU full. The runtime now releases the cache
  after building the encoder. Steady-state memory is unchanged, and the multiplexing OOMs above
  happen in training, after that point.
- **OOMs inside NCCL.** Native TP4 and native PP2 with 4 first-stage layers failed in NCCL
  collectives. TP4 was repeated with `NCCL_DEBUG=WARN` (logging only), and NCCL reported
  `ncclCuMemAlloc ... Cuda failure 2 'out of memory'` on the failing ranks.
- **Shared GPUs.** Another user's solver campaign and benchmark processes ran on GPUs 0–6. Two runs
  exceeded 1% mean foreign SM and were re-measured once the GPUs were quiet:
  - native at 1024/32: 3.82 s and 3.81 s with 2.4–3.3% foreign SM, versus 3.79 s clean;
  - DistTrain at 2048/64: 16.76 s and 20.21 s with 7.9–9.9% foreign SM, versus 16.35 s clean.

  Every reported run had at most 0.43% mean foreign SM. One OOM (native `nat_p2f6` at 2048/16) first
  happened while another process held 3.4 GiB. It was repeated, and the repeat ran out of memory with
  0.5 GiB of foreign memory on the failing GPU.
- **One run per point, 5 measured steps.** Differences below about 3% are not resolved, for example
  DistTrain TP2 vs TP2/PP2 at seq 4096.

## Reproduce

`1b-1b-3b`:

```bash
HF_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct   # config, tokenizer and processor only
cat > cases.txt <<'EOF'
nat_t1p1 native --language-tp 1 --language-pp 1
nat_t2p1 native --language-tp 2 --language-pp 1 --sequence-parallel
mux_t1p1 multiplexing --language-tp 1 --language-pp 1 --mux-encoder-schedule interleaved
mux_t2p1 multiplexing --language-tp 2 --language-pp 1 --sequence-parallel --mux-encoder-schedule interleaved
dt44_t1p1 disttrain --vision-gpus 4 --language-tp 1 --language-pp 1
dt26_t2p3 disttrain --vision-gpus 2 --language-tp 2 --language-pp 3 --sequence-parallel
EOF
HF_MODEL=$HF_MODEL MODEL_SIZE=1b-1b-3b CASES_FILE=cases.txt SEQS="2048 4096 8192 16384" GBSS="16 32 64" \
REPS=1 WARMUP_ITERS=3 MEASURE_ITERS=5 LOG_INTERVAL=4 QUIET_THRESHOLD=30 \
EXTRA_WORKLOAD_ARGS="--no-gradient-accumulation-fusion" \
bash examples/models/qwen/qwen3_omni/run_omni_grid.sh
```

`3b-3b-9b` (the candidate layouts above; `NCCL_DEBUG=WARN` only adds NCCL's error details to the logs):

```bash
cat > cases_9b.txt <<'EOF'
nat_t1 native --language-tp 1 --language-pp 1
nat_t2 native --language-tp 2 --language-pp 1 --sequence-parallel
nat_t4 native --language-tp 4 --language-pp 1 --sequence-parallel
nat_p2 native --language-tp 1 --language-pp 2
nat_p2f4 native --language-tp 1 --language-pp 2 --language-first-stage-layers 4
nat_p2f6 native --language-tp 1 --language-pp 2 --language-first-stage-layers 6
nat_t2p2f4 native --language-tp 2 --language-pp 2 --sequence-parallel --language-first-stage-layers 4
mux_t1 multiplexing --language-tp 1 --language-pp 1 --mux-encoder-schedule interleaved
mux_t2 multiplexing --language-tp 2 --language-pp 1 --sequence-parallel --mux-encoder-schedule interleaved
mux_t4 multiplexing --language-tp 4 --language-pp 1 --sequence-parallel --mux-encoder-schedule interleaved
mux_t8 multiplexing --language-tp 8 --language-pp 1 --sequence-parallel --mux-encoder-schedule interleaved
mux_t2p2 multiplexing --language-tp 2 --language-pp 2 --sequence-parallel --mux-encoder-schedule deferred
dt44_t1 disttrain --vision-gpus 4 --language-tp 1 --language-pp 1
dt44_t2 disttrain --vision-gpus 4 --language-tp 2 --language-pp 1 --sequence-parallel
dt44_t4 disttrain --vision-gpus 4 --language-tp 4 --language-pp 1 --sequence-parallel
dt44_t2p2 disttrain --vision-gpus 4 --language-tp 2 --language-pp 2 --sequence-parallel
dt26_t1p3 disttrain --vision-gpus 2 --language-tp 1 --language-pp 3
dt26_t2p3 disttrain --vision-gpus 2 --language-tp 2 --language-pp 3 --sequence-parallel
EOF
NCCL_DEBUG=WARN HF_MODEL=$HF_MODEL MODEL_SIZE=3b-3b-9b CASES_FILE=cases_9b.txt \
SEQS="1024 2048 4096 8192 16384" GBSS="16 32 64" \
REPS=1 WARMUP_ITERS=3 MEASURE_ITERS=5 LOG_INTERVAL=4 QUIET_THRESHOLD=30 QUIET_WINDOW=30 \
EXTRA_WORKLOAD_ARGS="--no-gradient-accumulation-fusion" \
bash examples/models/qwen/qwen3_omni/run_omni_grid.sh
```

`run_omni_grid.sh` drives `compare_training.py` through the shared
`examples/models/qwen/qwen3_vl/run_throughput_comparison.sh` runner, which provides the GPU gate and
interference monitor. Before each point, `plan_omni_grid.py` leaves an `unsupported` or `oom_inferred`
marker for cases it does not launch. It also skips cases that already finished at that point, so the
same command resumes an interrupted grid. `summarize_omni_grid.py` writes the per-point table to
`grid_summary.json`, with OOM memory details and failure rows.

## Implementation summary

- **Qwen3-Omni model** (`modeling_qwen3_omni/thinker_model.py`, `model.py`, `qwen3_omni_provider.py`):
  - `add_encoder`/`add_decoder` are honored: encoder-only replicas build only the towers, and a
    decoder-only first stage consumes external features.
  - DistTrain support: equal image/clip shards per encoder rank, and pack/unpack of vision, DeepStack
    and audio features in shard order.
  - `encode_multimodal()` for encoder replicas.
  - Several audio clips per sample are now supported (the per-clip trimming used to require one clip
    per sample).
  - `pg_collection` is exposed on the model wrapper, which training with decentralized process
    groups requires.
- **Multiplexing** (`training/multiplexing/adapters.py`, `runtime.py`):
  - Model adapters for Qwen3-VL (one vision stream) and Qwen3-Omni (vision + audio streams). Audio
    clips are partitioned item-wise like images.
  - Lazy micro-batch fetching in the interleaved schedule.
  - Teardown matches native `pretrain`.
  - PyTorch's cache is released after the encoder is built, so NCCL can allocate its communicators.
- **Mock data** (`MockVLMSFTDatasetConfig`): optional white-noise audio clips (`num_audios`,
  `audio_seconds`).
- **Benchmark driver** (`compare_training.py`, `omni_benchmark.py`): `--model-size` (`1b-1b-3b`,
  `3b-3b-9b`), and `--language-first-stage-layers` for an uneven first pipeline stage.
- **Grid tooling**:
  - `plan_omni_grid.py`: rejects unsupported cases before launch, infers OOM, resumes.
  - `summarize_omni_grid.py`: OOM memory details; NCCL allocation failures count as OOM.
  - The runner repeats an OOM when other processes held more than 2 GiB on the failing GPU
    (`gpu_activity.py oom-foreign`).
- **Measurement** (`qwen3_vl/throughput.py`): pluggable FLOPs function and MFU, plus per-rank device
  memory and memory outside PyTorch. `omni_benchmark.py` provides the models and the audio FLOPs.

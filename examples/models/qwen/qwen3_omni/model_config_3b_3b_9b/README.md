# Qwen3-Omni benchmark configuration: 3B vision + 3B audio + 9B dense LLM

Configuration export, 2026-11-16. Copy this directory into the other repository.
The experiment trains the **thinker only**, with random initialization and mock
image/audio inputs. No weights are included.

| File | Purpose |
|---|---|
| `config.json` | Hugging Face configuration matching the architecture documented in `../README.benchmark.md`: 21 vision heads and 20 audio heads |
| `config.current.json` | Configuration matching the current `../omni_benchmark.py`: 24 vision heads and 32 audio heads, permitting encoder TP up to 8 |
| `megatron_model_config.json` | Explicit Megatron model settings, including the dense LLM overrides; shared by both variants |
| `experiment.json` | Provenance, parameter counts, workload, and compatibility notes |

The two variants have the same parameter counts but different attention head
dimensions. Choose the variant explicitly when comparing results.

| Component | Architecture | Parameters |
|---|---|---:|
| Vision | 27 layers, hidden 2688, FFN 10752, DeepStack layers 8/16/24, output 4096 | 2,990,991,616 |
| Audio | 38 layers, hidden 2560, FFN 10240, output 4096 | 3,030,571,136 |
| Dense LLM | 48 layers, hidden 4096, FFN 9728, 32 query heads, 8 KV heads, head dimension 128 | 8,997,191,680 |
| Total thinker | Includes tower projectors and untied LLM embeddings/output | 15,018,754,432 |

Counts are calculated from module parameter shapes. The LLM vocabulary is
152064, and the embedding and output matrices are separate. Vision uses
16x16 patches, temporal patch size 2, spatial merge size 2, and a learned
2304-position embedding table. Audio uses 128 mel bins and the original
convolutional frontend (480 channels, three stride-2 convolutions).

## Load in another repository

Use a Transformers installation with Qwen3-Omni support. Configuration loading
does not require Megatron Bridge or a model download:

```python
import json
from pathlib import Path

from transformers import Qwen3OmniMoeConfig

export_dir = Path("model_config_3b_3b_9b")
# Select config.current.json here to match the current benchmark script.
payload = json.loads((export_dir / "config.json").read_text())
config = Qwen3OmniMoeConfig.from_dict(payload)
thinker = config.thinker_config
vision_config = thinker.vision_config
audio_config = thinker.audio_config
llm_config = thinker.text_config
```

To instantiate the HF thinker on the meta device for shape inspection without
allocating its weights:

```python
import torch
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

with torch.device("meta"):
    model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker)
```

Initialize real weights through the target repository's training/model builder
when running the model. Reuse the tokenizer and processor from
`Qwen/Qwen3-Omni-30B-A3B-Instruct`; they are not bundled here.

## Preserve the effective benchmark model

The HF `model_type` retains `moe` for compatibility with Qwen3-Omni classes.
`text_config.mlp_only_layers` lists **every layer, 0 through 47**, selecting a
dense SwiGLU MLP throughout. The inherited expert fields are inactive in the HF
model. When adapting to another runtime, explicitly disable experts.

Bridge's conversion also needs the following overrides because its provider
reads the inherited expert fields independently of `mlp_only_layers`:

```python
from megatron.bridge import AutoBridge

provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
provider.num_moe_experts = None
provider.moe_ffn_hidden_size = None
provider.ffn_hidden_size = 9728
```

`megatron_model_config.json` records the effective model settings for porting.
It is a settings reference, not a serialized `ConfigContainer`: convert the
`silu` activation name to the target runtime's callable and `bfloat16` dtype
names to its dtype objects. Select TP, PP, SP, and runtime `seq_length`
separately. The inherited 65536 maximum position length is not the benchmark's
runtime sequence length.

Two implementation details matter when matching training behavior:

- The HF text config has `attention_dropout=0.0`. The Bridge benchmark does
  not override the Megatron-Core default `attention_dropout=0.1`; that effective
  value is recorded in `megatron_model_config.json`. Bridge hidden dropout is
  0.0. Set the target runtime's attention dropout explicitly for a matched run.
- Preserve Q/K RMSNorm and multimodal RoPE (`mrope_section=[24,20,20]`, base
  1000000). Bridge selects its non-interleaved rotary implementation; the HF
  config retains the original HF rotary metadata. Use the corresponding
  runtime's Qwen3-Omni rotary implementation when porting.

The full HF configurations retain the base model's talker/code2wav metadata,
with `enable_audio_output=false`. Those components were not built or trained
in this experiment; instantiate the thinker for the approximately 15B model.

## Workload for a matched throughput test

The documented benchmark used BF16 mixed precision, distributed Adam, FP32
gradient reduction, micro-batch 1, global batch 16/32/64, and no activation
recompute, CUDA graphs, or gradient-accumulation fusion. Each 1024 sequence
tokens contributes one 448x448 image and one 15-second audio clip. Images yield
784 patches / 196 LLM tokens; clips yield 1500 mel frames / 195 LLM tokens.
The grid used runtime sequence lengths 1024/2048/4096/8192/16384, with 3 warmup
and 5 measured steps. See `../README.benchmark.md` for the measured parallel
layouts and results.

The export was checked for JSON validity, architecture overrides, head
divisibility, dense-layer coverage, parameter counts, and absence of private
paths. Model construction and execution were not run as part of this export.

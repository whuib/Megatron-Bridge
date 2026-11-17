#!/bin/bash
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

# Repeated, order-rotated throughput runs of native / DistTrain / multiplexing Qwen3-VL training
# with one shared workload, followed by a contract-checked summary. Configure with environment
# variables; every *_ARGS value is passed verbatim to compare_training.py.
#
# Layout sweeps: set CASES_FILE to a file with one case per line, "<name> <mode> <layout args...>"
# (blank lines and lines starting with # are ignored). Without it, one case per mode in MODES
# uses the matching *_ARGS.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

NPROC=${NPROC:-8}
REPS=${REPS:-3}
WARMUP_ITERS=${WARMUP_ITERS:-10}
MEASURE_ITERS=${MEASURE_ITERS:-40}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MODES=${MODES:-"native disttrain multiplexing"}
# nemo_experiments/ is gitignored, so results are never committed by accident.
OUT_DIR=${OUT_DIR:-"$REPO_ROOT/nemo_experiments/qwen3_vl_throughput/$(date +%Y%m%d_%H%M%S)"}
LAUNCH=${LAUNCH:-"uv run python -m torch.distributed.run"}
SUMMARIZE=${SUMMARIZE:-"uv run python"}
# Shared by every mode: the workload and the training features that must match.
WORKLOAD_ARGS=${WORKLOAD_ARGS:-"--num-layers 12 --seq-length 8192 --global-batch-size 32 --micro-batch-size 2 --num-images 8 --image-size 256 --recompute full"}
# Mode-specific parallel layouts on NPROC GPUs.
NATIVE_ARGS=${NATIVE_ARGS:-"--language-tp 1 --language-pp 2"}
DISTTRAIN_ARGS=${DISTTRAIN_ARGS:-"--language-tp 1 --language-pp 2 --vision-gpus 4 --encoder-tp 1"}
MULTIPLEXING_ARGS=${MULTIPLEXING_ARGS:-"--language-tp 1 --language-pp 2 --encoder-tp 1"}
# Shared-GPU hygiene (nvidia-smi pmon): before each run wait until no process exceeded QUIET_THRESHOLD %
# SM for QUIET_WINDOW s; during the run account foreign SM time per GPU, and redo a run whose foreign
# mean SM exceeded MAX_FOREIGN_SM % on any GPU (up to MAX_RETRIES times). GPU_GATE=0 disables both.
GPU_GATE=${GPU_GATE:-1}
QUIET_THRESHOLD=${QUIET_THRESHOLD:-1}
QUIET_WINDOW=${QUIET_WINDOW:-60}
QUIET_MAX_WAIT=${QUIET_MAX_WAIT:-7200}
MAX_FOREIGN_SM=${MAX_FOREIGN_SM:-1.0}
# An OOM while other processes held more than this much memory (MiB) on the failing GPU is repeated, not
# reported. Taken from PyTorch's OOM message (GPU total - free - this process); for an OOM inside NCCL,
# which has no such message, from the largest foreign process the monitor saw during the run.
OOM_FOREIGN_FB_MIB=${OOM_FOREIGN_FB_MIB:-2048}
MAX_RETRIES=${MAX_RETRIES:-3}
GPU_ACTIVITY=${GPU_ACTIVITY:-"python $SCRIPT_DIR/gpu_activity.py"}
# Entry point (relative to the repository root); e.g. examples/models/qwen/qwen3_omni/compare_training.py.
COMPARE_SCRIPT=${COMPARE_SCRIPT:-examples/models/qwen/qwen3_vl/compare_training.py}

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

case_names=()
case_lines=()
if [[ -n "${CASES_FILE:-}" ]]; then
  while read -r name mode args; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    case_names+=("$name")
    case_lines+=("$mode $args")
  done < "$CASES_FILE"
  cp "$CASES_FILE" "$OUT_DIR/cases.txt"
else
  read -r -a mode_list <<< "$MODES"
  for mode in "${mode_list[@]}"; do
    case "$mode" in
      native) mode_args=$NATIVE_ARGS ;;
      disttrain) mode_args=$DISTTRAIN_ARGS ;;
      multiplexing) mode_args=$MULTIPLEXING_ARGS ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    case_names+=("$mode")
    case_lines+=("$mode $mode_args")
  done
fi
num_cases=${#case_names[@]}

for rep in $(seq 1 "$REPS"); do
  # Rotate the order every repetition so drift on a shared machine is not attributed to one case.
  for offset in $(seq 0 $((num_cases - 1))); do
    index=$(( (offset + rep - 1) % num_cases ))
    name=${case_names[$index]}
    read -r mode case_args <<< "${case_lines[$index]}"
    run="$name.rep$rep"
    for attempt in $(seq 0 "$MAX_RETRIES"); do
      if [[ "$GPU_GATE" == 1 ]]; then
        # shellcheck disable=SC2086
        $GPU_ACTIVITY wait-quiet --threshold "$QUIET_THRESHOLD" --window "$QUIET_WINDOW" \
          --max-wait "$QUIET_MAX_WAIT" 2>> "$OUT_DIR/$run.gate.log" \
          || echo "$run: GPUs not quiet after ${QUIET_MAX_WAIT}s; running anyway (see interference record)"
      fi
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$OUT_DIR/$run.gpus_before.csv" || true
      # shellcheck disable=SC2086
      $LAUNCH --standalone --nproc_per_node="$NPROC" \
        "$COMPARE_SCRIPT" --mode "$mode" $case_args $WORKLOAD_ARGS \
        --train-iters $((WARMUP_ITERS + MEASURE_ITERS)) --warmup-iters "$WARMUP_ITERS" \
        --log-interval "$LOG_INTERVAL" --throughput-json "$OUT_DIR/$run.json" \
        > "$OUT_DIR/$run.log" 2>&1 &
      launcher=$!
      monitor=
      if [[ "$GPU_GATE" == 1 ]]; then
        # shellcheck disable=SC2086
        $GPU_ACTIVITY monitor --root-pid "$launcher" --threshold "$QUIET_THRESHOLD" \
          --out "$OUT_DIR/$run.interference.json" 2>> "$OUT_DIR/$run.gate.log" &
        monitor=$!
      fi
      status=0
      wait "$launcher" || status=$?
      [[ -n "$monitor" ]] && { wait "$monitor" || true; }
      if [[ $status -ne 0 ]]; then
        echo "$run: FAILED ($(grep -m1 -oE 'OutOfMemoryError|ValueError: [^\"]*|RuntimeError: [^\"]*' "$OUT_DIR/$run.log" | head -c 160))"
        foreign_fb=$(python "$SCRIPT_DIR/gpu_activity.py" oom-foreign --log "$OUT_DIR/$run.log" \
          --interference "$OUT_DIR/$run.interference.json" 2>/dev/null || echo 0)
        if grep -qE "OutOfMemoryError|CUDA out of memory|Cuda failure 2 'out of memory'|Failed to CUDA calloc" "$OUT_DIR/$run.log" \
            && [[ $foreign_fb -gt $OOM_FOREIGN_FB_MIB ]] && [[ $attempt -lt $MAX_RETRIES ]]; then
          mkdir -p "$OUT_DIR/contaminated"
          for file in "$OUT_DIR/$run".*; do mv "$file" "$OUT_DIR/contaminated/$(basename "$file").attempt$attempt"; done
          echo "$run: OOM while a foreign process held ${foreign_fb} MiB; waiting for quiet GPUs and repeating"
          continue
        fi
        break
      fi
      foreign=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_foreign_mean_sm_percent"])' \
        "$OUT_DIR/$run.interference.json" 2>/dev/null || echo 0)
      echo "$run: $(grep -m1 '^THROUGHPUT' "$OUT_DIR/$run.log" || echo 'finished without a THROUGHPUT line') foreign_mean_sm=$foreign"
      if python -c 'import sys; sys.exit(float(sys.argv[1]) <= float(sys.argv[2]))' "$foreign" "$MAX_FOREIGN_SM" \
          && [[ $attempt -lt $MAX_RETRIES ]]; then
        mkdir -p "$OUT_DIR/contaminated"
        for file in "$OUT_DIR/$run".*; do mv "$file" "$OUT_DIR/contaminated/$(basename "$file").attempt$attempt"; done
        echo "$run: foreign GPU work above ${MAX_FOREIGN_SM}% SM; waiting for quiet GPUs and repeating"
        continue
      fi
      break
    done
  done
done

$SUMMARIZE "$SCRIPT_DIR/summarize_throughput.py" "$OUT_DIR"

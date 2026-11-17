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

# Qwen3-Omni native / DistTrain / multiplexing throughput over a sequence-length x global-batch grid.
# Every (SEQ, GBS) point runs the cases of CASES_FILE through run_throughput_comparison.sh (same
# workload, quiet-GPU gate, interference monitor). Failed runs, including OOM, are kept as such.
# plan_omni_grid.py does not launch a case whose layout is rejected before launch, or that ran out of
# memory at a point with no larger sequence and global batch; it leaves a marker with the reason.
#
#   HF_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct MODEL_SIZE=3b-3b-9b CASES_FILE=cases.txt \
#   SEQS="1024 2048 4096 8192" GBSS="16 32 64" bash examples/models/qwen/qwen3_omni/run_omni_grid.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

: "${HF_MODEL:?set HF_MODEL to a local Qwen3-Omni directory (config, tokenizer, processor)}"
: "${CASES_FILE:?set CASES_FILE to '<name> <mode> <layout args>' lines}"
: "${MODEL_SIZE:?set MODEL_SIZE to a size of omni_benchmark.MODEL_SIZES (e.g. 1b-1b-3b)}"
SEQS=${SEQS:-"2048 4096 8192"}
GBSS=${GBSS:-"16 32 64"}
GRID_DIR=${GRID_DIR:-"$REPO_ROOT/nemo_experiments/qwen3_omni_grid/$(date +%Y%m%d_%H%M%S)"}
EXTRA_WORKLOAD_ARGS=${EXTRA_WORKLOAD_ARGS:-""}
SUMMARIZE=${SUMMARIZE:-"uv run python"}
NPROC=${NPROC:-8}
# SELECT_TOP=K with SCREEN_GBS=G: above GBS G, run only each mode's K fastest layouts measured at GBS G.
SELECT_ARGS=""
if [[ -n "${SELECT_TOP:-}" ]]; then
  SELECT_ARGS="--select-top $SELECT_TOP --screen-gbs ${SCREEN_GBS:?SCREEN_GBS is required with SELECT_TOP}"
fi

mkdir -p "$GRID_DIR"
cp "$CASES_FILE" "$GRID_DIR/cases.txt"
for seq in $SEQS; do
  for gbs in $GBSS; do
    point="$GRID_DIR/seq${seq}_gbs${gbs}"
    workload="--hf-model $HF_MODEL --model-size $MODEL_SIZE --seq-length $seq --global-batch-size $gbs --micro-batch-size 1 $EXTRA_WORKLOAD_ARGS"
    echo "== seq=$seq gbs=$gbs"
    mkdir -p "$point"
    # shellcheck disable=SC2086
    $SUMMARIZE "$SCRIPT_DIR/plan_omni_grid.py" --grid-dir "$GRID_DIR" --cases "$CASES_FILE" --seq "$seq" \
      --gbs "$gbs" --world-size "$NPROC" --workload-args "$workload" --out "$point/cases.launch.txt" $SELECT_ARGS
    [[ -s "$point/cases.launch.txt" ]] || continue
    OUT_DIR="$point" CASES_FILE="$point/cases.launch.txt" NPROC="$NPROC" \
    COMPARE_SCRIPT=examples/models/qwen/qwen3_omni/compare_training.py WORKLOAD_ARGS="$workload" \
    SUMMARIZE="$SUMMARIZE" \
      bash "$REPO_ROOT/examples/models/qwen/qwen3_vl/run_throughput_comparison.sh" || true
  done
done
$SUMMARIZE "$SCRIPT_DIR/summarize_omni_grid.py" "$GRID_DIR"

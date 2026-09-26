#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

device="${DEVICE:-cuda}"
warmup="${WARMUP:-50}"
repeats="${REPEATS:-200}"
sizes="${SIZES:-256 512}"
models="${MODELS:-}"
output="${OUTPUT:-outputs/performance/inference_benchmark.json}"

args=(--device "$device" --warmup "$warmup" --repeats "$repeats" --sizes $sizes --output "$output")
if [[ -n "$models" ]]; then
    args+=(--models $models)
fi

python scripts/benchmark_inference.py "${args[@]}"

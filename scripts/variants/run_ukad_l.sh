#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/../.."

MODE="${1:-pilot}"
VARIANT=ukad_l
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHON_BIN
WORKER_MODE=0

export DATASETS="${DATASETS:-$PWD/datasets}"
export WORKBENCH="${WORKBENCH:-$PWD/workbench}"

case "$MODE" in
    __worker)
        if [[ "$#" -ne 4 ]]; then
            echo "internal usage: $0 __worker DATASET SEED GPU" >&2
            exit 2
        fi
        TASK_DATASETS=("$2")
        TASK_SEEDS=("$3")
        GPUS=("$4")
        WORKER_MODE=1
        ;;
    pilot)
        read -r -a GPUS <<< "${RUN_GPUS:-0 1 2 3}"
        TASK_DATASETS=(isic busi glas cvc)
        TASK_SEEDS=(2981 6142 2981 1187)
        ;;
    pilot2)
        read -r -a GPUS <<< "${RUN_GPUS:-0 1 2 3}"
        TASK_DATASETS=(busi cvc glas isic)
        TASK_SEEDS=(2981 2981 6142 6142)
        ;;
    busi|cvc|glas|isic)
        read -r -a GPUS <<< "${RUN_GPUS:-0 1 2}"
        TASK_DATASETS=("$MODE" "$MODE" "$MODE")
        TASK_SEEDS=(2981 6142 1187)
        ;;
    all)
        read -r -a GPUS <<< "${RUN_GPUS:-0 1 2 3 4 5 6 7}"
        TASK_DATASETS=(busi busi busi cvc cvc cvc glas glas glas isic isic isic)
        TASK_SEEDS=(2981 6142 1187 2981 6142 1187 2981 6142 1187 2981 6142 1187)
        ;;
    *)
        echo "usage: $0 [pilot|pilot2|busi|cvc|glas|isic|all]" >&2
        exit 2
        ;;
esac

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "RUN_GPUS must contain at least one GPU index" >&2
    exit 2
fi
if [[ ! -d "$DATASETS" ]]; then
    echo "Dataset root does not exist: $DATASETS" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable does not exist: $PYTHON_BIN" >&2
    exit 2
fi

mkdir -p "$WORKBENCH/logs/_runs"
mkdir -p "$WORKBENCH/locks"

is_complete() {
    local metrics="$1"
    [[ -f "$metrics" ]] || return 1
    awk -F, '
        NR > 1 && $1 ~ /^[0-9]+$/ { if ($1 + 0 > maximum) maximum = $1 + 0 }
        END { exit !(maximum >= 399) }
    ' "$metrics"
}

is_running() {
    local experiment="$1"
    local seed="$2"
    pgrep -u "$(id -u)" -f -- \
        "train.py .*--experiment ${experiment} --dataseed ${seed}" >/dev/null
}

run_task() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    local experiment="${VARIANT}_${dataset}"
    local run_name="${experiment}_dataseed${seed}"
    local config="configs/variants/${VARIANT}/${dataset}.yaml"
    local metrics="$WORKBENCH/logs/${run_name}/metrics.csv"
    local log="$WORKBENCH/logs/_runs/${run_name}.log"
    local lock="$WORKBENCH/locks/${run_name}.lock"

    exec 9> "$lock"
    if ! flock -n 9; then
        echo "[skip locked] $run_name"
        return 0
    fi
    if is_complete "$metrics"; then
        echo "[skip complete] $run_name"
        return 0
    fi
    if is_running "$experiment" "$seed"; then
        echo "[skip running] $run_name"
        return 0
    fi

    echo "[GPU $gpu] starting $run_name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" train.py \
        --config "$config" \
        --experiment "$experiment" \
        --dataseed "$seed" >> "$log" 2>&1
}

if [[ "$WORKER_MODE" -eq 1 ]]; then
    run_task "${TASK_DATASETS[0]}" "${TASK_SEEDS[0]}" "${GPUS[0]}"
    exit $?
fi

gpu_count=${#GPUS[@]}
for task_index in "${!TASK_DATASETS[@]}"; do
    gpu="${GPUS[$((task_index % gpu_count))]}"
    dataset="${TASK_DATASETS[$task_index]}"
    seed="${TASK_SEEDS[$task_index]}"
    experiment="${VARIANT}_${dataset}"
    run_name="${experiment}_dataseed${seed}"
    config="configs/variants/${VARIANT}/${dataset}.yaml"
    log="$WORKBENCH/logs/_runs/${run_name}.log"
    launcher_log="$WORKBENCH/logs/_runs/${run_name}.launcher.log"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run GPU $gpu] $run_name config=$config log=$log"
        continue
    fi

    nohup bash "$SCRIPT_PATH" __worker "$dataset" "$seed" "$gpu" \
        >> "$launcher_log" 2>&1 < /dev/null &
    echo "[launched] $run_name GPU=$gpu PID=$! log=$log"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run complete; no jobs were submitted."
else
    echo "All requested UKAD-L jobs were submitted in the background."
fi

#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

_model_opts() {
    local opts="all" d
    for d in configs/baselines/*/; do [[ -d "$d" ]] && opts="$opts $(basename "$d")"; done
    echo "$opts" | tr " " "|"
}

dataset="${1:-}"
model="${2:-all}"
TRAINING_SEED=1029
read -r -a DATASEEDS <<< "${DATASEEDS:-2981 6142 1187}"
SEQUENTIAL="${SEQUENTIAL:-0}"
RUN_DEVICES="${RUN_DEVICES:-}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${WORKBENCH:-workbench}/logs/_runs"
mkdir -p "$LOG_DIR"

if [[ -z "$dataset" ]]; then
    echo "usage: $0 <busi|cvc|glas|isic> [$(_model_opts)]" >&2
    exit 2
fi
case "$dataset" in busi|cvc|glas|isic) ;; *) echo "unknown dataset: $dataset" >&2; exit 2 ;; esac
case "$SEQUENTIAL" in 0|1) ;; *) echo "SEQUENTIAL must be 0 or 1" >&2; exit 2 ;; esac

resolve_config() {
    local candidate="$1"
    if [[ -f "configs/baselines/${candidate}/${dataset}.yaml" ]]; then
        config="configs/baselines/${candidate}/${dataset}.yaml"
        experiment="${candidate}_${dataset}"
    else
        echo "unknown model: ${candidate} (no baseline ${dataset} config)" >&2
        exit 2
    fi
}

case "$model" in
    all)
        models=()
        for d in configs/baselines/*/; do
            m="$(basename "$d")"
            [[ -f "configs/baselines/${m}/${dataset}.yaml" ]] && models+=("$m")
        done
        ;;
    *)
        resolve_config "$model"
        models=("$model")
        ;;
esac

run_model() {
    local m="$1"
    resolve_config "$m"

    local pids=()
    local devices=()
    local run_values=()
    for dataseed in "${DATASEEDS[@]}"; do
        run_values+=("dataseed${dataseed}")
    done
    if [[ -n "$RUN_DEVICES" ]]; then
        read -r -a devices <<< "$RUN_DEVICES"
        if [[ "${#devices[@]}" -eq 1 ]]; then
            while [[ "${#devices[@]}" -lt "${#run_values[@]}" ]]; do devices+=("${devices[0]}"); done
        fi
        if [[ "${#devices[@]}" -ne "${#run_values[@]}" ]]; then
            echo "RUN_DEVICES needs one entry per run, or a single entry to share" >&2
            return 2
        fi
    fi
    local index=0
    for run_id in "${run_values[@]}"; do
        local log="${LOG_DIR}/${experiment}_${run_id}.log"
        local args=(
            python train.py
            --config "$config"
            --experiment "$experiment"
        )
        args+=(--dataseed "${run_id#dataseed}")
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] ${experiment} ${run_id} training_seed=${TRAINING_SEED} log=${log}"
        elif [[ "$SEQUENTIAL" == "1" ]]; then
            echo "[running] ${experiment} ${run_id} training_seed=${TRAINING_SEED} log=${log}"
            if ! "${args[@]}" 2>&1 | tee "$log"; then
                echo "[failed] ${experiment} ${run_id}; inspect ${log}" >&2
                return 1
            fi
        else
            if [[ -n "$RUN_DEVICES" ]]; then
                CUDA_VISIBLE_DEVICES="${devices[$index]}" nohup "${args[@]}" > "$log" 2>&1 &
                local device="${devices[$index]}"
            else
                nohup "${args[@]}" > "$log" 2>&1 &
                local device="${CUDA_VISIBLE_DEVICES:-all}"
            fi
            local pid=$!
            pids+=("$pid")
            echo "[launched] ${experiment} ${run_id} device=${device} PID=${pid} log=${log}"
        fi
        index=$((index + 1))
    done
    if [[ "$SEQUENTIAL" == "1" || "$DRY_RUN" == "1" ]]; then
        echo "${experiment} complete (${run_values[*]}; training seed ${TRAINING_SEED})."
        return 0
    fi
    echo "Waiting for ${#pids[@]} concurrent seed jobs (${experiment}) ..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[failed] PID=${pid}; inspect its log" >&2
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        return 1
    fi

    echo "${experiment} complete (${run_values[*]}; training seed ${TRAINING_SEED})."
}

for m in "${models[@]}"; do
    run_model "$m"
done

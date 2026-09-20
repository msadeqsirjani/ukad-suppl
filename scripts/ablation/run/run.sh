#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT="${CONFIG_ROOT:-configs/ablation}"
. scripts/ablation/groups.sh

_ablation_opts() {
    local opts="all paper appendix decoder" d
    for d in ${ROOT}/ukad_*/; do [[ -d "$d" ]] && opts="$opts $(basename "$d")"; done
    echo "$opts" | tr " " "|"
}

dataset="${1:-}"
ablation="${2:-all}"
DATASEED="${DATASEED:-2981}"
SEQUENTIAL="${SEQUENTIAL:-0}"
RUN_DEVICES="${RUN_DEVICES:-}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${WORKBENCH:-workbench}/logs/_runs"
mkdir -p "$LOG_DIR"

if [[ -z "$dataset" ]]; then
    echo "usage: $0 <busi|cvc|glas|isic> [$(_ablation_opts)]" >&2
    exit 2
fi
case "$dataset" in busi|cvc|glas|isic) ;; *) echo "unknown dataset: $dataset" >&2; exit 2 ;; esac
case "$SEQUENTIAL" in 0|1) ;; *) echo "SEQUENTIAL must be 0 or 1" >&2; exit 2 ;; esac

resolve_config() {
    local candidate="$1"
    if [[ -f "${ROOT}/${candidate}/${dataset}.yaml" ]]; then
        config="${ROOT}/${candidate}/${dataset}.yaml"
        experiment="${candidate}_${dataset}"
    else
        echo "unknown ablation: ${candidate} (no ${ROOT}/${candidate}/${dataset}.yaml)" >&2
        exit 2
    fi
}

case "$ablation" in
    all)
        ablations=()
        for d in ${ROOT}/ukad_*/; do
            a="$(basename "$d")"
            [[ -f "${ROOT}/${a}/${dataset}.yaml" ]] && ablations+=("$a")
        done
        ;;
    paper|appendix|decoder)
        check_ablation_groups "$ROOT"
        ablations=()
        while read -r a; do
            [[ -f "${ROOT}/${a}/${dataset}.yaml" ]] && ablations+=("$a")
        done < <(ablation_group "$ablation")
        ;;
    *)
        resolve_config "$ablation"
        ablations=("$ablation")
        ;;
esac

if [[ "${#ablations[@]}" -eq 0 ]]; then
    echo "no ukad ablation configs found for dataset: ${dataset}" >&2
    exit 2
fi

if [[ -n "$RUN_DEVICES" ]]; then
    read -r -a devices <<< "$RUN_DEVICES"
    if [[ "${#devices[@]}" -eq 1 ]]; then
        while [[ "${#devices[@]}" -lt "${#ablations[@]}" ]]; do devices+=("${devices[0]}"); done
    fi
    if [[ "${#devices[@]}" -ne "${#ablations[@]}" ]]; then
        echo "RUN_DEVICES needs one entry per ablation, or a single entry to share" >&2
        exit 2
    fi
fi

pids=()
index=0
for a in "${ablations[@]}"; do
    resolve_config "$a"
    log="${LOG_DIR}/${experiment}_dataseed${DATASEED}.log"
    args=(python train.py --config "$config" --experiment "$experiment" --dataseed "$DATASEED")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] ${experiment} dataseed${DATASEED} log=${log}"
    elif [[ "$SEQUENTIAL" == "1" ]]; then
        echo "[running] ${experiment} dataseed${DATASEED} log=${log}"
        if ! "${args[@]}" 2>&1 | tee "$log"; then
            echo "[failed] ${experiment}; inspect ${log}" >&2
            exit 1
        fi
    else
        if [[ -n "$RUN_DEVICES" ]]; then
            CUDA_VISIBLE_DEVICES="${devices[$index]}" nohup "${args[@]}" > "$log" 2>&1 &
            device="${devices[$index]}"
        else
            nohup "${args[@]}" > "$log" 2>&1 &
            device="${CUDA_VISIBLE_DEVICES:-all}"
        fi
        pid=$!
        pids+=("$pid")
        echo "[launched] ${experiment} dataseed${DATASEED} device=${device} PID=${pid} log=${log}"
    fi
    index=$((index + 1))
done

if [[ "$SEQUENTIAL" == "1" || "$DRY_RUN" == "1" ]]; then
    echo "ablation run complete (dataset=${dataset}, dataseed=${DATASEED})."
    exit 0
fi

echo "Waiting for ${#pids[@]} concurrent ablation jobs (dataset=${dataset}) ..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[failed] PID=${pid}; inspect its log" >&2
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    exit 1
fi

echo "ablation run complete (dataset=${dataset}, dataseed=${DATASEED})."

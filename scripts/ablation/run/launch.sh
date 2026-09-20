#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "$0")/../../.."
. scripts/ablation/groups.sh

ROOT="${CONFIG_ROOT:-configs/ablation}"
SEEDS="${SEEDS:-2981 6142 1187}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
DONE_EPOCH="${DONE_EPOCH:-399}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
WORK="${WORKBENCH:-workbench}"
LOG_DIR="${WORK}/logs/_runs"

dataset="${1:-}"
selector="${2:-all}"

usage() {
    echo "usage: [SEEDS=\"2981 6142 1187\"] [GPUS=\"0 1\"] $0 <busi|cvc|glas|isic> [all|paper|appendix|decoder|<ablation>]" >&2
    exit 2
}

[[ -n "$dataset" ]] || usage
case "$dataset" in busi|cvc|glas|isic) ;; *) echo "unknown dataset: $dataset" >&2; usage ;; esac

ablations=()
case "$selector" in
    all)
        for d in ${ROOT}/ukad_*/; do
            a="$(basename "$d")"
            [[ -f "${ROOT}/${a}/${dataset}.yaml" ]] && ablations+=("$a")
        done
        ;;
    paper|appendix|decoder)
        check_ablation_groups "$ROOT" || exit 2
        while read -r a; do
            [[ -f "${ROOT}/${a}/${dataset}.yaml" ]] && ablations+=("$a")
        done < <(ablation_group "$selector")
        ;;
    *)
        [[ -f "${ROOT}/${selector}/${dataset}.yaml" ]] || { echo "unknown ablation: ${selector}" >&2; usage; }
        ablations=("$selector")
        ;;
esac
[[ "${#ablations[@]}" -gt 0 ]] || { echo "no ablation configs for dataset: ${dataset}" >&2; exit 2; }

read -r -a devices <<< "$GPUS"
read -r -a seeds <<< "$SEEDS"
[[ "${#devices[@]}" -gt 0 ]] || { echo "GPUS must list at least one index" >&2; exit 2; }
[[ "${#seeds[@]}" -gt 0 ]] || { echo "SEEDS must list at least one seed" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "python not found: ${PYTHON_BIN}" >&2; exit 2; }

is_complete() {
    [[ -f "$1" ]] || return 1
    awk -F, -v target="$DONE_EPOCH" '
        NR > 1 && $1 ~ /^[0-9]+$/ { if ($1 + 0 > seen) seen = $1 + 0 }
        END { exit !(seen >= target) }
    ' "$1"
}

is_running() {
    pgrep -u "$(id -u)" -f -- "train.py .*--experiment $1 --dataseed $2" >/dev/null
}

mkdir -p "$LOG_DIR"
index=0
launched=0
for seed in "${seeds[@]}"; do
    for a in "${ablations[@]}"; do
        experiment="${a}_${dataset}"
        run="${experiment}_dataseed${seed}"
        log="${LOG_DIR}/${run}.log"
        if [[ "$FORCE" != "1" ]] && is_complete "${WORK}/logs/${run}/metrics.csv"; then
            echo "[skip complete] $run"
            continue
        fi
        if is_running "$experiment" "$seed"; then
            echo "[skip running] $run"
            continue
        fi
        gpu="${devices[$((index % ${#devices[@]}))]}"
        index=$((index + 1))
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run GPU $gpu] $run log=$log"
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PYTHON_BIN" train.py \
            --config "${ROOT}/${a}/${dataset}.yaml" \
            --experiment "$experiment" \
            --dataseed "$seed" >> "$log" 2>&1 < /dev/null &
        echo "[launched GPU $gpu] $run PID=$! log=$log"
        launched=$((launched + 1))
    done
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run complete (dataset=${dataset}, selector=${selector})."
else
    echo "${launched} job(s) submitted in the background (dataset=${dataset}, selector=${selector})."
fi

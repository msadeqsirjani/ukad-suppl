#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT_DIR="${OUT_DIR:-outputs/performance/ablations}"
ROOT="${CONFIG_ROOT:-configs/ablation}"
. scripts/ablation/groups.sh

_ablation_opts() {
    local opts="all paper appendix decoder" d
    for d in ${ROOT}/ukad_*/; do [[ -d "$d" ]] && opts="$opts $(basename "$d")"; done
    echo "$opts" | tr " " "|"
}

dataset="${1:-}"
ablation="${2:-all}"
read -r -a DATASEEDS <<< "${DATASEEDS:-${DATASEED:-2981 6142 1187}}"

if [[ -z "$dataset" ]]; then
    echo "usage: $0 <busi|cvc|glas|isic> [$(_ablation_opts)]" >&2
    exit 2
fi
case "$dataset" in busi|cvc|glas|isic) ;; *) echo "unknown dataset: $dataset" >&2; exit 2 ;; esac

resolve_config() {
    local candidate="$1"
    if [[ -f "${ROOT}/${candidate}/${dataset}.yaml" ]]; then
        config="${ROOT}/${candidate}/${dataset}.yaml"
        experiment="${candidate}_${dataset}"
        output="${OUT_DIR}/${candidate}_${dataset}.json"
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

mkdir -p ${OUT_DIR}

for a in "${ablations[@]}"; do
    resolve_config "$a"
    eval_config="configs/data/${dataset}/val_dataloader.yaml"
    checkpoints=()
    for dataseed in "${DATASEEDS[@]}"; do
        checkpoint_dir="${WORKBENCH:-workbench}/ckpts/${experiment}_dataseed${dataseed}"
        mapfile -t matches < <(find "$checkpoint_dir" -maxdepth 1 -type f -name 'best_*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
        if [[ "${#matches[@]}" -eq 0 ]]; then
            echo "no validation-selected best checkpoint in ${checkpoint_dir}" >&2
            exit 2
        fi
        checkpoints+=("${matches[0]}")
    done
    args=(
        python scripts/eval/aggregate.py
        --experiment "$experiment"
        --dataset "$dataset"
        --config "$config"
        --eval_config "$eval_config"
        --checkpoint "${checkpoints[@]}"
        --device "${DEVICE:-auto}"
        --output "$output"
        --dataseeds "${DATASEEDS[@]}"
    )
    "${args[@]}"
done

echo "${dataset} ablation evaluation complete (ablation=${ablation})."

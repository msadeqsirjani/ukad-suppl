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

if [[ -z "$dataset" ]]; then
    echo "usage: $0 <busi|cvc|glas|isic> [$(_model_opts)]" >&2
    exit 2
fi

case "$dataset" in busi|cvc|glas|isic) ;; *) echo "unknown dataset: $dataset" >&2; exit 2 ;; esac

resolve_config() {
    local candidate="$1"
    if [[ -f "configs/variants/${candidate}/${dataset}.yaml" ]]; then
        config="configs/variants/${candidate}/${dataset}.yaml"
        experiment="${candidate}_${dataset}"
        output="outputs/performance/variants/${candidate}_${dataset}.json"
    elif [[ -f "configs/baselines/${candidate}/${dataset}.yaml" ]]; then
        config="configs/baselines/${candidate}/${dataset}.yaml"
        experiment="${candidate}_${dataset}"
        output="outputs/performance/${candidate}_${dataset}.json"
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

for m in "${models[@]}"; do
    resolve_config "$m"
    checkpoints=()
    eval_config="configs/data/${dataset}/val_dataloader.yaml"
    run_ids=()
    for dataseed in "${DATASEEDS[@]}"; do
        run_ids+=("dataseed${dataseed}")
    done
    for run_id in "${run_ids[@]}"; do
        checkpoint_dir="${WORKBENCH:-workbench}/ckpts/${experiment}_${run_id}"
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
    )
    args+=(--dataseeds "${DATASEEDS[@]}")
    "${args[@]}"
done

echo "${dataset} evaluation complete (model=${model})."

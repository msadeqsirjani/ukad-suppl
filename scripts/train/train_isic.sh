#!/usr/bin/env bash


set -u
here="$(dirname "$0")"
model="${1:-all}"
if [[ "$model" == "-h" || "$model" == "--help" ]]; then
    opts="all"
    for d in "$here"/../../configs/baselines/*/; do [[ -d "$d" ]] && opts="$opts|$(basename "$d")"; done
    echo "usage: $0 [$opts]" >&2
    exit 0
fi
exec "$here/train.sh" isic "$model"

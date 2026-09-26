#!/usr/bin/env bash

set -u
here="$(dirname "$0")"
ablation="${1:-all}"
if [[ "$ablation" == "-h" || "$ablation" == "--help" ]]; then
    opts="all|paper|appendix|decoder"
    for d in "$here"/../../../configs/ablation/ukad_*/; do [[ -d "$d" ]] && opts="$opts|$(basename "$d")"; done
    echo "usage: $0 [$opts]" >&2
    exit 0
fi
exec "$here/eval.sh" glas "$ablation"

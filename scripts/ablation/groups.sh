#!/usr/bin/env bash

PAPER_ABLATIONS=(
    ukad_no_displacement
    ukad_no_kan
    ukad_no_displacement_no_kan
    ukad_no_context
)

DECODER_ABLATIONS=(
    ukad_conv_decoder
    ukad_kan_decoder
    ukad_mlp_decoder
)

APPENDIX_ABLATIONS=(
    ukad_hard_gate
    ukad_independent_gate
    ukad_no_adaptive_context
    ukad_no_boundary
    ukad_no_boundary_loss
    ukad_no_decoder_gate
    ukad_no_deep_supervision
)

ablation_group() {
    case "$1" in
        paper) printf '%s\n' "${PAPER_ABLATIONS[@]}" ;;
        appendix) printf '%s\n' "${APPENDIX_ABLATIONS[@]}" ;;
        decoder) printf '%s\n' "${DECODER_ABLATIONS[@]}" ;;
        *) return 1 ;;
    esac
}

check_ablation_groups() {
    local root="$1" d a known unassigned=()
    known="$(printf '%s\n' "${PAPER_ABLATIONS[@]}" "${APPENDIX_ABLATIONS[@]}" "${DECODER_ABLATIONS[@]}")"
    for d in ${root}/ukad_*/; do
        [[ -d "$d" ]] || continue
        a="$(basename "$d")"
        grep -qxF "$a" <<< "$known" || unassigned+=("$a")
    done
    if [[ "${#unassigned[@]}" -gt 0 ]]; then
        echo "unassigned ablation(s), add them to scripts/ablation/groups.sh: ${unassigned[*]}" >&2
        return 1
    fi
}

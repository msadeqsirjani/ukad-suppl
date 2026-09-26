import argparse
import json
import os
import re
import statistics
from copy import deepcopy
from pathlib import Path

from gen_report import (
    BLANK,
    DATASETS,
    SEG_METRICS,
    HERE,
    ROOT,
    fmt_num,
    fmt_pct,
    load,
    validate_result,
    write_report,
)

GROUPS_FILE = os.path.join(ROOT, "scripts", "ablation", "groups.sh")
ABLATION_DIR = os.path.join(HERE, "performance", "ablations")
REFERENCE = "ukad_b"
REFERENCE_NAME = "UKAD-B (full)"

ABLATION_NAMES = {
    "ukad_no_displacement": r"w/o displacement",
    "ukad_no_kan": r"w/o Group-Rational Form",
    "ukad_no_displacement_no_kan": r"w/o displacement + KAN",
    "ukad_no_context": r"w/o context branch",
    "ukad_conv_decoder": r"conv decoder",
    "ukad_kan_decoder": r"KAN decoder",
    "ukad_mlp_decoder": r"MLP decoder",
    "ukad_hard_gate": r"hard gate",
    "ukad_independent_gate": r"independent gate",
    "ukad_no_adaptive_context": r"w/o adaptive context",
    "ukad_no_boundary": r"w/o boundary module",
    "ukad_no_boundary_loss": r"w/o boundary loss",
    "ukad_no_decoder_gate": r"w/o decoder gate",
    "ukad_no_deep_supervision": r"w/o deep supervision",
}

GROUP_CAPTIONS = {
    "decoder": "Decoder-block comparison for UKAD-B. Each row uses one block type at every decoder level, against the depth-dependent hybrid decoder of the full model.",
    "paper": "Component ablation of UKAD-B. Each row removes one component from the full model.",
    "appendix": "Design-choice ablation of UKAD-B. Each row changes one design decision in the full model.",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("paper", "appendix", "decoder", "all"), default="all")
    parser.add_argument("--output_dir", default=os.path.join(HERE, "report"))
    return parser.parse_args()


def read_groups(path):
    text = Path(path).read_text()
    groups = {}
    for match in re.finditer(r"^(PAPER|APPENDIX|DECODER)_ABLATIONS=\((.*?)\)", text, re.M | re.S):
        groups[match.group(1).lower()] = match.group(2).split()
    missing = {"paper", "appendix", "decoder"} - groups.keys()
    if missing:
        raise ValueError(f"{path} does not define {sorted(missing)}")
    return groups


def load_ablation(arm, dataset):
    path = os.path.join(ABLATION_DIR, f"{arm}_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        result = json.load(f)
    validate_result(result, path, arm, dataset)
    return deepcopy(result)


def active_datasets(rows):
    return [dataset for dataset in DATASETS if any(row[1].get(dataset) is not None for row in rows[1:])]


def parameter_cell(result_by_dataset):
    values = [
        result["model_info"]["params"] / 1e6
        for result in result_by_dataset.values()
        if result is not None and result["model_info"].get("params") is not None
    ]
    return f"{statistics.fmean(values):.2f}" if values else BLANK


def ablation_table(group, arms, metrics):
    rows = [(REFERENCE_NAME, {dataset: load(REFERENCE, dataset) for dataset in DATASETS})]
    rows.extend((ABLATION_NAMES.get(arm, arm), {dataset: load_ablation(arm, dataset) for dataset in DATASETS}) for arm in arms)

    datasets = active_datasets(rows)
    if not datasets:
        return None

    arrows = {True: r"$\uparrow$", False: r"$\downarrow$"}
    columns = "lc" + "c" * (len(metrics) * len(datasets))
    header = [r"\multirow{2}{*}{Model}", r"\multirow{2}{*}{Params (M)}"]
    header.extend(rf"\multicolumn{{{len(metrics)}}}{{c}}{{{DATASETS[dataset]}}}" for dataset in datasets)
    metric_header = ["", ""]
    for _ in datasets:
        metric_header.extend(f"{name}\\,{arrows[higher_better]}" for _, name, higher_better, _, _ in metrics)

    lines = [r"\begin{table}[t]", r"\centering"]
    lines.append(
        rf"\caption{{{GROUP_CAPTIONS[group]} Results are reported as mean\,$\pm$\,standard deviation "
        + r"across data splits. The full model is shaded, and a dash denotes a result not yet available.}"
    )
    lines.extend(
        [
            rf"\label{{tab:ablation-{group}}}",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
        ]
    )
    start = 3
    rules = []
    for _ in datasets:
        end = start + len(metrics) - 1
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(rules))
    lines.extend([" & ".join(metric_header) + r" \\", r"\midrule"])

    for index, (name, results) in enumerate(rows):
        cells = [r"\textbf{" + name + r"}" if index == 0 else name, parameter_cell(results)]
        for dataset in datasets:
            for key, _, _, percentage, decimals in metrics:
                result = results.get(dataset)
                if result is None or key not in result["summary"]:
                    cells.append(BLANK)
                    continue
                summary = result["summary"][key]
                cells.append(fmt_pct(summary) if percentage else fmt_num(summary, decimals))
        row = " & ".join(cells) + r" \\"
        if index == 0:
            row = r"\rowcolor{red!5} " + row
        lines.append(row)
        if index == 0:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines)


def main(args):
    groups = read_groups(GROUPS_FILE)
    metrics = {metric[0]: metric for metric in SEG_METRICS}
    selected = ("paper", "appendix", "decoder") if args.group == "all" else (args.group,)

    for group in selected:
        table = ablation_table(group, groups[group], [metrics[key] for key in ("val/iou", "val/dice_score", "val/hd95")])
        output = os.path.join(args.output_dir, f"ablation_{group}.tex")
        if table is None:
            print(f"warning: no {group} ablation results found in {ABLATION_DIR}, skipped {output}")
            continue
        write_report(output, [table])


if __name__ == "__main__":
    main(get_args())

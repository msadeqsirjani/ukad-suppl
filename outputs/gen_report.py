import argparse
import json
import os
import statistics
import sys
from copy import deepcopy
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = str(Path(HERE).parent)
sys.path.insert(0, ROOT)

EXPECTED_INPUT_SHAPES = {
    "busi": [1, 3, 256, 256],
    "cvc": [1, 3, 256, 256],
    "glas": [1, 3, 512, 512],
    "isic": [1, 3, 512, 512],
}

DATASETS = {
    "busi": "BUSI",
    "cvc": "CVC-ClinicDB",
    "glas": "GlaS",
    "isic": "ISIC 2018",
}

MODEL_ORDER = [
    "unet",
    "attunet",
    "unetpp",
    "unext_s",
    "unext",
    "unext_l",
    "mednext",
    "umamba",
    "nnunet_resenc",
    "rollingunet",
    "rollingunet_m",
    "rollingunet_l",
    "ukagnet",
    "ukan",
    "ufunkan",
    "cglknet",
    "ukanplus",
    "adakan",
    "ukad_s",
    "ukad_b",
    "ukad_l",
]

MODEL_NAMES = {
    "unet": "U-Net",
    "attunet": "Attention U-Net",
    "unetpp": "U-Net++",
    "unext_s": "UNeXt-S",
    "unext": "UNeXt-B",
    "unext_l": "UNeXt-L",
    "rollingunet": "Rolling-UNet-S",
    "rollingunet_m": "Rolling-UNet-M",
    "rollingunet_l": "Rolling-UNet-L",
    "ukan": "U-KAN",
    "ukanplus": "UKAN+",
    "ukagnet": "UKAGNet",
    "ufunkan": "U-FunKAN",
    "umamba": "U-Mamba",
    "adakan": "AdaKAN",
    "mednext": "MedNeXt",
    "nnunet_resenc": "nnU-Net ResEnc",
    "cglknet": "CGLKNet",
    "ukad": "UKAD",
    "ukad_s": "UKAD-S",
    "ukad_b": "UKAD-B",
    "ukad_l": "UKAD-L",
}

OURS = {"ukad_s", "ukad_b", "ukad_l"}
VARIANT_MODELS = {"ukad_s", "ukad_b", "ukad_l"}
BLANK = "-"

SEG_METRICS = [
    ("val/iou", "IoU", True, True, 2),
    ("val/dice_score", "Dice", True, True, 2),
    ("val/recall", "Recall", True, True, 2),
    ("val/specificity", "Specificity", True, True, 2),
    ("val/precision", "Precision", True, True, 2),
    ("val/hd", "HD", False, False, 2),
    ("val/hd95", "HD95", False, False, 2),
]


def load_costs():
    path = os.path.join(HERE, "performance", "costs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


COSTS = load_costs()


def model_info(data, model, dataset):
    if model in data[dataset]:
        return data[dataset][model]["model_info"]
    return COSTS.get(f"{model}_{dataset}", {})


def load(model, dataset):
    parts = [HERE, "performance"]
    if model in VARIANT_MODELS:
        parts.append("variants")
    p = os.path.join(*parts, f"{model}_{dataset}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        result = json.load(f)
    validate_result(result, p, model, dataset)
    return deepcopy(result)


def validate_result(result, path, model, dataset):
    required = {"experiment", "dataset", "evaluation", "summary", "model_info"}
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    expected_experiment = f"{model}_{dataset}"
    if result["experiment"] != expected_experiment:
        raise ValueError(f"{path} claims experiment {result['experiment']!r}, " f"expected {expected_experiment!r}")
    if result["dataset"] != dataset:
        raise ValueError(f"{path} claims dataset {result['dataset']!r}, expected {dataset!r}")
    expected_split = "validation"
    if result["evaluation"].get("split") != expected_split:
        raise ValueError(f"{path} has the wrong evaluation split")
    expected_prefix = "val/"
    if not result["summary"] or any(not key.startswith(expected_prefix) for key in result["summary"]):
        raise ValueError(f"{path} has metrics from the wrong split")
    if result["model_info"].get("input_shape") != EXPECTED_INPUT_SHAPES[dataset]:
        raise ValueError(f"{path} profiles {result['model_info'].get('input_shape')}, expected " f"{EXPECTED_INPUT_SHAPES[dataset]} for {dataset}")


def fmt_pct(m):
    mean = m["mean"] * 100
    if m.get("n_runs") == 1:
        return f"{mean:.2f}"
    std = m["std"] * 100
    return f"{mean:.2f}\\,$\\pm$\\,{std:.2f}"


def fmt_num(m, dec):
    mean = m["mean"]
    if m.get("n_runs") == 1:
        return f"{mean:.{dec}f}"
    std = m["std"]
    return f"{mean:.{dec}f}\\,$\\pm$\\,{std:.{dec}f}"


EFFICIENCY_METRICS = [
    ("params", 1e6, "Params (M)"),
    ("macs", 1e9, "MACs (G)"),
    ("flops", 1e9, "FLOPs (G)"),
]


def efficiency_info(data, model, dataset):
    if model in data:
        return data[model]["model_info"]
    return COSTS.get(f"{model}_{dataset}", {})


def efficiency_table(source_dataset, spatial, caption_datasets):
    models = list(MODEL_ORDER)
    data = {m: load(m, source_dataset) for m in models}
    data = {m: d for m, d in data.items() if d is not None}

    metric_values = {
        key: [(value / scale) if (value := efficiency_info(data, m, source_dataset).get(key)) is not None else None for m in models]
        for key, scale, _ in EFFICIENCY_METRICS
    }
    best = {key: best_indices(metric_values[key], False) for key, _, _ in EFFICIENCY_METRICS}
    second = {key: second_indices(metric_values[key], False) for key, _, _ in EFFICIENCY_METRICS}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Model efficiency (parameters, MACs and FLOPs), applicable to {caption_datasets}. "
        + r"Blue and red indicate the best and second-best results per column, respectively; "
        + r"a dash denotes a result not yet available.}"
    )
    lines.append(rf"\label{{tab:efficiency-{spatial}}}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & " + " & ".join(lbl for _, _, lbl in EFFICIENCY_METRICS) + r" \\")
    lines.append(r"\midrule")

    for i, m in enumerate(models):
        name = MODEL_NAMES[m]
        cells = [name]
        for key, _, _ in EFFICIENCY_METRICS:
            v = metric_values[key][i]
            if v is None:
                cells.append(BLANK)
                continue
            s = f"{v:.2f}"
            if i in best[key]:
                s = r"\textcolor{blue}{" + s + r"}"
            elif i in second[key]:
                s = r"\textcolor{red}{" + s + r"}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def best_indices(values, higher_better):
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return set()
    if higher_better:
        best = max(v for _, v in valid)
    else:
        best = min(v for _, v in valid)
    return {i for i, v in valid if abs(v - best) < 1e-12}


def second_indices(values, higher_better):
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return set()
    best = max(v for _, v in valid) if higher_better else min(v for _, v in valid)
    remaining = [(i, v) for i, v in valid if abs(v - best) >= 1e-12]
    if not remaining:
        return set()
    second = max(v for _, v in remaining) if higher_better else min(v for _, v in remaining)
    return {i for i, v in remaining if abs(v - second) < 1e-12}


def seg_table(metrics, label, caption, show_cost=True):
    models = list(MODEL_ORDER)
    data = {
        dataset: {model: result for model in models if (result := load(model, dataset)) is not None}
        for dataset in DATASETS
    }
    best = {}
    second = {}
    parameter_text = {}
    flops_text = {}
    for model in models:
        parameter_values = [
            value / 1e6 for dataset in DATASETS if (value := model_info(data, model, dataset).get("params")) is not None
        ]
        flops_values = [value / 1e9 for dataset in DATASETS if (value := model_info(data, model, dataset).get("flops")) is not None]
        parameter_text[model] = f"{statistics.fmean(parameter_values):.2f}" if parameter_values else BLANK
        flops_text[model] = f"{statistics.fmean(flops_values):.2f}" if flops_values else BLANK

    for dataset in DATASETS:
        for key, _, higher_better, _, _ in metrics:
            values = [data[dataset][model]["summary"][key]["mean"] if model in data[dataset] else None for model in models]
            best[dataset, key] = best_indices(values, higher_better)
            second[dataset, key] = second_indices(values, higher_better)

    arrows = {True: r"$\uparrow$", False: r"$\downarrow$"}
    lead = 3 if show_cost else 1
    columns = ("lcc" if show_cost else "l") + "c" * (len(metrics) * len(DATASETS))
    dataset_header = [r"\multirow{2}{*}{Method}"]
    if show_cost:
        dataset_header.extend([r"\multirow{2}{*}{Params (M)}", r"\multirow{2}{*}{FLOPs (G)}"])
    dataset_header.extend(rf"\multicolumn{{{len(metrics)}}}{{c}}{{{name}}}" for name in DATASETS.values())
    metric_header = [""] * lead
    for _ in DATASETS:
        metric_header.extend(f"{name}\\,{arrows[higher_better]}" for _, name, higher_better, _, _ in metrics)

    lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering"]
    cost_note = r"Parameters and FLOPs are averaged across dataset configurations. " if show_cost else ""
    lines.append(
        rf"\caption{{{caption} Results are reported as mean\,$\pm$\,standard deviation. "
        + r"\textcolor{blue}{Blue} and \textcolor{red}{Red} indicate the best and second-best results "
        + r"within each dataset, respectively. "
        + cost_note
        + r"The proposed model is shaded, and a dash denotes a result not yet available.}"
    )
    lines.extend(
        [
            rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(dataset_header) + r" \\",
        ]
    )
    start = lead + 1
    rules = []
    for _ in DATASETS:
        end = start + len(metrics) - 1
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(rules))
    lines.extend([" & ".join(metric_header) + r" \\", r"\midrule"])

    separated = False
    for model_index, model in enumerate(models):
        if model in OURS and not separated:
            lines.append(r"\midrule")
            separated = True
        name = MODEL_NAMES[model]
        if model in OURS:
            name = r"\textbf{" + name + r"}"
        cells = [name] + ([parameter_text[model], flops_text[model]] if show_cost else [])
        for dataset in DATASETS:
            for key, _, _, percentage, decimals in metrics:
                if model not in data[dataset]:
                    cells.append(BLANK)
                    continue
                summary = data[dataset][model]["summary"][key]
                value = fmt_pct(summary) if percentage else fmt_num(summary, decimals)
                if model_index in best[dataset, key]:
                    value = r"\textcolor{blue}{\textbf{" + value + r"}}"
                elif model_index in second[dataset, key]:
                    value = r"\textcolor{red}{\textbf{" + value + r"}}"
                cells.append(value)
        row = " & ".join(cells) + r" \\"
        if model in OURS:
            row = r"\rowcolor{red!5} " + row
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", r"\end{landscape}"])
    return "\n".join(lines)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def missing_results(result_dir):
    result_dir = Path(result_dir)
    expected = [
        result_dir / f"{model}_{dataset}.json"
        for model in MODEL_ORDER
        for dataset in DATASETS
        if (Path(ROOT) / "configs" / "baselines" / model / f"{dataset}.yaml").is_file()
    ]
    expected.extend(
        result_dir / "variants" / f"{model}_{dataset}.json"
        for model in VARIANT_MODELS
        for dataset in DATASETS
        if (Path(ROOT) / "configs" / "variants" / model / f"{dataset}.yaml").is_file()
    )
    if len(expected) != 76:
        raise RuntimeError(f"canonical report contract requires 76 configured results, found {len(expected)}")
    return [path for path in expected if not path.is_file()]


def render_doc(tables):
    doc = []
    doc.append(r"\PassOptionsToPackage{table}{xcolor}")
    doc.append(r"\documentclass[sigconf,nonacm]{acmart}")
    doc.append(r"\settopmatter{printacmref=false}")
    doc.append(r"\usepackage{booktabs}")
    doc.append(r"\usepackage{graphicx}")
    doc.append(r"\usepackage{amsmath}")
    doc.append(r"\usepackage{amssymb}")
    doc.append(r"\usepackage{tabularx}")
    doc.append(r"\usepackage{multirow}")
    doc.append(r"\usepackage{pdflscape}")
    doc.append(r"\usepackage{xcolor}")
    doc.append(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}")
    doc.append(r"\renewcommand{\arraystretch}{1.1}")
    doc.append("")
    doc.append(r"\begin{document}")
    doc.append(r"\pagestyle{empty}")
    doc.append(r"\thispagestyle{empty}")
    doc.append(r"\onecolumn")
    doc.append("")
    for table in tables:
        doc.append(table)
        doc.append("")
    doc.append(r"\end{document}")
    return "\n".join(doc) + "\n"


def write_report(output, tables):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(render_doc(tables))
    print("wrote", output)


def main(args):
    result_dir = Path(HERE) / "performance"
    if not any(result_dir.glob("*.json")):
        raise FileNotFoundError(f"no results found in {result_dir}")

    metrics = {metric[0]: metric for metric in SEG_METRICS}
    tables = [
        seg_table(
            [metrics[key] for key in ("val/iou", "val/dice_score", "val/hd95")],
            "tab:seg-overlap",
            "Segmentation overlap and boundary performance across all datasets and models.",
        ),
        seg_table(
            [metrics[key] for key in ("val/recall", "val/precision", "val/specificity", "val/hd")],
            "tab:seg-other",
            "Additional segmentation performance across all datasets and models.",
            show_cost=False,
        ),
    ]

    output = args.output or os.path.join(HERE, "report", "performance_report.tex")
    missing = missing_results(result_dir)
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remainder = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        print(f"warning: {len(missing)} result(s) still missing, left blank:\n{preview}{remainder}")
    write_report(output, tables)


if __name__ == "__main__":
    main(get_args())

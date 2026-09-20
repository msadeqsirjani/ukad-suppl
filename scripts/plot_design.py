import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import gen_ablation_report as ablation
import gen_report as report
import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "design.png"
DATASETS = ("busi", "cvc", "glas")
LABELS = ("BUSI", "CVC", "GlaS")
ARMS = (
    ("ukad_b", "UKAD-B", style.BAR_UKAD),
    ("ukad_hard_gate", "Hard Gate", style.BAR[0]),
    ("ukad_independent_gate", "Independent Gate", style.BAR[3]),
    ("ukad_no_adaptive_context", "W/O Adaptive Context", style.BAR[2]),
    ("ukad_no_boundary", "W/O Boundary", style.BAR[4]),
    ("ukad_no_decoder_gate", "W/O Decoder Gate", style.BAR[5]),
    ("ukad_no_deep_supervision", "W/O Deep Supervision", style.BAR[1]),
)


def iou(model, dataset):
    if model == "ukad_b":
        result = report.load(model, dataset)
    else:
        result = ablation.load_ablation(model, dataset)
    if result is None:
        raise SystemExit(f"missing {model}_{dataset}")
    return 100.0 * result["summary"]["val/iou"]["mean"], 100.0 * result["summary"]["val/iou"]["std"]


def collect():
    rows = []
    for model, name, color in ARMS:
        means, stds = zip(*(iou(model, dataset) for dataset in DATASETS))
        rows.append({"name": name, "color": color, "mean": np.array(means), "std": np.array(stds)})
    return rows


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    style.faceted_bars(collect(), LABELS, "IoU (%)", get_args().out)

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
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "decoder.png"
DATASETS = ("busi", "cvc", "glas", "isic")
LABELS = ("BUSI", "CVC", "GlaS", "ISIC")
ARMS = (
    ("ukad_b", "Hybrid", style.BAR_UKAD),
    ("ukad_conv_decoder", "Convolution", style.BAR[0]),
    ("ukad_mlp_decoder", "MLP", style.BAR[2]),
    ("ukad_kan_decoder", "KAN", style.BAR[3]),
)


def iou(model, dataset):
    result = report.load(model, dataset) if model == "ukad_b" else ablation.load_ablation(model, dataset)
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

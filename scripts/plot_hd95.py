import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import gen_report as report
import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "hd95.png"
DATASETS = ("busi", "cvc", "glas", "isic")
LABELS = ("BUSI", "CVC", "GlaS", "ISIC")
MODELS = (
    ("unet", "U-Net", style.BAR[0]),
    ("ukan", "U-KAN", style.BAR[3]),
    ("adakan", "AdaKAN", style.BAR[4]),
    ("rollingunet_l", "Rolling-UNet-L", style.BAR[5]),
    ("ukad_b", "UKAD-B", style.BAR_UKAD),
)


def hd95(model, dataset):
    result = report.load(model, dataset)
    if result is None:
        raise SystemExit(f"missing {model}_{dataset}")
    metric = result["summary"]["val/hd95"]
    return metric["mean"], metric["std"]


def collect():
    rows = []
    for model, name, color in MODELS:
        means, stds = zip(*(hd95(model, dataset) for dataset in DATASETS))
        rows.append({"name": name, "color": color, "mean": np.array(means), "std": np.array(stds)})
    return rows


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    style.faceted_bars(collect(), LABELS, "HD95 (pixels)", get_args().out, floor=0.0, ncol=3)

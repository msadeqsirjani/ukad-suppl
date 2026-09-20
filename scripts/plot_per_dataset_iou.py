import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import gen_report as report
import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "per_dataset_iou.png"
DATASETS = ("busi", "cvc", "glas", "isic")
TITLE = {"busi": "BUSI", "cvc": "CVC", "glas": "GlaS", "isic": "ISIC"}
MODELS = (
    ("unet", "U-Net", style.BAR[0]),
    ("ukan", "U-KAN", style.BAR[3]),
    ("adakan", "AdaKAN", style.BAR[4]),
    ("rollingunet_l", "Rolling-UNet-L", style.BAR[5]),
    ("ukad_b", "UKAD-B", style.BAR_UKAD),
)


def collect(datasets):
    rows = []
    for model, name, color in MODELS:
        means = []
        stds = []
        for dataset in datasets:
            result = report.load(model, dataset)
            if result is None:
                raise SystemExit(f"missing {model}_{dataset}")
            means.append(100.0 * result["summary"]["val/iou"]["mean"])
            stds.append(100.0 * result["summary"]["val/iou"]["std"])
        rows.append({"name": name, "color": color, "mean": np.array(means), "std": np.array(stds)})
    return rows


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    style.faceted_bars(collect(args.datasets), tuple(TITLE[d] for d in args.datasets), "IoU (%)", args.out, figsize=(5.0, 3.0), ncol=3)

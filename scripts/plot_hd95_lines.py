import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
DS = ["busi", "cvc", "glas", "isic"]
TITLE = ["BUSI", "CVC", "GlaS", "ISIC"]
MODELS = [
    ("U-Net", "unet"),
    ("Attention U-Net", "attunet"),
    ("U-Net++", "unetpp"),
    ("UNeXt-B", "unext"),
    ("U-Mamba", "umamba"),
    ("Rolling-UNet-L", "rollingunet_l"),
    ("UKAGNet", "ukagnet"),
    ("U-KAN", "ukan"),
    ("U-FunKAN", "ufunkan"),
    ("CGLKNet", "cglknet"),
    ("UKAN+", "ukanplus"),
    ("AdaKAN", "adakan"),
    ("UKAD-S", "ukad_s"),
    ("UKAD-B", "ukad_b"),
    ("UKAD-L", "ukad_l"),
]
BASE_COLOR = "#1683E6"
OURS_COLOR = "#F28C28"


def summary(stem, ds):
    for folder in ("", "variants/"):
        p = ROOT / "outputs" / "performance" / folder / f"{stem}_{ds}.json"
        if p.exists():
            return json.load(open(p))["summary"]["val/hd95"]
    raise FileNotFoundError(stem)


def hd95(stem, ds):
    return summary(stem, ds)["mean"]


def std(stem, ds):
    return summary(stem, ds)["std"]


def _unused(stem, ds):
    for folder in ("", "variants/"):
        p = ROOT / "outputs" / "performance" / folder / f"{stem}_{ds}.json"
        if p.exists():
            return json.load(open(p))["summary"]["val/hd95"]["mean"]
    raise FileNotFoundError(stem)


style.apply()
fig, axes = plt.subplots(3, 5, figsize=(9.0, 4.8), sharex=True, sharey=True)
xs = np.arange(len(DS))

for k, (name, stem) in enumerate(MODELS):
    ax = axes[k // 5, k % 5]
    vals = [hd95(stem, d) for d in DS]
    errs = [std(stem, d) for d in DS]
    color = OURS_COLOR if name.startswith("UKAD") else BASE_COLOR
    ax.errorbar(xs, vals, yerr=errs, fmt="o-", markersize=5.6, linewidth=1.6, color=color,
                ecolor="#C8CDD3", elinewidth=0.7, capsize=0.0, markeredgewidth=0.0, zorder=2)
    ax.set_title(name)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(TITLE, rotation=30, ha="right")
    style.restyle(ax)
    ax.tick_params(which="both", length=0)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    if k % 5 == 0:
        ax.set_ylabel("HD95 (px)")

fig.tight_layout()
out = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "hd95_lines.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print(f"wrote {out}")

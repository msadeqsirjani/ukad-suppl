import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

import gen_report as report
import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "pareto.png"
DATASETS = ("busi", "cvc", "glas", "isic")
TITLE = {"busi": "BUSI", "cvc": "CVC", "glas": "GlaS", "isic": "ISIC"}
FAMILIES = (
    ("UNeXt", ("unext_s", "unext", "unext_l"), style.TAB[0]),
    ("Rolling-UNet", ("rollingunet", "rollingunet_m", "rollingunet_l"), style.TAB[1]),
    ("UKAD", ("ukad_s", "ukad_b", "ukad_l"), style.TAB[2]),
)


def series(models, dataset):
    xs, ys, sds = [], [], []
    for model in models:
        result = report.load(model, dataset)
        if result is None:
            raise SystemExit(f"missing {model}_{dataset}")
        xs.append(result["model_info"]["params"] / 1e6)
        ys.append(100.0 * result["summary"]["val/iou"]["mean"])
        stat = result["summary"]["val/iou"]
        sds.append(100.0 * stat["std"] / np.sqrt(stat["n_runs"]))
    return np.array(xs), np.array(ys), np.array(sds)


def draw(out_path):
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(3.75, 3.75))
    axes = axes.ravel()
    for j, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        for name, models, color in FAMILIES:
            xs, ys, sds = series(models, dataset)
            ax.fill_between(xs, ys - sds, ys + sds, color=color, alpha=0.08, linewidth=0.0, zorder=1)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.4,
                marker="o",
                markersize=4.2,
                markeredgewidth=0.0,
                label=name if j == 0 else None,
                zorder=3 if name == "UKAD" else 2,
            )
        ax.set_xscale("log")
        ax.set_title(TITLE[dataset], pad=3)
        ax.set_xlabel("Params (M)", fontsize=style.SIZE - 3.0)
        style.clean(ax)
        ax.set_ylabel("IoU (%)", fontsize=style.SIZE - 3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    style.place_legend(fig, handles, labels, fontsize=style.SIZE - 5.0, anchor=(0.5, 0.975), loc="upper center", ncol=3)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.145, top=0.855, wspace=0.42, hspace=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    draw(get_args().out)

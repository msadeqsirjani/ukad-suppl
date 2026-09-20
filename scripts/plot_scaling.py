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
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "scaling.png"
DATASETS = ("busi", "cvc", "glas", "isic")
FAMILIES = (
    ("UNeXt", ("unext_s", "unext", "unext_l"), style.OKABE[3], "o"),
    ("Rolling-UNet", ("rollingunet", "rollingunet_m", "rollingunet_l"), style.OKABE[5], "s"),
    ("UKAD", ("ukad_s", "ukad_b", "ukad_l"), style.UKAD, "*"),
)


def mean_iou(model):
    values = []
    for dataset in DATASETS:
        result = report.load(model, dataset)
        if result is None:
            raise SystemExit(f"missing {model}_{dataset}")
        values.append(result["summary"]["val/iou"]["mean"])
    return 100.0 * float(np.mean(values))


def params_m(model):
    result = report.load(model, "busi")
    return result["model_info"]["params"] / 1e6


def collect():
    families = []
    for name, models, color, marker in FAMILIES:
        xs = [params_m(model) for model in models]
        ys = [mean_iou(model) for model in models]
        families.append({"name": name, "x": xs, "y": ys, "color": color, "marker": marker})
    return families


def draw(families, out_path):
    style.apply()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for family in families:
        ours = family["name"] == "UKAD"
        ax.plot(
            family["x"],
            family["y"],
            color=family["color"],
            linewidth=1.6 if ours else 1.1,
            linestyle="-" if ours else (0, (3.5, 2.2)),
            zorder=3,
        )
        ax.scatter(
            family["x"],
            family["y"],
            s=78 if ours else 32,
            marker=family["marker"],
            color=family["color"],
            zorder=4,
            label=family["name"],
            linewidths=0.0,
        )
        for x, y in zip(family["x"], family["y"]):
            offset, ha = ((0, 10), "center") if ours else ((0, -13), "center")
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=offset,
                ha=ha,
                fontsize=style.SIZE,
                color=family["color"],
            )
    ys = [y for family in families for y in family["y"]]
    xs = [x for family in families for x in family["x"]]
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Mean IoU (%)")
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.35, max(xs) * 1.25)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.55)
    style.restyle(ax)
    style.legend_top(fig, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main(args):
    draw(collect(), args.out)


if __name__ == "__main__":
    main(get_args())

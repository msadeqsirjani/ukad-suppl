import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter, ScalarFormatter

import gen_report as report
import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "efficiency.png"
DEFAULT_DATASETS = ("busi", "cvc", "glas", "isic")
FLOPS_DATASETS = DEFAULT_DATASETS

OURS = "ukad"
OURS_COLOR = style.UKAD
MODEL_COLOR = style.MODEL_COLOR

LABELS = {
    "unext": (-2, -16, "center", True),
    "ukan": (0, 11, "center", False),
    "rollingunet": (-26, -14, "right", True),
    "ukagnet": (0, 15, "center", True),
    "ufunkan": (11, 2, "left", True),
    "adakan": (0, 11, "center", True),
    "cglknet": (0, -24, "center", True),
    "ukad": (12, 7, "left", False),
    "umamba": (2, -21, "center", True),
    "unetpp": (14, 15, "left", True),
    "unet": (0, -13, "center", True),
    "attunet": (13, 3, "left", True),
}

VARIANT_FAMILY = {
    "unext_s": "unext",
    "unext": "unext",
    "unext_l": "unext",
    "rollingunet": "rollingunet",
    "rollingunet_m": "rollingunet",
    "rollingunet_l": "rollingunet",
    "ukad_s": "ukad",
    "ukad_b": "ukad",
    "ukad_l": "ukad",
}

FAMILY_PICK = {
    "ukad": "ukad_l",
}

FAMILY_NAME = {
    "unext": "UNeXt",
    "rollingunet": "Rolling-UNet",
    "ukad": "UKAD",
}

Y_TICKS = (75, 76, 78, 80, 82, 83)

SIZE_REFS_M = (5, 15, 30)
FONT = style.SIZE
LABEL_SCALE = 1.15
TEXT_INK = style.INK


def size_to_area(params_m):
    return 12.0 + 11.0 * params_m


def collect(datasets, flops_datasets=FLOPS_DATASETS):
    rows = []
    for model in report.MODEL_ORDER:
        results = [report.load(model, d) for d in datasets]
        if any(r is None for r in results):
            absent = [d for d, r in zip(datasets, results) if r is None]
            print(f"skip {model}: no result for {', '.join(absent)}")
            continue
        cost = [report.load(model, d) for d in flops_datasets]
        cost = [r for r in cost if r is not None] or results
        flops = [r["model_info"]["flops"] for r in cost]
        iou = [r["summary"]["val/iou"]["mean"] for r in results]
        rows.append(
            {
                "model": model,
                "family": VARIANT_FAMILY.get(model, model),
                "name": FAMILY_NAME.get(VARIANT_FAMILY.get(model, model), report.MODEL_NAMES[model]),
                "flops_g": sum(flops) / len(flops) / 1e9,
                "params_m": sum(r["model_info"]["params"] for r in cost) / len(cost) / 1e6,
                "iou": 100.0 * sum(iou) / len(iou),
            }
        )
    return best_per_family(rows)


def best_per_family(rows):
    best = {}
    for row in rows:
        family = row["family"]
        pick = FAMILY_PICK.get(family)
        if pick is not None and row["model"] != pick:
            continue
        current = best.get(family)
        if current is None or row["iou"] > current["iou"]:
            best[family] = row
    return [best[f] for f in dict.fromkeys(row["family"] for row in rows) if f in best]


def frontier(rows):
    front = []
    best = float("-inf")
    for r in sorted(rows, key=lambda r: r["flops_g"]):
        if r["iou"] > best:
            front.append(r)
            best = r["iou"]
    return front


def style_axes():
    style.apply()
    plt.rcParams.update(
        {
            "font.size": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "legend.title_fontsize": FONT,
        }
    )


def draw(rows, out_path):
    style_axes()
    fig, ax = plt.subplots(figsize=(5.2, 3.51))

    ours_row = max(rows, key=lambda r: r["iou"] if r["family"] == OURS else float("-inf"))
    ax.axhline(
        ours_row["iou"],
        color=OURS_COLOR,
        linewidth=0.7,
        linestyle=(0, (4, 3)),
        alpha=0.45,
        zorder=0,
    )

    front = frontier(rows)
    ax.plot(
        [r["flops_g"] for r in front],
        [r["iou"] for r in front],
        color="#C5CBD3",
        linestyle=(0, (5, 3)),
        linewidth=0.9,
        zorder=1,
    )

    for r in rows:
        ours = r["family"] == OURS
        ax.scatter(
            r["flops_g"],
            r["iou"],
            s=size_to_area(r["params_m"]) * 3.4 if ours else size_to_area(r["params_m"]),
            marker="*" if ours else "o",
            color=MODEL_COLOR.get(r["family"], "#9BA7B4"),
            edgecolors="none",
            alpha=1.0 if ours else 0.9,
            zorder=5 if ours else 3,
        )

    for r in rows:
        ours = r["family"] == OURS
        label = r["name"]
        dx, dy, ha, leader = LABELS.get(r["family"], (11, 5, "left", False))
        arrow = dict(arrowstyle="-", color="#B0B0B0", linewidth=0.6, shrinkA=0.0, shrinkB=3.0) if leader else None
        ax.annotate(
            label,
            (r["flops_g"], r["iou"]),
            textcoords="offset points",
            xytext=(dx * LABEL_SCALE, dy * LABEL_SCALE),
            ha=ha,
            va="center",
            fontsize=FONT,
            fontweight="normal",
            color=TEXT_INK,
            arrowprops=arrow,
            zorder=6,
        )

    flops = [r["flops_g"] for r in rows]
    iou = [r["iou"] for r in rows]
    span = max(iou) - min(iou)
    ax.set_xscale("log")
    ax.set_xlim(min(flops) / 2.4, max(flops) * 11.0)
    ax.set_ylim(min(iou) - 0.10 * span, max(iou) + 0.16 * span)

    unet_iou = next((r["iou"] for r in rows if r["family"] == "unet"), None)
    if unet_iou is not None:
        ax.axhspan(ax.get_ylim()[0], unet_iou, color="#C0504D", alpha=0.06, linewidth=0, zorder=0)
        ax.annotate(
            "Below Plain U-Net",
            (ax.get_xlim()[0], ax.get_ylim()[0]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=FONT - 1,
            color="#9C4340",
            zorder=2,
        )

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="out", length=3.0, color="#8A8A8A")

    ax.set_xlabel("Mean FLOPs (G)")
    ax.set_ylabel("Mean IoU (%)")
    low, high = ax.get_ylim()
    ticks = [t for t in Y_TICKS if low <= t <= high]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.annotate(
        f"{ours_row['iou']:.2f}",
        (ax.get_xlim()[1], ours_row["iou"]),
        textcoords="offset points",
        xytext=(-2, 5),
        ha="right",
        fontsize=FONT,
        color=OURS_COLOR,
        fontweight="normal",
    )

    style.clean(ax)

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=style.SLATE,
            markeredgecolor=style.INK,
            markeredgewidth=0.8,
            markersize=4.5 + 0.16 * p,
            label=f"{p} M",
        )
        for p in SIZE_REFS_M
    ]
    legend = ax.legend(
        handles=size_handles,
        loc="lower right",
        ncol=1,
        title="Params (M)",
        fontsize=FONT - 2.0,
        title_fontsize=FONT - 2.0,
        frameon=True,
        fancybox=False,
        edgecolor=style.INK,
        framealpha=1.0,
        borderaxespad=0.5,
        labelspacing=0.75,
        handlelength=2.2,
        handletextpad=0.9,
        borderpad=0.45,
    )
    legend.get_title().set_color(TEXT_INK)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--flops-datasets", nargs="+", default=list(FLOPS_DATASETS))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main(args):
    rows = collect(tuple(args.datasets), tuple(args.flops_datasets))
    if len(rows) < 2:
        raise SystemExit("not enough results to plot")
    draw(rows, args.out)


if __name__ == "__main__":
    main(get_args())

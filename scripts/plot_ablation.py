"""Plot paired IoU and HD95 ablation effects, separately for each dataset."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as style


ROOT = Path(__file__).resolve().parent.parent
PERF = ROOT / "outputs" / "performance"
OUT = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "ablation.png"
SPLITS = ("dataseed2981", "dataseed6142", "dataseed1187")
DEFAULT = "variants/ukad_b"
DATASETS = (("busi", "BUSI"), ("cvc", "CVC"), ("glas", "GlaS"), ("isic", "ISIC"))
COMPONENTS = (
    ("Displacement", "ukad_no_displacement", style.BLUE),
    ("Group-Rational Form", "ukad_no_kan", style.SAND),
    ("Displacement + Group-Rational Form", "ukad_no_displacement_no_kan", style.UKAD),
    ("Context Branch", "ukad_no_context", style.TEAL),
    ("Hybrid Decoder", "ukad_kan_decoder", style.LILAC),
    ("Adaptive Context", "ukad_no_adaptive_context", style.ROSE),
)


def runs(name, dataset, metric):
    record = json.loads((PERF / f"{name}_{dataset}.json").read_text())
    return np.array([record["per_run"][split][metric] for split in SPLITS])


def paired_effect(arm, dataset, metric, scale):
    """Positive values favor retaining the component, split by split."""
    return scale * (runs(DEFAULT, dataset, metric) - runs(f"ablations/{arm}", dataset, metric))


def limits(metric, scale):
    values = []
    for dataset, _ in DATASETS:
        for _, arm, _ in COMPONENTS:
            effect = paired_effect(arm, dataset, metric, scale)
            values.extend((effect.mean() - effect.std(ddof=1), effect.mean() + effect.std(ddof=1), 0.0))
    step = 1.0 if metric == "val/iou" else 0.5
    return step * np.floor((min(values) - 0.10) / step), step * np.ceil((max(values) + 0.10) / step), step


def draw_panel(ax, dataset, title, x_limits, y_limits):
    x_min, x_max, _ = x_limits
    y_min, y_max, _ = y_limits
    zero_y = (0.0 - y_min) / (y_max - y_min)
    # The two shaded quadrants give the bivariate effect an immediate reading:
    # upper-right improves both measures, while lower-left worsens both.
    ax.axvspan(0.0, x_max, ymin=zero_y, ymax=1.0, color="#EDF6EE", zorder=-2)
    ax.axvspan(x_min, 0.0, ymin=0.0, ymax=zero_y, color="#FBEFF0", zorder=-2)
    ax.axvline(0.0, color=style.INK, linewidth=0.75, zorder=1)
    ax.axhline(0.0, color=style.INK, linewidth=0.75, zorder=1)
    for label, arm, color in COMPONENTS:
        iou = paired_effect(arm, dataset, "val/iou", 100.0)
        hd95 = paired_effect(arm, dataset, "val/hd95", -1.0)
        ax.errorbar(
            iou.mean(),
            hd95.mean(),
            xerr=iou.std(ddof=1),
            yerr=hd95.std(ddof=1),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.75,
            capsize=1.5,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
            label=label,
        )
    ax.set_xlim(*x_limits[:2])
    ax.set_ylim(*y_limits[:2])
    ax.set_title(title, fontsize=style.SIZE - 1.0, pad=3)
    ax.xaxis.set_major_locator(MultipleLocator(x_limits[2]))
    ax.yaxis.set_major_locator(MultipleLocator(y_limits[2]))
    ax.grid(True, color=style.GRID, linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", labelsize=style.SIZE - 4.2)
    style.box(ax)
    if dataset == "busi":
        ax.text(
            0.97,
            0.96,
            "better\nboth",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=style.SIZE - 6.2,
            color="#44764B",
        )
        ax.text(
            0.03,
            0.04,
            "worse\nboth",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=style.SIZE - 6.2,
            color="#9A555A",
        )


def main():
    style.apply()
    x_limits = limits("val/iou", 100.0)
    y_limits = limits("val/hd95", -1.0)
    fig, axes = plt.subplots(2, 2, figsize=(3.75, 3.75), sharex=True, sharey=True)
    for ax, (dataset, title) in zip(axes.ravel(), DATASETS):
        draw_panel(ax, dataset, title, x_limits, y_limits)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\Delta$IoU (pp; right is better)", fontsize=style.SIZE - 4.0, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta$HD95 (px; up is better)", fontsize=style.SIZE - 4.0, labelpad=2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.place_legend(fig, handles, labels, fontsize=style.SIZE - 5.3, anchor=(0.5, 0.985), loc="upper center", ncol=2)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.13, top=0.82, wspace=0.27, hspace=0.38)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as style


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "main_ablation.png"
DATASETS = ("BUSI", "CVC", "GlaS", "ISIC")
COMPONENTS = (
    ("Displ.", "Displacement", style.TAB[0], (-0.28, 0.39, -0.30, -0.14), (0.69, 0.15, 0.70, 0.38)),
    ("Rational", "Group-Rational Form", style.TAB[1], (-0.41, 0.03, 0.24, -0.27), (0.57, 0.45, 0.54, 0.34)),
    ("Context", "Context Branch", style.TAB[2], (0.24, 0.11, 0.24, 0.20), (1.44, 0.21, 0.54, 0.10)),
    ("Hybrid", "Hybrid KAN Decoder", style.TAB[4], (0.09, 0.86, 0.28, 0.33), (0.44, 0.55, 0.05, 0.32)),
    ("Adaptive", "Adaptive Context", style.TAB[6], (0.11, 0.78, 0.09, 1.04), (0.95, 0.36, 0.20, 0.28)),
)


def main():
    style.apply()
    # Match the Pareto panel: four datasets in a near-square 2x2 layout.
    fig, axes = plt.subplots(2, 2, figsize=(3.75, 3.75), sharey=True)
    axes = axes.ravel()
    positions = np.arange(len(COMPONENTS))
    for dataset_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        values = [component[3][dataset_index] for component in COMPONENTS]
        stds = [component[4][dataset_index] for component in COMPONENTS]
        colors = [component[2] for component in COMPONENTS]
        ax.bar(
            positions,
            values,
            yerr=stds,
            color=colors,
            width=0.66,
            edgecolor="none",
            error_kw={"ecolor": style.INK, "elinewidth": 0.55, "capsize": 1.4},
            zorder=3,
        )
        ax.axhline(0.0, color=style.UKAD, linewidth=0.85, linestyle=(0, (3.2, 2.0)), zorder=2)
        ax.set_title(dataset, fontsize=style.SIZE, pad=4)
        style.clean(ax)
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("Component", fontsize=style.SIZE - 3.0, labelpad=0.0)
        # Match the baseline of Figure 3's ``Params (M)`` labels.  Its log
        # tick labels otherwise make Matplotlib place the x label lower.
        ax.xaxis.set_label_coords(0.5, -0.17)
        ax.set_ylabel(r"Paired $\Delta$ IoU (%)", fontsize=style.SIZE - 3.0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=component[2], label=component[0])
        for component in COMPONENTS
    ]
    # Use abbreviated keys so all five bar categories fit in the same one-row
    # legend area used by the Pareto figure.
    style.place_legend(fig, handles, fontsize=style.SIZE - 5.0, anchor=(0.5, 0.975), loc="upper center", ncol=5)
    # Match the Pareto figure exactly, including the lower margin reserved for
    # its x-axis labels, so the paired subplot areas have the same dimensions.
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.145, top=0.855, wspace=0.42, hspace=0.55)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()

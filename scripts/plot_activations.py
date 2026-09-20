import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style
import paper_viz as viz

OUT_PATH = viz.FIGURES / "activations.png"
WHICH = ("inner", "outer")
GROUP_COLORS = style.GROUP
LEGEND_LABELS = [f"G = {i + 1}" for i in range(8)]
MAIN_FONT = style.SIZE
MAIN_DATASETS = viz.DATASETS


def draw(seed, out_path):
    viz.style_axes()
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.0), sharex=True, sharey=True)
    xs = np.linspace(-2.5, 2.5, 401)
    handles = []
    for col, dataset in enumerate(viz.DATASETS):
        for row, which in enumerate(WHICH):
            ax = axes[row, col]
            a, b = viz.load_group_rational("ukad_b", dataset, seed, which=which)
            for group in range(a.shape[0]):
                ys = viz.rational_curve(a[group], b[group], xs)
                line = ax.plot(xs, ys, color=GROUP_COLORS[group], linewidth=1.25, label=f"$g={group + 1}$")
                if row == 0 and col == 0:
                    handles.extend(line)
            ax.plot(xs, xs, color="#B8B8B8", linewidth=0.7, linestyle=(0, (3.5, 2.2)), zorder=0)
            ax.axhline(0.0, color="#EEEEEE", linewidth=0.5, zorder=0)
            ax.axvline(0.0, color="#EEEEEE", linewidth=0.5, zorder=0)
            if row == 0:
                ax.set_title(viz.DATASET_TITLE[dataset])
            if col == 0:
                ax.set_ylabel(rf"{which.capitalize()} $\phi$")
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal", adjustable="box")
            style.restyle(ax)
            ax.yaxis.grid(False)
    for ax in axes[1]:
        ax.set_xlabel(r"$t$")
    fig.tight_layout()
    style.legend_above(fig, axes, handles, LEGEND_LABELS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def draw_main(seed, out_path):
    viz.style_axes()
    plt.rcParams.update(
        {
            "font.size": MAIN_FONT,
            "axes.titlesize": MAIN_FONT,
            "axes.labelsize": MAIN_FONT,
            "xtick.labelsize": MAIN_FONT,
            "ytick.labelsize": MAIN_FONT,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.35), sharex=True, sharey=True)
    xs = np.linspace(-2.5, 2.5, 401)
    handles = []
    for col, dataset in enumerate(MAIN_DATASETS):
        ax = axes[col]
        a, b = viz.load_group_rational("ukad_b", dataset, seed, which="outer")
        for group in range(a.shape[0]):
            ys = viz.rational_curve(a[group], b[group], xs)
            line = ax.plot(xs, ys, color=GROUP_COLORS[group], linewidth=1.25, label=f"$g={group + 1}$")
            if col == 0:
                handles.extend(line)
        ax.plot(xs, xs, color="#B8B8B8", linewidth=0.7, linestyle=(0, (3.5, 2.2)), zorder=0)
        ax.set_title(viz.DATASET_TITLE[dataset])
        if col == 0:
            ax.set_ylabel(r"outer $\phi$")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$t$")
        style.restyle(ax)
        ax.yaxis.grid(False)
    fig.tight_layout()
    style.legend_above(fig, axes, handles, LEGEND_LABELS, ncol=4, fontsize=MAIN_FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=viz.SPLIT_SEED)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--main", action="store_true")
    return parser.parse_args()


def main(args):
    if args.main:
        draw_main(args.seed, args.out)
    else:
        draw(args.seed, args.out)


if __name__ == "__main__":
    main(get_args())

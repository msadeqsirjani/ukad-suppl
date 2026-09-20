import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures" / "contrast.png"
COLUMNS = (
    ("U-Net", style.MODEL_COLOR["unet"], "Fixed $\\phi$", "Fixed Grid"),
    ("U-KAN", style.MODEL_COLOR["ukan"], "Learned $\\phi$", "Fixed Grid"),
    ("UKAD (ours)", style.MODEL_COLOR["ukad"], "Group-Rational $\\phi$", "Adaptive Grid"),
)
GRID_X, GRID_Y = 6, 4


def activations(index, xs):
    if index == 0:
        return [np.maximum(xs, 0.0)]
    if index == 1:
        return [0.85 * xs + 0.40 * np.sin(1.7 * xs)]
    return [
        xs / (1.0 + 0.16 * np.abs(xs)),
        (xs + 0.34 * xs**2) / (1.0 + 0.52 * xs**2),
        1.55 * xs / (1.0 + 0.85 * np.abs(xs)),
        (xs + 0.55 * xs**3) / (1.0 + 1.25 * xs**2),
    ]


def grid_points():
    xs, ys = np.meshgrid(np.linspace(0.12, 0.88, GRID_X), np.linspace(0.16, 0.84, GRID_Y))
    return xs, ys


def offsets(xs, ys):
    dx = 0.044 * np.sin(3.1 * np.pi * ys) + 0.026 * np.cos(2.2 * np.pi * xs)
    dy = 0.039 * np.cos(2.6 * np.pi * xs) - 0.022 * np.sin(1.9 * np.pi * ys)
    return dx, dy


def tint(color):
    rgb = np.array(to_rgb(color))
    return tuple(1.0 - 0.07 * (1.0 - rgb))


def draw_activation(ax, index, color):
    xs = np.linspace(-2.5, 2.5, 401)
    curves = activations(index, xs)
    for k, ys in enumerate(curves):
        ax.plot(xs, ys, color=color, alpha=1.0 - 0.13 * k, linewidth=1.2 if k == 0 else 1.0, zorder=3)
    ax.axhline(0.0, color=style.GRID, linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=style.GRID, linewidth=0.6, zorder=1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(0.78)
    style.box(ax)


def draw_grid(ax, index, color):
    xs, ys = grid_points()
    if index < 2:
        ax.scatter(xs, ys, s=12, color=color, alpha=0.85, edgecolors="none", zorder=3)
    else:
        dx, dy = offsets(xs, ys)
        ax.scatter(xs, ys, s=9, facecolors="none", edgecolors=style.SLATE, linewidths=0.5, alpha=0.55, zorder=2)
        ax.quiver(xs, ys, dx, dy, color=color, angles="xy", scale_units="xy", scale=1.0, width=0.010, headwidth=3.6, headlength=4.4, zorder=3)
        ax.scatter(xs + dx, ys + dy, s=12, color=color, edgecolors="none", zorder=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.03, 0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(0.78)
    style.box(ax)


def draw(out_path):
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(3.9, 2.45), gridspec_kw={"height_ratios": (1.0, 1.0)})
    for index, (name, color, phi, grid) in enumerate(COLUMNS):
        top, bottom = axes[0, index], axes[1, index]
        draw_activation(top, index, color)
        draw_grid(bottom, index, color)
        for ax in (top, bottom):
            ax.set_facecolor(tint(color))
        top.set_title(name, color=color, fontsize=style.SIZE - 0.5, pad=3)
        top.set_xlabel(phi, fontsize=style.SIZE - 1.0, labelpad=2)
        bottom.set_xlabel(grid, fontsize=style.SIZE - 1.0, labelpad=2)
    fig.tight_layout(w_pad=0.6, h_pad=1.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    draw(get_args().out)

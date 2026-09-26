from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np

_LIBERATION = Path("/usr/share/fonts/truetype/liberation")
for _name in (
    "LiberationSerif-Regular.ttf",
    "LiberationSerif-Bold.ttf",
    "LiberationSerif-Italic.ttf",
    "LiberationSerif-BoldItalic.ttf",
):
    _path = _LIBERATION / _name
    if _path.exists():
        fm.fontManager.addfont(str(_path))

FONT = "Liberation Serif"
SIZE = 12.0

TAB = ("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF")

BLUE = TAB[0]
SAND = TAB[1]
TEAL = TAB[2]
UKAD = TAB[3]
LILAC = TAB[4]
BROWN = TAB[5]
ROSE = TAB[6]
SLATE = TAB[7]
OLIVE = TAB[8]
INK = "#333333"
GRID = "#E6E6E6"

OKABE = (BLUE, TEAL, SAND, LILAC, ROSE, SLATE, OLIVE)
BAR = ("#1683E6", "#607D8B", "#F28C28", "#7554D8", "#F04470", "#E5A623", "#2687D4")
BAR_UKAD = "#38A84A"
GROUP = (BLUE, SAND, TEAL, UKAD, LILAC, SLATE, ROSE, OLIVE)
HEAT = plt.get_cmap("cividis_r")
GATE = HEAT
CONTOUR = (85, 168, 104)
CONTOUR_GT = BLUE
CONTOUR_PRED = UKAD
OVERLAY = {
    "tp": np.array(to_rgb(TEAL)),
    "fp": np.array(to_rgb(UKAD)),
    "fn": np.array(to_rgb(BLUE)),
}
MODEL_COLOR = {
    "unet": BLUE,
    "attunet": TEAL,
    "unetpp": OLIVE,
    "unext": SAND,
    "rollingunet": SLATE,
    "ukan": LILAC,
    "ukanplus": ROSE,
    "ukagnet": BROWN,
    "ufunkan": LILAC,
    "umamba": SLATE,
    "adakan": ROSE,
    "cglknet": TAB[9],
    "mednext": "#393B79",
    "nnunet_resenc": "#7B4173",
    "ukad": UKAD,
}


def apply():
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.serif": (FONT, "Times New Roman", "Nimbus Roman", "Times"),
            "mathtext.fontset": "stix",
            "font.size": SIZE,
            "axes.titlesize": SIZE,
            "axes.labelsize": SIZE,
            "xtick.labelsize": SIZE,
            "ytick.labelsize": SIZE,
            "legend.fontsize": SIZE,
            "legend.title_fontsize": SIZE,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 0.0,
            "ytick.major.size": 0.0,
            "xtick.minor.size": 0.0,
            "ytick.minor.size": 0.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": INK,
            "legend.framealpha": 1.0,
            "axes.grid": False,
            "axes.titlepad": 4.0,
        }
    )


def box(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK)
    ax.tick_params(which="both", length=0.0, width=0.7, color=INK, direction="in", top=True, right=True)


def clean(ax):
    box(ax)
    ax.grid(False)
    ax.set_axisbelow(True)


def restyle(ax):
    box(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def legend_ncol(_n=None):
    return 4


def _pad_legend(handles, labels, ncol=4):
    handles = list(handles)
    labels = list(labels)
    while len(handles) < ncol or len(handles) % ncol != 0:
        handles.append(plt.Line2D([], [], linestyle="None", marker="None", color="none", label="\u00a0"))
        labels.append("\u00a0")
    nrow = len(handles) // ncol
    order_h, order_l = [], []
    for col in range(ncol):
        for row in range(nrow):
            index = row * ncol + col
            order_h.append(handles[index])
            order_l.append(labels[index])
    return order_h, order_l


def place_legend(fig, handles, labels=None, fontsize=None, anchor=(0.5, 1.0), loc="upper center", ncol=4):
    if fontsize is None:
        fontsize = SIZE
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    handles, labels = _pad_legend(handles, labels, ncol)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor=INK,
        framealpha=1.0,
        fontsize=fontsize,
        handlelength=1.15,
        handletextpad=0.4,
        columnspacing=1.15,
        borderpad=0.25,
        labelspacing=0.35,
        borderaxespad=0.0,
    )


def legend_above(fig, axes, handles, labels=None, pad=0.015, ncol=4, fontsize=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    top = max(inverse.transform_bbox(ax.get_tightbbox(renderer)).y1 for ax in np.ravel(axes))
    place_legend(fig, handles, labels, fontsize=fontsize, anchor=(0.5, top + pad), loc="lower center", ncol=ncol)


def legend_top(fig, ncol=None, fontsize=None):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    place_legend(fig, handles, labels, fontsize=fontsize)


def tight_ylim(means, stds, floor=None):
    lo = float(np.min(np.asarray(means) - np.asarray(stds)))
    hi = float(np.max(np.asarray(means) + np.asarray(stds)))
    pad = max(0.35, 0.22 * (hi - lo) if hi > lo else 0.8)
    low = lo - pad
    if floor is not None:
        low = max(floor, low)
    return low, hi + pad


def _save(fig, out_path, rect=(0.0, 0.0, 1.0, 0.91), w_pad=1.2):
    fig.tight_layout(rect=rect, w_pad=w_pad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def faceted_dots(rows, labels, ylabel, out_path, figsize=None, floor=None):
    apply()
    n_ds = len(labels)
    n_arm = len(rows)
    if figsize is None:
        figsize = (1.4 * n_ds + 0.75, 2.3)
    fig, axes = plt.subplots(1, n_ds, figsize=figsize, sharey=False)
    if n_ds == 1:
        axes = [axes]
    xs = np.arange(n_arm)
    for j, ax in enumerate(axes):
        means = np.array([row["mean"][j] for row in rows])
        stds = np.array([row["std"][j] for row in rows])
        ax.plot(xs, means, color="#D5C6B0", linewidth=0.8, zorder=1)
        for i, row in enumerate(rows):
            ax.errorbar(
                i,
                row["mean"][j],
                yerr=row["std"][j],
                fmt="o",
                color=row["color"],
                ecolor=INK,
                elinewidth=0.55,
                capsize=1.3,
                markersize=5.4,
                markeredgewidth=0.0,
                zorder=3,
            )
        ax.set_xlim(-0.45, n_arm - 0.55)
        ax.set_ylim(*tight_ylim(means, stds, floor=floor))
        ax.set_xticks([])
        ax.set_title(labels[j], pad=3)
        restyle(ax)
        if j == 0:
            ax.set_ylabel(ylabel)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=row["color"],
            linestyle="",
            markersize=5.4,
            markeredgewidth=0.0,
            label=row["name"],
        )
        for row in rows
    ]
    place_legend(fig, handles)
    _save(fig, out_path)


def delta_forest(rows, labels, out_path, xlabel=r"$\Delta$ IoU", figsize=None):
    apply()
    ref = rows[0]
    others = rows[1:]
    n_ds = len(labels)
    n_arm = len(others)
    if figsize is None:
        figsize = (1.7 * n_ds + 2.3, 0.42 * n_arm + 1.35)
    fig, axes = plt.subplots(1, n_ds, figsize=figsize, sharey=True)
    if n_ds == 1:
        axes = [axes]
    ys = np.arange(n_arm)
    for j, ax in enumerate(axes):
        ax.axvline(0.0, color=UKAD, linewidth=0.85, linestyle=(0, (3.2, 2.0)), zorder=1)
        for i, row in enumerate(others):
            delta = row["mean"][j] - ref["mean"][j]
            ax.errorbar(
                delta,
                i,
                xerr=row["std"][j],
                fmt="o",
                color=row["color"],
                ecolor=INK,
                elinewidth=0.55,
                capsize=1.3,
                markersize=5.4,
                markeredgewidth=0.0,
                zorder=3,
            )
        ax.set_yticks(ys)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(labels[j], pad=3)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        box(ax)
        ax.set_xlabel(r"$\Delta$ IoU (\%)")
    axes[0].invert_yaxis()
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=row["color"], markersize=4.8, label=row["name"]) for row in others]
    place_legend(fig, handles, ncol=3)
    _save(fig, out_path, rect=(0.0, 0.0, 1.0, 0.79), w_pad=0.9)


def value_heatmap(rows, labels, out_path, figsize=None):
    apply()
    names = [row["name"] for row in rows]
    mat = np.stack([row["mean"] for row in rows], axis=0)
    col_lo = mat.min(axis=0)
    col_hi = mat.max(axis=0)
    scaled = (mat - col_lo) / (col_hi - col_lo + 1e-8)
    if figsize is None:
        figsize = (4.5, 0.42 * len(names) + 1.05)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(scaled, cmap=HEAT, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(names)), names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7.0, color=INK)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    box(ax)
    ax.tick_params(direction="out", top=False, right=False, length=0)
    ax.set_xlabel("IoU (%)  |  color scaled within each dataset")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def faceted_bars(rows, labels, ylabel, out_path, figsize=None, floor=None, ncol=4):
    apply()
    n_ds = len(labels)
    n_arm = len(rows)
    if figsize is None:
        figsize = (1.35 * n_ds + 0.7, 2.25)
    fig, axes = plt.subplots(1, n_ds, figsize=figsize, sharey=False)
    if n_ds == 1:
        axes = [axes]
    xs = np.arange(n_arm)
    for j, ax in enumerate(axes):
        means = np.array([row["mean"][j] for row in rows])
        stds = np.array([row["std"][j] for row in rows])
        colors = [row["color"] for row in rows]
        ax.bar(
            xs,
            means,
            width=0.72,
            color=colors,
            yerr=stds,
            ecolor=INK,
            capsize=1.2,
            error_kw={"elinewidth": 0.55, "capthick": 0.55},
            zorder=3,
            linewidth=0.0,
            edgecolor="none",
        )
        ax.set_xlim(-0.7, n_arm - 0.3)
        ax.set_ylim(*tight_ylim(means, stds, floor=floor))
        ax.set_xticks([])
        ax.set_title(labels[j], pad=3)
        restyle(ax)
        if j == 0:
            ax.set_ylabel(ylabel)
    handles = [Patch(facecolor=row["color"], edgecolor="none", label=row["name"]) for row in rows]
    place_legend(fig, handles, ncol=ncol)
    _save(fig, out_path)

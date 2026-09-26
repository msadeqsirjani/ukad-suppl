import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style
import paper_viz as viz

OUT_PATH = viz.FIGURES / "displacement.png"
COLUMNS = (
    "Reference image +\nground-truth contour",
    r"Mean offset magnitude" "\n" r"$|\delta|$ (bottleneck px)",
    "Displacement direction\n(sparse field)",
)


def draw_row(axes, image, gt, dx, dy, r_max):
    contour = viz.draw_contour(image, gt, color=style.CONTOUR, width=max(1, image.shape[0] // 280))
    axes[0].imshow(contour)
    mag = np.sqrt(dx**2 + dy**2)
    h, w = dx.shape
    H, W = image.shape[:2]
    mag_up = cv2.resize(mag, (W, H), interpolation=cv2.INTER_NEAREST)
    im = axes[1].imshow(mag_up, cmap=style.HEAT, vmin=0.0, vmax=r_max, interpolation="nearest")
    ys = np.linspace(0, H - 1, h)
    xs = np.linspace(0, W - 1, w)
    grid_x, grid_y = np.meshgrid(xs, ys)
    step = 2 if h <= 32 else 3
    axes[2].imshow(contour)
    axes[2].quiver(
        grid_x[::step, ::step],
        grid_y[::step, ::step],
        (dx * (W / max(w, 1)))[::step, ::step],
        (dy * (H / max(h, 1)))[::step, ::step],
        color="#E13B3F",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0055,
        pivot="mid",
        headwidth=3.5,
        headlength=3.9,
        headaxislength=3.5,
        linewidth=0.2,
        edgecolor="#741317",
        alpha=0.94,
    )
    return im


def cached_case(dataset_name, seed, device):
    cache_path = Path(tempfile.gettempdir()) / f"ukad_displacement_{dataset_name}_{seed}.npz"
    if cache_path.exists():
        with np.load(cache_path) as payload:
            return tuple(payload[key] for key in ("image", "gt", "dx", "dy", "r_max"))

    dataset = viz.val_dataset(dataset_name, seed)
    index = viz.pick_index(dataset_name, dataset)
    image_t, mask_t = viz.tensors(dataset, index)
    size = int(image_t.shape[-1])
    image = viz.display_image(dataset, index, size)
    gt = viz.as_mask(mask_t.detach().cpu().numpy())
    print(f"{dataset_name} index={index} file={viz.sample_name(dataset, index)}", flush=True)
    model = viz.load_model("ukad_b", dataset_name, seed, device)
    _ = viz.predict(model, image_t, device)
    dx, dy, r_max = viz.displacement_maps(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(cache_path, image=image, gt=gt, dx=dx, dy=dy, r_max=r_max)
    return image, gt, dx, dy, np.asarray(r_max)


def draw(seed, device_name, out_path):
    viz.style_axes()
    device = viz.get_device(device_name)
    fig, axes = plt.subplots(len(viz.DATASETS), 3, figsize=(7.2, 5.85), constrained_layout=True)
    color_mappable = None
    for row, dataset_name in enumerate(viz.DATASETS):
        image, gt, dx, dy, r_max = cached_case(dataset_name, seed, device)
        r_max = float(r_max)
        color_mappable = draw_row(axes[row], image, gt, dx, dy, r_max)
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.7)
                spine.set_color(style.INK)
            if row == 0:
                ax.set_title(COLUMNS[col], pad=6, fontweight="bold", linespacing=1.05)
            if col == 0:
                ax.set_ylabel(viz.DATASET_TITLE[dataset_name], fontweight="bold", labelpad=8)
    cbar = fig.colorbar(color_mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Mean magnitude (bottleneck px)", labelpad=6)
    cbar.set_ticks(np.arange(0, r_max + 1))
    cbar.ax.tick_params(labelsize=style.SIZE, length=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=viz.SPLIT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main(args):
    draw(args.seed, args.device, args.out or OUT_PATH)


if __name__ == "__main__":
    main(get_args())

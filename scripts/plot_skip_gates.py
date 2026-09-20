import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style
import paper_viz as viz

OUT_PATH = viz.FIGURES / "skip_gates.png"
LEVELS = (4, 1)
LEVEL_TITLE = {4: "Alpha 4", 1: "Alpha 1"}


def draw(seed, device_name, out_path):
    viz.style_axes()
    device = viz.get_device(device_name)
    fig, axes = plt.subplots(len(viz.DATASETS), 3, figsize=(7.0, 5.8), gridspec_kw={"wspace": 0.05, "hspace": 0.06})
    for row, dataset_name in enumerate(viz.DATASETS):
        dataset = viz.val_dataset(dataset_name, seed)
        index = viz.pick_index(dataset_name, dataset)
        image_t, mask_t = viz.tensors(dataset, index)
        size = int(image_t.shape[-1])
        image = viz.display_image(dataset, index, size)
        gt = viz.as_mask(mask_t.detach().cpu().numpy())
        print(f"{dataset_name} index={index} file={viz.sample_name(dataset, index)}")
        model = viz.load_model("ukad_b", dataset_name, seed, device)
        maps = viz.skip_gate_maps(model, image_t, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        contour = viz.draw_contour(image, gt, color=style.CONTOUR, width=max(1, image.shape[0] // 280))
        height, width = image.shape[:2]
        ax = axes[row, 0]
        ax.imshow(contour)
        ax.set_xticks([])
        ax.set_yticks([])
        style.box(ax)
        ax.tick_params(length=0)
        if row == 0:
            ax.set_title("Image + GT", pad=4)
        ax.set_ylabel(viz.DATASET_TITLE[dataset_name])
        color_mappable = None
        for col, level in enumerate(LEVELS, start=1):
            ax = axes[row, col]
            heat = maps[level]
            heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
            color_mappable = ax.imshow(heat, cmap=style.GATE, vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            style.box(ax)
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(LEVEL_TITLE[level], pad=4)
    cbar = fig.colorbar(color_mappable, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Gate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=viz.SPLIT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    draw(args.seed, args.device, args.out)

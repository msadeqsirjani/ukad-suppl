import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style
import paper_viz as viz

OUT_PATH = viz.FIGURES / "qualitative.png"
DISPLAY_TITLE = {
    "attunet": "Attention U-Net",
    "rollingunet_l": "Rolling-UNet",
    "ukad_l": "UKAD",
}
MAIN_MODELS = ("attunet", "unetpp", "rollingunet_l", "ukan", "adakan", "ukad_l")
MAIN_INDICES = (0, 2, 0, 0)


def columns(names):
    return ("Image", "GT") + tuple(DISPLAY_TITLE.get(name, viz.MODEL_TITLE[name]) for name in names)


COLUMNS = columns(viz.QUAL_MODELS)
PAGE_WIDTH = 8.2


def panel_image(kind, image, gt, pred=None):
    if kind == "image":
        return image.astype(np.float32) / 255.0
    if kind == "gt":
        mask = np.zeros_like(image, dtype=np.float32)
        mask[..., :] = 0.08
        mask[viz.as_mask(gt)] = (0.95, 0.95, 0.95)
        return mask
    base = image.astype(np.float32) / 255.0
    gt = viz.as_mask(gt)
    pred = viz.as_mask(pred)
    both = gt & pred
    missed = gt & ~pred
    extra = pred & ~gt
    for mask, color, alpha in (
        (both, style.TEAL, 0.62),
        (missed, style.CONTOUR_GT, 0.76),
        (extra, style.CONTOUR_PRED, 0.76),
    ):
        if mask.any():
            base[mask] = (1.0 - alpha) * base[mask] + alpha * np.asarray(to_rgb(color), dtype=np.float32)
    return np.clip(base, 0.0, 1.0)


def draw_case(axes_row, dataset_name, dataset, index, seed, device, title_row, label, preds=None):
    image_t, mask_t = viz.tensors(dataset, index)
    size = int(image_t.shape[-1])
    image = viz.display_image(dataset, index, size)
    gt = viz.as_mask(mask_t.detach().cpu().numpy())
    print(f"{dataset_name} index={index} file={viz.sample_name(dataset, index)} size={size}")
    if preds is None:
        preds = {}
        for model_name in viz.QUAL_MODELS:
            model = viz.load_model(model_name, dataset_name, seed, device)
            preds[model_name] = viz.predict(model, image_t, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    panels = [("image", None), ("gt", None)] + [(name, preds[name]) for name in viz.QUAL_MODELS]
    for col, (kind, pred) in enumerate(panels):
        ax = axes_row[col]
        ax.imshow(panel_image(kind, image, gt, pred))
        if pred is not None:
            is_ukad = kind.startswith("ukad")
            ax.text(
                0.035,
                0.965,
                f"IoU {100 * viz.mask_iou(pred, gt):.1f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=style.SIZE - 4.0,
                color=style.INK,
                bbox={
                    "facecolor": "#FFF3F2" if is_ukad else "white",
                    "edgecolor": style.UKAD if is_ukad else style.INK,
                    "linewidth": 0.6,
                    "boxstyle": "square,pad=0.2",
                },
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.15 if kind.startswith("ukad") else 0.7)
            spine.set_color(style.UKAD if kind.startswith("ukad") else style.INK)
        if title_row:
            ax.set_title(
                COLUMNS[col],
                fontsize=style.SIZE - 2.0,
                pad=3,
                fontweight="bold" if kind.startswith("ukad") else "normal",
                color=style.UKAD if kind.startswith("ukad") else style.INK,
            )
        if col == 0:
            ax.set_ylabel(label)


def draw(seed, device_name, out_path, rows, start, representative=False, indices=None):
    viz.style_axes()
    device = viz.get_device(device_name)
    n_rows = len(viz.DATASETS) * rows
    figsize = (PAGE_WIDTH, PAGE_WIDTH / len(COLUMNS) * n_rows + 0.35)
    fig, axes = plt.subplots(n_rows, len(COLUMNS), figsize=figsize, gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    axes = np.atleast_2d(axes)
    row = 0
    for dataset_name in viz.DATASETS:
        dataset = viz.val_dataset(dataset_name, seed)
        if indices is None:
            selector = viz.representative_indices if representative else viz.best_indices
            chosen = selector(dataset_name, dataset, seed, device, count=start + rows)[start:]
        else:
            chosen = [indices[viz.DATASETS.index(dataset_name)]]
        cases = [viz.tensors(dataset, index) for index in chosen]
        predictions = {}
        for model_name in viz.QUAL_MODELS:
            model = viz.load_model(model_name, dataset_name, seed, device)
            predictions[model_name] = viz.predict_batch(model, (image for image, _ in cases), device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for case_index, index in enumerate(chosen):
            preds = {name: masks[case_index] for name, masks in predictions.items()}
            draw_case(
                axes[row],
                dataset_name,
                dataset,
                index,
                seed,
                device,
                row == 0,
                viz.DATASET_TITLE[dataset_name],
                preds=preds,
            )
            row += 1
    handles = [
        Patch(facecolor=style.TEAL, edgecolor="none", label="Agreement"),
        Patch(facecolor=style.CONTOUR_GT, edgecolor="none", label="Missed ground truth"),
        Patch(facecolor=style.CONTOUR_PRED, edgecolor="none", label="Extra prediction"),
    ]
    fig.tight_layout()
    style.legend_above(fig, axes, handles, pad=0.005, ncol=3, fontsize=style.SIZE - 2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=viz.SPLIT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--representative", action="store_true", help="select pre-specified morphology cases without model predictions")
    parser.add_argument("--main", action="store_true", help="render the compact baseline-versus-UKAD panel used in the main paper")
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    return parser.parse_args()


def main(args):
    global COLUMNS
    rows = args.rows if args.rows is not None else 1
    if args.main:
        viz.QUAL_MODELS = MAIN_MODELS
        COLUMNS = columns(viz.QUAL_MODELS)
        if args.indices is None and not args.representative and rows == 1:
            args.indices = MAIN_INDICES
    start = 0
    out_path = args.out
    if out_path is None:
        out_path = viz.FIGURES / "qualitative.png"
    draw(args.seed, args.device, out_path, rows, start, representative=args.representative, indices=args.indices)


if __name__ == "__main__":
    main(get_args())

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.colors import to_rgb
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")

FIGURES = ROOT / "docs" / "paper" / "iclr2027-9page" / "figures"
DATASETS = ("busi", "cvc", "glas", "isic")
DATASET_TITLE = {
    "busi": "BUSI",
    "cvc": "CVC",
    "glas": "GlaS",
    "isic": "ISIC",
}
SHORT_TITLE = {
    "busi": "BUSI",
    "cvc": "CVC",
    "glas": "GlaS",
    "isic": "ISIC",
}
SPLIT_SEED = 2981
QUAL_MODELS = (
    "unet",
    "attunet",
    "unetpp",
    "rollingunet_l",
    "umamba",
    "ukan",
    "ukanplus",
    "adakan",
    "cglknet",
    "ukad_l",
)
QUAL_SELECTOR = "ukad_b"
QUAL_OURS = ("ukad_b", "ukad_l")
QUAL_RIVALS = ("unet", "ukan", "adakan")
QUAL_MIN_IOU = 0.75
MODEL_TITLE = {
    "unet": "U-Net",
    "attunet": "Attention U-Net",
    "unetpp": "U-Net++",
    "rollingunet_l": "Rolling-UNet-L",
    "umamba": "U-Mamba",
    "cglknet": "CGLKNet",
    "ukanplus": "UKAN+",
    "ukan": "U-KAN",
    "adakan": "AdaKAN",
    "ukad_s": "UKAD-S",
    "ukad_b": "UKAD-B",
    "ukad_l": "UKAD-L",
}


def workbench():
    return Path(os.environ.get("WORKBENCH", ROOT / "workbench"))


def config_path(model, dataset):
    kind = "variants" if model.startswith("ukad") else "baselines"
    return ROOT / "configs" / kind / model / f"{dataset}.yaml"


def best_checkpoint(model, dataset, seed):
    folder = workbench() / "ckpts" / f"{model}_{dataset}_dataseed{seed}"
    matches = sorted(folder.glob("best_*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"no best checkpoint in {folder}")
    return matches[0]


def get_device(name="auto"):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model, dataset, seed, device):
    from src import models
    from src.utils.serialization_utils import load_config

    net = models.get(load_config(config_path(model, dataset))["model"]).to(device)
    payload = torch.load(best_checkpoint(model, dataset, seed), map_location=device, weights_only=False)
    net.load_state_dict(payload.get("state_dict", payload), strict=True)
    del payload
    net.eval()
    return net


def val_dataset(dataset, seed):
    from src import data
    from src.utils.serialization_utils import load_config

    os.environ["DATASEED"] = str(seed)
    loader = data.get(load_config(ROOT / "configs" / "data" / dataset / "val_dataloader.yaml"))
    return loader.dataset


def as_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] <= 4 else mask[..., 0]
    return mask > 0.5


def display_image(dataset, index, size):
    from src.utils.img_utils import read_img

    row = dataset.sampler[index]
    image = read_img(dataset.root / row["image"], dtype="uint8", channels=3)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def sample_name(dataset, index):
    return Path(str(dataset.sampler[index]["image"])).name


def tensors(dataset, index):
    sample = dataset[index]
    image = sample["image"]
    mask = sample["mask"]
    if mask.ndim == 3:
        mask = mask[0]
    return image, mask


def logits_to_mask(out):
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("out", out.get("logits", next(iter(out.values()))))
    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3:
        out = out[0]
    return (torch.sigmoid(out) > 0.5).detach().cpu().numpy()


def predict(model, image, device):
    with torch.inference_mode():
        batch = image if image.ndim == 4 else image.unsqueeze(0)
        return logits_to_mask(model(batch.to(device)))


def predict_batch(model, images, device):
    """Return binary masks for equally sized images in one forward pass."""
    with torch.inference_mode():
        batch = torch.stack(list(images)).to(device)
        out = model(batch)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("out", out.get("logits", next(iter(out.values()))))
        if out.ndim == 4:
            out = out[:, 0]
        return (torch.sigmoid(out) > 0.5).detach().cpu().numpy()


def _features(mask, image=None):
    m = as_mask(mask)
    area = float(m.mean())
    if area == 0.0:
        return None
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
    peri = float((m & ~eroded).sum())
    pixels = float(m.sum())
    compactness = (peri**2) / (4.0 * math.pi * pixels + 1e-6)
    ncomp = int(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    contrast = 0.0
    if image is not None:
        gray = image.mean(axis=-1) if image.ndim == 3 else image
        inside = gray[m]
        outside = gray[~m]
        if inside.size and outside.size:
            contrast = abs(float(inside.mean()) - float(outside.mean()))
    return {
        "area": area,
        "compactness": compactness,
        "ncomp": ncomp,
        "border": float(border.mean()),
        "contrast": contrast,
    }


def overlay_heatmap(image, heat, cmap=None, alpha=0.50, vmax=None):
    import matplotlib.pyplot as plt
    import paper_style

    if cmap is None:
        cmap = paper_style.HEAT
    cmap_obj = cmap if not isinstance(cmap, str) else plt.get_cmap(cmap)
    heat = np.asarray(heat, dtype=np.float32)
    peak = float(vmax) if vmax is not None else float(heat.max())
    heat = heat / (peak + 1e-8)
    height, width = image.shape[:2]
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    color = cmap_obj(np.clip(heat, 0.0, 1.0))[..., :3]
    base = image.astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)


def candidate_rows(dataset_name, dataset):
    from src.utils.img_utils import read_img

    rows = []
    for index in range(len(dataset.sampler)):
        row = dataset.sampler[index]
        mask = np.squeeze(read_img(dataset.root / row["mask"], dtype="float32", normalize=255.0, channels=1))
        image = None
        if dataset_name in ("busi", "isic"):
            image = read_img(dataset.root / row["image"], dtype="uint8", channels=3)
            if image.shape[:2] != mask.shape[:2]:
                image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        feats = _features(mask, image)
        if feats is None or feats["border"] > 0.3:
            continue
        feats["index"] = index
        rows.append(feats)
    return rows


def mask_iou(pred, gt):
    pred = as_mask(pred)
    gt = as_mask(gt)
    union = float(np.logical_or(pred, gt).sum())
    if union == 0.0:
        return 0.0
    return float(np.logical_and(pred, gt).sum()) / union


def iou_table(dataset_name, dataset, seed, device, models, pool):
    table = {}
    batch_size = 8
    for name in models:
        net = load_model(name, dataset_name, seed, device)
        table[name] = {}
        for offset in range(0, len(pool), batch_size):
            indices = pool[offset : offset + batch_size]
            samples = [tensors(dataset, index) for index in indices]
            preds = predict_batch(net, (image for image, _ in samples), device)
            for index, pred, (_, mask) in zip(indices, preds, samples):
                table[name][index] = mask_iou(pred, mask.detach().cpu().numpy())
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return table


def best_indices(dataset_name, dataset, seed, device, count=1, candidate_limit=24):
    rows = candidate_rows(dataset_name, dataset)
    pool = [row["index"] for row in rows] or list(range(len(dataset.sampler)))
    if len(pool) > candidate_limit:
        step = (len(pool) - 1) / float(candidate_limit - 1)
        pool = [pool[round(i * step)] for i in range(candidate_limit)]
    table = iou_table(dataset_name, dataset, seed, device, QUAL_OURS + QUAL_RIVALS, pool)
    ours = {index: min(table[name][index] for name in QUAL_OURS) for index in pool}
    rivals = {index: max(table[name][index] for name in QUAL_RIVALS) for index in pool}
    margin = {index: ours[index] - rivals[index] for index in pool}
    strong = [index for index in pool if ours[index] >= QUAL_MIN_IOU] or pool
    strong.sort(key=lambda index: (-margin[index], -ours[index], index))
    for index in strong[:count]:
        print(f"{dataset_name} index={index} ours={ours[index]:.4f} rivals={rivals[index]:.4f} margin={margin[index]:.4f}")
    return strong[:count]


def representative_indices(dataset_name, dataset, seed, device, count=1):
    """Choose pre-specified morphology cases without reading model predictions."""
    primary = pick_index(dataset_name, dataset)
    indices = [primary]
    while len(indices) < count:
        candidate = pick_index_alt(dataset_name, dataset)
        if candidate not in indices:
            indices.append(candidate)
        else:
            break
    for index in indices:
        print(f"{dataset_name} morphology-selected index={index}")
    return indices


def pick_index(dataset_name, dataset):
    rows = candidate_rows(dataset_name, dataset)
    if not rows:
        return 0
    if dataset_name == "busi":
        cand = [row for row in rows if 0.03 <= row["area"] <= 0.10]
        pool = cand or rows
        return min(pool, key=lambda row: row["contrast"])["index"]
    if dataset_name == "cvc":
        cand = [row for row in rows if 0.04 <= row["area"] <= 0.20]
        pool = cand or rows
        return max(pool, key=lambda row: row["compactness"])["index"]
    if dataset_name == "glas":
        cand = [row for row in rows if 0.20 <= row["area"] <= 0.65]
        pool = cand or rows
        return max(pool, key=lambda row: row["ncomp"])["index"]
    cand = [row for row in rows if 0.04 <= row["area"] <= 0.12]
    pool = cand or rows
    return min(pool, key=lambda row: row["contrast"])["index"]


def pick_index_alt(dataset_name, dataset):
    primary = pick_index(dataset_name, dataset)
    rows = [row for row in candidate_rows(dataset_name, dataset) if row["index"] != primary]
    if not rows:
        return (primary + 1) % max(len(dataset.sampler), 1)
    if dataset_name == "busi":
        cand = [row for row in rows if 0.12 <= row["area"] <= 0.35]
        pool = cand or rows
        return max(pool, key=lambda row: row["area"])["index"]
    if dataset_name == "cvc":
        cand = [row for row in rows if 0.03 <= row["area"] <= 0.25]
        pool = cand or rows
        return min(pool, key=lambda row: row["compactness"])["index"]
    if dataset_name == "glas":
        cand = [row for row in rows if 0.15 <= row["area"] <= 0.55]
        pool = cand or rows
        return min(pool, key=lambda row: row["ncomp"])["index"]
    cand = [row for row in rows if 0.15 <= row["area"] <= 0.40]
    pool = cand or rows
    return max(pool, key=lambda row: row["area"])["index"]


def contours(mask, width):
    from scipy.ndimage import binary_erosion

    mask = as_mask(mask)
    inner = binary_erosion(mask, iterations=width, border_value=1)
    return mask & ~inner


def overlay_contours(image, pred, gt):
    import paper_style

    out = image.astype(np.float32) / 255.0
    width = max(3, round(image.shape[0] / 64))
    for mask, color in ((gt, paper_style.CONTOUR_GT), (pred, paper_style.CONTOUR_PRED)):
        edge = contours(mask, width)
        if edge.any():
            out[edge] = to_rgb(color)
    return np.clip(out, 0.0, 1.0)


def draw_contour(image, mask, color=(0, 220, 70), width=1):
    out = np.ascontiguousarray(image.copy())
    contours, _ = cv2.findContours(as_mask(mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, width)
    return out


def displacement_maps(model):
    disp = model._model.bott.kad.displace
    raw = disp._last_raw
    gate = disp._last_gate
    py = disp.max_radius * gate * torch.tanh(raw[:, :, 0])
    px = disp.max_radius * gate * torch.tanh(raw[:, :, 1])
    dx = px.mean(dim=1)[0].detach().cpu().numpy()
    dy = py.mean(dim=1)[0].detach().cpu().numpy()
    return dx, dy, float(disp.max_radius)


def skip_gate_maps(model, image, device):
    stored = {}
    hooks = []
    net = model._model

    def attach(level):
        module = getattr(net, f"dec{level}_gate", None)
        if module is None:
            return

        def _hook(_module, _inputs, output, key=level):
            stored[key] = output[0, 0].detach().cpu().numpy()

        hooks.append(module.psi.register_forward_hook(_hook))

    for level in (4, 3, 2, 1):
        attach(level)
    predict(model, image, device)
    for hook in hooks:
        hook.remove()
    return stored


def load_group_rational(model, dataset, seed, which="inner"):
    payload = torch.load(best_checkpoint(model, dataset, seed), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    prefix = f"_model.bott.kad.{which}"
    a = state[f"{prefix}.a"].detach().cpu().numpy()
    b = state[f"{prefix}.b"].detach().cpu().numpy()
    return a, b


def rational_curve(a, b, x):
    p = np.zeros_like(x)
    for k, coef in enumerate(a):
        p = p + coef * np.power(x, k)
    q = np.zeros_like(x)
    for k, coef in enumerate(b, start=1):
        q = q + coef * np.power(x, k)
    return p / (1.0 + np.abs(q))


def style_axes():
    import paper_style as style

    style.apply()

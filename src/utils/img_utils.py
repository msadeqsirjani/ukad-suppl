from pathlib import Path

import numpy as np
import cv2

from .io_utils import decode_raw, decode_mat


def _as_hwc(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(
            f"Expected 2 or 3 dimensional image, found `ndim`: {img.ndim}."
        )
    return img


def decode_png(read_path):
    img = cv2.imread(read_path, -1)
    img = _as_hwc(img)
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def read_img(
    read_path, dtype=None, shape=None, normalize=None, channels=None, **kwargs
):

    read_path = Path(read_path)
    if not read_path.exists():
        raise ValueError(f"Source path does not exist: {read_path}.")
    ext = read_path.suffix.lower()

    if ext in (".png", ".jpeg", ".jpg"):
        img = decode_png(read_path)
    elif ext in (".raw", ".bin"):
        img = decode_raw(read_path, **kwargs)
    elif ext in (".mat",):
        img = decode_mat(read_path, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized filename extension found: {read_path}. Only `.png`, `.jpeg`, `.jpg`, `.raw`, `.bin` or `.mat` are supported."
        )

    if shape is not None:
        if all(shape):
            img = cv2.resize(img, shape[:2], interpolation=cv2.INTER_LINEAR)
            assert img.shape == tuple(
                shape
            ), f"`shape` mismatch: `{img.shape}` != `{shape}`."
    if channels is not None:
        if img.shape[-1] == 1 and channels == 3:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 3 and channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        elif img.shape[-1] != channels:
            raise ValueError(
                f"Cannot convert image with {img.shape[-1]} channels to {channels}."
            )
    if normalize is not None:
        img = img / float(normalize)
    if dtype is not None:
        img = img.astype(np.dtype(dtype))

    return img

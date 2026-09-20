"""Download GlaS and emit the three U-KAN paper-comparison splits.

Source: Kaggle mirror of the MICCAI'2015 GlaS / Warwick QU dataset
        https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation
        (the original warwick.ac.uk download now requires login)
Requires: pip install kaggle  +  ~/.kaggle/kaggle.json

Output structure (BMP converted to PNG):
    ${DATASETS}/glas/
        train_1.png
        train_1_anno.png
        testA_1.png
        testA_1_anno.png
        ...
    ${WORKBENCH}/data_lists/glas/split_{2981,6142,1187}/
        train.csv, val.csv, protocol.json

U-FunKAN's legacy baseline rows were sourced from U-KAN. U-KAN combines the
official training and test cases, then applies
``sklearn.model_selection.train_test_split`` independently with split seeds
2981, 6142, and 1187. The split is unstratified and image-level; the held-out
20 percent is both validation and reported evaluation, with no independent
test partition.

Matches config:
    root: ${DATASETS}/glas
    load_params: [{dtype: uint8}, {normalize: 255., dtype: float32}]

Usage:
    python scripts/data/prepare_glas.py \\
        --datasets /path/to/datasets \\
        --workbench /path/to/workbench
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "sani84/glasmiccai2015-gland-segmentation"
SPLIT_SEEDS = (2981, 6142, 1187)
EXPECTED_SAMPLES = 165
EXPECTED_COUNTS = {"train": 132, "val": 33}


def kaggle_download(dataset: str, dest: Path):
    try:
        import kaggle
    except ImportError:
        print("  install kaggle:  pip install kaggle")
        print("  set API key:     https://www.kaggle.com/docs/api")
        raise

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dataset} via Kaggle API ...")
    kaggle.api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)


def download_glas(raw_dir: Path) -> Path:
    extract_dir = raw_dir / "extracted"
    if extract_dir.exists() and (
        any(extract_dir.rglob("*.bmp")) or any(extract_dir.rglob("*.png"))
    ):
        print(f"  skip download (exists): {extract_dir}")
        return extract_dir
    kaggle_download(KAGGLE_DATASET, extract_dir)
    return extract_dir


def bmp_to_png(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if "_anno" in src.stem:
        img = ((img > 0).astype(np.uint8)) * 255
    cv2.imwrite(str(dst), img)
    return True


def convert_and_collect(extract_dir: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    bmp_files = list(extract_dir.rglob("*.bmp"))
    if not bmp_files:

        for f in extract_dir.rglob("*.png"):
            target = dst_root / f.name
            if not target.exists():
                shutil.copy2(f, target)
        return

    for bmp in bmp_files:
        png_name = bmp.stem + ".png"
        dst = dst_root / png_name
        if not dst.exists():
            bmp_to_png(bmp, dst)

    print(f"  converted {len(bmp_files)} BMP → PNG in {dst_root}")


def collect_pairs(root: Path):
    images = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() == ".png" and "_anno" not in p.name
    )
    pairs = []
    for img in images:
        mask = root / (img.stem + "_anno" + img.suffix)
        if mask.exists():
            pairs.append((img.name, mask.name))
    return pairs


def make_csv(pairs, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["image", "mask"]).to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: {len(pairs)} samples")


def write_split(lists_dir: Path, split_seed: int, splits: dict, metadata: dict):
    protocol_dir = lists_dir / f"split_{split_seed}"
    for split, split_pairs in splits.items():
        make_csv(split_pairs, protocol_dir / f"{split}.csv")
    with open(protocol_dir / "protocol.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_paper_split(all_pairs, split_seed: int):
    """Reproduce U-KAN's unstratified all-image sklearn split."""
    train, val = train_test_split(
        list(all_pairs),
        test_size=0.2,
        random_state=split_seed,
        shuffle=True,
        stratify=None,
    )
    return {"train": train, "val": val}


def paper_metadata(split_seed: int, splits: dict):
    return {
        "schema_version": 1,
        "protocol": "paper_compat",
        "compatibility_target": "U-KAN",
        "dataset": "GlaS",
        "description": "U-KAN unstratified random 80/20 all-image split",
        "split_seed": split_seed,
        "splitter": "sklearn.model_selection.train_test_split",
        "splitter_parameters": {
            "test_size": 0.2,
            "random_state": split_seed,
            "shuffle": True,
            "stratify": None,
        },
        "split_unit": "image",
        "source_order": "lexicographically sorted image paths",
        "exact_upstream_membership_verified": False,
        "counts": {name: len(values) for name, values in splits.items()},
        "independent_test": False,
        "validation_is_reported_evaluation": True,
        "contains_official_test_leakage": True,
    }


def main(args):
    datasets = Path(args.datasets)
    workbench = Path(args.workbench)
    dst_root = datasets / "glas"
    lists_dir = workbench / "data_lists" / "glas"
    raw_dir = datasets / "_glas_raw"

    if dst_root.exists() and any(dst_root.glob("*.png")):
        print(f"  {dst_root} already populated, skipping download.")
    else:
        extract_dir = download_glas(raw_dir)
        convert_and_collect(extract_dir, dst_root)

    all_pairs = collect_pairs(dst_root)
    if not all_pairs:
        raise RuntimeError(
            f"No image-mask pairs found in {dst_root}; expected "
            "'name.png' + 'name_anno.png' pairs"
        )

    if len(all_pairs) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"GlaS paper protocol requires {EXPECTED_SAMPLES} image-mask pairs, "
            f"found {len(all_pairs)}"
        )
    unsupported = sorted(set(args.split_seeds) - set(SPLIT_SEEDS))
    if unsupported or len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError(
            f"--split_seeds must be unique values from {SPLIT_SEEDS}; "
            f"got {args.split_seeds}"
        )
    for split_seed in args.split_seeds:
        splits = build_paper_split(all_pairs, split_seed)
        counts = {name: len(values) for name, values in splits.items()}
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(
                f"GlaS split {split_seed} expected {EXPECTED_COUNTS}, got {counts}"
            )
        write_split(
            lists_dir,
            split_seed,
            splits,
            paper_metadata(split_seed, splits),
        )

    if raw_dir.exists():
        print(f"  removing {raw_dir} ...")
        shutil.rmtree(raw_dir)

    print(f"\nDone. {len(all_pairs)} total pairs.")
    print(f"  images : {dst_root}")
    print(f"  lists  : {lists_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True, help="$DATASETS root")
    p.add_argument("--workbench", required=True, help="$WORKBENCH root")
    p.add_argument(
        "--split_seeds",
        nargs="+",
        type=int,
        default=list(SPLIT_SEEDS),
        help="U-KAN data split seeds (default: 2981 6142 1187)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())

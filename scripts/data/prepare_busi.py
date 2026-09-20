"""Download BUSI and emit the three U-KAN paper-comparison splits.

Kaggle dataset: aryashah2k/breast-ultrasound-images-dataset
Requires: pip install kaggle  +  ~/.kaggle/kaggle.json

Output structure:
    ${DATASETS}/BUSI/
        benign (1).png
        benign (1)_mask.png
        ...
    ${WORKBENCH}/data_lists/BUSI/split_{2981,6142,1187}/
        train.csv, val.csv, protocol.json

U-FunKAN's legacy baseline rows were sourced from U-KAN. U-KAN applies
``sklearn.model_selection.train_test_split`` independently with split seeds
2981, 6142, and 1187. The split is unstratified and image-level; the held-out
20 percent is both validation and reported evaluation, with no independent
test partition.

Matches config:
    root: ${DATASETS}/BUSI
    load_params: [{dtype: uint8}, {normalize: 255., dtype: float32}]

Usage:
    python scripts/data/prepare_busi.py \\
        --datasets /path/to/datasets \\
        --workbench /path/to/workbench
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "aryashah2k/breast-ultrasound-images-dataset"
CLASSES = ("benign", "malignant")
SPLIT_SEEDS = (2981, 6142, 1187)
EXPECTED_SAMPLES = 647
EXPECTED_COUNTS = {"train": 517, "val": 130}


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


def flatten(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for cls in CLASSES:
        cls_dir = src_dir / cls
        if not cls_dir.exists():

            cls_dir = src_dir / "Dataset_BUSI_with_GT" / cls
        if not cls_dir.exists():
            print(f"  WARN: {cls} folder not found, skipping.")
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                target = dst_dir / f.name
                if not target.exists():
                    shutil.copy2(f, target)
                    moved += 1
    print(f"  copied {moved} files to {dst_dir}")


def collect_pairs(root: Path):
    images = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and "_mask" not in p.name
        and any(p.name.startswith(c) for c in CLASSES)
    )
    pairs = []
    for img in images:

        mask = root / (img.stem + "_mask" + img.suffix)
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


def build_paper_split(pairs, split_seed: int):
    """Reproduce U-KAN's unstratified image-level sklearn split."""
    train, val = train_test_split(
        list(pairs),
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
        "dataset": "BUSI",
        "description": "U-KAN unstratified random 80/20 image split",
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
        "patient_group_independence_guaranteed": False,
    }


def main(args):
    datasets = Path(args.datasets)
    workbench = Path(args.workbench)
    dst_root = datasets / "BUSI"
    lists_dir = workbench / "data_lists" / "BUSI"
    raw_dir = datasets / "_busi_raw"

    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"  {dst_root} already populated, skipping download.")
    else:
        kaggle_download(KAGGLE_DATASET, raw_dir)
        flatten(raw_dir, dst_root)

    pairs = collect_pairs(dst_root)
    if not pairs:
        raise RuntimeError(
            f"No image-mask pairs found in {dst_root}; expected "
            "'image.png' + 'image_mask.png' pairs"
        )

    if len(pairs) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"BUSI paper protocol requires {EXPECTED_SAMPLES} benign/malignant "
            f"image-mask pairs, found {len(pairs)}"
        )
    unsupported = sorted(set(args.split_seeds) - set(SPLIT_SEEDS))
    if unsupported or len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError(
            f"--split_seeds must be unique values from {SPLIT_SEEDS}; "
            f"got {args.split_seeds}"
        )
    for split_seed in args.split_seeds:
        splits = build_paper_split(pairs, split_seed)
        counts = {name: len(values) for name, values in splits.items()}
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(
                f"BUSI split {split_seed} expected {EXPECTED_COUNTS}, got {counts}"
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

    print(f"\nDone. {len(pairs)} total pairs.")
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

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "balraj98/cvcclinicdb"
SPLIT_SEEDS = (2981, 6142, 1187)
EXPECTED_SAMPLES = 612
EXPECTED_COUNTS = {"train": 489, "val": 123}

IMAGE_SUBDIRS = ("Original", "images", "PNG", "CVC-ClinicDB/Original")
MASK_SUBDIRS = ("Ground Truth", "masks", "PNG_masks", "CVC-ClinicDB/Ground Truth")


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


def find_subdir(root: Path, candidates):
    for c in candidates:
        d = root / c
        if d.exists() and any(d.iterdir()):
            return d
    return None


def _is_binary_mask_dir(d: Path, n_check: int = 5) -> bool:
    import cv2

    samples = sorted(d.glob("*.png"))[:n_check]
    if not samples:
        return False
    for f in samples:
        arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        near_binary = ((arr < 15) | (arr > 240)).mean()
        if near_binary < 0.95:
            return False
    return True


def organize(raw_dir: Path, dst_root: Path):
    img_dir = find_subdir(raw_dir, IMAGE_SUBDIRS)
    mask_dir = find_subdir(raw_dir, MASK_SUBDIRS)

    if img_dir is None or mask_dir is None:

        png_dirs = [
            d for d in raw_dir.rglob("*") if d.is_dir() and any(d.glob("*.png"))
        ]
        if len(png_dirs) >= 2:
            png_dirs = sorted(png_dirs, key=lambda d: d.name)
            img_dir, mask_dir = png_dirs[0], png_dirs[1]
        else:
            print(f"  WARN: could not locate image/mask dirs in {raw_dir}")
            return

    if _is_binary_mask_dir(img_dir) and not _is_binary_mask_dir(mask_dir):
        print(f"  swap: detected masks in {img_dir.name}, images in {mask_dir.name}")
        img_dir, mask_dir = mask_dir, img_dir

    dst_img = dst_root / "Original"
    dst_mask = dst_root / "Ground Truth"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    for f in sorted(img_dir.glob("*.png")):
        t = dst_img / f.name
        if not t.exists():
            shutil.copy2(f, t)

    for f in sorted(mask_dir.glob("*.png")):
        t = dst_mask / f.name
        if not t.exists():
            shutil.copy2(f, t)

    print(f"  organized into {dst_root}")


def collect_pairs(root: Path):
    img_dir = root / "Original"
    mask_dir = root / "Ground Truth"

    if not img_dir.exists():
        img_dir = find_subdir(root, IMAGE_SUBDIRS) or root
    if not mask_dir.exists():
        mask_dir = find_subdir(root, MASK_SUBDIRS) or root

    imgs = sorted(img_dir.glob("*.png"))
    pairs = []
    for img in imgs:
        mask = mask_dir / img.name
        if not mask.exists():
            mask = mask_dir / (img.stem + ".png")
        if mask.exists():
            img_rel = img.relative_to(root)
            mask_rel = mask.relative_to(root)
            pairs.append((str(img_rel), str(mask_rel)))
    return pairs


def make_csv(pairs, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["image", "mask"]).to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: {len(pairs)} samples")


def write_split(
    lists_dir: Path, split_seed: int, splits: dict, metadata: dict
) -> None:
    protocol_dir = lists_dir / f"split_{split_seed}"
    for split, split_pairs in splits.items():
        make_csv(split_pairs, protocol_dir / f"{split}.csv")
    protocol_dir.mkdir(parents=True, exist_ok=True)
    with open(protocol_dir / "protocol.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_paper_split(pairs, split_seed: int):
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
        "dataset": "CVC",
        "description": "U-KAN unstratified random 80/20 frame split",
        "split_seed": split_seed,
        "splitter": "sklearn.model_selection.train_test_split",
        "splitter_parameters": {
            "test_size": 0.2,
            "random_state": split_seed,
            "shuffle": True,
            "stratify": None,
        },
        "split_unit": "frame",
        "source_order": "lexicographically sorted image paths",
        "exact_upstream_membership_verified": False,
        "counts": {name: len(values) for name, values in splits.items()},
        "independent_test": False,
        "validation_is_reported_evaluation": True,
        "contains_video_sequence_leakage": True,
    }


def main(args):
    datasets = Path(args.datasets)
    workbench = Path(args.workbench)
    dst_root = datasets / "CVC"
    lists_dir = workbench / "data_lists" / "CVC"
    raw_dir = datasets / "_cvc_raw"

    if dst_root.exists() and any(dst_root.rglob("*.png")):
        print(f"  {dst_root} already populated, skipping download.")
    else:
        kaggle_download(KAGGLE_DATASET, raw_dir)
        organize(raw_dir, dst_root)

    pairs = collect_pairs(dst_root)
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found in {dst_root}")

    if len(pairs) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"CVC paper protocol requires {EXPECTED_SAMPLES} frame-mask pairs, "
            f"found {len(pairs)}"
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
                f"CVC split {split_seed} expected {EXPECTED_COUNTS}, got {counts}"
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

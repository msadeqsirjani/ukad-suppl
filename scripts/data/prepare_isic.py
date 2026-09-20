"""Download ISIC 2018 Task 1 and emit the three paper-comparison splits.

Official challenge archive (no authentication required):
    https://isic-challenge-data.s3.amazonaws.com/2018/

ISIC 2018 lesion boundary segmentation is the dataset shared by the UNeXt and
Rolling-UNet reference implementations. The same unstratified image-level
``sklearn.model_selection.train_test_split`` used for BUSI, CVC-ClinicDB and
GlaS is applied here with split seeds 2981, 6142 and 1187.

Output structure:
    ${DATASETS}/ISIC2018/
        ISIC_0000000.png
        ISIC_0000000_mask.png
        ...
    ${WORKBENCH}/data_lists/ISIC2018/split_{2981,6142,1187}/
        train.csv, val.csv, protocol.json

Usage:
    python scripts/data/prepare_isic.py \\
        --datasets /path/to/datasets \\
        --workbench /path/to/workbench
"""

import argparse
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ARCHIVE_HOST = "https://isic-challenge-data.s3.amazonaws.com/2018"
ARCHIVES = {
    "images": f"{ARCHIVE_HOST}/ISIC2018_Task1-2_Training_Input.zip",
    "masks": f"{ARCHIVE_HOST}/ISIC2018_Task1_Training_GroundTruth.zip",
}
SPLIT_SEEDS = (2981, 6142, 1187)
EXPECTED_SAMPLES = 2594
EXPECTED_COUNTS = {"train": 2075, "val": 519}
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
DOWNLOAD_TIMEOUT = 120
DOWNLOAD_ATTEMPTS = 5
MASK_SUFFIX = "_segmentation"


def _progress(done, total):
    pct = done * 100 // total if total else 0
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    print(f"\r  [{bar}] {pct}%  {done//1024//1024}MB", end="", flush=True)


def _stream(url: str, partial: Path):
    resume = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if resume:
        headers["Range"] = f"bytes={resume}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        if resume and response.status != 206:
            resume = 0
        total = int(response.headers.get("Content-Length") or 0) + resume
        read = resume
        print(f"  downloading {url}")
        if resume:
            print(f"  resuming at {resume // 1024 // 1024}MB")
        with open(partial, "ab" if resume else "wb") as stream:
            while chunk := response.read(1 << 20):
                stream.write(chunk)
                read += len(chunk)
                _progress(read, total)
    print()
    if total and read != total:
        raise OSError(f"incomplete transfer: received {read} of {total} bytes")


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  skip download (exists): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _stream(url, partial)
        except urllib.error.HTTPError as error:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"cannot download {url}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            done = partial.stat().st_size if partial.exists() else 0
            reason = getattr(error, "reason", None) or error
            print(f"  attempt {attempt}/{DOWNLOAD_ATTEMPTS} interrupted ({reason})")
            if attempt == DOWNLOAD_ATTEMPTS:
                raise SystemExit(f"cannot download {url}: {reason}\n" f"{done // 1024 // 1024}MB kept at {partial}; rerun to resume.") from error
            print(f"  keeping {done // 1024 // 1024}MB and retrying with a range request")
            continue
        partial.rename(dest)
        return


def unzip(archive: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.rglob("*.jpg")) or any(out_dir.rglob("*.png")):
        print(f"  skip extract (already populated): {out_dir}")
        return
    print(f"  extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(out_dir)


def _to_png(src: Path, dst: Path, binarize: bool):
    if dst.exists():
        return
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read {src}")
    if binarize:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = ((image > 0).astype(np.uint8)) * 255
    cv2.imwrite(str(dst), image)


def convert(raw_dir: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in raw_dir.rglob("ISIC_*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and MASK_SUFFIX not in p.stem)
    masks = {p.stem.replace(MASK_SUFFIX, ""): p for p in raw_dir.rglob(f"ISIC_*{MASK_SUFFIX}.png")}
    converted = 0
    for image in images:
        mask = masks.get(image.stem)
        if mask is None:
            continue
        _to_png(image, dst_root / f"{image.stem}.png", binarize=False)
        _to_png(mask, dst_root / f"{image.stem}_mask.png", binarize=True)
        converted += 1
        if converted % 250 == 0:
            print(f"\r  converted {converted}/{len(images)} pairs", end="", flush=True)
    print(f"\r  converted {converted} pairs to {dst_root}")


def collect_pairs(root: Path):
    images = sorted(p for p in root.iterdir() if p.suffix.lower() == ".png" and "_mask" not in p.name)
    pairs = []
    for image in images:
        mask = root / (image.stem + "_mask" + image.suffix)
        if mask.exists():
            pairs.append((image.name, mask.name))
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
        "compatibility_target": "UNeXt/Rolling-UNet",
        "dataset": "ISIC2018",
        "description": "unstratified random 80/20 image split of ISIC 2018 Task 1",
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
    dst_root = datasets / "ISIC2018"
    lists_dir = workbench / "data_lists" / "ISIC2018"
    raw_dir = datasets / "_isic_raw"

    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"  {dst_root} already populated, skipping download.")
    else:
        for name, url in ARCHIVES.items():
            archive = raw_dir / Path(url).name
            download(url, archive)
            unzip(archive, raw_dir / name)
        convert(raw_dir, dst_root)

    pairs = collect_pairs(dst_root)
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found in {dst_root}; expected " "'ISIC_XXXXXXX.png' + 'ISIC_XXXXXXX_mask.png' pairs")
    if len(pairs) != EXPECTED_SAMPLES:
        raise RuntimeError(f"ISIC 2018 Task 1 requires {EXPECTED_SAMPLES} image-mask pairs, " f"found {len(pairs)}")
    unsupported = sorted(set(args.split_seeds) - set(SPLIT_SEEDS))
    if unsupported or len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError(f"--split_seeds must be unique values from {SPLIT_SEEDS}; got {args.split_seeds}")
    for split_seed in args.split_seeds:
        splits = build_paper_split(pairs, split_seed)
        counts = {name: len(values) for name, values in splits.items()}
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(f"ISIC split {split_seed} expected {EXPECTED_COUNTS}, got {counts}")
        write_split(lists_dir, split_seed, splits, paper_metadata(split_seed, splits))

    if raw_dir.exists() and not args.keep_downloads:
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
        help="data split seeds (default: 2981 6142 1187)",
    )
    p.add_argument(
        "--keep_downloads",
        action="store_true",
        help="retain the downloaded archives and extracted originals",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())

import argparse
from pathlib import Path

from prepare_isic import ARCHIVE_HOST, collect_pairs, convert, download, make_csv, unzip

ARCHIVES = {
    "images": f"{ARCHIVE_HOST}/ISIC2018_Task1-2_Test_Input.zip",
    "masks": f"{ARCHIVE_HOST}/ISIC2018_Task1_Test_GroundTruth.zip",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--workbench", required=True)
    return parser.parse_args()


def main(args):
    datasets, workbench = Path(args.datasets), Path(args.workbench)
    raw_dir = datasets / "ISIC2018_test_raw"
    dst_root = datasets / "ISIC2018_test"
    for url in ARCHIVES.values():
        archive = raw_dir / Path(url).name
        download(url, archive)
        unzip(archive, raw_dir / archive.stem)
    convert(raw_dir, dst_root)
    make_csv(collect_pairs(dst_root), workbench / "data_lists" / "ISIC2018" / "test.csv")


if __name__ == "__main__":
    main(get_args())

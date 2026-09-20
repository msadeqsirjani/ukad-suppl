import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path

from flops import compute_model_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ("busi", "cvc", "glas", "isic")
CONFIG_ROOTS = ("configs/baselines", "configs/variants")
KEEP = ("params", "trainable", "macs", "flops", "size_mb", "input_shape")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "performance", "costs.json"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def configs():
    for config_root in CONFIG_ROOTS:
        for model_dir in sorted(d for d in (Path(ROOT) / config_root).iterdir() if d.is_dir()):
            for dataset in DATASETS:
                path = model_dir / f"{dataset}.yaml"
                if path.is_file():
                    yield f"{model_dir.name}_{dataset}", path


def main(args):
    costs = {}
    for key, path in configs():
        try:
            info = compute_model_info(str(path), device=args.device)
        except Exception as exc:
            print(f"skip {key}: {exc}", flush=True)
            continue
        costs[key] = {name: info[name] for name in KEEP if name in info}
        print(f"{key:34s} {costs[key]['params'] / 1e6:8.2f}M {costs[key]['flops'] / 1e9:8.2f}G", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(costs, f, indent=2, sort_keys=True)
    print("wrote", args.output)


if __name__ == "__main__":
    main(get_args())

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PER_IMAGE = ROOT / "outputs" / "performance" / "per_image"
DATASETS = ("busi", "cvc", "glas", "isic")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", nargs="+", default=["ukad_l", "ukad_b"])
    parser.add_argument("--baselines", nargs="+", default=["adakan", "rollingunet_l", "ukan"])
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "performance" / "paired_bootstrap.json")
    return parser.parse_args()


def load(name, dataset):
    return json.loads((PER_IMAGE / f"{name}_{dataset}.json").read_text())["runs"]


def strata(ours, base, datasets):
    out = []
    for dataset in datasets:
        a, b = load(ours, dataset), load(base, dataset)
        for run_key in a:
            if a[run_key]["mask_pixels"] != b[run_key]["mask_pixels"]:
                raise ValueError(f"image order differs for {ours}/{base} {dataset} {run_key}")
            out.append(100 * (np.array(a[run_key]["iou"]) - np.array(b[run_key]["iou"])))
    return out


def bootstrap(diffs, n_boot, rng):
    observed = np.mean([d.mean() for d in diffs])
    samples = np.zeros(n_boot)
    for d in diffs:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        samples += d[idx].mean(1)
    samples /= len(diffs)
    low, high = np.percentile(samples, [2.5, 97.5])
    p = min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    wins = sum(int((d > 0).sum()) for d in diffs)
    losses = sum(int((d < 0).sum()) for d in diffs)
    return {"delta": observed, "ci_low": low, "ci_high": high, "p": p, "image_wins": wins, "image_losses": losses}


def main(args):
    rng = np.random.default_rng(args.seed)
    results = []
    for ours in args.ours:
        for base in args.baselines:
            row = {"ours": ours, "baseline": base, "all": bootstrap(strata(ours, base, DATASETS), args.n_boot, rng)}
            for dataset in DATASETS:
                row[dataset] = bootstrap(strata(ours, base, [dataset]), args.n_boot, rng)
            results.append(row)
            a = row["all"]
            per = " ".join(f"{d}:{row[d]['delta']:+.2f}(p={row[d]['p']:.3f})" for d in DATASETS)
            print(f"{ours} vs {base}: {a['delta']:+.2f} [{a['ci_low']:+.2f}, {a['ci_high']:+.2f}] p={a['p']:.4f} | {per}")
    args.output.write_text(json.dumps({"n_boot": args.n_boot, "seed": args.seed, "results": results}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main(get_args())

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "outputs" / "performance"
SPLITS = ("dataseed2981", "dataseed6142", "dataseed1187")
DATASETS = ("busi", "cvc", "glas", "isic")
BASELINES = (
    "unet",
    "attunet",
    "unetpp",
    "cglknet",
    "unext",
    "rollingunet_l",
    "umamba",
    "ukagnet",
    "ukan",
    "ufunkan",
    "ukanplus",
    "adakan",
    "mednext",
    "nnunet_resenc",
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", nargs="+", default=["variants/ukad_l", "variants/ukad_b"])
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--metric", default="val/iou")
    parser.add_argument("--output", type=Path, default=PERF / "paired_runs.json")
    return parser.parse_args()


def values(name, metric):
    return np.array([json.loads((PERF / f"{name}_{d}.json").read_text())["per_run"][s][metric] for d in DATASETS for s in SPLITS])


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def main(args):
    results = []
    for ours in args.ours:
        rows = []
        for base in args.baselines:
            diff = 100 * (values(ours, args.metric) - values(base, args.metric))
            rows.append({"baseline": base, "delta": diff.mean(), "wins": int((diff > 0).sum()), "p": wilcoxon(diff).pvalue})
        for row, adjusted in zip(rows, holm(np.array([r["p"] for r in rows]))):
            row["p_holm"] = adjusted
            print(f"{Path(ours).name} vs {row['baseline']:14s} {row['delta']:+.2f} {row['wins']:2d}/12 p={row['p']:.4f} holm={adjusted:.4f}")
        results.append({"ours": ours, "rows": rows})
    args.output.write_text(json.dumps({"metric": args.metric, "pairs": 12, "results": results}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main(get_args())

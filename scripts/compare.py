import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor import EPOCH_W, MEAN_W, NAME_W, SEED_W, pretty, split_name


def curve(path):
    out = {}
    best = None
    epoch = -1
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                raw_epoch = (row.get("epoch") or "").strip()
                if raw_epoch:
                    epoch = int(float(raw_epoch))
                raw_iou = (row.get("val/iou") or "").strip()
                if raw_iou and epoch >= 0:
                    value = float(raw_iou)
                    if best is None or value > best:
                        best = value
                    out[epoch] = best
    except (OSError, ValueError):
        pass
    return out


def at(points, epoch):
    reached = [e for e in points if e <= epoch]
    return points[max(reached)] if reached else None


def collect(root, dataset):
    runs = {}
    for run in sorted(Path(root).glob(f"*_{dataset}_dataseed*")):
        metrics = run / "metrics.csv"
        if not metrics.is_file():
            continue
        parts = split_name(run.name)
        if parts is None:
            continue
        variant, _, seed = parts
        points = curve(metrics)
        if points:
            runs.setdefault(variant, {})[seed] = points
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--ref", default="ukad")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    root = args.root or os.path.join(os.environ.get("WORKBENCH", "workbench"), "logs")
    runs = collect(root, args.dataset)
    if args.ref not in runs:
        print(f"reference {args.ref!r} not found; have: {', '.join(sorted(runs))}")
        return
    reference = runs.pop(args.ref)
    seeds = sorted({s for per_seed in runs.values() for s in per_seed})

    header = f"{'Variant':{NAME_W}s}" + "".join(f"{seed:>{SEED_W}d} " for seed in seeds) + f"{'Mean gap':>{MEAN_W}s}{'Epoch':>{EPOCH_W}s}"
    print(f"\n  {args.dataset.upper()}   matched-epoch gap vs {pretty(args.ref)}")
    print("  " + "─" * len(header))
    print("  " + header)
    print("  " + "─" * len(header))

    table = []
    for variant, per_seed in runs.items():
        gaps = {}
        epochs = []
        for seed, points in per_seed.items():
            epoch = max(points)
            base = at(reference.get(seed, {}), epoch)
            if base is None:
                continue
            gaps[seed] = (points[epoch] - base) * 100
            epochs.append(epoch)
        if gaps:
            table.append((variant, gaps, epochs))

    table.sort(key=lambda row: -sum(row[1].values()) / len(row[1]))
    for variant, gaps, epochs in table:
        body = ""
        for seed in seeds:
            text = f"{gaps[seed]:+.2f}" if seed in gaps else "-"
            body += f"{text:>{SEED_W}s} "
        values = list(gaps.values())
        mean = sum(values) / len(values)
        text = f"{mean:+.2f}" if len(values) < 2 else f"{mean:+.2f}±{statistics.stdev(values):.2f}"
        low, high = min(epochs), max(epochs)
        span = f"{low}" if low == high else f"{low}-{high}"
        print(f"  {pretty(variant):{NAME_W}s}{body}{text:>{MEAN_W}s}{span:>{EPOCH_W}s}")
    print()


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

DATASETS = ("busi", "cvc", "glas", "isic")
LIVE_WINDOW = 300


def summarize(path):
    epoch = -1
    best = None
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                raw_epoch = (row.get("epoch") or "").strip()
                if raw_epoch:
                    epoch = max(epoch, int(float(raw_epoch)))
                raw_iou = (row.get("val/iou") or "").strip()
                if raw_iou:
                    value = float(raw_iou)
                    if best is None or value > best:
                        best = value
    except (OSError, ValueError):
        pass
    return epoch + 1, best


def elapsed_by_run():
    out = {}
    try:
        raw = subprocess.run(["ps", "-eo", "etimes,cmd"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for line in raw.splitlines():
        found = re.search(r"^\s*(\d+)\s.*--experiment (\S+) --dataseed (\d+)", line)
        if found:
            key = f"{found.group(2)}_dataseed{found.group(3)}"
            out[key] = max(out.get(key, 0), int(found.group(1)))
    return out


def split_name(name):
    found = re.match(r"^(.*)_dataseed(\d+)$", name)
    if not found:
        return None
    experiment, seed = found.group(1), int(found.group(2))
    for dataset in DATASETS:
        if experiment.endswith("_" + dataset):
            return experiment[: -len(dataset) - 1], dataset, seed
    return experiment, "?", seed


def collect(root, patterns, max_epochs):
    running = elapsed_by_run()
    groups = defaultdict(dict)
    for run in sorted({run for pattern in patterns for run in Path(root).glob(pattern)}):
        metrics = run / "metrics.csv"
        if not metrics.is_file():
            continue
        parts = split_name(run.name)
        if parts is None:
            continue
        variant, dataset, seed = parts
        epoch, best = summarize(metrics)
        groups[(dataset, variant)][seed] = {
            "epoch": epoch,
            "best": best,
            "alive": run.name in running,
            "age": time.time() - metrics.stat().st_mtime,
        }
    return groups


ACRONYMS = {
    "kan": "KAN",
    "kad": "KAD",
    "hkd": "HKD",
    "ds": "DS",
    "ukad": "UKAD",
}
NAME_W = 30
SEED_W = 8
MEAN_W = 18
EPOCH_W = 12


def pretty(variant):
    if variant == "ukad_b":
        return "UKAD-B"
    if variant == "ukad_l":
        return "UKAD-L"
    if variant == "ukad_s":
        return "UKAD-S"
    stem = variant[5:] if variant.startswith("ukad_") else variant
    return " ".join(ACRONYMS.get(word, word.capitalize()) for word in stem.split("_"))


def cell(entry, max_epochs):
    if entry is None:
        return " " * (SEED_W + 1)
    stalled = entry["epoch"] < max_epochs and not (entry["alive"] and entry["age"] < LIVE_WINDOW)
    iou = f"{entry['best'] * 100:.2f}" if entry["best"] is not None else "-"
    return f"{'!' if stalled else ' '}{iou:>{SEED_W - 1}s} "


def progress(per_seed, max_epochs):
    epochs = [e["epoch"] for e in per_seed.values()]
    low, high = min(epochs), max(epochs)
    if low >= max_epochs:
        return f"{max_epochs}/{max_epochs}"
    span = f"{low}" if low == high else f"{low}-{high}"
    return f"{span}/{max_epochs}"


def render(root, patterns, max_epochs):
    groups = collect(root, patterns, max_epochs)
    if not groups:
        return "no runs match " + " ".join(patterns)
    lines = []
    total = done = 0
    for dataset in sorted({d for d, _ in groups}):
        rows = {v: s for (d, v), s in groups.items() if d == dataset}
        seeds = sorted({seed for s in rows.values() for seed in s})
        header = f"{'Variant':{NAME_W}s}" + "".join(f"{seed:>{SEED_W}d} " for seed in seeds) + f"{'IoU':>{MEAN_W}s}{'Epoch':>{EPOCH_W}s}"
        lines.append("")
        lines.append(f"  {dataset.upper()}")
        lines.append("  " + "\u2500" * len(header))
        lines.append("  " + header)
        lines.append("  " + "\u2500" * len(header))

        def mean_of(per_seed):
            scores = [e["best"] for e in per_seed.values() if e["best"] is not None]
            return sum(scores) / len(scores) if scores else None

        ordered = sorted(rows, key=lambda v: (mean_of(rows[v]) is None, -(mean_of(rows[v]) or 0.0)))
        for variant in ordered:
            per_seed = rows[variant]
            total += len(per_seed)
            done += sum(1 for e in per_seed.values() if e["epoch"] >= max_epochs)
            score = mean_of(per_seed)
            scores = [e["best"] for e in per_seed.values() if e["best"] is not None]
            if score is None:
                text = "-"
            elif len(scores) > 1:
                text = f"{score * 100:.2f}±{statistics.stdev(scores) * 100:.2f}"
            else:
                text = f"{score * 100:.2f}"
            mean = f"{text:>{MEAN_W}s}"
            body = "".join(cell(per_seed.get(seed), max_epochs) for seed in seeds)
            lines.append(f"  {pretty(variant):{NAME_W}s}{body}{mean}{progress(per_seed, max_epochs):>{EPOCH_W}s}")
    head = f"  {time.strftime('%Y-%m-%d %H:%M:%S')}    {total} runs    {done} done    {total - done} live    ! = stalled"
    return head + "\n" + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="*", default=["ukad*"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--watch", type=float, default=0.0)
    args = parser.parse_args()

    root = args.root or os.path.join(os.environ.get("WORKBENCH", "workbench"), "logs")
    if not args.watch:
        print(render(root, args.patterns, args.max_epochs))
        return
    try:
        while True:
            print("\033[2J\033[H" + render(root, args.patterns, args.max_epochs), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

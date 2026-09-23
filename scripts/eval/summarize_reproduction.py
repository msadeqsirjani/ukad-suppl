import csv
from pathlib import Path

LOGS = Path("outputs/reproduction_check/logs")
OUTPUT = Path("outputs/reproduction_check/summary.csv")


def main():
    rows = {}
    for log in sorted(LOGS.glob("*.log")):
        for line in log.read_text().splitlines():
            if not line.startswith("RESULT "):
                continue
            _, file, run, fallback, diff = line.split()
            rows[(file, run)] = {
                "file": file,
                "run": run,
                "fallback_batches": int(fallback.split("=")[1]),
                "max_abs_diff": float(diff.split("=")[1]),
            }
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "run", "fallback_batches", "max_abs_diff"])
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    exact = sum(row["max_abs_diff"] == 0 for row in rows.values())
    fallback = sum(row["fallback_batches"] > 0 for row in rows.values())
    print(f"{len(rows)} runs, {exact} exact, {fallback} with fallback")


if __name__ == "__main__":
    main()

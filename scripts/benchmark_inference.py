"""Benchmark inference latency and GPU memory for all paper models.

The benchmark measures the model forward pass only: it excludes data loading,
preprocessing, checkpoint I/O, and post-processing.  Models run in eval mode
with inference mode and FP32 inputs.  The 256 and 512 runs load the matching
BUSI and ISIC configuration, respectively, so resolution-specific models use
their paper configuration.

Example:
    python scripts/benchmark_inference.py --device cuda --output outputs/performance/inference.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402
from src.utils.serialization_utils import load_config  # noqa: E402


MODEL_LABELS = {
    "unet": "U-Net",
    "attunet": "Attention U-Net",
    "unetpp": "U-Net++",
    "unext_s": "UNeXt-S",
    "unext": "UNeXt-B",
    "unext_l": "UNeXt-L",
    "umamba": "U-Mamba",
    "rollingunet": "Rolling-UNet-S",
    "rollingunet_m": "Rolling-UNet-M",
    "rollingunet_l": "Rolling-UNet-L",
    "ukagnet": "UKAGNet",
    "ukan": "U-KAN",
    "ufunkan": "U-FunKAN",
    "cglknet": "CGLKNet",
    "ukanplus": "UKAN+",
    "adakan": "AdaKAN",
    "ukad_s": "UKAD-S",
    "ukad_b": "UKAD-B",
    "ukad_l": "UKAD-L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument("--warmup", type=int, default=50, help="Untimed warm-up iterations.")
    parser.add_argument("--repeats", type=int, default=200, help="Timed forward passes per model/resolution.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512], choices=[256, 512])
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_LABELS), default=list(MODEL_LABELS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "performance" / "inference_benchmark.json",
        help="JSON output path; a CSV with the same stem is also written.",
    )
    parser.add_argument("--strict", action="store_true", help="Stop instead of recording a model/configuration error.")
    return parser.parse_args()


def config_path(model_name: str, size: int) -> Path:
    # The architecture configurations are resolution agnostic except where a
    # baseline explicitly encodes a spatial size.  BUSI and ISIC provide the
    # paper's 256 and 512 configurations, respectively.
    dataset = "busi" if size == 256 else "isic"
    candidates = (
        ROOT / "configs" / "baselines" / model_name / f"{dataset}.yaml",
        ROOT / "configs" / "variants" / model_name / f"{dataset}.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    paths = ", ".join(str(path.relative_to(ROOT)) for path in candidates)
    raise FileNotFoundError(f"No {size}-pixel configuration found for {model_name}: {paths}")


def load_network(path: Path, device: torch.device) -> torch.nn.Module:
    config = load_config(path)
    if "model" not in config:
        raise ValueError(f"Expected a stage config with a model entry: {path}")
    network = models.get(config["model"])._model
    return network.to(device).eval()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(network: torch.nn.Module, size: int, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    in_channels = getattr(network, "in_channels", 3)
    x = torch.randn(1, in_channels, size, size, device=device, dtype=torch.float32)

    for _ in range(warmup):
        _ = network(x)
    synchronize(device)

    # Reset after warm-up so the peak is the stable inference allocation.  The
    # reset counter begins at current allocation, which includes model weights.
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = network(x)
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    return {
        "latency_mean_ms": float(sum(elapsed_ms) / len(elapsed_ms)),
        "latency_median_ms": float(torch.tensor(elapsed_ms).median().item()),
        "latency_std_ms": float(torch.tensor(elapsed_ms).std(unbiased=False).item()),
        "throughput_images_per_s": float(1000.0 / (sum(elapsed_ms) / len(elapsed_ms))),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "batch_size": 1,
        "timing": "CUDA events; synchronized per iteration; model forward only",
    }


def write_results(output: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    keys = ["model", "label", "input_size", "config", "params", "latency_median_ms", "latency_mean_ms", "latency_std_ms", "throughput_images_per_s", "peak_allocated_mb", "peak_reserved_mb", "error"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available CUDA device.")

    metadata = environment(device) | {"warmup": args.warmup, "repeats": args.repeats}
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for size in args.sizes:
            row: dict[str, Any] = {"model": model_name, "label": MODEL_LABELS[model_name], "input_size": size}
            network = None
            try:
                path = config_path(model_name, size)
                row["config"] = str(path.relative_to(ROOT))
                torch.cuda.empty_cache()
                network = load_network(path, device)
                row["params"] = sum(parameter.numel() for parameter in network.parameters())
                row.update(benchmark(network, size, device, args.warmup, args.repeats))
                print(f"{row['label']:20} {size:>3}px  {row['latency_median_ms']:8.3f} ms  {row['peak_allocated_mb']:8.1f} MB")
            except Exception as error:  # Keep the all-model report useful if one baseline fails.
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"{row['label']:20} {size:>3}px  ERROR: {row['error']}", file=sys.stderr)
                if args.strict:
                    raise
            finally:
                del network
                torch.cuda.empty_cache()
            rows.append(row)

    write_results(args.output, metadata, rows)
    print(f"Wrote {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main(parse_args())

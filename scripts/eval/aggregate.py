import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils.protocol import (
    SEGMENTATION_DATASETS,
    SEGMENTATION_SPLIT_SEEDS,
    TRAINING_SEED,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", choices=SEGMENTATION_DATASETS, required=True)
    parser.add_argument("--config", required=True, help="model/training experiment YAML")
    parser.add_argument(
        "--eval_config",
        required=True,
        help="validation dataloader for the evaluated dataset",
    )
    parser.add_argument("--dataseeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--input_shape", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_checkpoint(model, checkpoint, device):
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)


def main(args):
    import torch

    from flops import compute_model_info
    from src import data, models
    from src.models._seg_metrics import seg_indicators
    from src.utils.serialization_utils import load_config

    run_keys = list(args.dataseeds or SEGMENTATION_SPLIT_SEEDS)
    if len(args.checkpoint) != len(run_keys):
        raise ValueError("one --checkpoint is required per protocol run")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.environ["DATASEED"] = str(run_keys[0])
    experiment_config = load_config(args.config)
    model_info = compute_model_info(args.config, args.input_shape, device=str(device))
    model_config = experiment_config["model"]
    class_name = model_config["class_name"]

    if class_name != "UltrasoundSegmentationModel":
        raise ValueError(f"unsupported model class: {class_name}")
    metric_function = seg_indicators
    task = "segmentation"
    split = "validation"
    prefix = "val"
    independent = False
    selection = "best checkpoint selected on validation IoU"

    per_run = {}
    example_counts = {}
    for run_key, checkpoint in zip(run_keys, args.checkpoint):
        os.environ["DATASEED"] = str(run_key)
        run_label = f"dataseed{run_key}"

        model = models.get(model_config).to(device)
        load_checkpoint(model, checkpoint, device)
        loader = data.get(load_config(args.eval_config))
        totals = {}
        count = 0
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device)
                target = batch["mask"].to(device)
                values = metric_function(model(image), target)
                batch_size = int(image.shape[0])
                count += batch_size
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        if count == 0:
            raise ValueError(f"{split} dataloader is empty for {run_label}")
        per_run[run_label] = {f"{prefix}/{name}": total / count for name, total in totals.items()}
        example_counts[run_label] = count

    summary = {}
    for name in next(iter(per_run.values())):
        samples = [values[name] for values in per_run.values()]
        summary[name] = {
            "mean": statistics.fmean(samples),
            "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "n_runs": len(samples),
        }

    aggregation = "mean and sample standard deviation across data splits 2981, 6142 and 1187; fixed training RNG 1029"
    metric_semantics = "strict >0.5 threshold; medpy binary indicators per batch, weighted by batch size"

    result = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "task": task,
        "training_seed": TRAINING_SEED,
        "split_seeds": run_keys,
        "evaluation": {
            "split": split,
            "independent": independent,
            "selection": selection,
            "aggregation": aggregation,
            "metric_semantics": metric_semantics,
            "config": args.eval_config,
        },
        "config_provenance": {"path": args.config},
        "architecture": {
            "parameters": model_info["params"],
            "trainable_parameters": model_info["trainable"],
        },
        "checkpoints": {label: str(path) for label, path in zip(per_run, args.checkpoint)},
        "n_examples": example_counts,
        "per_run": per_run,
        "summary": summary,
        "model_info": model_info,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(get_args())

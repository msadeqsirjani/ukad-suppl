import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from per_image_iou import image_iou

PERF = ROOT / "outputs" / "performance"
TEST_CONFIG = "configs/data/isic_test/test_dataloader.yaml"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", type=Path, default=PERF / "isic_test")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def evaluate(model_name, device):
    import torch

    from src import data, models
    from src.models._seg_metrics import seg_indicators
    from src.utils.serialization_utils import load_config

    record = json.loads((PERF / f"{model_name}_isic.json").read_text())
    runs = {}
    for run_key, checkpoint in record["checkpoints"].items():
        os.environ["DATASEED"] = run_key.removeprefix("dataseed")
        model = models.get(load_config(record["config_provenance"]["path"])["model"]).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        model.eval()
        loader = data.get(load_config(TEST_CONFIG))
        totals, count, ious, hd95s = {}, 0, [], []
        with torch.inference_mode():
            for batch in loader:
                image, target = batch["image"].to(device), batch["mask"].to(device)
                logits = model(image)
                for name, value in seg_indicators(logits, target).items():
                    totals[name] = totals.get(name, 0.0) + value * image.shape[0]
                count += image.shape[0]
                iou, _, hd = image_iou(logits, target)
                ious += iou
                hd95s += hd
        runs[run_key] = {"batch": {name: total / count for name, total in totals.items()}, "iou": ious, "hd95": hd95s, "n": count}
    return {"model": model_name, "checkpoints": record["checkpoints"], "runs": runs}


def summarize(result):
    rows = {
        "iou": [run["batch"]["iou"] * 100 for run in result["runs"].values()],
        "dice": [run["batch"]["dice_score"] * 100 for run in result["runs"].values()],
        "hd95_img": [statistics.fmean(run["hd95"]) for run in result["runs"].values()],
    }
    return {name: (statistics.fmean(values), statistics.stdev(values)) for name, values in rows.items()}


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        out = args.out_dir / f"{Path(model_name).name}.json"
        if out.exists() and not args.force:
            result = json.loads(out.read_text())
        else:
            result = evaluate(model_name, args.device)
            out.write_text(json.dumps(result, indent=2) + "\n")
        stats = summarize(result)
        print(f"{Path(model_name).name:16}" + "  ".join(f"{name}={mean:.2f}±{std:.2f}" for name, (mean, std) in stats.items()))


if __name__ == "__main__":
    main(get_args())

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.protocol import SEGMENTATION_DATASETS

PERF = ROOT / "outputs" / "performance"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SEGMENTATION_DATASETS, default=list(SEGMENTATION_DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", type=Path, default=PERF / "per_image")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def image_iou(logits, target):
    import torch

    pred = torch.sigmoid(logits) > 0.5
    true = target > 0.5
    inter = (pred & true).flatten(1).sum(1).float()
    union = (pred | true).flatten(1).sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    return iou.tolist(), true.flatten(1).sum(1).tolist(), image_hd95(pred.cpu().numpy(), true.cpu().numpy())


def image_hd95(pred, true):
    from medpy.metric.binary import hd95

    values = []
    for p, t in zip(pred, true):
        if p.any() and t.any():
            values.append(float(hd95(p, t)))
        elif p.any() or t.any():
            values.append(float(math.hypot(*p.shape[-2:])))
        else:
            values.append(0.0)
    return values


def evaluate(model_name, dataset, device):
    import torch

    from src import data, models
    from src.utils.serialization_utils import load_config

    record = json.loads((PERF / f"{model_name}_{dataset}.json").read_text())
    runs = {}
    for run_key, checkpoint in record["checkpoints"].items():
        os.environ["DATASEED"] = run_key.removeprefix("dataseed")
        model = models.get(load_config(record["config_provenance"]["path"])["model"]).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        model.eval()
        loader = data.get(load_config(record["evaluation"]["config"]))
        ious, pixels, hd95s = [], [], []
        with torch.inference_mode():
            for batch in loader:
                iou, count, hd = image_iou(model(batch["image"].to(device)), batch["mask"].to(device))
                ious += iou
                pixels += count
                hd95s += hd
        runs[run_key] = {"iou": ious, "mask_pixels": pixels, "hd95": hd95s}
    return {"model": model_name, "dataset": dataset, "checkpoints": record["checkpoints"], "runs": runs}


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        for dataset in args.datasets:
            out = args.out_dir / f"{Path(model_name).name}_{dataset}.json"
            if out.exists() and not args.force:
                print(f"skip {out}")
                continue
            result = evaluate(model_name, dataset, args.device)
            out.write_text(json.dumps(result, indent=2) + "\n")
            means = {k: round(100 * sum(v["iou"]) / len(v["iou"]), 2) for k, v in result["runs"].items()}
            hd95 = {k: round(sum(v["hd95"]) / len(v["hd95"]), 2) for k, v in result["runs"].items()}
            print(f"wrote {out} iou={means} hd95={hd95}")


if __name__ == "__main__":
    main(get_args())

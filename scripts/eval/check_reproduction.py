import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output_dir", default="outputs/reproduction_check/runs")
    return parser.parse_args()


def check_run(payload, label, checkpoint):
    import torch
    from medpy.metric.binary import hd, hd95

    from src import data, models
    from src.models._seg_metrics import seg_indicators
    from src.utils.serialization_utils import load_config

    os.environ["DATASEED"] = label.removeprefix("dataseed")
    model = models.get(load_config(payload["config_provenance"]["path"])["model"]).cuda()
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    model.eval()
    fallback, totals, count = 0, {}, 0
    with torch.inference_mode():
        for batch in data.get(load_config(payload["evaluation"]["config"])):
            output = model(batch["image"].cuda())
            pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
            tgt = (batch["mask"] > 0.5).numpy()
            try:
                hd(pred, tgt)
                hd95(pred, tgt)
            except Exception:
                fallback += 1
            size = int(output.shape[0])
            count += size
            for name, value in seg_indicators(output, batch["mask"].cuda()).items():
                totals[name] = totals.get(name, 0.0) + value * size
    stored = payload["per_run"][label]
    metrics = {
        name: {"stored": stored[f"val/{name}"], "recomputed": total / count, "diff": total / count - stored[f"val/{name}"]}
        for name, total in totals.items()
    }
    return {"checkpoint": checkpoint, "n_examples": count, "fallback_batches": fallback, "metrics": metrics}


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in args.files:
        payload = json.loads(Path(file).read_text())
        runs = {label: check_run(payload, label, checkpoint) for label, checkpoint in payload["checkpoints"].items()}
        target = output_dir / Path(file).relative_to("outputs/performance").as_posix().replace("/", "__")
        target.write_text(json.dumps({"source": file, "runs": runs}, indent=2) + "\n")
        print(f"checked {file}", flush=True)


if __name__ == "__main__":
    main(get_args())

import glob
import os
import re

import torch


def find_latest_checkpoint(ckpt_dir):
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last):
        return last
    files = glob.glob(os.path.join(ckpt_dir, "*.pt")) + glob.glob(
        os.path.join(ckpt_dir, "*.ckpt")
    )
    if not files:
        return None

    def _epoch(path):
        match = re.search(r"epoch[_=](\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(files, key=lambda p: (_epoch(p), os.path.getmtime(p)))


def torch_load(model, ckpt, **kwargs):
    checkpoint = torch.load(ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        state_dict = {name[7:]: checkpoint[name] for name in checkpoint}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, **kwargs)
    return model


def pythonify_logs(logs):
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in logs.items()}


def split_loss_logs(value):
    if isinstance(value, dict):
        return value["loss"], pythonify_logs(value)
    elif isinstance(value, (list, tuple)):
        loss, logs = value
        return loss, pythonify_logs(logs)
    else:
        return value, {"loss": value.item()}

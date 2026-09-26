import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from thop import profile, clever_format

from src import models
from src.models import nets
from src.models.nets.baselines.ukad import GroupRational
from src.utils.serialization_utils import load_config


def count_group_rational(module, inputs, output):
    x = inputs[0]
    ops = x.numel() * (module.degree_p + module.degree_q + 2)
    module.total_ops += torch.DoubleTensor([int(ops)])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="stage yaml (configs/stage*/*.yaml) or net yaml (configs/nets/*.yaml).",
    )
    p.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=None,
        help="N C H W. If omitted, infers 512 for GlaS and ISIC, 256 otherwise.",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def load_network(cfg):
    if "model" in cfg:
        return models.get(cfg["model"])._model
    return nets.get(cfg)


def infer_input_shape(net, config_path=None):
    in_ch = getattr(net, "in_channels", None)
    if in_ch is None:
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                in_ch = m.in_channels
                break
    if in_ch is None:
        in_ch = 3
    config_name = str(config_path or "").lower()
    if "glas" in config_name or "isic" in config_name or "_512" in config_name:
        spatial = 512
    else:
        spatial = 256
    return [1, in_ch, spatial, spatial]


def compute_model_info(config_path, input_shape=None, device="cpu"):
    cfg = load_config(config_path)
    net = load_network(cfg).eval()

    shape = list(input_shape) if input_shape else infer_input_shape(net, config_path)
    assert len(shape) == 4, f"input_shape must be N C H W, got {shape}"

    dev = torch.device(device)
    net = net.to(dev)
    x = torch.randn(*shape, device=dev)

    macs, _ = profile(
        net,
        inputs=(x,),
        verbose=False,
        custom_ops={
            GroupRational: count_group_rational,
        },
    )
    flops = 2 * macs

    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024**2)

    return {
        "config": config_path,
        "input_shape": shape,
        "input_shape_source": "explicit" if input_shape else "dataset_inferred",
        "params": int(n_params),
        "trainable": int(n_trainable),
        "macs": int(macs),
        "flops": int(flops),
        "size_mb": float(size_mb),
    }


def main(args):
    info = compute_model_info(args.config, args.input_shape, args.device)
    params_h, macs_h, flops_h = clever_format([info["params"], info["macs"], info["flops"]], "%.3f")

    print(f"\n  config       : {info['config']}")
    print(f"  input shape  : {tuple(info['input_shape'])}")
    print(f"  device       : {args.device}")
    print("  --")
    print(f"  params       : {info['params']:,}  ({params_h})")
    print(f"  trainable    : {info['trainable']:,}")
    print(f"  MACs         : {info['macs']:,}  ({macs_h})")
    print(f"  FLOPs (2×M)  : {info['flops']:,}  ({flops_h})")
    print(f"  size (fp32)  : {info['size_mb']:.2f} MB")


if __name__ == "__main__":
    main(get_args())

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import argparse
from pathlib import Path
from argparse import RawTextHelpFormatter

import lightning
import torch
from src import trainers, models, data
from src.utils.serialization_utils import load_config
from src.utils.torch_utils import find_latest_checkpoint
from src.utils.protocol import (
    SEGMENTATION_DATASETS,
    SEGMENTATION_SPLIT_SEEDS,
    TRAINING_SEED,
)


def get_args():
    parser = argparse.ArgumentParser(usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to experiment configuration file (*.yaml).",
        metavar="",
    )

    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="how much of training dataset to use (default: 1.0).",
        metavar="",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="how much of validation dataset to use (default: 1.0).",
        metavar="",
    )

    parser.add_argument(
        "--dataseed",
        type=int,
        default=None,
        help="data split seed; must be one of 2981, 6142, 1187 (default: 2981).",
        metavar="",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="optional experiment name.",
        metavar="",
    )
    return parser.parse_args()


def main(args):
    dataset = Path(args.config).stem
    if dataset not in SEGMENTATION_DATASETS:
        raise ValueError(f"cannot infer supported dataset from config: {args.config}")

    split_seed = args.dataseed or SEGMENTATION_SPLIT_SEEDS[0]
    if split_seed not in SEGMENTATION_SPLIT_SEEDS:
        raise ValueError(f"dataseed must be one of {SEGMENTATION_SPLIT_SEEDS}")
    os.environ["DATASEED"] = str(split_seed)

    lightning.seed_everything(TRAINING_SEED, workers=True)
    torch.backends.cudnn.benchmark = True

    experiment = args.experiment or "_".join(Path(args.config).with_suffix("").parts[-2:])
    os.environ["EXPERIMENT_NAME"] = f"{experiment}_dataseed{split_seed}"

    cfg = load_config(args.config)

    trainer = trainers.get(
        cfg["trainer"],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    model = models.get(cfg["model"])
    train_loader = data.get(cfg["data"]["train_dataloaders"][0])
    val_loader = data.get(cfg["data"]["val_dataloaders"][0])

    resume_ckpt = None
    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    if ckpt_cb is not None and getattr(ckpt_cb, "dirpath", None):
        resume_ckpt = find_latest_checkpoint(ckpt_cb.dirpath)
    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")
    else:
        print("No existing checkpoint found; starting from scratch.")

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    from datetime import datetime

    cmd_args = get_args()
    for k, v in vars(cmd_args).items():
        print(f"\t{k:20}: {v}")

    start_time = datetime.now()
    print(f"\n{start_time}: Script `{Path(__file__).name}` has started.")
    main(cmd_args)
    end_time = datetime.now()
    print(
        f"\n{end_time}: Script `{Path(__file__).name}` has stopped.\n"
        f"Elapsed time: {end_time - start_time} (hours : minutes : seconds : microseconds)."
    )

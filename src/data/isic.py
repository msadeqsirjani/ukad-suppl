from torch.utils.data import DataLoader

from . import datasets
from ._transforms import seg_transform


def isic(dataset, split="val", **kwargs):
    db = datasets.get(dataset, transform=seg_transform(split, 512))
    return DataLoader(db, **kwargs)

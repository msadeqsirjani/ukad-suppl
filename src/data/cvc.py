from torch.utils.data import DataLoader

from . import datasets
from ._transforms import seg_transform


def cvc(dataset, split="val", **kwargs):
    db = datasets.get(dataset, transform=seg_transform(split, 256))
    return DataLoader(db, **kwargs)

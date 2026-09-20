from torch.utils.data import DataLoader

from .busi import busi
from .cvc import cvc
from .glas import glas
from .isic import isic

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(
        identifier,
        module_objects={
            "busi": busi,
            "cvc": cvc,
            "glas": glas,
            "isic": isic,
        },
        **kwargs,
    )

    if isinstance(obj, DataLoader):
        return obj
    raise ValueError(f"Could not interpret data instance: {obj}.")

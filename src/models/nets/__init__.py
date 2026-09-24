from .neuraloperator.no import NeuralOperator

from .baselines import (
    AttentionUNet,
    RollingUNet,
    UNet,
    UNetPP,
    UNeXt,
    UKAGNet,
    UKAN,
    UKANPlus,
    UKAD,
    UMamba,
    AdaKAN,
    CGLKNet,
    NNUNetResEnc,
    MedNeXt,
    LKMUNet,
    CMUNeXt,
)

from src.utils.torch_utils import torch_load
from src.utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(
        identifier,
        module_objects={
            "NeuralOperator": NeuralOperator,
            "AttentionUNet": AttentionUNet,
            "RollingUNet": RollingUNet,
            "UNet": UNet,
            "UNetPP": UNetPP,
            "UNeXt": UNeXt,
            "UKAGNet": UKAGNet,
            "UKAN": UKAN,
            "UKANPlus": UKANPlus,
            "UKAD": UKAD,
            "UMamba": UMamba,
            "AdaKAN": AdaKAN,
            "CGLKNet": CGLKNet,
            "NNUNetResEnc": NNUNetResEnc,
            "MedNeXt": MedNeXt,
            "LKMUNet": LKMUNet,
            "CMUNeXt": CMUNeXt,
        },
        **kwargs,
    )

    if checkpoint:
        obj = torch_load(obj, checkpoint, strict=True)
    return obj

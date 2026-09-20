from torch import nn
from torch.nn import functional as F

from .funkan import FUNKAN

from ..layers import (
    conv1x1,
    conv3x3,
    conv5x5,
    ConvBlock,
    ResidualEncoderBlock,
    ResidualDecoderBlock,
)


class NeuralOperator(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(16, 32),
        backbone="funkan",
        embedding=None,
        lifting=None,
        projection=None,
        depth=3,
        skip=False,
        **kwargs,
    ):
        super(NeuralOperator, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        assert len(filters) - 1 >= 1, "at least 2 filters are required"
        self.skip = bool(skip)

        if embedding is None:
            embedding = "conv5x5" if len(filters) <= 2 else "conv3x3"
        if embedding == "conv5x5":
            self.embedding = conv5x5(in_ch, filters[0])
        elif embedding == "conv3x3":
            self.embedding = conv3x3(in_ch, filters[0])
        else:
            raise ValueError(f"Unrecognized `embedding` found: {embedding}.")

        if lifting == "u-enc":
            self.lifting = nn.ModuleList(
                [
                    ResidualEncoderBlock(filters[i], filters[i + 1])
                    for i in range(len(filters) - 1)
                ]
            )
        else:
            self.lifting = nn.ModuleList(
                [
                    ConvBlock(filters[i], filters[i + 1], layer="conv3x3")
                    for i in range(len(filters) - 1)
                ]
            )

        assert backbone == "funkan", f"Unrecognized `backbone` found: {backbone}."
        self.backbone = nn.ModuleList(
            [FUNKAN(filters[-1], filters[-1], **kwargs) for _ in range(depth)]
        )

        filters = filters[::-1]
        if projection == "u-dec":
            self.projection = nn.ModuleList(
                [
                    ResidualDecoderBlock(filters[i], filters[i + 1])
                    for i in range(len(filters) - 1)
                ]
            )
        elif projection in (None, "conv1x1", "conv3x3"):
            projection_layer = "conv1x1" if projection is None else projection
            self.projection = nn.ModuleList(
                [
                    ConvBlock(filters[i], filters[i + 1], layer=projection_layer)
                    for i in range(len(filters) - 1)
                ]
            )
        else:
            raise ValueError(f"Unrecognized `projection` found: {projection}.")

        self.restoration = conv1x1(filters[-1], out_ch)

    def forward(self, x):
        x = self.embedding(x)
        feats = {}
        for i, layer in enumerate(self.lifting):
            if self.skip:
                x, feat = layer(x, return_feature=True)
                feats[f"enc-{i}"] = feat
            else:
                x = layer(x)
        for layer in self.backbone:
            x = x + layer(x)
        for j, layer in enumerate(self.projection):
            if self.skip:
                x = layer(x, feats[f"enc-{i - j}"])
            else:
                x = layer(x)
        x = self.restoration(F.relu(x))
        return x

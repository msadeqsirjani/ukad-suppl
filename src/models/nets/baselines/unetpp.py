import torch
from torch import nn
from torch.nn import functional as F

from ._blocks import DoubleConv


class UNetPP(nn.Module):

    def __init__(self, in_ch, out_ch=None, filters=(32, 64, 128, 256, 512), **kwargs):
        super(UNetPP, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        filters = list(filters)
        assert len(filters) >= 2, "UNetPP requires at least 2 filter widths"

        self.depth = len(filters)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.nodes = nn.ModuleDict()
        for i in range(self.depth):
            for j in range(self.depth - i):
                if j == 0:
                    in_c = in_ch if i == 0 else filters[i - 1]
                else:
                    in_c = filters[i] * j + filters[i + 1]
                self.nodes[f"{i}_{j}"] = DoubleConv(in_c, filters[i])

        self.restoration = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        feat = {}
        for i in range(self.depth):
            feat[(i, 0)] = self.nodes[f"{i}_0"](x if i == 0 else self.pool(feat[(i - 1, 0)]))

        for j in range(1, self.depth):
            for i in range(self.depth - j):
                up = F.interpolate(
                    feat[(i + 1, j - 1)],
                    size=feat[(i, 0)].shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                cat = torch.cat([feat[(i, k)] for k in range(j)] + [up], dim=1)
                feat[(i, j)] = self.nodes[f"{i}_{j}"](cat)

        return self.restoration(feat[(0, self.depth - 1)])

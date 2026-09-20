"""Attention U-Net matching the standard 34.88 M implementation."""

import torch
from torch import nn
from torch.nn import functional as F

from ._blocks import DoubleConv


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size=None):
        if size is None:
            return self.conv(x)
        # Keep the official operation order while supporting odd-sized inputs.
        x = F.interpolate(x, size=size, mode="nearest")
        return self.conv[1:](x)


class AttentionGate(nn.Module):

    def __init__(self, gating_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(64, 128, 256, 512, 1024),
        inter_ch=None,
        **kwargs,
    ):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        filters = list(filters)
        assert len(filters) >= 2, "AttentionUNet requires at least 2 filter widths"

        self.depth = len(filters)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.encoders = nn.ModuleList(
            [
                DoubleConv(in_ch if i == 0 else filters[i - 1], filters[i])
                for i in range(self.depth)
            ]
        )
        self.upconvs = nn.ModuleDict(
            {str(j): UpConv(filters[j + 1], filters[j]) for j in range(self.depth - 1)}
        )
        self.attentions = nn.ModuleDict(
            {
                str(j): AttentionGate(
                    filters[j],
                    filters[j],
                    int(inter_ch) if inter_ch is not None else max(filters[j] // 2, 1),
                )
                for j in range(self.depth - 1)
            }
        )
        self.decoders = nn.ModuleDict(
            {
                str(j): DoubleConv(2 * filters[j], filters[j])
                for j in range(self.depth - 1)
            }
        )
        self.restoration = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else self.pool(x))
            if i < self.depth - 1:
                skips.append(x)

        for j in range(self.depth - 2, -1, -1):
            x = self.upconvs[str(j)](x, skips[j].shape[2:])
            skip = self.attentions[str(j)](x, skips[j])
            x = self.decoders[str(j)](torch.cat([skip, x], dim=1))
        return self.restoration(x)

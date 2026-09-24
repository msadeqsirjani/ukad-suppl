import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def conv_block(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
    )


def up_conv(ch_in, ch_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), *conv_block(ch_in, ch_out))


def pointwise_mlp(ch_in, ch_out, hidden):
    return [
        nn.Conv2d(ch_in, hidden, kernel_size=1),
        nn.GELU(),
        nn.BatchNorm2d(hidden),
        nn.Conv2d(hidden, ch_out, kernel_size=1),
        nn.GELU(),
        nn.BatchNorm2d(ch_out),
    ]


def fusion_conv(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=2),
        nn.GELU(),
        nn.BatchNorm2d(ch_in),
        *pointwise_mlp(ch_in, ch_out, ch_out * 4),
    )


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=k, groups=ch_in, padding=k // 2), nn.GELU(), nn.BatchNorm2d(ch_in))),
                    *pointwise_mlp(ch_in, ch_in, ch_in * 4),
                )
                for _ in range(depth)
            ]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        return self.up(self.block(x))


class CMUNeXt(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dims=(16, 32, 128, 160, 256), depths=(1, 1, 1, 3, 1), kernels=(3, 3, 7, 7, 7)):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(in_ch, dims[0])
        self.encoders = nn.ModuleList(CMUNeXtBlock(dims[max(i - 1, 0)], dims[i], depth=depths[i], k=kernels[i]) for i in range(len(dims)))
        self.ups = nn.ModuleList(up_conv(dims[i], dims[i - 1]) for i in range(len(dims) - 1, 0, -1))
        self.fusions = nn.ModuleList(fusion_conv(dims[i - 1] * 2, dims[i - 1]) for i in range(len(dims) - 1, 0, -1))
        self.head = nn.Conv2d(dims[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        for i, encoder in enumerate(self.encoders):
            x = encoder(self.pool(x) if i else x)
            skips.append(x)
        for up, fusion, skip in zip(self.ups, self.fusions, reversed(skips[:-1])):
            x = fusion(torch.cat((skip, up(x)), dim=1))
        return self.head(x)

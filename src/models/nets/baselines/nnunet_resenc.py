from collections.abc import Sequence

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, bias: bool = True, act: bool = True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
        ]
        if act:
            layers.append(nn.LeakyReLU(inplace=True))
        super().__init__(*layers)


class BasicBlockD(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, act=False)
        skip = []
        if stride != 1:
            skip.append(nn.AvgPool2d(stride, stride))
        if in_ch != out_ch:
            skip.append(ConvNormAct(in_ch, out_ch, 1, bias=False, act=False))
        self.skip = nn.Sequential(*skip)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class NNUNetResEnc(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        filters: Sequence[int] = (32, 64, 128, 256, 512, 512, 512),
        strides: Sequence[int] = (1, 2, 2, 2, 2, 2, 2),
        blocks_per_stage: Sequence[int] = (1, 3, 4, 6, 6, 6, 6),
        decoder_blocks_per_stage: Sequence[int] = (1, 1, 1, 1, 1, 1),
        kernel_size: int = 3,
    ):
        super().__init__()
        n = len(filters)
        assert len(strides) == n and len(blocks_per_stage) == n and len(decoder_blocks_per_stage) == n - 1
        self.stem = ConvNormAct(in_ch, filters[0], kernel_size)
        self.encoder = nn.ModuleList()
        prev = filters[0]
        for f, s, b in zip(filters, strides, blocks_per_stage):
            self.encoder.append(nn.Sequential(*[BasicBlockD(prev if i == 0 else f, f, kernel_size, s if i == 0 else 1) for i in range(b)]))
            prev = f
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for k in range(1, n):
            below, skip = filters[-k], filters[-(k + 1)]
            convs = [ConvNormAct(2 * skip if i == 0 else skip, skip, kernel_size) for i in range(decoder_blocks_per_stage[k - 1])]
            self.decoder.append(nn.Sequential(*convs))
            self.upsamples.append(nn.ConvTranspose2d(below, skip, strides[-k], strides[-k]))
            self.seg_layers.append(nn.Conv2d(skip, out_ch, 1))
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, BasicBlockD):
            nn.init.zeros_(module.conv2[1].weight)
            nn.init.zeros_(module.conv2[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = skips[-1]
        for k, (up, stage) in enumerate(zip(self.upsamples, self.decoder)):
            x = stage(torch.cat((up(x), skips[-(k + 2)]), 1))
        return self.seg_layers[-1](x)

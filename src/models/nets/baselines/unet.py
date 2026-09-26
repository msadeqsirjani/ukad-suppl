import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(64, 128, 256, 512, 1024),
        bilinear=False,
        **kwargs,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        filters = list(filters)
        assert len(filters) >= 2, "UNet requires at least 2 filter widths"

        factor = 2 if bilinear else 1
        encoder_widths = filters[:-1] + [filters[-1] // factor]
        self.inc = DoubleConv(in_ch, encoder_widths[0])
        self.downs = nn.ModuleList(
            [
                Down(encoder_widths[i - 1], encoder_widths[i])
                for i in range(1, len(encoder_widths))
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(
                    filters[j + 1],
                    filters[j] if j == 0 else filters[j] // factor,
                    bilinear,
                )
                for j in range(len(filters) - 2, -1, -1)
            ]
        )
        self.restoration = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))

        x = features[-1]
        for up, skip in zip(self.ups, reversed(features[:-1])):
            x = up(x, skip)
        return self.restoration(x)

import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {
    "S": dict(exp_r=[2] * 9, block_counts=[2] * 9),
    "B": dict(exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], block_counts=[2] * 9),
    "M": dict(exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3]),
    "L": dict(exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3], block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3]),
}


class MedNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=True):
        super().__init__()
        self.do_res = do_res
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv2 = nn.Conv2d(in_channels, exp_r * in_channels, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(exp_r * in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.conv3(self.act(self.conv2(self.norm(self.conv1(x)))))
        return x + x1 if self.do_res else x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels)

    def forward(self, x):
        x1 = super().forward(x)
        return x1 + self.res_conv(x) if self.resample_do_res else x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.ConvTranspose2d(in_channels, out_channels, 1, stride=2)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels)

    def forward(self, x):
        x1 = F.pad(super().forward(x), (1, 0, 1, 0))
        if self.resample_do_res:
            x1 = x1 + F.pad(self.res_conv(x), (1, 0, 1, 0))
        return x1


class OutBlock(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv_out = nn.ConvTranspose2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv_out(x)


class MedNeXt(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, size="B", n_channels=32, kernel_size=3, do_res=True, do_res_up_down=True, **kwargs):
        super().__init__()
        exp_r, counts = PRESETS[size]["exp_r"], PRESETS[size]["block_counts"]
        widths = [n_channels * 2**i for i in range(5)]

        def stage(i, w):
            return nn.Sequential(*[MedNeXtBlock(w, w, exp_r[i], kernel_size, do_res) for _ in range(counts[i])])

        self.stem = nn.Conv2d(in_ch, n_channels, kernel_size=1)
        for i in range(4):
            setattr(self, f"enc_block_{i}", stage(i, widths[i]))
            setattr(self, f"down_{i}", MedNeXtDownBlock(widths[i], widths[i + 1], exp_r[i + 1], kernel_size, do_res_up_down))
        self.bottleneck = stage(4, widths[4])
        for i in reversed(range(4)):
            setattr(self, f"up_{i}", MedNeXtUpBlock(widths[i + 1], widths[i], exp_r[8 - i], kernel_size, do_res_up_down))
            setattr(self, f"dec_block_{i}", stage(8 - i, widths[i]))
        self.out_0 = OutBlock(n_channels, out_ch)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i in range(4):
            skips.append(getattr(self, f"enc_block_{i}")(x))
            x = getattr(self, f"down_{i}")(skips[-1])
        x = self.bottleneck(x)
        for i in reversed(range(4)):
            x = getattr(self, f"dec_block_{i}")(skips[i] + getattr(self, f"up_{i}")(x))
        return self.out_0(x)

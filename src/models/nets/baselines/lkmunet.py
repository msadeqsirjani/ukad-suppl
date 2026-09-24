import copy

import torch
from torch import nn
from torch.nn import functional as F


def _mamba(dim, d_state, d_conv, expand):
    from mamba_ssm import Mamba

    return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=False)


class BiMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_forw = _mamba(dim, d_state, d_conv, expand)
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)
        self.mamba_backw = _mamba(dim, d_state, d_conv, expand)
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()

    def forward(self, tokens):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = self.norm(tokens.float())
            forw = self.mamba_forw(tokens)
            backw = self.mamba_backw(tokens.flip(1)).flip(1)
            return self.out_proj(forw + backw)


class PixelMamba(nn.Module):
    def __init__(self, dim, window, **mamba_kwargs):
        super().__init__()
        self.window = window
        self.mamba = BiMamba(dim, **mamba_kwargs)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.window if h % self.window == 0 and w % self.window == 0 else 1
        nh, nw = h // p, w // p
        tokens = x.reshape(b, c, nh, p, nw, p).permute(0, 3, 5, 2, 4, 1).reshape(b * p * p, nh * nw, c)
        y = self.mamba(tokens).reshape(b, p, p, nh, nw, c).permute(0, 5, 3, 1, 4, 2).reshape(b, c, h, w)
        return y + x


class PatchMamba(nn.Module):
    def __init__(self, dim, pool, **mamba_kwargs):
        super().__init__()
        self.pool = pool
        self.mamba = BiMamba(dim, **mamba_kwargs)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.pool if h % self.pool == 0 and w % self.pool == 0 else 1
        z = F.avg_pool2d(x, p)
        nh, nw = z.shape[-2:]
        y = self.mamba(z.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, nh, nw)
        return F.interpolate(y, scale_factor=p, mode="nearest") + x


def _conv_norm(in_ch, out_ch, kernel, stride, bias, act):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding=(kernel - 1) // 2, bias=bias),
        nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
    ]
    if act:
        layers.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, conv_bias=True):
        super().__init__()
        self.conv1 = _conv_norm(in_ch, out_ch, 3, stride, conv_bias, True)
        self.conv2 = _conv_norm(out_ch, out_ch, 3, 1, conv_bias, False)
        self.act = nn.LeakyReLU(inplace=True)
        skip = []
        if stride != 1:
            skip.append(nn.AvgPool2d(stride, stride))
        if in_ch != out_ch:
            skip.append(_conv_norm(in_ch, out_ch, 1, 1, False, False))
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


def _res_stage(in_ch, out_ch, stride, blocks, conv_bias):
    return nn.Sequential(ResBlockD(in_ch, out_ch, stride, conv_bias), *[ResBlockD(out_ch, out_ch, 1, conv_bias) for _ in range(blocks - 1)])


def _per_stage(value, n):
    return [int(value)] * n if isinstance(value, int) else [int(v) for v in value]


class LKMUNet(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=1,
        filters=(32, 64, 128, 256, 512, 512),
        strides=None,
        blocks_per_stage=2,
        decoder_blocks_per_stage=2,
        window_scale=5,
        d_state=16,
        d_conv=4,
        expand=2,
        conv_bias=True,
    ):
        super().__init__()
        filters = [int(f) for f in filters]
        n = len(filters)
        strides = _per_stage(strides if strides is not None else [1] + [2] * (n - 1), n)
        enc_blocks = _per_stage(blocks_per_stage, n)
        dec_blocks = _per_stage(decoder_blocks_per_stage, n - 1)
        if not len(strides) == len(enc_blocks) == n or len(dec_blocks) != n - 1:
            raise ValueError("strides/blocks_per_stage need n entries and decoder_blocks_per_stage n - 1")
        mamba_kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        scales = [2 ** ((n - s + 1) // 2 - 1) for s in range(n)]
        self.windows = [window_scale * p for p in scales]
        self.pools = scales

        self.stem = _conv_norm(in_ch, filters[0], 3, 1, conv_bias, True)
        chans = [filters[0]] + filters
        self.stages = nn.ModuleList(_res_stage(chans[s], filters[s], strides[s], enc_blocks[s], conv_bias) for s in range(n))
        self.pixel_mamba = nn.ModuleList(PixelMamba(filters[s], self.windows[s], **mamba_kwargs) for s in range(n))
        self.patch_mamba = nn.ModuleList(PatchMamba(filters[s], self.pools[s], **mamba_kwargs) for s in range(n))

        levels = range(n - 2, -1, -1)
        self.decoder = nn.ModuleList(_res_stage(2 * filters[s], filters[s], 1, dec_blocks[i], conv_bias) for i, s in enumerate(levels))
        self.upsamples = nn.ModuleList(nn.ConvTranspose2d(filters[s + 1], filters[s], strides[s + 1], strides[s + 1], bias=conv_bias) for s in levels)
        self.heads = nn.ModuleList(nn.Conv2d(filters[s], out_ch, 1) for s in levels)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, ResBlockD):
            nn.init.zeros_(module.conv2[1].weight)
            nn.init.zeros_(module.conv2[1].bias)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage, pixel, patch in zip(self.stages, self.pixel_mamba, self.patch_mamba):
            x = patch(pixel(stage(x)))
            skips.append(x)
        x = skips[-1]
        for i, (up, block) in enumerate(zip(self.upsamples, self.decoder)):
            x = block(torch.cat((up(x), skips[-(i + 2)]), 1))
        return self.heads[-1](x)

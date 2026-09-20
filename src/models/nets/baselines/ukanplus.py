import math

import torch
from torch import nn
from torch.nn import functional as F

from .ukan import (
    ConvLayer,
    DecoderConvLayer,
    KANBlock,
    KANLinear,
    PatchEmbed,
)


class KANSelectiveKernel(nn.Module):

    def __init__(self, ch, branches=2, reduction=16, min_dim=32, grid_size=5, spline_order=3):
        super().__init__()
        self.branches = branches
        self.ch = ch
        dim = max(ch // reduction, min_dim)
        groups = math.gcd(ch, 32)
        self.paths = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    ch,
                    ch,
                    kernel_size=3,
                    padding=1 + i,
                    dilation=1 + i,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for i in range(branches)
        )
        self.squeeze = KANLinear(ch, dim, grid_size=grid_size, spline_order=spline_order)
        self.squeeze_norm = nn.LayerNorm(dim)
        self.select = nn.ModuleList(KANLinear(dim, ch, grid_size=grid_size, spline_order=spline_order) for _ in range(branches))

    def forward(self, x):
        feats = torch.stack([path(x) for path in self.paths], dim=1)
        pooled = feats.sum(dim=1).mean(dim=(2, 3))
        z = self.squeeze_norm(self.squeeze(pooled))
        logits = torch.stack([proj(z) for proj in self.select], dim=1)
        weights = logits.softmax(dim=1)[..., None, None]
        return (feats * weights).sum(dim=1)


class ConvSpatialAttention(nn.Module):

    def __init__(self, ch, kernel_size=7):
        super().__init__()
        self.reduce = nn.Conv2d(ch, 1, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        peak = x.amax(dim=1, keepdim=True)
        conv = self.reduce(x)
        weight = torch.sigmoid(self.spatial(torch.cat([avg, peak, conv], dim=1)))
        return x * weight


class KSCSFusion(nn.Module):

    def __init__(self, ch, branches=2, reduction=16, spatial_kernel=7, grid_size=5, spline_order=3):
        super().__init__()
        self.channel = KANSelectiveKernel(ch, branches, reduction, grid_size=grid_size, spline_order=spline_order)
        self.spatial = ConvSpatialAttention(ch, spatial_kernel)

    def forward(self, x):
        return x + self.channel(x) + self.spatial(x)


class UKANPlus(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        embed_dims=(128, 160, 256),
        no_kan=False,
        drop_rate=0.0,
        fusion_branches=2,
        fusion_reduction=16,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        if len(embed_dims) != 3:
            raise ValueError("UKANPlus expects exactly three embedding widths")
        c3, c4, c5 = embed_dims
        if c3 % 8:
            raise ValueError("the first UKANPlus embedding width must be divisible by 8")
        c1, c2 = c3 // 8, c3 // 4

        self.encoder1 = ConvLayer(in_ch, c1)
        self.encoder2 = ConvLayer(c1, c2)
        self.encoder3 = ConvLayer(c2, c3)

        self.fusion_e1 = KSCSFusion(c1, fusion_branches, fusion_reduction)
        self.fusion_e2 = KSCSFusion(c2, fusion_branches, fusion_reduction)
        self.fusion_e3 = KSCSFusion(c3, fusion_branches, fusion_reduction)

        self.patch_embed3 = PatchEmbed(c3, c4)
        self.block1 = nn.ModuleList([KANBlock(c4, drop_rate, no_kan)])
        self.norm3 = nn.LayerNorm(c4)

        self.patch_embed4 = PatchEmbed(c4, c5)
        self.block2 = nn.ModuleList([KANBlock(c5, drop_rate, no_kan)])
        self.norm4 = nn.LayerNorm(c5)

        self.decoder1 = DecoderConvLayer(c5, c4)
        self.dblock1 = nn.ModuleList([KANBlock(c4, drop_rate, no_kan)])
        self.dnorm3 = nn.LayerNorm(c4)

        self.decoder2 = DecoderConvLayer(c4, c3)
        self.dblock2 = nn.ModuleList([KANBlock(c3, drop_rate, no_kan)])
        self.dnorm4 = nn.LayerNorm(c3)

        self.decoder3 = DecoderConvLayer(c3, c2)
        self.decoder4 = DecoderConvLayer(c2, c1)
        self.decoder5 = DecoderConvLayer(c1, c1)

        self.fusion_d3 = KSCSFusion(c3, fusion_branches, fusion_reduction)
        self.fusion_d4 = KSCSFusion(c2, fusion_branches, fusion_reduction)
        self.fusion_d5 = KSCSFusion(c1, fusion_branches, fusion_reduction)

        self.restoration = nn.Conv2d(c1, out_ch, kernel_size=1)

    @staticmethod
    def _tokens_to_map(x, h, w):
        b = x.shape[0]
        return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _upsample(x, size):
        return F.relu(F.interpolate(x, size=size, mode="bilinear"))

    def forward(self, x):
        input_size = x.shape[2:]
        t1 = self.fusion_e1(F.relu(F.max_pool2d(self.encoder1(x), 2, 2)))
        t2 = self.fusion_e2(F.relu(F.max_pool2d(self.encoder2(t1), 2, 2)))
        t3 = self.fusion_e3(F.relu(F.max_pool2d(self.encoder3(t2), 2, 2)))

        out, h, w = self.patch_embed3(t3)
        for block in self.block1:
            out = block(out, h, w)
        out = self.norm3(out)
        t4 = self._tokens_to_map(out, h, w)

        out, h, w = self.patch_embed4(t4)
        for block in self.block2:
            out = block(out, h, w)
        out = self._tokens_to_map(self.norm4(out), h, w)

        out = self._upsample(self.decoder1(out), t4.shape[2:]) + t4
        _, _, h, w = out.shape
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock1:
            out = block(out, h, w)
        out = self._tokens_to_map(self.dnorm3(out), h, w)

        out = self._upsample(self.decoder2(out), t3.shape[2:]) + t3
        _, _, h, w = out.shape
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock2:
            out = block(out, h, w)
        out = self._tokens_to_map(self.dnorm4(out), h, w)

        out = self._upsample(self.decoder3(self.fusion_d3(out)), t2.shape[2:]) + t2
        out = self._upsample(self.decoder4(self.fusion_d4(out)), t1.shape[2:]) + t1
        out = self._upsample(self.decoder5(self.fusion_d5(out)), input_size)
        return self.restoration(out)

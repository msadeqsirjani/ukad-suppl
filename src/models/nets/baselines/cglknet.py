import timm
import torch
from torch import nn
from torchvision.ops import DeformConv2d


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class LargeKernelDeformFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.offset1 = nn.Conv2d(dim, 2 * 7 * 7, kernel_size=3, padding=1)
        self.dcn1 = DeformConv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.offset2 = nn.Conv2d(dim, 2 * 5 * 5, kernel_size=3, padding=1)
        self.dcn2 = DeformConv2d(dim, dim, kernel_size=5, padding=6, dilation=3, groups=dim)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out1 = self.dcn1(x, self.offset1(x))
        out2 = self.dcn2(out1, self.offset2(out1))
        stacked = torch.stack([out1, out2], dim=1)
        weights = torch.softmax(torch.stack([self.gap(out1), self.gap(out2)], dim=1), dim=1)
        return (stacked * weights).sum(1)


class PDLEA(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.lk_fusion = LargeKernelDeformFusion(dim)
        self.norm2 = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.act(self.proj_in(self.norm1(x)))
        y = self.lk_fusion(y)
        x = x + y
        return x + self.mlp(self.norm2(x))


class DLEM(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.pdlea1 = PDLEA(dim, mlp_ratio)
        self.pdlea2 = PDLEA(dim, mlp_ratio)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, p, g):
        x = self.act(self.norm(p + g))
        x = self.pdlea1(x)
        x = self.pdlea2(x)
        return self.proj_out(x)


class LocalContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim))

    def forward(self, x):
        return self.proj(self.dwconv(x))


class GlobalContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1), nn.BatchNorm2d(1))

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True)[0]
        return self.proj(torch.cat([avg, mx], dim=1))


class GBM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_ctx = LocalContext(dim)
        self.global_ctx = GlobalContext()
        self.fuse_bn = nn.BatchNorm2d(dim)
        self.select = nn.Conv2d(dim, 2, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, skip, context):
        fused = self.fuse_bn(self.local_ctx(skip) + self.global_ctx(context))
        weights = torch.softmax(self.select(fused), dim=1)
        w_skip, w_ctx = weights[:, 0:1], weights[:, 1:2]
        skip_att = skip + w_skip * skip
        ctx_att = context + w_ctx * context
        cross_gate = torch.sigmoid(skip_att + ctx_att)
        skip_att = skip_att * cross_gate
        ctx_att = ctx_att * cross_gate
        channel_weights = torch.softmax(torch.stack([self.gap(skip_att), self.gap(ctx_att)], dim=1), dim=1)
        stacked = torch.stack([skip_att, ctx_att], dim=1)
        combined = (stacked * channel_weights).sum(1)
        gated = torch.sigmoid(combined) * skip
        return self.out_bn(self.out_proj(gated))


class PatchExpand(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(dim // 2)

    def forward(self, x):
        return self.norm(self.shuffle(self.expand(x)))


class CGLKNet(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        backbone="maxvit_rmlp_tiny_rw_256",
        mlp_ratio=4.0,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.in_ch = in_ch
        self.encoder = timm.create_model(backbone, features_only=True, pretrained=False, out_indices=(1, 2, 3, 4))
        dims = self.encoder.feature_info.channels()
        if len(dims) != 4 or any(dims[i] * 2 != dims[i + 1] for i in range(3)):
            raise ValueError("CGLKNet expects a 4-stage backbone with doubling channel widths")

        self.gbm3 = GBM(dims[2])
        self.dlem3 = DLEM(dims[2], mlp_ratio)
        self.up3 = PatchExpand(dims[3])

        self.gbm2 = GBM(dims[1])
        self.dlem2 = DLEM(dims[1], mlp_ratio)
        self.up2 = PatchExpand(dims[2])

        self.gbm1 = GBM(dims[0])
        self.dlem1 = DLEM(dims[0], mlp_ratio)
        self.up1 = PatchExpand(dims[1])

        self.final_up = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] * 16, kernel_size=1),
            nn.PixelShuffle(4),
        )
        self.restoration = nn.Conv2d(dims[0], out_ch, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] != 3:
            x = x[:, :3]

        m1, m2, m3, m4 = self.encoder(x)

        up = self.up3(m4)
        gate = self.gbm3(m3, up)
        d3 = self.dlem3(up, gate)

        up = self.up2(d3)
        gate = self.gbm2(m2, up)
        d2 = self.dlem2(up, gate)

        up = self.up1(d2)
        gate = self.gbm1(m1, up)
        d1 = self.dlem1(up, gate)

        out = self.final_up(d1)
        return self.restoration(out)

import math

import torch
from torch import nn
from torch.nn import functional as F


class DWBNReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class BernsteinKANLinear(nn.Module):
    def __init__(self, in_features, out_features, order=3, base_activation=nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bernstein_coeff = nn.Parameter(torch.empty(in_features, order + 1))
        self.base_scale = nn.Parameter(torch.ones(in_features))
        self.bernstein_scale = nn.Parameter(torch.ones(in_features))
        binom = torch.tensor([math.comb(order, r) for r in range(order + 1)], dtype=torch.float32)
        self.register_buffer("binom", binom)
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.bernstein_coeff, std=0.1)

    def bernstein(self, x):
        z = x.clamp(0.0, 1.0).unsqueeze(-1)
        r = torch.arange(self.order + 1, device=x.device, dtype=x.dtype)
        basis = self.binom.to(x.dtype) * z.pow(r) * (1 - z).pow(self.order - r)
        return (basis * self.bernstein_coeff).sum(-1)

    def forward(self, x):
        phi = self.base_scale * self.base_activation(x) + self.bernstein_scale * self.bernstein(x)
        return F.linear(phi, self.base_weight)


class BernsteinKANLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, order=3, drop=0.0):
        super().__init__()
        self.fc1 = BernsteinKANLinear(in_features, hidden_features, order)
        self.fc2 = BernsteinKANLinear(hidden_features, out_features, order)
        self.fc3 = BernsteinKANLinear(out_features, out_features, order)
        self.dwconv1 = DWBNReLU(hidden_features)
        self.dwconv2 = DWBNReLU(out_features)
        self.dwconv3 = DWBNReLU(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.dwconv1(self.fc1(x), h, w)
        x = self.dwconv2(self.fc2(x), h, w)
        x = self.dwconv3(self.fc3(x), h, w)
        return self.drop(x)


class AdaptKAN(nn.Module):
    def __init__(self, dim, order=3, drop=0.0, bottleneck_ratio=1, scale_init=0.1):
        super().__init__()
        hidden = max(1, dim // bottleneck_ratio)
        self.norm = nn.LayerNorm(dim)
        self.kan = BernsteinKANLayer(dim, dim, dim, order, drop)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.up = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        left = self.kan(self.norm(x), h, w)
        right = self.up(self.act(self.down(x)))
        return self.drop(left + self.scale * right)


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def _heads(self, proj, x):
        b, n, c = x.shape
        return proj(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, x):
        b, n, c = x.shape
        q = self._heads(self.q, x).softmax(dim=-1)
        k = self._heads(self.k, x).softmax(dim=-2)
        v = self._heads(self.v, x)
        context = self.attn_drop(k.transpose(-2, -1) @ v)
        out = (q @ context).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class EffiKANBlock(nn.Module):
    def __init__(self, dim, num_heads=8, order=3, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, drop)
        self.adapt_kan = AdaptKAN(dim, order, drop)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x))
        x = x + self.adapt_kan(x, h, w)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        fan_out = self.proj.kernel_size[0] * self.proj.kernel_size[1]
        fan_out *= self.proj.out_channels
        fan_out //= self.proj.groups
        nn.init.normal_(self.proj.weight, 0, math.sqrt(2.0 / fan_out))
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AdaKAN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        embed_dims=(256, 320, 512),
        num_heads=8,
        bernstein_order=3,
        drop_rate=0.0,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        if len(embed_dims) != 3:
            raise ValueError("AdaKAN expects exactly three embedding widths")
        c3, c4, c5 = embed_dims
        if c3 % 8:
            raise ValueError("the first AdaKAN embedding width must be divisible by 8")
        if c4 % num_heads or c5 % num_heads:
            raise ValueError("the tokenized embedding widths must divide num_heads")
        c1, c2 = c3 // 8, c3 // 4

        self.encoder1 = ConvLayer(in_ch, c1)
        self.encoder2 = ConvLayer(c1, c2)
        self.encoder3 = ConvLayer(c2, c3)

        self.patch_embed3 = PatchEmbed(c3, c4)
        self.block1 = nn.ModuleList([EffiKANBlock(c4, num_heads, bernstein_order, drop_rate)])
        self.norm3 = nn.LayerNorm(c4)

        self.patch_embed4 = PatchEmbed(c4, c5)
        self.block2 = nn.ModuleList([EffiKANBlock(c5, num_heads, bernstein_order, drop_rate)])
        self.norm4 = nn.LayerNorm(c5)

        self.decoder1 = DecoderConvLayer(c5, c4)
        self.dblock1 = nn.ModuleList([EffiKANBlock(c4, num_heads, bernstein_order, drop_rate)])
        self.dnorm3 = nn.LayerNorm(c4)

        self.decoder2 = DecoderConvLayer(c4, c3)
        self.dblock2 = nn.ModuleList([EffiKANBlock(c3, num_heads, bernstein_order, drop_rate)])
        self.dnorm4 = nn.LayerNorm(c3)

        self.decoder3 = DecoderConvLayer(c3, c2)
        self.decoder4 = DecoderConvLayer(c2, c1)
        self.decoder5 = DecoderConvLayer(c1, c1)
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
        t1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t2 = F.relu(F.max_pool2d(self.encoder2(t1), 2, 2))
        t3 = F.relu(F.max_pool2d(self.encoder3(t2), 2, 2))

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

        out = self._upsample(self.decoder3(out), t2.shape[2:]) + t2
        out = self._upsample(self.decoder4(out), t1.shape[2:]) + t1
        out = self._upsample(self.decoder5(out), input_size)
        return self.restoration(out)

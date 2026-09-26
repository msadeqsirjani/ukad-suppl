import math

import torch
from torch import nn
from torch.nn import functional as F

from ._blocks import DoubleConv


class DropPath(nn.Module):

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DWConv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.point_conv(self.dwconv(x))


class Lo2(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(2 * in_features, hidden_features)
        self.fc6 = nn.Linear(2 * hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv(hidden_features)
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(2 * hidden_features)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.apply(_init_weights)

    @staticmethod
    def _roll_tokens(x, h, w, spatial_dim, direction=1):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w).contiguous()
        channels = torch.chunk(x, c, dim=1)
        x = torch.cat(
            [
                torch.roll(channel, direction * shift, dims=spatial_dim)
                for channel, shift in zip(channels, range(c))
            ],
            dim=1,
        )
        return x.reshape(b, c, h * w).transpose(1, 2).contiguous()

    def forward(self, x, h, w):
        b, _, c = x.shape

        x1 = self._roll_tokens(x, h, w, spatial_dim=2)
        x1 = self.drop(self.act1(self.fc1(x1)))
        x1 = self.drop(self.fc2(self._roll_tokens(x1, h, w, spatial_dim=3)))

        x2 = self._roll_tokens(x, h, w, spatial_dim=3, direction=-1)
        x2 = self.drop(self.act1(self.fc3(x2)))
        x2 = self.drop(self.fc4(self._roll_tokens(x2, h, w, spatial_dim=2)))

        x1 = self.fc5(self.norm1(torch.cat([x1 + x, x2 + x], dim=2)))
        x1 = self.drop(x1) + x

        x2 = x.transpose(1, 2).view(b, c, h, w)
        x2 = self.norm2(self.act2(self.dwconv(x2)))
        x2 = x2.flatten(2).transpose(1, 2)

        return self.drop(self.fc6(torch.cat([x1, x2], dim=2)))


class Lo2Block(nn.Module):

    def __init__(self, dim, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Lo2(dim, hidden_features=dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x, h, w):
        return self.drop_path(self.mlp(x, h, w))


class FeatureIncentive(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_ch)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(self.act(x.flatten(2).transpose(1, 2)))
        return x, h, w


class DecoderDoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _tokens_to_map(x, h, w):
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class RollingUNet(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(16, 32, 64, 128, 256),
        drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        assert len(filters) == 5, "RollingUNet expects exactly 5 filter widths"
        c1, c2, c3, c4, c5 = filters

        self.conv1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = FeatureIncentive(c3, c4)
        self.FIBlock2 = FeatureIncentive(c4, c5)
        self.FIBlock3 = FeatureIncentive(c5, c4)
        dpr = torch.linspace(0, drop_path_rate, 3).tolist()
        self.block1 = nn.ModuleList([Lo2Block(c4, drop=drop_rate, drop_path=dpr[0])])
        self.block2 = nn.ModuleList(
            [Lo2Block(c5, drop=drop_rate + 0.1, drop_path=dpr[1])]
        )
        self.block3 = nn.ModuleList([Lo2Block(c4, drop=drop_rate, drop_path=dpr[0])])
        self.norm1 = nn.LayerNorm(c4)
        self.norm2 = nn.LayerNorm(c5)
        self.norm3 = nn.LayerNorm(c4)

        self.FIBlock4 = nn.Conv2d(c4, c3, 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(c3)
        self.decoder3 = DecoderDoubleConv(c3, c2)
        self.decoder2 = DecoderDoubleConv(c2, c1)
        self.decoder1 = DecoderDoubleConv(c1, 8)
        self.restoration = nn.Conv2d(8, out_ch, kernel_size=1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(self.pool1(t1))
        t3 = self.conv3(self.pool2(t2))
        x = self.pool3(t3)

        x, h, w = self.FIBlock1(x)
        for block in self.block1:
            x = block(x, h, w)
        t4 = _tokens_to_map(self.norm1(x), h, w)

        x = self.pool4(t4)
        x, h, w = self.FIBlock2(x)
        for block in self.block2:
            x = block(x, h, w)
        x = _tokens_to_map(self.norm2(x), h, w)

        x, h, w = self.FIBlock3(x)
        x = self._up(_tokens_to_map(x, h, w), t4.shape[2:]) + t4
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        for block in self.block3:
            x = block(x, h, w)
        x = _tokens_to_map(self.norm3(x), h, w)

        x = self._up(F.relu(self.dbn4(self.FIBlock4(x))), t3.shape[2:]) + t3
        x = self._up(self.decoder3(x), t2.shape[2:]) + t2
        x = self._up(self.decoder2(x), t1.shape[2:]) + t1
        return self.restoration(self.decoder1(x))

import math

import torch
from torch import nn
from torch.nn import functional as F


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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class ShiftMLP(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        shift_size=5,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = int(shift_size)
        self.pad = self.shift_size // 2
        self.apply(_init_weights)

    def _shift(self, x, h, w, dim):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        chunks = torch.chunk(x, self.shift_size, dim=1)
        shifts = range(-self.pad, self.pad + 1)
        x = torch.cat(
            [torch.roll(chunk, shift, dim) for chunk, shift in zip(chunks, shifts)],
            dim=1,
        )
        x = x.narrow(2, self.pad, h).narrow(3, self.pad, w)
        return x.reshape(b, c, h * w).transpose(1, 2)

    def forward(self, x, h, w):
        x = self.fc1(self._shift(x, h, w, dim=2))
        x = self.drop(self.act(self.dwconv(x, h, w)))
        x = self.fc2(self._shift(x, h, w, dim=3))
        return self.drop(x)


class ShiftedBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=1.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ShiftMLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x, h, w):
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class OverlapPatchEmbed(nn.Module):

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
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


def _tokens_to_map(x, h, w):
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class UNeXt(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(16, 32, 128, 160, 256),
        drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        assert len(filters) == 5, "UNeXt expects exactly 5 channel widths"
        c1, c2, c3, c4, c5 = filters

        self.encoder1 = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.encoder2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.encoder3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(c1)
        self.ebn2 = nn.BatchNorm2d(c2)
        self.ebn3 = nn.BatchNorm2d(c3)

        dpr = torch.linspace(0, drop_path_rate, 3).tolist()
        self.patch_embed3 = OverlapPatchEmbed(c3, c4)
        self.block1 = nn.ModuleList(
            [ShiftedBlock(c4, drop=drop_rate, drop_path=dpr[0])]
        )
        self.norm3 = nn.LayerNorm(c4)
        self.patch_embed4 = OverlapPatchEmbed(c4, c5)
        self.block2 = nn.ModuleList(
            [ShiftedBlock(c5, drop=drop_rate, drop_path=dpr[1])]
        )
        self.norm4 = nn.LayerNorm(c5)

        self.decoder1 = nn.Conv2d(c5, c4, 3, padding=1)
        self.decoder2 = nn.Conv2d(c4, c3, 3, padding=1)
        self.decoder3 = nn.Conv2d(c3, c2, 3, padding=1)
        self.decoder4 = nn.Conv2d(c2, c1, 3, padding=1)
        self.decoder5 = nn.Conv2d(c1, c1, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(c4)
        self.dbn2 = nn.BatchNorm2d(c3)
        self.dbn3 = nn.BatchNorm2d(c2)
        self.dbn4 = nn.BatchNorm2d(c1)

        self.dblock1 = nn.ModuleList(
            [ShiftedBlock(c4, drop=drop_rate, drop_path=dpr[0])]
        )
        self.dnorm3 = nn.LayerNorm(c4)
        self.dblock2 = nn.ModuleList(
            [ShiftedBlock(c3, drop=drop_rate, drop_path=dpr[1])]
        )
        self.dnorm4 = nn.LayerNorm(c3)
        self.restoration = nn.Conv2d(c1, out_ch, kernel_size=1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[2:]
        t1 = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t2 = F.relu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t3 = F.relu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))

        x, h, w = self.patch_embed3(t3)
        for block in self.block1:
            x = block(x, h, w)
        t4 = _tokens_to_map(self.norm3(x), h, w)

        x, h, w = self.patch_embed4(t4)
        for block in self.block2:
            x = block(x, h, w)
        x = _tokens_to_map(self.norm4(x), h, w)

        x = F.relu(self._up(self.dbn1(self.decoder1(x)), t4.shape[2:])) + t4
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        for block in self.dblock1:
            x = block(x, h, w)
        x = _tokens_to_map(self.dnorm3(x), h, w)

        x = F.relu(self._up(self.dbn2(self.decoder2(x)), t3.shape[2:])) + t3
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        for block in self.dblock2:
            x = block(x, h, w)
        x = _tokens_to_map(self.dnorm4(x), h, w)

        x = F.relu(self._up(self.dbn3(self.decoder3(x)), t2.shape[2:])) + t2
        x = F.relu(self._up(self.dbn4(self.decoder4(x)), t1.shape[2:])) + t1
        x = F.relu(self._up(self.decoder5(x), input_size))
        return self.restoration(x)

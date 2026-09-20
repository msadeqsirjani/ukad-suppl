import math

import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1)
        self.register_buffer("grid", grid.contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline,
                )

    def b_splines(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected (*, {self.in_features}) input, got {tuple(x.shape)}"
            )

        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, k + 1 :] - x)
                / (self.grid[:, k + 1 :] - self.grid[:, 1:-k])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected (*, {self.in_features}) grid input, got {tuple(x.shape)}"
            )
        if y.shape != (x.shape[0], self.in_features, self.out_features):
            raise ValueError(
                "coefficient targets must have shape "
                f"{(x.shape[0], self.in_features, self.out_features)}, "
                f"got {tuple(y.shape)}"
            )

        bases = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(bases, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if not self.enable_standalone_scale_spline:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected (*, {self.in_features}) input, got {tuple(x.shape)}"
            )
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(
            self.b_splines(x).reshape(x.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return base + spline

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        activation = l1_fake.sum()
        probabilities = l1_fake / activation
        entropy = -torch.sum(probabilities * probabilities.log())
        return regularize_activation * activation + regularize_entropy * entropy


class DWBNReLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class KANLayer(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        no_kan=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear = nn.Linear if no_kan else KANLinear

        self.fc1 = linear(in_features, hidden_features)
        self.fc2 = linear(hidden_features, out_features)
        self.fc3 = linear(out_features, out_features)
        self.dwconv1 = DWBNReLU(hidden_features)
        self.dwconv2 = DWBNReLU(out_features)
        self.dwconv3 = DWBNReLU(out_features)
        self.drop = nn.Dropout(drop)
        self._init_standard_layers()

    def _init_standard_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1]
                fan_out *= module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _apply_linear(self, layer, x):
        b, n, c = x.shape
        return layer(x.reshape(b * n, c)).reshape(b, n, -1).contiguous()

    def forward(self, x, h, w):
        x = self.dwconv1(self._apply_linear(self.fc1, x), h, w)
        x = self.dwconv2(self._apply_linear(self.fc2, x), h, w)
        x = self.dwconv3(self._apply_linear(self.fc3, x), h, w)
        return x


class KANBlock(nn.Module):

    def __init__(self, dim, drop=0.0, no_kan=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = KANLayer(dim, dim, dim, drop=drop, no_kan=no_kan)

    def forward(self, x, h, w):
        return x + self.layer(self.norm(x), h, w)


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


class UKAN(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch=None,
        embed_dims=(128, 160, 256),
        no_kan=False,
        drop_rate=0.0,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        if len(embed_dims) != 3:
            raise ValueError("UKAN expects exactly three embedding widths")
        c3, c4, c5 = embed_dims
        if c3 % 8:
            raise ValueError("the first UKAN embedding width must be divisible by 8")
        c1, c2 = c3 // 8, c3 // 4

        self.encoder1 = ConvLayer(in_ch, c1)
        self.encoder2 = ConvLayer(c1, c2)
        self.encoder3 = ConvLayer(c2, c3)

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

import torch
from torch import nn
from torch.nn import functional as F


UKAGNET_UPSTREAM_REPOSITORY = "https://github.com/IvanDrokin/torch-conv-kan"
UKAGNET_UPSTREAM_COMMIT = "d65e9ccfbe47ddd655a73f85a365f166db54eb3a"
UKAGNET_UPSTREAM_VARIANT = "UKAGNet depth=5 layers=2 conv non-bottleneck width_scale=1"


class KAGNConv2d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        degree=3,
        dropout=0.0,
        affine=True,
    ):
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_ch % groups:
            raise ValueError("in_ch must be divisible by groups")
        if out_ch % groups:
            raise ValueError("out_ch must be divisible by groups")
        if degree < 0:
            raise ValueError("degree must be non-negative")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.degree = degree
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.base_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_ch // groups,
                    out_ch // groups,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups=1,
                    bias=False,
                )
                for _ in range(groups)
            ]
        )
        self.layer_norm = nn.ModuleList(
            [nn.BatchNorm2d(out_ch // groups, affine=affine) for _ in range(groups)]
        )
        self.poly_weights = nn.Parameter(
            torch.empty(
                groups,
                out_ch // groups,
                (in_ch // groups) * (degree + 1),
                kernel_size,
                kernel_size,
            )
        )
        self.beta_weights = nn.Parameter(torch.empty(degree + 1))

        for conv in self.base_conv:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity="linear")
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (kernel_size**2 * in_ch * (degree + 1.0)),
        )

    def _beta(self, n, m):
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    def _gram_poly(self, x):
        p0 = torch.ones_like(x)
        if self.degree == 0:
            return p0
        p1 = x
        grams = [p0, p1]
        for i in range(2, self.degree + 1):
            p2 = x * p1 - self._beta(i - 1, i) * p0
            grams.append(p2)
            p0, p1 = p1, p2
        return torch.cat(grams, dim=1)

    def _forward_group(self, x, group_index):
        basis = self.base_conv[group_index](self.base_activation(x))
        x = torch.tanh(x).contiguous()
        if self.dropout is not None:
            x = self.dropout(x)
        grams = self.base_activation(self._gram_poly(x))
        y = F.conv2d(
            grams,
            self.poly_weights[group_index],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return self.base_activation(self.layer_norm[group_index](y + basis))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_ch:
            raise ValueError(
                f"expected NCHW input with {self.in_ch} channels, got {tuple(x.shape)}"
            )
        chunks = torch.split(x, self.in_ch // self.groups, dim=1)
        return torch.cat(
            [self._forward_group(chunk, i) for i, chunk in enumerate(chunks)], dim=1
        )


class UKAGNet(nn.Module):
    upstream_repository = UKAGNET_UPSTREAM_REPOSITORY
    upstream_commit = UKAGNET_UPSTREAM_COMMIT
    upstream_variant = UKAGNET_UPSTREAM_VARIANT

    def __init__(
        self,
        input_channels=3,
        num_classes=1,
        unet_depth=5,
        unet_layers=2,
        groups=1,
        width_scale=1,
        use_bottleneck=False,
        mixer_type="conv",
        degree=3,
        affine=True,
        dropout=0.0,
    ):
        super().__init__()
        if unet_depth < 2:
            raise ValueError("unet_depth must be at least 2")
        if unet_layers < 1:
            raise ValueError("unet_layers must be at least 1")
        if not isinstance(width_scale, int) or width_scale < 1:
            raise ValueError("width_scale must be a positive integer")
        if use_bottleneck:
            raise ValueError("the reported UKAGNet baseline is non-bottleneck")
        if mixer_type != "conv":
            raise ValueError("the reported UKAGNet baseline uses mixer_type='conv'")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.unet_depth = unet_depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        widths = [16 * width_scale * 2**i for i in range(unet_depth)]
        for depth_index, width in enumerate(widths):
            in_width = input_channels if depth_index == 0 else widths[depth_index - 1]
            stride = 1 if depth_index == 0 else 2
            layers = [
                KAGNConv2d(
                    in_width,
                    width,
                    stride=stride,
                    groups=groups,
                    degree=degree,
                    dropout=0.0 if depth_index == 0 else dropout,
                    affine=affine,
                )
            ]
            layers.extend(
                KAGNConv2d(
                    width,
                    width,
                    groups=groups,
                    degree=degree,
                    dropout=dropout,
                    affine=affine,
                )
                for _ in range(unet_layers - 1)
            )
            self.encoder.append(nn.Sequential(*layers))

        for depth_index in reversed(range(unet_depth - 1)):
            width = widths[depth_index]
            layers = [
                KAGNConv2d(
                    3 * width,
                    width,
                    groups=groups,
                    degree=degree,
                    dropout=dropout,
                    affine=affine,
                )
            ]
            layers.extend(
                KAGNConv2d(
                    width,
                    width,
                    groups=groups,
                    degree=degree,
                    dropout=dropout,
                    affine=affine,
                )
                for _ in range(unet_layers - 1)
            )
            self.decoder.append(nn.Sequential(*layers))

        self.output = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"expected NCHW input with {self.input_channels} channels, got {tuple(x.shape)}"
            )
        output_size = x.shape[-2:]
        skips = []
        for block_index, block in enumerate(self.encoder):
            x = block(x)
            if block_index < self.unet_depth - 1:
                skips.append(x)

        for block in self.decoder:
            skip = skips.pop()
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=True
            )
            x = block(torch.cat([x, skip], dim=1))

        x = self.output(x)
        if x.shape[-2:] != output_size:
            raise RuntimeError(
                f"UKAGNet output shape {tuple(x.shape[-2:])} does not match input {output_size}"
            )
        return x

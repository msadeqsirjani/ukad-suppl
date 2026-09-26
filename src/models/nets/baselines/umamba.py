from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _as_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected a 2-D size, got {value!r}")
    return int(value[0]), int(value[1])


def _conv_output_size(
    size: tuple[int, int], stride: tuple[int, int]
) -> tuple[int, int]:
    return tuple((axis - 1) // step + 1 for axis, step in zip(size, stride))


class MambaLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "UMamba requires `mamba_ssm` (and `causal-conv1d`) built for "
                "this environment's CUDA/PyTorch version."
            ) from exc

        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        if channels != self.dim:
            raise ValueError(
                f"patch-token Mamba expected {self.dim} channels, got {channels}"
            )
        sequence = x.reshape(batch, channels, -1).transpose(1, 2)
        sequence = self.mamba(self.norm(sequence))
        return sequence.transpose(1, 2).reshape(batch, channels, *spatial)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        spatial_dim = math.prod(spatial)
        if spatial_dim != self.dim:
            raise ValueError(
                "channel-token Mamba was planned for a feature map with "
                f"{self.dim} pixels, got {tuple(spatial)} ({spatial_dim} pixels). "
                "Use the input_size matching this model's config."
            )
        sequence = x.flatten(2)
        sequence = self.mamba(self.norm(sequence))
        return sequence.reshape(batch, channels, *spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        device_type = x.device.type if x.device.type in ("cpu", "cuda") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if self.channel_token:
                return self.forward_channel_token(x)
            return self.forward_patch_token(x)


class BasicResBlock(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int | Sequence[int] = 1,
        projection: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        stride = _as_pair(stride)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=conv_bias
        )
        self.norm1 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=conv_bias
        )
        self.norm2 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        if projection or in_ch != out_ch or stride != (1, 1):
            self.shortcut = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=conv_bias
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


def _residual_stage(
    in_ch: int,
    out_ch: int,
    stride: int | Sequence[int],
    blocks: int,
    conv_bias: bool,
) -> nn.Sequential:
    if blocks < 1:
        raise ValueError("each U-Mamba stage needs at least one residual block")
    return nn.Sequential(
        BasicResBlock(in_ch, out_ch, stride, projection=True, conv_bias=conv_bias),
        *[
            BasicResBlock(out_ch, out_ch, conv_bias=conv_bias)
            for _ in range(blocks - 1)
        ],
    )


class ResidualMambaEncoder(nn.Module):
    def __init__(
        self,
        input_size: int | Sequence[int],
        input_channels: int,
        features_per_stage: Sequence[int],
        strides: Sequence[int | Sequence[int]],
        blocks_per_stage: Sequence[int],
        d_state: int,
        d_conv: int,
        expand: int,
        conv_bias: bool,
    ):
        super().__init__()
        n_stages = len(features_per_stage)
        if not (
            len(strides) == len(blocks_per_stage) == n_stages
        ):
            raise ValueError("filters, strides, and blocks_per_stage must have equal length")

        self.output_channels = tuple(int(v) for v in features_per_stage)
        self.strides = tuple(_as_pair(v) for v in strides)
        size = _as_pair(input_size)
        feature_map_sizes: list[tuple[int, int]] = []
        for stride in self.strides:
            size = _conv_output_size(size, stride)
            feature_map_sizes.append(size)

        stem_channels = self.output_channels[0]
        self.stem = _residual_stage(
            input_channels,
            stem_channels,
            stride=1,
            blocks=int(blocks_per_stage[0]),
            conv_bias=conv_bias,
        )

        stages = []
        mamba_layers = []
        in_ch = stem_channels
        for stage_index, (out_ch, stride, blocks, map_size) in enumerate(
            zip(
                self.output_channels,
                self.strides,
                blocks_per_stage,
                feature_map_sizes,
            )
        ):
            stages.append(
                _residual_stage(
                    in_ch,
                    out_ch,
                    stride=stride,
                    blocks=int(blocks),
                    conv_bias=conv_bias,
                )
            )

            has_mamba = bool(stage_index % 2) ^ bool(n_stages % 2)
            if has_mamba:
                pixels = math.prod(map_size)
                channel_token = pixels <= out_ch
                mamba_layers.append(
                    MambaLayer(
                        dim=pixels if channel_token else out_ch,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        channel_token=channel_token,
                    )
                )
            else:
                mamba_layers.append(nn.Identity())
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.feature_map_sizes = tuple(feature_map_sizes)
        self.channel_token_stages = tuple(
            index
            for index, layer in enumerate(self.mamba_layers)
            if isinstance(layer, MambaLayer) and layer.channel_token
        )
        self.mamba_stages = tuple(
            index
            for index, layer in enumerate(self.mamba_layers)
            if isinstance(layer, MambaLayer)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = mamba(stage(x))
            skips.append(x)
        return skips


class DecoderStage(nn.Module):
    def __init__(
        self,
        below_ch: int,
        skip_ch: int,
        blocks: int,
        has_skip: bool,
        conv_bias: bool,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.project = nn.Conv2d(below_ch, skip_ch, kernel_size=1, bias=conv_bias)
        self.blocks = _residual_stage(
            2 * skip_ch if has_skip else skip_ch,
            skip_ch,
            stride=1,
            blocks=blocks,
            conv_bias=conv_bias,
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None
    ) -> torch.Tensor:
        target = skip.shape[-2:] if skip is not None else tuple(2 * v for v in x.shape[-2:])
        x = self.project(F.interpolate(x, size=target, mode="nearest"))
        if self.has_skip:
            if skip is None:
                raise ValueError("decoder stage requires its encoder skip")
            x = torch.cat((x, skip), dim=1)
        return self.blocks(x)


class UMamba(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int | None = None,
        input_size: int | Sequence[int] = (256, 256),
        filters: Sequence[int] = (32, 64, 128, 256, 512, 512),
        strides: Sequence[int | Sequence[int]] | None = None,
        blocks_per_stage: int | Sequence[int] = 2,
        decoder_blocks_per_stage: int | Sequence[int] = 2,
        variant: str = "enc",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual: bool = False,
        conv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected UMamba arguments: {unknown}")
        if variant != "enc":
            raise ValueError(
                "this implementation is U-Mamba-Enc; use variant='enc'"
            )
        if out_ch is None:
            out_ch = in_ch
        if residual and in_ch != out_ch:
            raise ValueError("residual output requires in_ch == out_ch")

        filters = tuple(int(v) for v in filters)
        if len(filters) < 3:
            raise ValueError("UMamba requires at least three resolution stages")
        n_stages = len(filters)
        if strides is None:
            strides = (1,) + (2,) * (n_stages - 1)
        if isinstance(blocks_per_stage, int):
            encoder_blocks = [blocks_per_stage] * n_stages
        else:
            encoder_blocks = [int(v) for v in blocks_per_stage]
        if isinstance(decoder_blocks_per_stage, int):
            decoder_blocks = [decoder_blocks_per_stage] * (n_stages - 1)
        else:
            decoder_blocks = [int(v) for v in decoder_blocks_per_stage]
        if len(encoder_blocks) != n_stages:
            raise ValueError("blocks_per_stage must match the number of filters")
        if len(decoder_blocks) != n_stages - 1:
            raise ValueError("decoder_blocks_per_stage must have n_stages - 1 entries")

        for stage in range(math.ceil(n_stages / 2), n_stages):
            encoder_blocks[stage] = 1
        decoder_start = math.ceil((n_stages - 1) / 2 + 0.5)
        for stage in range(decoder_start, n_stages - 1):
            decoder_blocks[stage] = 1

        self.input_size = _as_pair(input_size)
        self.filters = filters
        self.variant = variant
        self.residual = bool(residual)
        self.encoder = ResidualMambaEncoder(
            input_size=self.input_size,
            input_channels=in_ch,
            features_per_stage=filters,
            strides=strides,
            blocks_per_stage=encoder_blocks,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conv_bias=conv_bias,
        )

        decoder = []
        for decoder_index, stage_index in enumerate(range(n_stages - 2, -1, -1)):
            has_skip = decoder_index < n_stages - 2
            decoder.append(
                DecoderStage(
                    below_ch=filters[stage_index + 1],
                    skip_ch=filters[stage_index],
                    blocks=decoder_blocks[decoder_index],
                    has_skip=has_skip,
                    conv_bias=conv_bias,
                )
            )
        self.decoder = nn.ModuleList(decoder)
        self.seg_heads = nn.ModuleList(
            nn.Conv2d(filters[index], out_ch, kernel_size=1)
            for index in range(n_stages - 2, -1, -1)
        )

        self.apply(self._init_he)

    @staticmethod
    def _init_he(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input = x
        skips = self.encoder(x)
        x = skips[-1]
        for decoder_index, stage in enumerate(self.decoder):
            skip = skips[-(decoder_index + 2)] if stage.has_skip else None
            x = stage(x, skip)
        x = self.seg_heads[-1](x)
        if x.shape[-2:] != model_input.shape[-2:]:
            x = F.interpolate(x, size=model_input.shape[-2:], mode="nearest")
        if self.residual:
            x = x + model_input
        return x

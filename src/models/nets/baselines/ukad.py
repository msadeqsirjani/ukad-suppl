import math

import torch
from torch import nn
from torch.nn import functional as F

from ._blocks import DoubleConv


class GroupRational(nn.Module):
    def __init__(self, ch, groups, degree_p=3, degree_q=2):
        super(GroupRational, self).__init__()
        assert ch % groups == 0, "channels must be divisible by groups"
        assert degree_p >= 1 and degree_q >= 1, "rational degrees must be >= 1"
        self.groups = groups
        self.degree_p = degree_p
        self.degree_q = degree_q
        a = torch.zeros(groups, degree_p + 1)
        a[:, 1] = 1.0
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(torch.empty(groups, degree_q))
        nn.init.normal_(self.b, mean=0.0, std=1e-2)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        xg = x.view(b, g, c // g, h, w)

        powers = [xg]
        for _ in range(2, max(self.degree_p, self.degree_q) + 1):
            powers.append(powers[-1] * xg)

        p = self.a[:, 0].view(1, g, 1, 1, 1).expand_as(xg)
        for k in range(1, self.degree_p + 1):
            p = p + self.a[:, k].view(1, g, 1, 1, 1) * powers[k - 1]

        q = torch.zeros_like(xg)
        for k in range(1, self.degree_q + 1):
            q = q + self.b[:, k - 1].view(1, g, 1, 1, 1) * powers[k - 1]

        return (p / (1.0 + q.abs())).view(b, c, h, w)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super(DropPath, self).__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop_path must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


def make_activation(kind, ch, groups, degree_p, degree_q):
    if kind == "rational":
        return GroupRational(ch, groups, degree_p, degree_q)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unknown activation kind: {kind!r}")


class GateHead(nn.Module):
    def __init__(self, ch, groups, gate_init_bias=4.0, hard_gate=False):
        super(GateHead, self).__init__()
        self.hard_gate = hard_gate
        self.conv = nn.Conv2d(ch, groups, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, gate_init_bias)

    def forward(self, x):
        soft = torch.sigmoid(self.conv(x))
        if not self.hard_gate:
            return soft
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()


class Displace(nn.Module):
    def __init__(self, ch, groups, max_radius=3.0, gate_init_bias=4.0, hard_gate=False):
        super(Displace, self).__init__()
        self.groups = groups
        self.max_radius = max_radius
        self.hard_gate = hard_gate
        self.offset = nn.Conv2d(ch, 3 * groups, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        with torch.no_grad():
            self.offset.bias.view(groups, 3)[:, 2] = gate_init_bias
        self._last_raw = None
        self._last_gate = None

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        raw = self.offset(x).view(b, g, 3, h, w)
        soft_gate = torch.sigmoid(raw[:, :, 2])
        if self.hard_gate:
            hard = (soft_gate > 0.5).float()
            gate = hard + soft_gate - soft_gate.detach()
        else:
            gate = soft_gate
        py = self.max_radius * gate * torch.tanh(raw[:, :, 0])
        px = self.max_radius * gate * torch.tanh(raw[:, :, 1])

        grid_x = gx.view(1, 1, h, w) + px * (2.0 / max(w - 1, 1))
        grid_y = gy.view(1, 1, h, w) + py * (2.0 / max(h - 1, 1))
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(b * g, h, w, 2)

        out = F.grid_sample(
            x.view(b * g, c // g, h, w),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        self._last_raw = raw
        self._last_gate = gate
        return out.view(b, c, h, w), gate

    def diagnostics(self):
        if self._last_raw is None:
            return {}
        offset = torch.tanh(self._last_raw[:, :, :2])
        return {
            "offset_radius_fraction": torch.linalg.vector_norm(offset, dim=2).clamp_max(1.0).mean(),
            "offset_saturation": (self._last_raw[:, :, :2].abs() > 2.0).float().mean(),
            "gate_mean": self._last_gate.mean(),
            "gate_binary_fraction": ((self._last_gate < 0.05) | (self._last_gate > 0.95)).float().mean(),
        }


class KAD(nn.Module):
    def __init__(self, ch, groups=8, max_radius=3.0, degree_p=3, degree_q=2, activation="rational", hard_gate=False):
        super(KAD, self).__init__()
        self.norm = nn.BatchNorm2d(ch)
        self.displace = Displace(ch, groups, max_radius, hard_gate=hard_gate)
        self.inner = make_activation(activation, ch, groups, degree_p, degree_q)
        self.mix = nn.Conv2d(ch, ch, kernel_size=1)
        self.outer = make_activation(activation, ch, groups, degree_p, degree_q)

    def forward(self, x):
        h = self.norm(x)
        h, gate = self.displace(h)
        h = self.inner(h)
        h = self.mix(h)
        return self.outer(h), gate


class DSC(nn.Module):
    def __init__(self, ch):
        super(DSC, self).__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class GlobalKANContext(nn.Module):
    def __init__(self, ch, groups=8, dilations=(1, 3, 5), kernel_size=5, degree_p=3, degree_q=2, activation="rational"):
        super(GlobalKANContext, self).__init__()
        self.dw = nn.ModuleList(
            [
                nn.Conv2d(
                    ch,
                    ch,
                    kernel_size=kernel_size,
                    padding=d * (kernel_size - 1) // 2,
                    dilation=d,
                    groups=ch,
                )
                for d in dilations
            ]
        )
        self.act = make_activation(activation, ch, groups, degree_p, degree_q)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        h = x
        for dw in self.dw:
            h = dw(h)
        h = self.act(h)
        return self.pw(h)


class KADBlock(nn.Module):
    def __init__(
        self,
        ch,
        groups=8,
        max_radius=3.0,
        degree_p=3,
        degree_q=2,
        context_dilations=(1, 3, 5),
        activation="rational",
        hard_gate=False,
        use_displacement=True,
        use_context=True,
        shared_gate=True,
        drop_path=0.0,
    ):
        super(KADBlock, self).__init__()
        self.drop_path = DropPath(drop_path)
        self.groups = groups
        self.use_displacement = use_displacement
        self.use_context = use_context
        self.shared_gate = shared_gate and use_displacement

        n_branches = 1
        if use_displacement:
            self.kad = KAD(ch, groups, max_radius, degree_p, degree_q, activation, hard_gate)
            n_branches += 1
        if use_context:
            self.context = GlobalKANContext(ch, groups, context_dilations, degree_p=degree_p, degree_q=degree_q, activation=activation)
            n_branches += 1
            if not self.shared_gate:
                self.context_gate = GateHead(ch, groups, hard_gate=hard_gate)
        self.dsc = DSC(ch)
        self.fuse = nn.Conv2d(n_branches * ch, ch, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(ch, 1, 1))

    def forward(self, x):
        _, c, _, _ = x.shape
        branches = []
        shared_gate_full = None
        if self.use_displacement:
            kad_out, gate = self.kad(x)
            shared_gate_full = gate.repeat_interleave(c // self.groups, dim=1)
            branches.append(shared_gate_full * kad_out)
        if self.use_context:
            ctx_out = self.context(x)
            if self.shared_gate:
                ctx_gate_full = shared_gate_full
            else:
                ctx_gate = self.context_gate(x)
                ctx_gate_full = ctx_gate.repeat_interleave(c // self.groups, dim=1)
            branches.append(ctx_gate_full * ctx_out)
        branches.append(self.dsc(x))
        y = torch.cat(branches, dim=1)
        return x + self.drop_path(self.scale * self.fuse(y))


class KANConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, degree_p=3, degree_q=2, activation="rational"):
        super(KANConvBlock, self).__init__()
        g = math.gcd(out_ch, groups)
        self.dw1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pw1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1 = make_activation(activation, out_ch, g, degree_p, degree_q)
        self.dw2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
        self.pw2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = make_activation(activation, out_ch, g, degree_p, degree_q)

    def forward(self, x):
        h = self.act1(self.norm1(self.pw1(self.dw1(x))))
        return self.act2(self.norm2(self.pw2(self.dw2(h))))


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shortcut(x) + self.body(x))


class MLPConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=2):
        super(MLPConvBlock, self).__init__()
        hidden = out_ch * expansion
        self.dw1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, out_ch, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.dw2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
        self.fc3 = nn.Conv2d(out_ch, hidden, kernel_size=1)
        self.fc4 = nn.Conv2d(hidden, out_ch, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm1(self.fc2(self.act(self.fc1(self.dw1(x)))))
        return self.norm2(self.fc4(self.act(self.fc3(self.dw2(h)))))


class HybridKANDecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        groups=8,
        degree_p=3,
        degree_q=2,
        activation="rational",
        kan_init=-2.0,
    ):
        super(HybridKANDecoderBlock, self).__init__()
        self.local = ResidualConvBlock(in_ch, out_ch)
        self.kan = KANConvBlock(in_ch, out_ch, groups, degree_p, degree_q, activation)
        self.kan_logit = nn.Parameter(torch.full((out_ch, 1, 1), float(kan_init)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.local(x) + torch.sigmoid(self.kan_logit) * self.kan(x))


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeparableResidualBlock, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                groups=out_ch,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shortcut(x) + self.body(x))


class AdaptiveScaleContext(nn.Module):
    def __init__(self, ch, dilations=(1, 2, 4)):
        super(AdaptiveScaleContext, self).__init__()
        if not dilations or any(int(d) <= 0 for d in dilations):
            raise ValueError("context dilations must be positive")
        self.norm = nn.BatchNorm2d(ch)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    ch,
                    ch,
                    kernel_size=3,
                    padding=int(d),
                    dilation=int(d),
                    groups=ch,
                    bias=False,
                )
                for d in dilations
            ]
        )
        hidden = max(ch // 8, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, len(dilations), kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.scale = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x):
        h = self.norm(x)
        weights = torch.softmax(self.router(h), dim=1)
        context = torch.zeros_like(h)
        for index, branch in enumerate(self.branches):
            context = context + weights[:, index : index + 1] * branch(h)
        return x + self.scale * self.fuse(context)


class PooledGlobalContextMixer(nn.Module):

    def __init__(self, ch, attention_dim=64, heads=4, pool_size=8, norm_groups=8):
        super(PooledGlobalContextMixer, self).__init__()
        if attention_dim <= 0:
            raise ValueError("global-context attention_dim must be positive")
        if heads <= 0 or attention_dim % heads:
            raise ValueError("global-context attention_dim must be divisible by heads")
        if pool_size <= 0:
            raise ValueError("global-context pool_size must be positive")

        norm_groups = min(int(norm_groups), int(ch))
        while ch % norm_groups:
            norm_groups -= 1

        self.attention_dim = int(attention_dim)
        self.heads = int(heads)
        self.head_dim = self.attention_dim // self.heads
        self.pool_size = int(pool_size)
        self.norm = nn.GroupNorm(norm_groups, ch)
        self.query = nn.Conv2d(ch, self.attention_dim, kernel_size=1, bias=False)
        self.key_value = nn.Conv2d(
            ch,
            2 * self.attention_dim,
            kernel_size=1,
            bias=False,
        )
        self.project = nn.Conv2d(self.attention_dim, ch, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.zeros(ch, 1, 1))

    def _heads(self, tensor):
        batch, channels, height, width = tensor.shape
        return tensor.flatten(2).transpose(1, 2).reshape(batch, height * width, self.heads, channels // self.heads).permute(0, 2, 1, 3)

    def forward(self, x):
        batch, _, height, width = x.shape
        normalized = self.norm(x)
        query = self._heads(self.query(normalized))

        pooled_size = (min(self.pool_size, height), min(self.pool_size, width))
        pooled = F.adaptive_avg_pool2d(normalized, pooled_size)
        key, value = self.key_value(pooled).chunk(2, dim=1)
        key = self._heads(key)
        value = self._heads(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        context = torch.matmul(attention, value)
        context = (
            context.permute(0, 2, 1, 3)
            .reshape(batch, height * width, self.attention_dim)
            .transpose(1, 2)
            .reshape(batch, self.attention_dim, height, width)
        )
        return x + self.residual_scale * self.project(context)


class UncertaintyBoundaryRefiner(nn.Module):
    def __init__(self, channels, refinement_ch=16):
        super(UncertaintyBoundaryRefiner, self).__init__()
        if refinement_ch <= 0:
            raise ValueError("refinement channels must be positive")

        def project(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, refinement_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(refinement_ch),
                nn.ReLU(inplace=True),
            )

        self.projections = nn.ModuleList(project(ch) for ch in channels)
        fused_ch = 2 * refinement_ch
        self.fuse = SeparableResidualBlock(len(channels) * refinement_ch, fused_ch)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(fused_ch, refinement_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refinement_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(refinement_ch, 1, kernel_size=1),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(
                fused_ch,
                fused_ch,
                kernel_size=3,
                padding=1,
                groups=fused_ch,
                bias=False,
            ),
            nn.BatchNorm2d(fused_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_ch, 1, kernel_size=1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, coarse_logits, features):
        size = coarse_logits.shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != size:
                feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            projected.append(feature)

        fused = self.fuse(torch.cat(projected, dim=1))
        boundary_logits = self.boundary_head(fused)
        probability = torch.sigmoid(coarse_logits.detach())
        uncertainty = 4.0 * probability * (1.0 - probability)
        correction = self.correction(fused)
        refined = coarse_logits + uncertainty * torch.sigmoid(boundary_logits) * correction
        return refined, boundary_logits


class Widen(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Widen, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class KANAttentionGate(nn.Module):
    def __init__(
        self,
        gating_ch,
        skip_ch,
        inter_ch,
        groups=8,
        degree_p=3,
        degree_q=2,
        activation="rational",
        mask_norm=True,
    ):
        super(KANAttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.act = make_activation(activation, inter_ch, groups, degree_p, degree_q)
        mask_layers = [nn.Conv2d(inter_ch, 1, kernel_size=1)]
        if mask_norm:
            mask_layers.append(nn.BatchNorm2d(1))
        mask_layers.append(nn.Sigmoid())
        self.psi = nn.Sequential(*mask_layers)

    def forward(self, g, x):
        return x * self.psi(self.act(self.W_g(g) + self.W_x(x)))


class UKAD(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        filters=(32, 64, 128, 256, 512),
        groups=8,
        max_radius=3.0,
        degree_p=3,
        degree_q=2,
        context_dilations=(1, 3, 5),
        residual=False,
        activation="rational",
        hard_gate=False,
        use_displacement=True,
        use_context=True,
        shared_gate=True,
        decoder_gate=True,
        decoder_block="conv",
        dropout=0.0,
        drop_path=0.0,
        deep_supervision=False,
        aux_weights=(0.2, 0.4),
        aux_target_mode="upsample",
        aux_anneal_start=None,
        attention_mask_norm=True,
        adaptive_context_levels=(),
        adaptive_context_dilations=(1, 2, 4),
        pooled_global_context=False,
        global_context_dim=64,
        global_context_heads=4,
        global_context_pool_size=8,
        boundary_refinement=False,
        refinement_channels=16,
        boundary_loss_weight=0.0,
        boundary_kernel_size=3,
    ):
        super(UKAD, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        if len(filters) != 5:
            raise ValueError("UKAD expects exactly 5 filter widths")
        if residual and in_ch != out_ch:
            raise ValueError("residual output requires in_ch == out_ch")
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if filters[-1] % groups:
            raise ValueError("bottleneck channels must be divisible by groups")
        if max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if activation not in ("rational", "relu"):
            raise ValueError(f"unknown activation kind: {activation!r}")
        if decoder_block not in ("conv", "kan", "mlp", "hybrid"):
            raise ValueError(f"unknown decoder_block kind: {decoder_block!r}")
        if aux_target_mode not in ("upsample", "native"):
            raise ValueError(f"unknown auxiliary target mode: {aux_target_mode!r}")
        if aux_anneal_start is not None and not 0.0 <= aux_anneal_start < 1.0:
            raise ValueError("aux_anneal_start must be in [0, 1)")
        adaptive_context_levels = tuple(int(level) for level in adaptive_context_levels)
        if len(set(adaptive_context_levels)) != len(adaptive_context_levels):
            raise ValueError("adaptive context levels must be unique")
        if any(level not in (1, 2, 3, 4) for level in adaptive_context_levels):
            raise ValueError("adaptive context levels must be selected from 1, 2, 3 and 4")
        if global_context_dim <= 0:
            raise ValueError("global_context_dim must be positive")
        if global_context_heads <= 0 or global_context_dim % global_context_heads:
            raise ValueError("global_context_dim must be divisible by global_context_heads")
        if global_context_pool_size <= 0:
            raise ValueError("global_context_pool_size must be positive")
        if boundary_refinement and out_ch != 1:
            raise ValueError("boundary refinement currently requires one output channel")
        if boundary_loss_weight < 0:
            raise ValueError("boundary loss weight must be non-negative")
        if boundary_kernel_size < 3 or boundary_kernel_size % 2 == 0:
            raise ValueError("boundary kernel size must be an odd integer >= 3")
        c1, c2, c3, c4, c5 = filters

        def inter_width(ch):
            return max((ch // 2 // groups) * groups, groups)

        def block(kind, level, in_ch, out_ch):
            if kind == "conv":
                return DoubleConv(in_ch, out_ch)
            if kind == "kan":
                return KANConvBlock(in_ch, out_ch, groups, degree_p, degree_q, activation)
            if kind == "mlp":
                return MLPConvBlock(in_ch, out_ch)
            if level in (4, 3):
                return HybridKANDecoderBlock(in_ch, out_ch, groups, degree_p, degree_q, activation)
            return ResidualConvBlock(in_ch, out_ch)

        self.in_channels = in_ch
        self.residual = bool(residual)
        self.decoder_gate = decoder_gate
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4_widen = Widen(c3, c4)
        self.enc4 = DoubleConv(c4, c4)
        self.bott_widen = Widen(c4, c5)
        self.bott = KADBlock(
            c5,
            groups,
            max_radius,
            degree_p,
            degree_q,
            context_dilations,
            activation=activation,
            hard_gate=hard_gate,
            use_displacement=use_displacement,
            use_context=use_context,
            shared_gate=shared_gate,
            drop_path=drop_path,
        )
        self.pooled_global_context = bool(pooled_global_context)
        self.global_context = (
            PooledGlobalContextMixer(
                c5,
                attention_dim=global_context_dim,
                heads=global_context_heads,
                pool_size=global_context_pool_size,
                norm_groups=groups,
            )
            if self.pooled_global_context
            else nn.Identity()
        )

        if decoder_gate:
            self.dec4_proj = nn.Conv2d(c5, c4, kernel_size=1)
            self.dec4_gate = KANAttentionGate(
                c4, c4, inter_width(c4), groups, degree_p, degree_q, activation=activation, mask_norm=attention_mask_norm
            )
            self.dec4 = block(decoder_block, 4, 2 * c4, c4)

            self.dec3_proj = nn.Conv2d(c4, c3, kernel_size=1)
            self.dec3_gate = KANAttentionGate(
                c3, c3, inter_width(c3), groups, degree_p, degree_q, activation=activation, mask_norm=attention_mask_norm
            )
            self.dec3 = block(decoder_block, 3, 2 * c3, c3)

            self.dec2_proj = nn.Conv2d(c3, c2, kernel_size=1)
            self.dec2_gate = KANAttentionGate(
                c2, c2, inter_width(c2), groups, degree_p, degree_q, activation=activation, mask_norm=attention_mask_norm
            )
            self.dec2 = block(decoder_block, 2, 2 * c2, c2)

            self.dec1_proj = nn.Conv2d(c2, c1, kernel_size=1)
            self.dec1_gate = KANAttentionGate(
                c1, c1, inter_width(c1), groups, degree_p, degree_q, activation=activation, mask_norm=attention_mask_norm
            )
            self.dec1 = block(decoder_block, 1, 2 * c1, c1)
        else:
            self.dec4_reduce = nn.Conv2d(c5, c4, kernel_size=1)
            self.dec4 = block(decoder_block, 4, c4, c4)

            self.dec3_reduce = nn.Conv2d(c4, c3, kernel_size=1)
            self.dec3 = block(decoder_block, 3, c3, c3)

            self.dec2_reduce = nn.Conv2d(c3, c2, kernel_size=1)
            self.dec2 = block(decoder_block, 2, c2, c2)

            self.dec1_reduce = nn.Conv2d(c2, c1, kernel_size=1)
            self.dec1 = block(decoder_block, 1, c1, c1)

        self.drop = nn.Dropout2d(float(dropout)) if dropout else nn.Identity()

        decoder_channels = {1: c1, 2: c2, 3: c3, 4: c4}
        self.scale_context = nn.ModuleDict(
            {str(level): AdaptiveScaleContext(decoder_channels[level], adaptive_context_dilations) for level in adaptive_context_levels}
        )

        self.deep_supervision = bool(deep_supervision)
        self.aux_weights = tuple(float(w) for w in aux_weights)
        self.aux_target_mode = aux_target_mode
        self.aux_anneal_start = aux_anneal_start
        if self.deep_supervision and len(self.aux_weights) != 2:
            raise ValueError("deep supervision expects two auxiliary weights")
        if self.deep_supervision:
            self.aux3 = nn.Conv2d(c3, out_ch, kernel_size=1)
            self.aux2 = nn.Conv2d(c2, out_ch, kernel_size=1)
        self._aux = []

        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)
        self.boundary_refinement = bool(boundary_refinement)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_kernel_size = int(boundary_kernel_size)
        if self.boundary_refinement:
            self.boundary_refiner = UncertaintyBoundaryRefiner((c1, c1, c2, c3), refinement_channels)
        self._boundary_logits = None

    def _up(self, x, skip):
        return F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

    def _decode(self, level, d):
        key = str(level)
        if key in self.scale_context:
            d = self.scale_context[key](d)
        if level > 1:
            d = self.drop(d)
        if self.deep_supervision and self.training and level in (3, 2):
            self._aux.append(getattr(self, f"aux{level}")(d))
        return d

    def deep_supervision_loss(self, target, criterion, progress=None):
        scale = 1.0
        if self.aux_anneal_start is not None and progress is not None:
            progress = min(max(float(progress), 0.0), 1.0)
            if progress > self.aux_anneal_start:
                scale = (1.0 - progress) / (1.0 - self.aux_anneal_start)
        total = target.new_zeros(())
        for logits, weight in zip(self._aux, self.aux_weights):
            if self.aux_target_mode == "native":
                prediction = logits
                aux_target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
            else:
                prediction = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
                aux_target = target
            value = criterion(prediction, aux_target)
            if isinstance(value, dict):
                value = value["loss"]
            total = total + scale * weight * value
        self._aux = []
        return total

    def boundary_supervision_loss(self, target):
        if not self.boundary_refinement or self._boundary_logits is None:
            return target.new_zeros(())

        logits = self._boundary_logits
        self._boundary_logits = None
        target = target.float()
        radius = self.boundary_kernel_size // 2
        dilated = F.max_pool2d(
            target,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=radius,
        )
        eroded = -F.max_pool2d(
            -target,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=radius,
        )
        boundary = (dilated - eroded).clamp_(0.0, 1.0)
        positive = boundary.sum()
        negative = boundary.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0).detach()
        bce = F.binary_cross_entropy_with_logits(logits, boundary, pos_weight=pos_weight)
        probability = torch.sigmoid(logits)
        intersection = (probability * boundary).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + boundary.sum() + 1.0)
        return self.boundary_loss_weight * 0.5 * (bce + dice)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected NCHW input with {self.in_channels} channels, got {tuple(x.shape)}")
        output_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.enc4_widen(self.pool(e3)))
        b = self.bott(self.bott_widen(self.pool(e4)))
        b = self.global_context(b)

        self._aux = []
        self._boundary_logits = None
        if self.decoder_gate:
            u4 = self.dec4_proj(self._up(b, e4))
            d4 = self._decode(4, self.dec4(torch.cat([u4, self.dec4_gate(u4, e4)], dim=1)))
            u3 = self.dec3_proj(self._up(d4, e3))
            d3 = self._decode(3, self.dec3(torch.cat([u3, self.dec3_gate(u3, e3)], dim=1)))
            u2 = self.dec2_proj(self._up(d3, e2))
            d2 = self._decode(2, self.dec2(torch.cat([u2, self.dec2_gate(u2, e2)], dim=1)))
            u1 = self.dec1_proj(self._up(d2, e1))
            d1 = self._decode(1, self.dec1(torch.cat([u1, self.dec1_gate(u1, e1)], dim=1)))
        else:
            d4 = self._decode(4, self.dec4(self.dec4_reduce(self._up(b, e4)) + e4))
            d3 = self._decode(3, self.dec3(self.dec3_reduce(self._up(d4, e3)) + e3))
            d2 = self._decode(2, self.dec2(self.dec2_reduce(self._up(d3, e2)) + e2))
            d1 = self._decode(1, self.dec1(self.dec1_reduce(self._up(d2, e1)) + e1))

        out = self.head(d1)
        if self.boundary_refinement:
            out, self._boundary_logits = self.boundary_refiner(out, (e1, d1, d2, d3))
        if self.residual:
            out = out + x
        if out.shape[-2:] != output_size:
            raise RuntimeError(f"UKAD output shape {tuple(out.shape[-2:])} does not match input {output_size}")
        return out

    def diagnostics(self):
        if hasattr(self.bott, "kad"):
            return self.bott.kad.displace.diagnostics()
        return {}

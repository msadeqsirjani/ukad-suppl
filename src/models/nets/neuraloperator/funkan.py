import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ResBlock


def hermite_function(n, x):
    if n == 0:
        return math.pi ** (-1 / 4) * torch.exp(-(x**2) / 2)
    elif n == 1:
        return math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x**2) / 2)
    else:
        return math.sqrt(2 / n) * x * hermite_function(n - 1, x) - math.sqrt(
            (n - 1) / n
        ) * hermite_function(n - 2, x)


class Hermite2d(nn.Module):

    def __init__(self, in_channels, r=6):
        super(Hermite2d, self).__init__()

        self.r = r
        self.offset = ResBlock(in_channels, in_channels * 2, bn=True)

    def forward(self, x):
        B, n, H, W = x.shape

        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="xy",
        )

        x_delta, y_delta = torch.split(self.offset(x), n, dim=1)

        x_grid = x_delta + x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = y_delta + y_grid.unsqueeze(0).unsqueeze(0)
        return torch.stack(
            [
                hermite_function(k, x_grid) * hermite_function(k, y_grid)
                for k in range(self.r)
            ],
            2,
        )


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, phi_shape, r=6):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if len(phi_shape) != 2 or any(int(size) <= 0 for size in phi_shape):
            raise ValueError("phi_shape must contain two positive spatial dimensions")
        self.phi_shape = tuple(int(size) for size in phi_shape)

        self.phi = nn.Parameter(torch.empty(in_channels, *self.phi_shape))
        nn.init.orthogonal_(self.phi)

        self.psi = Hermite2d(in_channels, r=r)

        self.theta = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {C}")
        if (H, W) != self.phi_shape:
            raise ValueError(
                "FUNKAN spatial parameter shape does not match its input: "
                f"configured phi_shape={self.phi_shape}, got feature map {(H, W)}. "
                "Set phi_shape to the NeuralOperator bottleneck size."
            )

        psi = self.psi(x)
        cost = torch.einsum(
            "bikd,id->bik", psi.flatten(start_dim=3), self.phi.flatten(start_dim=1)
        )
        x = torch.einsum(
            "bik,bikd->bid", F.softmax(cost, dim=2), psi.flatten(start_dim=3)
        ).view(B, self.in_channels, H, W)
        x = self.theta(x)

        return x

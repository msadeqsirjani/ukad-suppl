from torch import nn


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()

        if mid_ch is None:
            mid_ch = out_ch

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

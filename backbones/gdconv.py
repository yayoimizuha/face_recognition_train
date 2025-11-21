import torch
import torch.nn as nn


class GDConvHead(nn.Module):
    """Global Depthwise Convolution head.

    Input: (B, C, H, W) -> Depthwise Conv (H,W) -> 1x1 Conv -> BN -> (B, out_channels)
    サイズはコンストラクタで確定し forward はシンプルにします。
    """

    def __init__(self, in_channels: int, h: int, w: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.h = h
        self.w = w
        self.depthwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(self.h, self.w),
            groups=self.in_channels,
            bias=True,
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        assert h == self.h and w == self.w, "GDConvHead: input spatial size mismatch"
        x = self.depthwise(x)
        x = self.conv_1x1(x)
        x = self.batchnorm(x)
        x = x.view(b, -1)
        return x

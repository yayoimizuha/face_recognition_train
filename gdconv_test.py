import timm
import torch
import torch.nn as nn
from torchinfo import summary


class GDConvHead(nn.Module):
    """Global Depthwise Convolution head.

    入力特徴マップ (B, C, H, W) に対して、
    H×W の depthwise conv を掛けて (B, C, 1, 1) -> (B, C) を出力します。
    カーネルサイズは最初の forward 時に動的に決定します。
    """

    def __init__(self, in_channels: int, h: int, w: int, out_channels: int = None):
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
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        assert h == self.h and w == self.w
        x = self.depthwise(x)  # (B, C, 1, 1)
        x = self.conv_1x1(x)  # (B, out_channels, 1, 1)
        x = self.batchnorm(x) # (B, out_channels, 1, 1)
        x = x.view(b, -1)  # (B, out_channels)
        return x


if __name__ == "__main__":
    model = timm.create_model(
        model_name="hgnetv2_b3.ssld_stage2_ft_in1k",
        pretrained=True,
    )

    if hasattr(model, "head"):
        model.head = nn.Identity()

    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("feature map:", out.shape)

    assert out.dim() == 4  # (B, C, H, W)

    model.head = GDConvHead(*out.shape[1:])
    final_out = model(dummy)
    print("final out:", final_out.shape)

    summary(model, input_size=(1, 3, 224, 224))

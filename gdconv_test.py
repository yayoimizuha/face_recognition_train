import timm
import torch
import torch.nn as nn
from torchinfo import summary
from timm.layers import SelectAdaptivePool2d


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
        self.batchnorm_1 = nn.BatchNorm2d(num_features=self.in_channels)

        self.conv_1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.batchnorm_2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        assert h == self.h and w == self.w
        x = self.depthwise(x)  # (B, C, 1, 1)
        x = self.batchnorm_1(x)  # (B, C, 1, 1)
        x = self.conv_1x1(x)  # (B, out_channels, 1, 1)
        x = self.batchnorm_2(x)  # (B, out_channels, 1, 1)
        x = x.view(b, -1)  # (B, out_channels)
        return x


def strip_after_head_or_pool(model: nn.Module) -> None:
    """ "head" と SelectAdaptivePool2d の両方を探索し、より広い範囲を削除できる
    （= 先頭側に位置する）方を選び、その位置以降を Identity 化する。

    削除範囲サイズ = len(children) - index なので index が小さい方が削除範囲は大きい。
    どちらか一方のみ存在する場合はそれを採用。どちらも無ければ何もしない。
    """
    keys = list(model._modules.keys())
    target_indices = []
    for i, k in enumerate(keys):
        m = model._modules[k]
        if k == "head" and isinstance(m, nn.Module):
            target_indices.append(i)
        elif isinstance(m, nn.Module) and isinstance(m, SelectAdaptivePool2d):
            target_indices.append(i)
    if not target_indices:
        return  # 該当なし
    sentinel_idx = min(target_indices)  # 最も前方=最大削除範囲
    for k in keys[sentinel_idx:]:
        if isinstance(model._modules[k], nn.Module):
            model._modules[k] = nn.Identity()


class BaseWithGDConv(nn.Module):
    """ベースモデル出力の特徴マップ(4D)に GDConvHead を接続するだけの簡易ラッパー。"""

    def __init__(self, base: nn.Module, gd_head: nn.Module):
        super().__init__()
        self.base = base
        self.gd_head = gd_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.base(x)
        assert fm.dim() == 4, "Base model must return 4D feature map after stripping."
        return self.gd_head(fm)


if __name__ == "__main__":
    model = timm.create_model(
        model_name="mobilenetv4_conv_medium.e500_r256_in1k",
        pretrained=True,
    )

    print(model)
    summary(model, input_size=(1, 3, 112, 112))

    # 動的に head/SelectAdaptivePool2d 以降を無効化
    strip_after_head_or_pool(model)

    print(model)
    summary(model, input_size=(1, 3, 112, 112))

    model.cpu()

    dummy = torch.randn(4, 3, 112, 112)
    # 特徴マップ取得（strip 後は head/pool が Identity なのでそのまま forward）
    out = model(dummy)
    assert out.dim() == 4  # (B, C, H, W)
    print("feature map:", out.shape)

    # GDConvHead を構築し、シンプルなラッパーで接続
    gd_head = GDConvHead(*out.shape[1:], out_channels=512)
    wrapped = BaseWithGDConv(model, gd_head)

    final_out = wrapped(dummy)
    print("final out:", final_out.shape)

    print("Model with GDConvHead:", wrapped)
    summary(wrapped, input_size=(4, 3, 112, 112))

import torch
import torch.nn as nn
from timm.layers import SelectAdaptivePool2d


class GDConvHead(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.h = h
        self.w = w
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(h, w),
            groups=in_channels,
            bias=True,
        )
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        assert h == self.h and w == self.w
        x = self.depthwise(x)
        x = self.bn_in(x)
        x = self.conv_1x1(x)
        x = self.bn_out(x)
        x = x.view(b, -1)
        return x


def strip_after_head_or_pool(model: nn.Module) -> None:
    keys = list(model._modules.keys())
    target_indices = []
    for i, k in enumerate(keys):
        m = model._modules[k]
        if k == "head" and isinstance(m, nn.Module):
            target_indices.append(i)
        if k == "global_pool" and isinstance(m, nn.Module):
            target_indices.append(i)
        elif isinstance(m, nn.Module) and isinstance(m, SelectAdaptivePool2d):
            target_indices.append(i)
    if not target_indices:
        return
    sentinel_idx = min(target_indices)
    for k in keys[sentinel_idx:]:
        if isinstance(model._modules[k], nn.Module):
            model._modules[k] = nn.Identity()


class BaseWithGDConv(nn.Module):
    def __init__(self, base: nn.Module, gd_head: nn.Module):
        super().__init__()
        self.base = base
        self.gd_head = gd_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.base(x)
        assert fm.dim() == 4
        return self.gd_head(fm)


def build_gdconv_wrapper(base_model: nn.Module, out_channels: int, dummy_shape=(1, 3, 224, 224)) -> nn.Module:
    strip_after_head_or_pool(base_model)
    dummy = torch.zeros(*dummy_shape)
    with torch.no_grad():
        feats = base_model(dummy)
    if feats.dim() != 4:
        raise ValueError(f"Expected 4D feature map, got {tuple(feats.shape)}")
    _, c, h, w = feats.shape
    gd_head = GDConvHead(c, h, w, out_channels)
    return BaseWithGDConv(base_model, gd_head)

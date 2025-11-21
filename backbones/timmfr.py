"""
===============================================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Anjith George

This software is distributed under the terms described in the LICENSE file
located in the parent directory of this source code repository.

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""

import timm
import torch
import torch.nn as nn
import math
from .gdconv import GDConvHead


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "head" not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = False
            if module.bias is not None:
                bias = True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)


def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


def replace_activation_function(model: nn.Module, before, after):
    for name, module in model.named_children():
        if isinstance(module, before):
            setattr(model, name, after())
        else:
            replace_activation_function(module, before, after)


class TimmFRWrapperV2(nn.Module):
    """Wrap timm model and optionally replace its head with GDConv at init.

    GDConv適用時は初期化時にダミー入力で空間形状 (C,H,W) を取得し、モデルの `head` を
    `GDConvHead` に差し替えます。forward 内での分岐は行いません。
    """

    def __init__(
        self,
        model_name="edgenext_x_small",
        num_features=512,
        batchnorm=False,
        pretrained=True,
        dropout: float = 0.0,
        amp: torch.dtype | None = None,
        apply_gdconv: bool = False,
    ):
        super().__init__()
        self.featdim = num_features
        self.model_name = model_name
        self.amp_dtype = amp
        self.apply_gdconv = apply_gdconv
        if "untrained" in self.model_name:
            pretrained = False
        self.model = timm.create_model(self.model_name, pretrained=pretrained, drop_rate=dropout)
        if not self.apply_gdconv:
            self.model.reset_classifier(self.featdim)  # type: ignore
        else:
            # 一度 head を Identity にして空間特徴を取得
            if hasattr(self.model, "head"):
                self.model.head = nn.Identity()

            c_in, h_in, w_in = (3, 224, 224)
            dummy = torch.zeros(1, c_in, h_in, w_in)
            with torch.no_grad():
                feats = self.model(dummy)
            if feats.dim() != 4:
                raise ValueError(f"GDConv requires spatial feature map (B,C,H,W). Got {tuple(feats.shape)}")
            _, c, h, w = feats.shape
            # 差し替え: (B,C,H,W)->(B,featdim)
            self.model.head = GDConvHead(in_channels=c, h=h, w=w, out_channels=self.featdim)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype is not None)):
            out = self.model(x)
        return out.float()


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)

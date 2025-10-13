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
        if isinstance(module, nn.Linear) and 'head' not in name:
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
    """
    Wraps timm model
    """

    def __init__(self, model_name='edgenext_x_small', num_features=512, batchnorm=False, pretrained=True, dropout: float = 0.0, amp: torch.dtype | None = None):
        super().__init__()
        self.featdim = num_features
        self.model_name = model_name
        self.amp_dtype = amp

        self.model = timm.create_model(self.model_name, pretrained=pretrained, drop_rate=dropout)
        self.model.reset_classifier(self.featdim) #type: ignore

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype is not None)):
            x = self.model(x)
        # Ensure embeddings are float32 for downstream modules (e.g., distributed all_gather)
        return x.float()


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)


class GDConv1x1Head(nn.Module):
    """
    Global Depthwise Convolution followed by 1x1 Convolution to produce embeddings.

    This module expects a spatial feature map of shape (B, C, H, W). It applies:
      - Depthwise Conv2d with kernel size (H, W), stride=1, padding=0, groups=C
      - 1x1 Conv2d to project channels to the embedding dimension
      - Flatten to (B, E)
    """

    def __init__(self, in_channels: int, embedding_dim: int, kernel_size: tuple[int, int], bias: bool = False, use_bn: bool = True):
        super().__init__()
        kh, kw = kernel_size
        self.gdconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kh, kw),
            stride=1,
            padding=0,
            groups=in_channels,
            bias=False,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # Optional BN neck (ArcFace practice): stabilize embedding distribution
        if use_bn:
            bn = nn.BatchNorm1d(embedding_dim, eps=1e-05)
            # Match IResNet practice: gamma=1.0 and freeze scale (keep running stats)
            nn.init.constant_(bn.weight, 1.0)
            bn.weight.requires_grad = False
            self.bn: nn.Module = bn
        else:
            # Identity keeps forward simple and readable
            self.bn = nn.Identity()

        # Initialize GDConv to behave like per-channel global average pooling at start
        with torch.no_grad():
            # Depthwise weight shape: (C, 1, kh, kw)
            self.gdconv.weight.data.fill_(1.0 / float(kh * kw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.gdconv(x)  # (B, C, 1, 1)
        x = self.conv1x1(x)  # (B, E, 1, 1)
        x = x.flatten(1)  # (B, E)
        x = self.bn(x)
        return x


class TimmFRWithGDHead(nn.Module):
    """
    Wrap timm model and replace the GAP + FC head with GDConv + 1x1 Conv.

    The underlying timm model is used up to its spatial feature map (via forward_features).
    The GDConv+1x1 head is instantiated lazily on first forward based on feature map size.
    """

    def __init__(self, model_name: str = 'edgenext_x_small', num_features: int = 512, pretrained: bool = True, bias: bool = False, input_size: tuple[int, int] = (112, 112), dropout: float = 0.0, batchnorm: bool = True, amp: torch.dtype | None = None):
        super().__init__()
        self.model_name = model_name
        self.featdim = num_features
        self.bias = bias
        self.input_size = input_size
        self.amp_dtype = amp
        self.batchnorm = batchnorm
        # Create a timm model as a pure feature extractor (no classifier/global pool)
        self.model = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0, global_pool='', drop_rate=dropout)
        # Build GDConv head immediately based on a dummy 112x112 input
        self.head: nn.Module | None = None
        self._init_head_with_dummy()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'forward_features') and callable(getattr(self.model, 'forward_features')):
            return self.model.forward_features(x)  # type: ignore[attr-defined]
        # As a last resort, try the plain forward and assume it returns spatial features
        return self.model(x)

    def _init_head_with_dummy(self):
        # Run a dummy forward to infer (C,H,W) for the given input size
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
            feat = self._extract_features(dummy)
            if feat.dim() != 4:
                raise RuntimeError(f"Expected spatial features (B,C,H,W) but got shape {tuple(feat.shape)}")
            _, c, h, w = feat.shape
            self.head = GDConv1x1Head(in_channels=c, embedding_dim=self.featdim, kernel_size=(h, w), bias=self.bias, use_bn=self.batchnorm)
        if was_training:
            self.model.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype is not None)):
            feat = self._extract_features(x)
            # Rebuild head lazily if feature spatial size differs from initial dummy
            if feat.dim() != 4:
                raise RuntimeError(f"Expected spatial features (B,C,H,W) but got shape {tuple(feat.shape)}")
            _, c, h, w = feat.shape
            if self.head is None:
                self.head = GDConv1x1Head(in_channels=c, embedding_dim=self.featdim, kernel_size=(h, w), bias=self.bias, use_bn=self.batchnorm)
            else:
                # check kernel size
                ks = getattr(getattr(self.head, 'gdconv', None), 'kernel_size', None)
                if ks is None or ks != (h, w):
                    # re-create to match current spatial size
                    self.head = GDConv1x1Head(in_channels=c, embedding_dim=self.featdim, kernel_size=(h, w), bias=self.bias, use_bn=self.batchnorm)
            x = self.head(feat)
        # Ensure embeddings are float32 for downstream modules
        return x.float()


def get_timmfr_gdconv(model_name: str, **kwargs) -> nn.Module:
    """
    Create a timm-based face recognition backbone where the standard GAP + FC head is
    replaced with GDConv + 1x1 Conv that outputs embeddings of dimension `featdim`.

    Parameters (kwargs):
      - featdim (int): embedding dimension (default 512)
      - pretrained (bool): load pretrained weights for the timm backbone (default True)
      - bias (bool): whether to use bias in the final 1x1 conv (default False)

    Returns:
      nn.Module: a model that maps (B,3,H,W) -> (B, featdim)
    """
    return TimmFRWithGDHead(model_name=model_name, **kwargs)

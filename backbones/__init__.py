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
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2, replace_activation_function
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf,get_mbf_large

import torch


def get_model(name, **kwargs):
    if name == "r50":
        return iresnet50(False, **kwargs)
    elif name == 'edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False, **kwargs), rank_ratio=0.6)
    elif name == 'edgeface_xs_q':
        model = get_timmfrv2('edgenext_x_small', batchnorm=False, **kwargs)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', batchnorm=False, **kwargs)
    elif name == 'convnext-large':
        return get_timmfrv2('convnext_large.fb_in22k_ft_in1k', batchnorm=False, **kwargs)
    elif name == 'edgeface_base':
        return get_timmfrv2('edgenext_base', batchnorm=False, **kwargs)
    elif name == 'mobilenetv4_conv_medium':
        return get_timmfrv2('mobilenetv4_conv_medium', batchnorm=False, **kwargs)
    elif name == 'mobilenetv4_hybrid_medium':
        return get_timmfrv2('mobilenetv4_hybrid_medium', batchnorm=False, **kwargs)
    elif name == 'mobilenetv4_hybrid_medium_silu':
        model = get_timmfrv2('mobilenetv4_hybrid_medium', batchnorm=False, **kwargs)
        replace_activation_function(model, torch.nn.ReLU, torch.nn.SiLU)
        return model
    elif name == 'tf_efficientnetv2_b2':
        return get_timmfrv2('tf_efficientnetv2_b2', batchnorm=False, **kwargs)
    elif name == 'convnext_pico':
        return get_timmfrv2("convnextv2_pico.fcmae_ft_in1k", pretrained=True, **kwargs)
    elif name == 'edgeface_xxs_q':
        model = get_timmfrv2('edgenext_xx_small', batchnorm=False, **kwargs)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False, **kwargs), rank_ratio=0.5)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        amp = kwargs.get("amp", torch.float16)
        return get_mbf(fp16=fp16, num_features=num_features, amp=amp)

    elif name == "mbf_large":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        amp = kwargs.get("amp", torch.float16)
        return get_mbf_large(fp16=fp16, num_features=num_features, amp=amp)
    
    elif name.startswith('hf-hub:'):
        return get_timmfrv2(name, batchnorm=False, **kwargs)

    else:
        raise ValueError()

# add MobileFaceNet Backbone
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/mobilefacenet.py

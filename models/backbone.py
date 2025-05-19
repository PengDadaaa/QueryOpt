# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications by: Wang Peng 2025
# References:
# DETR:https://github.com/facebookresearch/detr

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

# from .position_encoding_mul import build_position_encoding
from .position import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer3": "0",'layer4': "1"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # self.deconv = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2,padding=0,groups=16)
        # self.deconv = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2, padding=0)
        mid_channel = 256
        self.deconv = nn.Sequential(nn.Conv2d(num_channels, mid_channel, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(mid_channel), #wp       
                            nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(mid_channel),#wp
                            nn.Conv2d(mid_channel, num_channels//2, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(num_channels//2), nn.ReLU())
        # self.deconv = nn.ConvTranspose2d(num_channels, mid_channel, kernel_size=4, stride=2, padding=1)
        # self.conv = nn.Conv2d(mid_channel, num_channels//2, kernel_size=1, stride=1, padding=0)
        # self.conv_layer3 = nn.Sequential(nn.Conv2d(num_channels//2, num_channels//2, kernel_size=1, stride=1, padding=0),
        #                     nn.BatchNorm2d(num_channels//2))
        self.conv_layer3 = nn.Conv2d(num_channels//2, num_channels//2, kernel_size=1, stride=1, padding=0)
        # self.conv_layer3 = nn.Conv2d(num_channels//2, num_channels, kernel_size=1, stride=1, padding=0)
        self.num_channels = num_channels
        
        for m in self.deconv:
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif not isinstance(m, (nn.ReLU)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # for m in self.conv_layer3:
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif not isinstance(m, (nn.ReLU)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_layer3.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)


        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            if name == "0":
                # x = self.conv_layer3(x) + F.interpolate(xs['1'],size=x.shape[-2:],mode='bilinear')
                # x = self.conv_layer3(x) + self.deconv(xs['1'])
                # x = self.conv_layer3(x) + self.conv(self.deconv(xs['1']))
                x = self.conv_layer3(x) + self.deconv(xs['1'])
                # x = x + self.deconv(xs['1'])
            # print(x.shape)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # backbone = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process())#, norm_layer=FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process())#, norm_layer=FrozenBatchNorm2d
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 1024   #wp
        
        # print(f'Total number of parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}')
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        start = 0
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x,start).to(x.tensors.dtype))
            start+= x.tensors.shape[-1]
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # train_backbone = args.lr_backbone > 0
    train_backbone = args.lr > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

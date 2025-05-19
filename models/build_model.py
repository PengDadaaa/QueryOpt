# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications by: Wang Peng 2025
# References:
# DETR:https://github.com/facebookresearch/detr

"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone 

from .Transformer import build_transformer


class MyModel(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, num_roll):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_roll = num_roll
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.hash_embed = nn.Linear(hidden_dim, 1,bias=False)# wp+ 1

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        

        self.input_proj_list = nn.ModuleList([nn.Conv2d(backbone.num_channels//2, hidden_dim, kernel_size=1),
                                              nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)])        

        self.backbone = backbone
        self.inference_roll = False
        

                
    def forward(self, samples: NestedTensor):

        hidden_dim = self.transformer.d_model
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs,masks = [],[]
        for i,feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj_list[i](src))
            masks.append(mask)
            assert mask is not None


        # 初始化一个列表来存储移位后的张量
        shifted_query_embed = []

        # 对输入张量进行四次循环移位操作
        for i in range(self.num_roll):
            shifted = torch.roll(self.query_embed.weight, shifts=i*hidden_dim//self.num_roll, dims=1)
            shifted_query_embed.append(shifted)



        query_embed = torch.cat(shifted_query_embed, dim=0)

        hs,attn_weights= self.transformer(srcs, masks, query_embed, pos)

        hs_ori = hs[-1]
        hs = F.normalize(hs, p=2, dim=-1)#*self.norm

        outputs_class = self.hash_embed(hs)

        out = {'pred': torch.tanh(outputs_class[-1].flatten(1)),
               'attn_weights':attn_weights,
               'hs':hs[-1],
               'hs_ori':hs_ori}


        return out


def build(args,code_length,num_roll):


    # device = torch.device(args.device)

    backbone = build_backbone(args)


    transformer = build_transformer(args,code_length)
    model = MyModel(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_roll = num_roll)

    return model

# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:27
# @brief      : 
"""

import torch
import torch.nn as nn

from src.models.base.blocks import ResidualBlock, NonLocalBlock

class Encoder(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolution = [16]
        num_res_blocks = 2
        layers = [nn.Conv2d(img_channels, channels[0], 3, 1, 1)]
        resolution = 256
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
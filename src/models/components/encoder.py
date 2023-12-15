# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:27
# @brief      : 
"""

import torch
import torch.nn as nn

from src.models.base.blocks import ResidualBlock, AttnBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, img_channels: int, hidden_size: int):
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
                # 256 16
                if resolution in attn_resolution:
                    layers.append(AttnBlock(in_channels))
                #
                if i != len(channels) - 2:
                    layers.append(DownSampleBlock(channels[i + 1]))
                    resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(AttnBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

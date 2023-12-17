# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:27
# @brief      : 
"""
import torch.nn as nn

from src.models.base.blocks import ResidualBlock, AttnBlock, UpSampleBlock, GroupNorm


class Decoder(nn.Module):
    def __init__(self, img_channels: int, hidden_size: int):
        super().__init__()
        attn_resolutions = [16]
        ch_mult = [128, 128, 256, 256, 512]
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions - 1]
        curr_res = 256 // 2 ** (num_resolutions - 1)

        layers = [
            nn.Conv2d(hidden_size, block_in, kernel_size=3, stride=1, padding=1),
            ResidualBlock(block_in, block_in),
            AttnBlock(block_in),
            ResidualBlock(block_in, block_in)
        ]

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for i_block in range(3):
                layers.append(ResidualBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    layers.append(AttnBlock(block_in))
            if i != 0:
                layers.append(UpSampleBlock(block_in))
                curr_res = curr_res * 2

        layers.append(GroupNorm(block_in))
        # layers.append(Swish())
        layers.append(nn.Conv2d(block_in, img_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

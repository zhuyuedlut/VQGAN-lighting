# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:42
# @brief      : 
"""
import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor):
        return self.gn(x)
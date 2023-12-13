# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:39
# @brief      : 
"""

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

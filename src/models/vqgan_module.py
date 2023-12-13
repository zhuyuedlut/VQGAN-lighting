# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:17
# @brief      : 
"""
import torch
import torch.nn as nn
class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.codebook = None

    def forward(x: torch.Tensor) -> torch.Tensor:
        pass


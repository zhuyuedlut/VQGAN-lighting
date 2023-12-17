# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:27
# @brief      : 
"""
from typing import Tuple

import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors: int, hidden_size: int, beta: float):
        super().__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.hidden_size = hidden_size
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.hidden_size)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.hidden_size)

        d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
             torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()  # moving average instead of hard codebook remapping
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss

# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:17
# @brief      :
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.models.components.codebook import Codebook
from src.models.components.decoder import Decoder
from src.models.components.encoder import Encoder


class VQGAN(nn.Module):
    def __init__(self, img_channels: int, hidden_size: int, num_codebook_vectors: int, beta: float):
        super().__init__()
        self.encoder = Encoder(img_channels=img_channels, hidden_size=hidden_size)
        self.decoder = Decoder(img_channels=img_channels, hidden_size=hidden_size)
        self.codebook = Codebook(num_codebook_vectors=num_codebook_vectors, hidden_size=hidden_size, beta=beta)
        self.quant_conv = nn.Conv2d(hidden_size, hidden_size, 1)
        self.quant_conv = nn.Conv2d(hidden_size, hidden_size, 1)

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

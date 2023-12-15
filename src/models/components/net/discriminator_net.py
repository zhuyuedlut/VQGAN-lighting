import functools

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels: int, num_filters_last: int=64,  n_layers: int = 3, norm_layer: nn.Module = nn.BatchNorm2d):
        super().__init__()

        use_bias = norm_layer.func == nn.InstanceNorm2d \
            if type(norm_layer) == functools.partial else norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding_size = 4
        sequence = [
            nn.Conv2d(image_channels, num_filters_last, kernel_size, stride=2, padding=padding_size),
            nn.LeakyReLU(0.2)
        ]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            sequence += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last, 
                    num_filters_last * num_filters_mult, 
                    kernel_size,
                    2 if i < n_layers else 1, padding_size, 
                    bias=use_bias
                ),
                norm_layer(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, 1, padding_size)]
            self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
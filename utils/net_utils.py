"""Reusable normalisation layers.

These are drop-in replacements for ``LayerNorm`` that follow the Dynamic Tanh
("DyT") formulation of *Transformers without Normalization*: a learnable scalar
inside a ``tanh``, followed by an affine transform. They are not used by the
default LLFDisc configuration but are kept for architecture experiments.
"""

import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """Dynamic Tanh normalisation with a selectable channel layout.

    Args:
        normalized_shape: Number of channels the affine parameters cover.
        channels_last: If truthy, the affine parameters broadcast over the last
            dimension (``(B, H, W, C)`` layout); otherwise they broadcast over
            the channel dimension of a ``(B, C, H, W)`` tensor.
        alpha_init_value: Initial value of the learnable ``tanh`` slope.
    """

    def __init__(self, normalized_shape, channels_last="channels_first", alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class DyT(nn.Module):
    """Channels-last Dynamic Tanh normalisation.

    Args:
        num_features: Number of features the affine parameters cover.
        alpha_init_value: Initial value of the learnable ``tanh`` slope.
    """

    def __init__(self, num_features, alpha_init_value=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

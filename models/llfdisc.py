"""LLFDisc network definition.

The model is a three-scale U-shaped encoder/decoder. Every scale is refined by an
``EnhancedLCA`` block that combines cross attention, an illumination-enhancement
feed-forward layer, noise-aware gating, dark-region enhancement and channel
recalibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class CAB(nn.Module):
    """Cross-Attention Block.

    Computes transposed (channel-wise) attention: queries come from ``x`` while
    keys and values come from ``y``, so the cost is linear in the number of
    pixels instead of quadratic. Depth-wise convolutions inject local context
    before the projections.

    Args:
        dim: Number of input/output channels.
        num_heads: Number of attention heads; ``dim`` must be divisible by it.
        bias: Whether the convolutions use a bias term.
    """

    def __init__(self, dim, num_heads, bias=False):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        # Learnable per-head temperature that rescales the attention logits.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: Query source, shape ``(B, C, H, W)``.
            y: Key/value source, shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, C, H, W)``.
        """
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2-normalising q and k turns the dot product into a cosine similarity,
        # which keeps the logits bounded and stabilises training.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    """Illumination-Enhancement Layer.

    A gated feed-forward block: the expanded features are split into two
    branches, each passed through a depth-wise convolution with a residual Tanh
    non-linearity, and the branches are then multiplied together. The
    multiplicative gate lets the layer amplify dark regions without saturating
    the bright ones.

    Args:
        dim: Number of input/output channels.
        ffn_expansion_factor: Hidden-width multiplier.
        bias: Whether the convolutions use a bias term.
    """

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                 stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                 stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    """Layer normalisation supporting channels-first 4-D tensors.

    ``nn.LayerNorm`` only normalises the trailing dimensions, so it cannot be
    applied directly to ``(B, C, H, W)`` feature maps. With
    ``data_format="channels_first"`` this module normalises over the channel
    axis instead.

    Args:
        normalized_shape: Number of channels to normalise over.
        eps: Constant added to the variance for numerical stability.
        data_format: Either ``"channels_first"`` or ``"channels_last"``.
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Global average pooling produces one descriptor per channel; a bottleneck MLP
    turns it into per-channel gains. The output activation is ``tanh`` rather
    than the usual ``sigmoid``, so a channel can be suppressed as well as
    amplified.

    Args:
        input_channels: Number of channels of the input feature map.
        reduction_ratio: Bottleneck reduction factor of the MLP.
    """

    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).reshape(b, c)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(b, c, 1, 1)
        return x * y

    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)


class EnhancedLCA(nn.Module):
    """Enhanced Lightweight Cross-Attention block.

    Pipeline: cross attention (:class:`CAB`) then illumination enhancement
    (:class:`IEL`), followed by three enhancement stages -- a noise-aware gate
    that suppresses amplified sensor noise, a dark-region enhancement branch,
    and channel attention -- and finally :class:`SEBlock` recalibration plus a
    residual connection to the block input.

    Args:
        dim: Number of input/output channels.
        num_heads: Number of attention heads used by :class:`CAB`.
        bias: Whether the attention convolutions use a bias term.
    """

    def __init__(self, dim, num_heads, bias=False):
        super(EnhancedLCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.se = SEBlock(dim)

        # Noise-aware gate: predicts a per-pixel keep probability so that noise
        # amplified along with the signal can be attenuated.
        self.noise_attention = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Dark-region enhancement branch.
        self.dark_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        # Channel attention applied to the enhanced features.
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        Args:
            x: Query feature map, shape ``(B, C, H, W)``.
            y: Key/value feature map, shape ``(B, C, H, W)``.

        Returns:
            Refined feature map of shape ``(B, C, H, W)``.
        """
        x_res = x

        # Cross attention followed by illumination enhancement.
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x

        # Noise perception and suppression.
        noise_map = self.noise_attention(x)
        x_denoised = x * noise_map

        # Dark-region enhancement.
        dark_features = self.dark_enhance(x_denoised)

        # Reweight the enhanced features channel-wise.
        channel_weights = self.channel_attention(dark_features)
        enhanced_features = dark_features * channel_weights

        # Merge the denoised and the enhanced features.
        x = x_denoised + enhanced_features

        # Squeeze-and-excitation recalibration.
        x = self.se(x)

        # Residual connection back to the block input.
        x = x + x_res

        return x


class LLFDisc(nn.Module):
    """LLFDisc low-light image enhancement network.

    Three-scale U-shaped architecture. The encoder halves the resolution twice
    while doubling the width; every scale is refined by an :class:`EnhancedLCA`
    block, and the decoder upsamples with transposed convolutions and adds the
    matching encoder features back in.

    Args:
        in_channels: Number of image channels (3 for RGB).
        filters: Channel width at the finest scale.

    Shape:
        Input ``(B, in_channels, H, W)`` in ``[0, 1]``;
        output ``(B, in_channels, H, W)`` in ``[-1, 1]`` (``tanh``).
    """

    def __init__(self, in_channels=3, filters=32):
        super(LLFDisc, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)

        self.lca1 = EnhancedLCA(filters, num_heads=4)
        self.lca2 = EnhancedLCA(filters * 2, num_heads=4)
        self.lca3 = EnhancedLCA(filters * 4, num_heads=4)

        self.down1 = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(filters * 2, filters * 4, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=4, stride=2, padding=1)

        self.final_conv = nn.Conv2d(filters, in_channels, kernel_size=3, padding=1)
        self._init_weights()

    def forward(self, x):
        x1 = F.relu(self.initial_conv(x))
        x1_enhanced = self.lca1(x1, x1)

        x2 = F.relu(self.down1(x1_enhanced))
        x2_enhanced = self.lca2(x2, x2)

        x3 = F.relu(self.down2(x2_enhanced))
        x3_enhanced = self.lca3(x3, x3)

        x = F.relu(self.up1(x3_enhanced))
        # Resize to match the skip connection: odd input sizes make the
        # transposed convolution output differ by one pixel.
        x = F.interpolate(x, size=x2_enhanced.shape[2:], mode='bilinear', align_corners=False)
        x = x + x2_enhanced

        x = F.relu(self.up2(x))
        # Same guard for the finest scale.
        x = F.interpolate(x, size=x1_enhanced.shape[2:], mode='bilinear', align_corners=False)
        x = x + x1_enhanced

        out = self.final_conv(x)
        return torch.tanh(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLFDisc().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    n_param = sum(p.nelement() for p in model.parameters())
    print(f"Parameters: {n_param / 2 ** 20:.4f} M")

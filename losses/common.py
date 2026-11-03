"""Loss terms shared by every ``CombinedLoss`` variant.

All functions take ``(pred, target)`` tensors of shape ``(B, C, H, W)`` with
values in ``[0, 1]`` and return a scalar tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


def color_loss(pred, target):
    """L1 distance between the global mean colour of the two batches.

    Penalises an overall colour cast without constraining local structure.
    """
    return torch.mean(torch.abs(torch.mean(pred, dim=[1, 2, 3]) - torch.mean(target, dim=[1, 2, 3])))


def psnr_loss(pred, target):
    """PSNR expressed as a loss.

    PSNR is a reward, so it is subtracted from a 40 dB reference to obtain a
    quantity that decreases as the reconstruction improves. The result may go
    negative once PSNR exceeds 40 dB, which is harmless for gradient descent.
    """
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)


def smooth_l1_loss(pred, target):
    """Huber (smooth L1) pixel reconstruction loss."""
    return F.smooth_l1_loss(pred, target)


def multiscale_ssim_loss(pred, target, max_val=1.0, power_factors=(0.5, 0.5)):
    """Multi-scale SSIM expressed as a loss, i.e. ``1 - MS-SSIM``.

    Note:
        ``power_factors`` is accepted for API compatibility but is not forwarded
        to ``ms_ssim``; the library defaults are used.
    """
    return 1.0 - ms_ssim(pred, target, data_range=max_val, size_average=True)


def gaussian_kernel(x, mu, sigma):
    """Unnormalised Gaussian kernel, used for soft histogram binning."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def histogram_loss(pred, target, bins=256, sigma=0.01):
    """L1 distance between the normalised intensity histograms of the batch.

    Histograms are accumulated per sample with ``torch.histc`` and normalised to
    sum to one, which makes the term a global tone-distribution constraint.

    Warning:
        ``torch.histc`` is not differentiable, so this term contributes no
        gradient. It is kept because it is part of the published loss
        configuration and is useful as a monitoring signal.

    Args:
        pred: Predicted batch, shape ``(B, C, H, W)``.
        target: Ground-truth batch, same shape.
        bins: Number of histogram bins over ``[0, 1]``.
        sigma: Kept for API compatibility; unused by the hard-binning path.

    Returns:
        Scalar tensor; a small positive constant if the distance degenerates.
    """
    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    pred_hist = torch.zeros((batch_size, bins), device=pred.device)
    target_hist = torch.zeros((batch_size, bins), device=target.device)

    for b in range(batch_size):
        pred_hist[b] = torch.histc(pred_flat[b], bins=bins, min=0.0, max=1.0)
        target_hist[b] = torch.histc(target_flat[b], bins=bins, min=0.0, max=1.0)

        # Normalise so the term is invariant to image resolution.
        pred_hist[b] = pred_hist[b] / (pred_hist[b].sum() + 1e-8)
        target_hist[b] = target_hist[b] / (target_hist[b].sum() + 1e-8)

    hist_distance = torch.mean(torch.abs(pred_hist - target_hist))

    # Guard against a degenerate (NaN or exactly zero) distance, which usually
    # means the inputs fell outside the [0, 1] binning range.
    if torch.isnan(hist_distance) or hist_distance == 0:
        print("Warning: histogram loss is NaN or zero")
        print(f"  pred range:   {pred.min().item()} - {pred.max().item()}")
        print(f"  target range: {target.min().item()} - {target.max().item()}")
        print(f"  pred hist sum:   {pred_hist.sum().item()}")
        print(f"  target hist sum: {target_hist.sum().item()}")
        return torch.tensor(1e-8, device=pred.device)

    return hist_distance


class get_Fre(nn.Module):
    """Splits a tensor into the amplitude and phase of its 2-D Fourier spectrum."""

    def __init__(self):
        super(get_Fre, self).__init__()

    def forward(self, dp):
        """
        Args:
            dp: Real-valued input tensor; the last two dims are transformed.

        Returns:
            Tuple ``(amplitude, phase)`` of the half-spectrum produced by
            ``torch.fft.rfft2``.
        """
        dp = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)

        return dp_amp, dp_pha

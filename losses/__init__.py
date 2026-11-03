"""Loss functions for LLFDisc.

The training script resolves the objective dynamically from the config keys
``loss_path`` and ``loss_name``, e.g.::

    loss_path: losses.combined
    loss_name: CombinedLossFull
"""

from .combined import CombinedLossFMse, CombinedLossFull, CombinedLossVgg
from .common import (
    color_loss,
    gaussian_kernel,
    get_Fre,
    histogram_loss,
    multiscale_ssim_loss,
    psnr_loss,
    smooth_l1_loss,
)
from .fourier import FourierGaussianKLLoss, FourierMseLoss
from .perceptual import VGGKLPerceptualLoss, VGGPerceptualLoss

# Default objective; ``CombinedLoss`` is the name used by the original configs.
CombinedLoss = CombinedLossFull

__all__ = [
    "CombinedLoss",
    "CombinedLossFMse",
    "CombinedLossFull",
    "CombinedLossVgg",
    "FourierGaussianKLLoss",
    "FourierMseLoss",
    "VGGKLPerceptualLoss",
    "VGGPerceptualLoss",
    "color_loss",
    "gaussian_kernel",
    "get_Fre",
    "histogram_loss",
    "multiscale_ssim_loss",
    "psnr_loss",
    "smooth_l1_loss",
]

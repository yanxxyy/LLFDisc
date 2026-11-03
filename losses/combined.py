"""Composite objectives used to train LLFDisc.

Three variants are exposed, all sharing the same seven-term structure and the
same ``alpha1``..``alpha7`` weights read from the config file:

===========================  ==================  ======================
Class                        Perceptual term     Frequency term
===========================  ==================  ======================
:class:`CombinedLossFull`    VGG feature KL      Fourier Gaussian KL
:class:`CombinedLossFMse`    VGG feature KL      Fourier MSE
:class:`CombinedLossVgg`     VGG feature MSE     Fourier Gaussian KL
===========================  ==================  ======================

Pick one from a config file via ``loss_path`` / ``loss_name``; every variant is
also aliased to ``CombinedLoss`` inside its own module for convenience.
"""

import torch
import torch.nn as nn

from .common import (
    color_loss,
    histogram_loss,
    multiscale_ssim_loss,
    psnr_loss,
    smooth_l1_loss,
)
from .fourier import FourierGaussianKLLoss, FourierMseLoss
from .perceptual import VGGKLPerceptualLoss, VGGPerceptualLoss


class _CombinedLossBase(nn.Module):
    """Weighted sum of seven reconstruction, perceptual and statistical terms.

    Subclasses only choose which perceptual and frequency modules to use; the
    weighting and reporting logic lives here.

    Args:
        device: Device the loss modules and inputs are placed on.
        opt: Parsed config mapping. Reads ``alpha1``..``alpha6`` (required) and
            ``alpha7`` (optional, defaults to 0.5):

            * ``alpha1`` -- smooth L1
            * ``alpha2`` -- perceptual
            * ``alpha3`` -- histogram
            * ``alpha4`` -- multi-scale SSIM
            * ``alpha5`` -- PSNR
            * ``alpha6`` -- colour
            * ``alpha7`` -- frequency domain
    """

    # Overridden by subclasses.
    perceptual_cls = VGGKLPerceptualLoss
    frequency_cls = FourierGaussianKLLoss
    frequency_key = 'frequency'

    def __init__(self, device, opt):
        super(_CombinedLossBase, self).__init__()
        self.device = device
        self.perceptual_loss_model = self.perceptual_cls(device)
        self.frequency_loss = self.frequency_cls()

        self.alpha1 = float(opt['alpha1'])
        self.alpha2 = float(opt['alpha2'])
        self.alpha3 = float(opt['alpha3'])
        self.alpha4 = float(opt['alpha4'])
        self.alpha5 = float(opt['alpha5'])
        self.alpha6 = float(opt['alpha6'])
        self.alpha7 = float(opt.get('alpha7', 0.5))

        # Unweighted value of each term from the most recent forward pass,
        # kept for logging via :meth:`print_loss_info`.
        self.loss_values = {}

        self._alpha_mapping = {
            'smooth_l1': 'alpha1',
            'perceptual': 'alpha2',
            'histogram': 'alpha3',
            'ms_ssim': 'alpha4',
            'psnr': 'alpha5',
            'color': 'alpha6',
            self.frequency_key: 'alpha7',
        }

    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: Network output, shape ``(B, C, H, W)``.
            y_pred: Ground-truth image, same shape.

        Returns:
            Scalar tensor holding the weighted total loss.

        Note:
            The argument names are inherited from the original code base and are
            the reverse of what they suggest: the training loop calls
            ``criterion(outputs, targets)``. Every term is symmetric or is
            applied consistently, so the ordering does not change the value.
        """
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)
        freq_l = self.frequency_loss(y_true, y_pred)

        self.loss_values = {
            'smooth_l1': smooth_l1_l.item(),
            'ms_ssim': ms_ssim_l.item(),
            'perceptual': perc_l.item(),
            'histogram': hist_l.item(),
            'psnr': psnr_l.item(),
            'color': color_l.item(),
            self.frequency_key: freq_l.item(),
        }

        total_loss = (self.alpha1 * smooth_l1_l
                      + self.alpha2 * perc_l
                      + self.alpha3 * hist_l
                      + self.alpha4 * ms_ssim_l
                      + self.alpha5 * psnr_l
                      + self.alpha6 * color_l
                      + self.alpha7 * freq_l)

        return torch.mean(total_loss)

    def print_loss_info(self, epoch=None):
        """Prints the raw and weighted value of every term from the last call."""
        epoch_str = f"Epoch {epoch} " if epoch is not None else ""
        print(f"\n===== {epoch_str}loss breakdown =====")

        print("Raw values:")
        for name, value in self.loss_values.items():
            print(f"  {name}: {value:.6f}")

        print("\nWeighted values:")
        weighted_losses = {}
        for loss_name, loss_value in self.loss_values.items():
            alpha_value = getattr(self, self._alpha_mapping[loss_name])
            weighted_value = alpha_value * loss_value
            weighted_losses[loss_name] = weighted_value
            print(f"  {loss_name} (weight {alpha_value:.4f}): {weighted_value:.6f}")

        print(f"\nTotal: {sum(weighted_losses.values()):.6f}")
        print("=" * 40)


class CombinedLossFull(_CombinedLossBase):
    """Full LLFDisc objective: VGG feature KL plus Fourier Gaussian KL."""

    perceptual_cls = VGGKLPerceptualLoss
    frequency_cls = FourierGaussianKLLoss
    frequency_key = 'fourier_kl'


class CombinedLossFMse(_CombinedLossBase):
    """Ablation: replaces the Fourier Gaussian KL term with a Fourier MSE."""

    perceptual_cls = VGGKLPerceptualLoss
    frequency_cls = FourierMseLoss
    frequency_key = 'fourier_mse'


class CombinedLossVgg(_CombinedLossBase):
    """Ablation: replaces the VGG feature KL term with a plain feature MSE."""

    perceptual_cls = VGGPerceptualLoss
    frequency_cls = FourierGaussianKLLoss
    frequency_key = 'fourier_kl'

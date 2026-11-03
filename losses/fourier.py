"""Frequency-domain loss terms.

Both variants compare the amplitude and the phase of the Fourier spectra of the
prediction and the target, but they differ in how:

* :class:`FourierGaussianKLLoss` -- fits a Gaussian to each spectrum and takes a
  closed-form KL divergence between them (the distribution-matching term used
  by the full LLFDisc loss).
* :class:`FourierMseLoss` -- a plain MSE between max-normalised spectra, used as
  an ablation baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_Fre


class FourierGaussianKLLoss(nn.Module):
    """KL divergence between Gaussian fits of the Fourier amplitude and phase.

    For every channel the amplitude and phase spectra are summarised by their
    mean and variance, and the two resulting Gaussians are compared in closed
    form. Matching distributions rather than individual coefficients makes the
    term insensitive to the exact spatial alignment of frequency content.

    Args:
        reduction: Kept for API compatibility; the loss always returns the mean
            over channels.
    """

    def __init__(self, reduction='mean'):
        super(FourierGaussianKLLoss, self).__init__()
        self.reduction = reduction
        self.fre_extractor = get_Fre()
        self.eps = 1e-8  # Guards against division by zero and log(0).

    def gaussian_kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """Closed-form KL divergence between two univariate Gaussians.

        ``KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
        = log(sigma2 / sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2 sigma2^2) - 1/2``

        Args:
            mu1, sigma1: Mean and standard deviation of the first Gaussian.
            mu2, sigma2: Mean and standard deviation of the second Gaussian.

        Returns:
            Scalar tensor holding the divergence.
        """
        # Standard deviations must stay strictly positive.
        sigma1 = torch.clamp(sigma1, min=self.eps)
        sigma2 = torch.clamp(sigma2, min=self.eps)

        term1 = torch.log(sigma2 / sigma1 + self.eps)
        term2 = (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2 * sigma2.pow(2))
        term3 = 0.5

        kl = term1 + term2 - term3
        return kl

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        batch_size, channels = pred.shape[:2]
        total_loss = 0.0

        # Transform each colour channel separately.
        for c in range(channels):
            pred_c = pred[:, c]
            target_c = target[:, c]

            pred_amp, pred_phase = self.fre_extractor(pred_c)
            target_amp, target_phase = self.fre_extractor(target_c)

            # Gaussian statistics of the amplitude spectrum.
            pred_amp_mean = pred_amp.mean()
            pred_amp_var = pred_amp.var()
            target_amp_mean = target_amp.mean()
            target_amp_var = target_amp.var()

            # Gaussian statistics of the phase spectrum.
            pred_phase_mean = pred_phase.mean()
            pred_phase_var = pred_phase.var()
            target_phase_mean = target_phase.mean()
            target_phase_var = target_phase.var()

            # KL is taken with the target as the reference distribution.
            amp_kl = self.gaussian_kl_divergence(
                target_amp_mean, torch.sqrt(target_amp_var),
                pred_amp_mean, torch.sqrt(pred_amp_var)
            )

            phase_kl = self.gaussian_kl_divergence(
                target_phase_mean, torch.sqrt(target_phase_var),
                pred_phase_mean, torch.sqrt(pred_phase_var)
            )

            total_loss += amp_kl + phase_kl

        return total_loss / channels


class FourierMseLoss(nn.Module):
    """MSE between max-normalised Fourier amplitude and phase spectra.

    Amplitude and phase are each divided by their maximum before the comparison,
    which keeps the two terms on a similar scale. Serves as the frequency-domain
    ablation baseline for :class:`FourierGaussianKLLoss`.

    Args:
        reduction: Kept for API compatibility; unused.
        amp_weight: Amplitude weight, stored for reference; the forward pass
            averages the two terms equally.
        phase_weight: Phase weight, stored for reference; see ``amp_weight``.
        freq_weight: Whether to weight frequency bands differently. Stored for
            reference; not applied in this implementation.
    """

    def __init__(self, reduction='mean', amp_weight=1.0, phase_weight=0.5, freq_weight=True):
        super(FourierMseLoss, self).__init__()
        self.reduction = reduction
        self.fre_extractor = get_Fre()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        self.freq_weight = freq_weight
        self.eps = 1e-8

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        pred_amp, pred_phase = self.fre_extractor(pred)
        target_amp, target_phase = self.fre_extractor(target)

        # Normalise each spectrum by its own maximum so that amplitude (which
        # spans several orders of magnitude) and phase are comparable.
        pred_amp_max = torch.max(pred_amp) + self.eps
        pred_phase_max = torch.max(pred_phase) + self.eps
        pred_amp = pred_amp / pred_amp_max
        pred_phase = pred_phase / pred_phase_max

        target_amp_max = torch.max(target_amp) + self.eps
        target_phase_max = torch.max(target_phase) + self.eps
        target_amp = target_amp / target_amp_max
        target_phase = target_phase / target_phase_max

        amp_loss = F.mse_loss(pred_amp, target_amp)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        return (amp_loss + phase_loss) / 2

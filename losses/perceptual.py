"""VGG-based perceptual losses.

Two variants are provided:

* :class:`VGGPerceptualLoss` -- MSE between VGG-19 feature maps.
* :class:`VGGKLPerceptualLoss` -- symmetric KL divergence between the channel
  and spatial distributions of the VGG-19 feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _vgg19_features(device, cut=16):
    """Returns the first ``cut`` layers of a frozen, ImageNet-pretrained VGG-19.

    Args:
        device: Device the feature extractor is moved to.
        cut: Number of leading layers to keep. The default of 16 ends at
            ``block3_conv3``, which captures texture without being too
            semantic.

    Returns:
        An ``nn.Sequential`` in eval mode with ``requires_grad=False``.
    """
    try:
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features[:cut]
    except AttributeError:
        # torchvision < 0.13 does not expose the ``weights`` enum.
        vgg = models.vgg19(pretrained=True).features[:cut]

    loss_model = vgg.to(device).eval()
    for param in loss_model.parameters():
        param.requires_grad = False
    return loss_model


class VGGPerceptualLoss(nn.Module):
    """Feature-space MSE between the prediction and the target.

    Args:
        device: Device on which the frozen VGG-19 is placed.
    """

    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_model = _vgg19_features(device)

    def forward(self, y_true, y_pred):
        target_device = next(self.loss_model.parameters()).device
        y_true, y_pred = y_true.to(target_device), y_pred.to(target_device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


class VGGKLPerceptualLoss(nn.Module):
    """Distribution-level perceptual loss on VGG-19 features.

    Instead of matching features element-wise, the feature maps are turned into
    probability distributions with a softmax and compared with a KL divergence
    along both the channel and the spatial axis. This is more tolerant of small
    spatial misalignments than a plain feature MSE.

    Args:
        device: Device on which the frozen VGG-19 is placed.
    """

    def __init__(self, device):
        super(VGGKLPerceptualLoss, self).__init__()
        self.loss_model = _vgg19_features(device)
        self.eps = 1e-8  # Guards the logarithm against zero probabilities.

    def kl_divergence(self, p, q):
        """Computes ``KL(p || q)`` after softmax-normalising both inputs.

        Args:
            p: Logits of the reference distribution, normalised over dim 1.
            q: Logits of the compared distribution, same shape as ``p``.

        Returns:
            Scalar tensor, averaged over the remaining dimensions.
        """
        p = F.softmax(p, dim=1) + self.eps
        q = F.softmax(q, dim=1) + self.eps

        kl_loss = torch.sum(p * torch.log(p / q), dim=1)
        return torch.mean(kl_loss)

    def feature_distribution_kl(self, feat1, feat2):
        """KL divergence between two feature maps, along channels and space.

        Args:
            feat1: Reference features, shape ``(B, C, H, W)``.
            feat2: Compared features, same shape.

        Returns:
            Scalar tensor: the sum of the channel-wise and spatial KL terms,
            averaged over the batch.
        """
        b, c, h, w = feat1.shape
        # Flatten the spatial dims so each sample becomes a (C, H*W) matrix.
        feat1_flat = feat1.view(b, c, -1)
        feat2_flat = feat2.view(b, c, -1)

        # KL across the channel axis.
        channel_kl = 0
        for i in range(b):
            channel_kl += self.kl_divergence(feat1_flat[i], feat2_flat[i])

        # KL across the spatial axis (matrix transposed, so dim 1 is space).
        spatial_kl = 0
        for i in range(b):
            spatial_kl += self.kl_divergence(feat1_flat[i].transpose(0, 1),
                                             feat2_flat[i].transpose(0, 1))

        return (channel_kl + spatial_kl) / b

    def forward(self, y_true, y_pred):
        target_device = next(self.loss_model.parameters()).device
        y_true = y_true.to(target_device)
        y_pred = y_pred.to(target_device)

        # Extract VGG features.
        feat_true = self.loss_model(y_true)
        feat_pred = self.loss_model(y_pred)

        # Compare their distributions.
        return self.feature_distribution_kl(feat_true, feat_pred)

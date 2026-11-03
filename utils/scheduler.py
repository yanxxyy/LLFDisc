"""Learning-rate schedules."""

from math import cos, pi


def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):
    """Linear warm-up followed by cosine annealing, applied in place.

    During the first ``warmup_epoch`` epochs the learning rate ramps linearly
    from 0 to ``lr_max``; afterwards it anneals along a cosine curve down to
    ``lr_min`` at ``max_epoch``.

    Args:
        optimizer: Optimiser whose param groups are updated.
        current_epoch: Zero-based index of the epoch about to start.
        max_epoch: Total number of epochs.
        lr_min: Final learning rate.
        lr_max: Peak learning rate reached at the end of warm-up.
        warmup_epoch: Number of warm-up epochs.

    Returns:
        The learning rate that was set.
    """
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
            1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

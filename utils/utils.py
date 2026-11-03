"""Small helpers: config parsing, dynamic imports, metrics and file utilities."""

import importlib
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


def parse_config(path=None):
    """Loads a YAML config file into a dict.

    Args:
        path: Path to the ``.yaml`` file.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If ``path`` is ``None`` or does not exist.
    """
    import yaml

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def dynamic_import(module_path, class_name):
    """Imports ``module_path`` and returns the attribute ``class_name`` from it.

    This is how the training and evaluation scripts stay agnostic of which
    network or loss is being used: both are named in the config file.

    Args:
        module_path: Dotted module path, e.g. ``"models.llfdisc"``.
        class_name: Name of the class or function to fetch from that module.

    Returns:
        The requested attribute.

    Raises:
        ImportError: If neither ``module_path`` nor ``src.<module_path>``
            can be imported.
        AttributeError: If the module has no such attribute.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        # Fall back to a ``src``-prefixed layout.
        module = importlib.import_module("src." + module_path)

    return getattr(module, class_name)


def check_file_exists(file_path):
    """Returns whether ``file_path`` exists (``False`` for ``None``)."""
    if not file_path:
        return False
    return os.path.exists(file_path)


def get_current_time(compact=True):
    """Returns the current local time as a string.

    Args:
        compact: If true, format as ``YYYYMMDDHHMM`` (safe inside file names);
            otherwise ``YYYY-MM-DD HH:MM:SS``.
    """
    now = datetime.now()
    if compact:
        return now.strftime("%Y%m%d%H%M")
    return now.strftime("%Y-%m-%d %H:%M:%S")


def count_image_files(folder_path, extensions=('.png',)):
    """Counts files with the given extensions under ``folder_path``, recursively.

    Args:
        folder_path: Root directory to walk.
        extensions: Lower-case extensions to count, e.g. ``('.png', '.jpg')``.

    Returns:
        The number of matching files.
    """
    count = 0
    for root, _dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count


def find_difference(arr1, arr2):
    """Returns the two one-sided set differences between ``arr1`` and ``arr2``.

    Handy for spotting unmatched file names between a low-light and a
    normal-light directory.

    Returns:
        Tuple ``(only_in_arr1, only_in_arr2)``, both as lists.
    """
    set1, set2 = set(arr1), set(arr2)
    return list(set1 - set2), list(set2 - set1)


def save_config(config, output_path):
    """Writes a config mapping to a plain-text file, one ``key: value`` per line.

    Saved alongside the checkpoints so every training run is reproducible.

    Args:
        config: Mapping (or any object) describing the run.
        output_path: Destination file path.

    Returns:
        ``True`` on success, ``False`` if writing failed.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(config, dict):
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(str(config))
        print(f"Config saved to: {output_path}")
        return True
    except OSError as e:
        print(f"Failed to save config: {e}")
        return False


def _match_brightness(pred, target):
    """Rescales ``pred`` so its mean brightness matches ``target``.

    This is the ``GT-mean`` protocol commonly reported in the low-light
    enhancement literature: it removes a global exposure offset before the
    metric is computed, so results are comparable across methods that normalise
    brightness differently.

    Args:
        pred: Predicted batch, shape ``(B, C, H, W)``.
        target: Ground-truth batch, same shape.

    Returns:
        The rescaled prediction, clamped to ``[0, 1]``.
    """
    mean_restored = pred.mean(axis=1).mean()
    mean_target = target.mean(axis=1).mean()
    return torch.clamp(pred * (mean_target / mean_restored), 0, 1)


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """Computes PSNR between two image batches.

    Args:
        img1: Predicted batch, shape ``(B, C, H, W)``, values in ``[0, 1]``.
        img2: Ground-truth batch, same shape.
        max_pixel_value: Dynamic range of the inputs.
        gt_mean: Whether to apply GT-mean brightness matching first.

    Returns:
        PSNR in dB as a float, or ``inf`` if the images are identical.
    """
    if gt_mean:
        img1 = _match_brightness(img1, img2)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """Computes SSIM between two image batches.

    Args:
        img1: Predicted batch, shape ``(B, C, H, W)``, values in ``[0, 1]``.
        img2: Ground-truth batch, same shape.
        max_pixel_value: Dynamic range of the inputs.
        gt_mean: Whether to apply GT-mean brightness matching first.

    Returns:
        SSIM in ``[0, 1]`` as a float.
    """
    if gt_mean:
        img1 = _match_brightness(img1, img2)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def calculate_lpips(img1, img2, lpips_fn, gt_mean=True):
    """Computes LPIPS between two image batches.

    Args:
        img1: Predicted batch, shape ``(B, C, H, W)``, values in ``[0, 1]``.
        img2: Ground-truth batch, same shape.
        lpips_fn: An initialised ``lpips.LPIPS`` module.
        gt_mean: Whether to apply GT-mean brightness matching first.

    Returns:
        LPIPS distance as a float; lower is better.
    """
    if gt_mean:
        img1 = _match_brightness(img1, img2)

    # LPIPS expects inputs in [-1, 1].
    lpips_val = lpips_fn(img1 * 2.0 - 1.0, img2 * 2.0 - 1.0).mean()
    return lpips_val.item()

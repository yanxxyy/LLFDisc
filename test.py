"""Reference-based evaluation: PSNR, SSIM and LPIPS on a paired test set.

Usage::

    python test.py --config configs/test.yaml

All metrics use the GT-mean protocol (the prediction is rescaled so its mean
brightness matches the ground truth) that is standard in the low-light
enhancement literature.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloaders
from utils import utils


def validate(model, dataloader, device, lpips_fn, verbose=True):
    """Runs the model over ``dataloader`` and averages PSNR, SSIM and LPIPS.

    Args:
        model: Network to evaluate; switched to eval mode.
        dataloader: Paired loader yielding ``(low, high)`` batches.
        device: Device to run on.
        lpips_fn: Initialised ``lpips.LPIPS`` module.
        verbose: Whether to print per-image metrics.

    Returns:
        Tuple ``(avg_psnr, avg_ssim, avg_lpips)``.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            # The network ends in tanh, so clamp before computing metrics.
            output = torch.clamp(model(low), 0, 1)

            psnr = utils.calculate_psnr(output, high)
            ssim = utils.calculate_ssim(output, high)
            lpips_score = utils.calculate_lpips(output, high, lpips_fn)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score

            if verbose:
                print(f'Image {idx + 1}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_score:.4f}')

    n = len(dataloader)
    return total_psnr / n, total_ssim / n, total_lpips / n


def main():
    import lpips

    parser = argparse.ArgumentParser(description='Evaluate LLFDisc (PSNR / SSIM / LPIPS)')
    parser.add_argument('--config', type=str, default='configs/test.yaml',
                        help='Path to the YAML config file (default: configs/test.yaml)')
    args = parser.parse_args()
    opt = utils.parse_config(args.config)

    weights_path = opt['weight_path']
    test_low = opt['test_low']
    test_high = opt['test_high']
    crop_size = int(opt['crop_size'])
    model_path = opt["model_path"]
    class_name = opt["class_name"]
    device = torch.device(opt['device'] if torch.cuda.is_available() else 'cpu')
    # Lower this (0 disables multiprocessing) on machines with few CPU cores.
    num_workers = int(opt.get('num_workers', 4))

    print(f'Using device: {device}')

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    print('LPIPS initialised with AlexNet')

    _, test_loader = create_dataloaders(None, None, test_low, test_high,
                                        crop_size=crop_size, batch_size=1,
                                        num_workers=num_workers)
    print(f'Test loader: {len(test_loader)} images')

    network_cls = utils.dynamic_import(model_path, class_name)
    model = network_cls().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    print('\nStarting evaluation...')
    avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, lpips_fn)

    print('\n' + '=' * 50)
    print('FINAL RESULTS:')
    print(f'Average PSNR:  {avg_psnr:.6f}')
    print(f'Average SSIM:  {avg_ssim:.6f}')
    print(f'Average LPIPS: {avg_lpips:.6f}')
    print('=' * 50)


if __name__ == '__main__':
    main()

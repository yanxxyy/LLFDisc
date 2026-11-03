"""No-reference evaluation: NIQE on unpaired low-light benchmarks.

Use this for datasets without ground truth (DICM, LIME, MEF, NPE, VV)::

    python test_niqe.py --config configs/test_niqe.yaml

Enhanced images are written to ``--output-dir`` (default ``results/<dataset>``)
and the mean NIQE over all valid images is reported. Requires the BasicSR
``niqe_pris_params.npz`` file; see ``utils/niqe_utils.py`` for how it is located.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_eval_dataloader
from utils import utils
from utils.niqe_utils import calculate_niqe


def validate(model, dataloader, device, result_dir, save_images=True):
    """Enhances every image in ``dataloader`` and averages their NIQE scores.

    Args:
        model: Network to evaluate; switched to eval mode.
        dataloader: Loader yielding ``(image, file_name)`` pairs.
        device: Device to run on.
        result_dir: Directory the enhanced images are written to.
        save_images: Whether to write the enhanced images to disk.

    Returns:
        Mean NIQE over the images that produced a finite score, or ``inf`` if
        none did. Lower is better.
    """
    model.eval()
    total_niqe = 0.0
    valid_images = 0

    if save_images:
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for low, filename in dataloader:
            low = low.to(device)
            # The network ends in tanh, so clamp into valid image range.
            output = torch.clamp(model(low), 0, 1)

            output_np = output[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)

            # NIQE and cv2.imwrite both expect 8-bit BGR.
            output_cv2 = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if save_images:
                cv2.imwrite(os.path.join(result_dir, filename[0]), output_cv2)

            try:
                niqe_score = calculate_niqe(output_cv2)
                # Degenerate outputs (e.g. a constant image) yield NaN or inf;
                # skip them so they cannot poison the average.
                if np.isnan(niqe_score) or np.isinf(niqe_score):
                    print(f'Image {filename[0]} - invalid NIQE score (NaN or Inf)')
                    continue

                total_niqe += niqe_score
                valid_images += 1
                print(f'Image {filename[0]} - NIQE: {niqe_score:.4f}')
            except Exception as e:
                print(f'Error processing image {filename[0]}: {e}')

    avg_niqe = total_niqe / valid_images if valid_images > 0 else float('inf')
    print(f'Processed {valid_images} valid images out of {len(dataloader)} total')
    return avg_niqe


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLFDisc with NIQE (no reference)')
    parser.add_argument('--config', type=str, default='configs/test_niqe.yaml',
                        help='Path to the YAML config file (default: configs/test_niqe.yaml)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for the enhanced images '
                             '(default: results/<test_low basename>)')
    args = parser.parse_args()
    opt = utils.parse_config(args.config)

    weights_path = opt['weight_path']
    test_low = opt['test_low']
    model_path = opt["model_path"]
    class_name = opt["class_name"]
    device = torch.device(opt['device'] if torch.cuda.is_available() else 'cpu')
    # Lower this (0 disables multiprocessing) on machines with few CPU cores.
    num_workers = int(opt.get('num_workers', 4))

    result_dir = args.output_dir or os.path.join('results', os.path.basename(test_low.rstrip('/\\')))

    print(f'Using device: {device}')

    test_loader = create_eval_dataloader(test_low, num_workers=num_workers)
    print(f'Test loader: {len(test_loader)} images')

    network_cls = utils.dynamic_import(model_path, class_name)
    model = network_cls().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    avg_niqe = validate(model, test_loader, device, result_dir)
    print(f'\nAverage NIQE: {avg_niqe:.6f}  (lower is better)')
    print(f'Enhanced images written to: {result_dir}')


if __name__ == '__main__':
    main()

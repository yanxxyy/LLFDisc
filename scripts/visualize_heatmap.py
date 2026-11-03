"""Heatmap visualisation and error analysis.

Renders a JET-colourmapped heatmap of each prediction and of its ground truth,
plus a HOT-colourmapped map of their absolute difference, which makes residual
error visible at a glance. Used for the qualitative figures in the paper.

Usage::

    python scripts/visualize_heatmap.py --pred results/lolv2_real \\
                                        --gt   data/LOLv2/Real_captured/Test/Normal \\
                                        --output figures/lolv2_real_cam
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import IMAGE_EXTENSIONS


def create_heatmap(image_array, colormap=cv2.COLORMAP_JET):
    """Renders a single-channel view of an image as a false-colour heatmap.

    Args:
        image_array: ``(H, W, 3)`` RGB or ``(H, W)`` grayscale ``uint8`` array.
        colormap: One of the ``cv2.COLORMAP_*`` constants.

    Returns:
        ``(H, W, 3)`` BGR ``uint8`` heatmap.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Min-max normalisation so the full colour range is always used.
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, colormap)


def calculate_heatmap_difference(heatmap1, heatmap2):
    """Renders the absolute difference of two heatmaps as its own heatmap.

    Args:
        heatmap1: First heatmap; defines the output resolution.
        heatmap2: Second heatmap; resized to match ``heatmap1`` if needed.

    Returns:
        ``(H, W, 3)`` BGR ``uint8`` difference heatmap (HOT colourmap).
    """
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))

    diff = np.abs(heatmap1.astype(np.float32) - heatmap2.astype(np.float32))
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return create_heatmap(diff_normalized, cv2.COLORMAP_HOT)


def load_image(image_path):
    """Loads an image as an RGB ``uint8`` array, or ``None`` if it cannot be read."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except OSError as e:
        print(f"Failed to load {image_path}: {e}")
        return None


def find_matching_image(pred_filename, gt_dir, strip_suffix=''):
    """Locates the ground-truth counterpart of a prediction file.

    Args:
        pred_filename: Base name of the prediction file.
        gt_dir: Directory holding the ground-truth images.
        strip_suffix: Suffix the inference script appended to the stem (e.g.
            ``"_enhanced"``), removed before matching.

    Returns:
        Path to the matching ground-truth image, or ``None``.
    """
    stem = os.path.splitext(pred_filename)[0]
    if strip_suffix and stem.endswith(strip_suffix):
        stem = stem[:-len(strip_suffix)]

    for ext in IMAGE_EXTENSIONS:
        gt_path = os.path.join(gt_dir, stem + ext)
        if os.path.exists(gt_path):
            return gt_path

    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Render heatmaps and difference maps')
    parser.add_argument('--pred', type=str, required=True,
                        help='Directory holding the predicted (enhanced) images')
    parser.add_argument('--gt', type=str, required=True,
                        help='Directory holding the ground-truth images')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory the heatmaps are written to')
    parser.add_argument('--strip-suffix', type=str, default='',
                        help='Suffix to strip from prediction stems when matching, '
                             'e.g. "_enhanced"')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.pred):
        raise FileNotFoundError(f'Prediction directory not found: {args.pred}')
    if not os.path.isdir(args.gt):
        raise FileNotFoundError(f'Ground-truth directory not found: {args.gt}')

    os.makedirs(args.output, exist_ok=True)

    pred_files = sorted(
        f for f in os.listdir(args.pred) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not pred_files:
        raise FileNotFoundError(f'No images found in {args.pred}')

    print(f'Found {len(pred_files)} prediction images')
    processed_count = 0

    for pred_filename in pred_files:
        print(f'Processing: {pred_filename}')

        pred_image = load_image(os.path.join(args.pred, pred_filename))
        if pred_image is None:
            continue

        gt_path = find_matching_image(pred_filename, args.gt, args.strip_suffix)
        if gt_path is None:
            print(f'  No matching ground-truth image for {pred_filename}')
            continue

        gt_image = load_image(gt_path)
        if gt_image is None:
            continue

        try:
            pred_heatmap = create_heatmap(pred_image, cv2.COLORMAP_JET)
            gt_heatmap = create_heatmap(gt_image, cv2.COLORMAP_JET)
            diff_heatmap = calculate_heatmap_difference(pred_heatmap, gt_heatmap)

            stem = os.path.splitext(pred_filename)[0]
            cv2.imwrite(os.path.join(args.output, f'{stem}_pred_heatmap.png'), pred_heatmap)
            cv2.imwrite(os.path.join(args.output, f'{stem}_gt_heatmap.png'), gt_heatmap)
            cv2.imwrite(os.path.join(args.output, f'{stem}_diff_heatmap.png'), diff_heatmap)

            processed_count += 1
        except Exception as e:
            print(f'  Error processing {pred_filename}: {e}')
            continue

    print(f'\nDone. Processed {processed_count} images.')
    print(f'Heatmaps written to: {args.output}')
    print('File naming:')
    print('  *_pred_heatmap.png - heatmap of the enhanced image')
    print('  *_gt_heatmap.png   - heatmap of the ground-truth image')
    print('  *_diff_heatmap.png - absolute difference between the two')


if __name__ == '__main__':
    main()

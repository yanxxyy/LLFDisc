"""Zoomed-crop figure generation.

Produces the "image plus magnified detail crops" layout used for qualitative
comparisons: the full image on top, with two user-selected regions enlarged
underneath and colour-coded boxes tying them back to their source locations.

Usage::

    python scripts/visualize_crops.py --input figures/ours.png \\
                                      --output figures/crops \\
                                      --box1 5 100 149 75 \\
                                      --box2 400 140 149 75
"""

import argparse
import os
import sys

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gen_simple_heatmap(pil_img):
    """Converts a PIL RGB image into a JET heatmap of the same size.

    Args:
        pil_img: Source image.

    Returns:
        ``(H, W, 3)`` RGB ``uint8`` heatmap.
    """
    gray = np.array(pil_img.convert('L'))
    gray_f = gray.astype(np.float32) / 255.0

    heat_color = cv2.applyColorMap(np.uint8(255 * gray_f), cv2.COLORMAP_JET)
    # cv2 returns BGR; matplotlib expects RGB.
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def save_figure(image_path, out_dir, boxes, scale_factor=2, colors=('red', 'blue'),
                heatmap_crops=True):
    """Renders one image with magnified detail crops appended below it.

    Args:
        image_path: Path to the source image.
        out_dir: Directory the figure is written to.
        boxes: Sequence of ``(x, y, w, h)`` crop regions in pixels.
        scale_factor: Integer magnification applied to each crop.
        colors: Outline colour per box, cycled if shorter than ``boxes``.
        heatmap_crops: If true the crops are taken from a heatmap rendering of
            the image, which exaggerates local contrast differences; if false
            they are taken from the image itself.

    Returns:
        Path to the saved figure.
    """
    raw_pil = Image.open(image_path).convert('RGB')
    raw_np = np.array(raw_pil)

    crop_source = gen_simple_heatmap(raw_pil) if heatmap_crops else raw_np

    # Reserve enough vertical space below the image for the tallest crop.
    max_enlarged_height = max(h * scale_factor for (_x, _y, _w, h) in boxes)
    combined_height = raw_np.shape[0] + max_enlarged_height
    combined_width = raw_np.shape[1]
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Top half: the original image.
    combined_image[:raw_np.shape[0], :raw_np.shape[1]] = raw_np

    enlarged_regions = []
    for i, (x, y, w, h) in enumerate(boxes):
        cropped = crop_source[y:y + h, x:x + w]
        # Nearest-neighbour magnification keeps pixel boundaries crisp.
        enlarged = cropped.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)

        # First crop is flush left, the others flush right.
        x_start = 0 if i == 0 else combined_width - enlarged.shape[1]
        y_start = raw_np.shape[0]

        eh, ew = enlarged.shape[:2]
        combined_image[y_start:y_start + eh, x_start:x_start + ew] = enlarged

        enlarged_regions.append((x_start, y_start, ew, eh, colors[i % len(colors)]))

    plt.figure(figsize=(10, 12))
    plt.imshow(combined_image)
    plt.axis('off')

    # Outline the source regions on the full image.
    for i, (x, y, w, h) in enumerate(boxes):
        plt.gca().add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor=colors[i % len(colors)], facecolor='none'))

    # Outline the magnified crops in matching colours.
    for (x_start, y_start, ew, eh, color) in enlarged_regions:
        plt.gca().add_patch(patches.Rectangle(
            (x_start, y_start), ew, eh, linewidth=3, edgecolor=color, facecolor='none'))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f'{stem}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description='Render zoomed-crop comparison figures')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='One or more source image paths')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory the figures are written to')
    parser.add_argument('--box1', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        default=[5, 100, 149, 75],
                        help='First crop region as x y w h (default: 5 100 149 75)')
    parser.add_argument('--box2', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        default=[400, 140, 149, 75],
                        help='Second crop region as x y w h (default: 400 140 149 75)')
    parser.add_argument('--scale', type=int, default=2,
                        help='Magnification applied to each crop (default: 2)')
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Take the crops from the image itself instead of a heatmap')
    return parser.parse_args()


def main():
    args = parse_args()
    boxes = [tuple(args.box1), tuple(args.box2)]

    for image_path in args.input:
        if not os.path.isfile(image_path):
            print(f'Skipping missing file: {image_path}')
            continue
        out_path = save_figure(image_path, args.output, boxes,
                               scale_factor=args.scale,
                               heatmap_crops=not args.no_heatmap)
        print(f'{image_path} -> {out_path}')


if __name__ == '__main__':
    main()

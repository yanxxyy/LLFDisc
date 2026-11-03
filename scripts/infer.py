"""Batch inference: enhance a folder of low-light images.

Usage::

    python scripts/infer.py --weights path/to/best.pth \\
                            --input  data/LOLv2/Real_captured/Test/Low \\
                            --output results/lolv2_real
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import IMAGE_EXTENSIONS
from models import LLFDisc


def parse_args():
    parser = argparse.ArgumentParser(description='Enhance a folder of low-light images')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory holding the low-light images')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory the enhanced images are written to')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override, e.g. "cuda:0" or "cpu" (default: auto)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many images (0 = no limit)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to the output file stem, e.g. "_enhanced"')
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f'Input directory not found: {args.input}')
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f'Checkpoint not found: {args.weights}')

    os.makedirs(args.output, exist_ok=True)

    print('Loading model...')
    model = LLFDisc().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f'Loaded weights from {args.weights}')

    image_files = sorted(
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        raise FileNotFoundError(f'No images found in {args.input}')
    if args.limit > 0:
        image_files = image_files[:args.limit]

    print(f'Found {len(image_files)} images, starting...')

    with torch.no_grad():
        for idx, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path).convert('RGB')
                stem = os.path.splitext(os.path.basename(img_path))[0]

                # HWC uint8 -> NCHW float in [0, 1].
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)

                enhanced = model(img_tensor)

                enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                # The network ends in tanh, so clip before quantising.
                enhanced_np = np.clip(enhanced_np, 0, 1)

                out_name = f'{stem}{args.suffix}.png'
                Image.fromarray((enhanced_np * 255).astype(np.uint8)).save(
                    os.path.join(args.output, out_name))

                print(f'[{idx + 1}/{len(image_files)}] {os.path.basename(img_path)} -> {out_name}')
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                continue

    print(f'\nDone. Enhanced images written to: {args.output}')


if __name__ == '__main__':
    main()

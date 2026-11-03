"""Model complexity benchmark: parameter count, FLOPs and inference latency.

Usage::

    python scripts/benchmark.py --size 256 --runs 50
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LLFDisc


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark LLFDisc complexity and latency')
    parser.add_argument('--size', type=int, default=256,
                        help='Side length of the square test input (default: 256)')
    parser.add_argument('--runs', type=int, default=50,
                        help='Timed forward passes to average over (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Untimed warm-up passes (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override, e.g. "cuda:0" or "cpu" (default: auto)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    model = LLFDisc().to(device).eval()
    x = torch.rand(1, 3, args.size, args.size, device=device)

    n_param = sum(p.nelement() for p in model.parameters())
    print(f'Parameters: {n_param / 2 ** 20:.4f} M ({n_param:,})')

    # thop is optional: the parameter count and timings are still useful without it.
    try:
        from thop import profile
        macs, _params = profile(model, inputs=(x,), verbose=False)
        print(f'FLOPs: {macs / 2 ** 30:.4f} G (at {args.size}x{args.size})')
    except ImportError:
        print('FLOPs: skipped (install "thop" to measure)')

    with torch.no_grad():
        # Warm-up: the first passes pay for CUDA context setup and autotuning.
        for _ in range(args.warmup):
            model(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.runs):
            model(x)
        # CUDA kernels are asynchronous, so synchronise before reading the clock.
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

    per_run = elapsed / args.runs
    print(f'Latency: {per_run * 1000:.3f} ms/image ({1 / per_run:.2f} FPS), '
          f'averaged over {args.runs} runs')


if __name__ == '__main__':
    main()

# LLFDisc: Low-Light Enhancement with Fourier Distribution Consistency

PyTorch implementation of LLFDisc, a lightweight U-shaped network for low-light
image enhancement trained with a frequency-domain distribution-consistency
objective.

Two ideas drive the method. First, the backbone refines every scale with an
**Enhanced Lightweight Cross-Attention (LCA)** block that couples linear-cost
channel attention with an illumination-enhancement gate, a noise-aware
suppression path and channel recalibration — brightening dark regions without
amplifying the sensor noise that lives there. Second, the objective adds a
**Fourier Gaussian KL** term that matches the *distributions* of the amplitude
and phase spectra rather than their individual coefficients, which restores
global tone and texture statistics while tolerating small spatial
misalignments.

## Contents

- [Results](#results)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference and visualisation](#inference-and-visualisation)
- [Configuration reference](#configuration-reference)
- [Architecture](#architecture)
- [Loss function](#loss-function)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Results

All numbers follow the GT-mean protocol standard in this literature: the
prediction is rescaled so its mean brightness matches the reference before the
metric is computed. PSNR and SSIM are reported from separate checkpoints, since
the two metrics peak at different epochs.

| Dataset | PSNR ↑ | SSIM ↑ |
| --- | --- | --- |
| LOL-v1 | 26.78 | 0.8435 |
| LOL-v2 Real-captured | 28.53 | 0.8836 |
| LOL-v2 Synthetic | 29.39 | 0.9439 |
| LSRW (Huawei) | 23.17 | 0.6332 |

On the unpaired benchmarks (DICM, LIME, MEF, NPE, VV) quality is measured with
NIQE; see [Evaluation](#evaluation).

Run `python scripts/benchmark.py` to report the parameter count, FLOPs and
latency on your own hardware.

## Installation

```bash
git clone https://github.com/<your-username>/LLFDisc.git
cd LLFDisc

conda create -n llfdisc python=3.9
conda activate llfdisc

# Install the PyTorch build matching your CUDA version first; see pytorch.org.
pip install torch torchvision

pip install -r requirements.txt
```

The LPIPS and VGG-19 perceptual terms download pretrained weights on first use,
so the initial run needs network access.

To evaluate NIQE you also need BasicSR's pristine-model parameters. Download
[`niqe_pris_params.npz`](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/niqe_pris_params.npz)
and place it in `utils/`, or point the `LLFDISC_NIQE_PARAMS` environment
variable at it.

## Dataset preparation

Place (or symlink) the datasets under `./datasets/`. The dataloader pairs
low-light and normal-light images by sorted file name, so corresponding files
must share a naming scheme that sorts identically.

```
datasets/
├── LOLv1/
│   ├── Train/{low,high}/
│   └── Test/{low,high}/
├── LOLv2/
│   ├── Real_captured/
│   │   ├── Train/{Low,Normal}/
│   │   └── Test/{Low,Normal}/
│   └── Synthetic/
│       ├── Train/{Low,Normal}/
│       └── Test/{Low,Normal}/
├── LSRW/
│   └── Huawei/
│       ├── Train/{low,high}/
│       └── Eval/{low,high}/
├── DICM/   # unpaired, NIQE only
├── LIME/
├── MEF/
├── NPE/
└── VV/
```

Any other layout works too — just edit the `train_low`, `train_high`,
`test_low` and `test_high` keys in the config file.

## Training

```bash
python train.py --config configs/lol_v1.yaml
```

Available configs: `lol_v1.yaml`, `lol_v2_real.yaml`, `lol_v2_syn.yaml`,
`lsrw_huawei.yaml`. Ablation variants live in `configs/ablation/` — see the
README there.

Training writes to `weight_save_path`: the best-PSNR, best-SSIM and best-LPIPS
weights, a periodic checkpoint every `checkpoint_epoch` epochs, the final-epoch
weights, and a copy of the resolved config. Superseded best-metric files are
deleted as training progresses, so the directory stays small.

Set `write_log: True` to also append a per-epoch log under `./log/`.

**Resuming.** Point `checkpoint_path` at an existing checkpoint and rerun the
same command. Model, optimiser, scheduler, AMP scaler, epoch counter and
best-metric history are all restored, and the script revalidates the restored
weights before continuing. A path that does not exist simply starts a fresh run,
which is why the shipped configs end in `/None`.

**Early stopping.** Training stops once `patience` consecutive epochs pass
without improving either PSNR or SSIM. The shipped values are deliberately large
(150–300), so in practice runs go the full `num_epochs`.

## Evaluation

Reference-based metrics on a paired test set:

```bash
python test.py --config configs/test.yaml
```

Edit `weight_path`, `test_low` and `test_high` in that config to select the
checkpoint and dataset. The script reports per-image and mean PSNR, SSIM and
LPIPS.

No-reference NIQE on an unpaired benchmark:

```bash
python test_niqe.py --config configs/test_niqe.yaml --output-dir results/DICM
```

Enhanced images are saved to the output directory alongside the reported mean
NIQE (lower is better).

## Inference and visualisation

Enhance a folder of images with no ground truth required:

```bash
python scripts/infer.py --weights weights/lol_v1/best_psnr.pth \
                        --input  datasets/LOLv1/Test/low \
                        --output results/lol_v1
```

Measure parameters, FLOPs and latency:

```bash
python scripts/benchmark.py --size 256 --runs 50
```

Render heatmaps of the prediction, the ground truth and their absolute
difference:

```bash
python scripts/visualize_heatmap.py --pred results/lol_v1 \
                                    --gt   datasets/LOLv1/Test/high \
                                    --output figures/lol_v1_cam
```

Render the "full image plus magnified detail crops" figures used for qualitative
comparison:

```bash
python scripts/visualize_crops.py --input figures/ours.png figures/gt.png \
                                  --output figures/crops \
                                  --box1 5 100 149 75 \
                                  --box2 400 140 149 75
```

## Configuration reference

The network and the objective are both resolved at runtime from the config, so
one script covers every experiment. `model_path` / `class_name` name the network
and `loss_path` / `loss_name` name the objective; both are imported with
`importlib`.

| Key | Meaning |
| --- | --- |
| `model_path`, `class_name` | Module and class of the network, e.g. `models.llfdisc` / `LLFDisc` |
| `loss_path`, `loss_name` | Module and class of the objective, e.g. `losses.combined` / `CombinedLossFull` |
| `train_low`, `train_high` | Training image directories |
| `test_low`, `test_high` | Test image directories |
| `weight_save_path` | Where best-metric and final weights are written |
| `checkpoint_save_path` | Where periodic checkpoints are written |
| `checkpoint_path` | Checkpoint to resume from; a non-existent path starts fresh |
| `checkpoint_epoch` | Save a periodic checkpoint every N epochs |
| `weight_path` | Probed for existence at start-up as a sanity check only |
| `device` | e.g. `cuda:0`; falls back to CPU when CUDA is unavailable |
| `batch_size`, `num_epochs` | Batch size and epoch budget |
| `learning_rate`, `min_learning_rate` | Initial LR and the cosine-annealing floor |
| `crop_size` | Random crop size; `0` trains on full images |
| `patience` | Early stop after N epochs without a PSNR or SSIM gain |
| `seed` | Seeds NumPy and Torch, and enables cuDNN determinism |
| `num_workers` | Dataloader worker processes; lower to 0 or 2 on machines with few cores or little RAM |
| `log_name`, `write_log`, `log_epochs` | Logging name, switch and interval |
| `alpha1` … `alpha7` | Loss-term weights, see below |
| `alpha8` | Reserved; read and printed but unused by the objective |
| `alpha_lpips` | Weight of the LPIPS term added by the training loop |

## Architecture

`models/llfdisc.py` defines a three-scale U-shaped encoder/decoder. The encoder
halves the resolution twice while doubling the width (32 → 64 → 128 channels);
the decoder upsamples with transposed convolutions and adds the matching encoder
features back in. Every scale is refined by an `EnhancedLCA` block:

- **`CAB`** — cross-attention that computes similarity across *channels* rather
  than pixels, so cost is linear in resolution. Queries, keys and values are
  produced by depth-wise convolutions, and q/k are L2-normalised so the logits
  stay bounded.
- **`IEL`** — illumination-enhancement layer. A gated feed-forward block whose
  two branches are multiplied together, letting it lift dark regions without
  saturating bright ones.
- **Noise-aware gate** — predicts a per-pixel keep probability that attenuates
  the noise otherwise amplified alongside the signal.
- **Dark-region enhancement** — a residual branch dedicated to the darkest
  areas, reweighted by channel attention.
- **`SEBlock`** — squeeze-and-excitation recalibration. Its output activation is
  `tanh` rather than `sigmoid`, so a channel can be suppressed as well as
  amplified.

The output passes through `tanh`, so predictions must be clamped to `[0, 1]`
before they are saved or scored; every script here does so.

## Loss function

`losses/combined.py` sums seven weighted terms:

| Weight | Term | Purpose |
| --- | --- | --- |
| `alpha1` | Smooth L1 | Pixel-level fidelity |
| `alpha2` | Perceptual (VGG-19) | Texture and structure realism |
| `alpha3` | Histogram | Global tone distribution |
| `alpha4` | Multi-scale SSIM | Multi-scale structural similarity |
| `alpha5` | PSNR | Direct optimisation of the reported metric |
| `alpha6` | Colour | Suppresses global colour cast |
| `alpha7` | Fourier Gaussian KL | Frequency-domain distribution consistency |

The training loop adds an LPIPS term on top, weighted by `alpha_lpips`.

Three variants let you ablate the two distribution-matching terms:

| Class | Perceptual term | Frequency term |
| --- | --- | --- |
| `CombinedLossFull` | VGG feature KL | Fourier Gaussian KL |
| `CombinedLossFMse` | VGG feature KL | Fourier MSE |
| `CombinedLossVgg` | VGG feature MSE | Fourier Gaussian KL |

Note that the histogram term uses `torch.histc`, which is not differentiable; it
contributes no gradient and functions as a monitoring signal.

## Repository layout

```
LLFDisc/
├── configs/                  # YAML experiment configs
│   └── ablation/             # ablation variants
├── data/
│   └── dataloader.py         # paired and unpaired datasets
├── losses/
│   ├── combined.py           # the three CombinedLoss variants
│   ├── common.py             # pixel, SSIM, histogram, colour terms
│   ├── fourier.py            # Fourier Gaussian KL and Fourier MSE
│   └── perceptual.py         # VGG-19 feature MSE and feature KL
├── models/
│   └── llfdisc.py            # CAB, IEL, SEBlock, EnhancedLCA, LLFDisc
├── scripts/
│   ├── benchmark.py          # parameters, FLOPs, latency
│   ├── infer.py              # batch inference
│   ├── visualize_crops.py    # magnified-crop figures
│   └── visualize_heatmap.py  # heatmap and difference maps
├── utils/
│   ├── net_utils.py          # DynamicTanh / DyT normalisation layers
│   ├── niqe_utils.py         # NIQE metric (adapted from BasicSR)
│   ├── scheduler.py          # warm-up plus cosine annealing
│   └── utils.py              # config parsing, dynamic import, metrics
├── train.py                  # training entry point
├── test.py                   # PSNR / SSIM / LPIPS evaluation
└── test_niqe.py              # NIQE evaluation
```

## Citation

If you find this work useful, please cite it:

```bibtex
@article{yan2026using,
  title={Using KL-divergence to focus frequency information in low-light image enhancement},
  author={Yan, XingYang and Huang, Xiaohong and Zhang, Zhao and You, Tian and Xu, Ziheng},
  journal={Pattern Recognition},
  pages={114555},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements

The NIQE implementation in `utils/niqe_utils.py` is adapted from
[BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0). The attention
and gated feed-forward designs draw on
[Restormer](https://github.com/swz30/Restormer) and
[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer). We thank the
authors of the LOL, LOL-v2 and LSRW datasets for making their data available.

## License

Released under the MIT License; see [LICENSE](LICENSE).

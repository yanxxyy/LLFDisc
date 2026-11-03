# Ablation configs

Each file reproduces one row of the ablation study. They differ from
`configs/lol_v2_real.yaml` only in the fields noted below, so any other change
you make to the base config should be mirrored here.

| Config | What changes |
| --- | --- |
| `loss_fourier_mse.yaml` | Fourier Gaussian KL term replaced by a plain Fourier MSE (`CombinedLossFMse`) |
| `loss_vgg_mse.yaml` | VGG feature KL term replaced by a plain feature MSE (`CombinedLossVgg`) |
| `alpha7_0.001.yaml` | Frequency-term weight `alpha7` set to 0.001 |
| `alpha7_0.01.yaml` | Frequency-term weight `alpha7` set to 0.01 |
| `alpha7_0.5.yaml` | Frequency-term weight `alpha7` set to 0.5 |
| `alpha7_1.0.yaml` | Frequency-term weight `alpha7` set to 1.0 |

Run one with:

```bash
python train.py --config configs/ablation/loss_fourier_mse.yaml
```

Architectural ablations (removing CAB, IEL or the SE block) are not expressed as
config keys; edit `EnhancedLCA` in `models/llfdisc.py` to drop the corresponding
sub-module.

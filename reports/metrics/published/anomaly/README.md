# VisA PatchCore frozen-test report

The official one-class VisA test rows were evaluated once on 2026-08-15 after
the model, 256×256 input, and normal-validation thresholds were frozen. Image
and pixel thresholds are the 99% and 99.5% quantiles of normal validation
scores. No anomalous test image or mask was used for calibration.

| Category | Test images | Image AUROC | Pixel AUROC | Dice | IoU | Normal FPR | CPU p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candle | 200 | 0.9553 | 0.9887 | 0.0575 | 0.0296 | 1.00% | 472.47 / 482.12 ms |
| capsules | 160 | 0.6667 | 0.9816 | 0.2920 | 0.1709 | 1.67% | 320.79 / 433.29 ms |
| pcb1 | 200 | 0.8798 | 0.9948 | 0.2053 | 0.1144 | 3.00% | 434.31 / 445.29 ms |
| **Macro mean** | — | **0.8339** | **0.9884** | **0.1849** | **0.1050** | **1.89%** | — |

CPU benchmark: Intel i5-12600KF, batch 1, 10 warmups, 100 images per category,
end-to-end wall time. Peak process RSS ranged from 1.62 to 2.06 GB. Training
used an RTX 5060 Ti for feature extraction; coreset selection is CPU-bound.

The exact machine-readable values and checkpoint hashes are in
[`evaluation_summary.json`](evaluation_summary.json). This directory contains
no VisA image, mask, workstation path, or model file. VisA is CC BY 4.0; see the
[official repository](https://github.com/amazon-science/spot-diff).

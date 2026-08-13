# v0.2 validation-only ablation results

The registered augmentation/resolution ablation did **not** promote a v2
model. All model and threshold choices used only the validation split; no test
metric was read or generated for these candidates.

Seven new runs (three seed-42 screen runs plus four confirmation runs) used
6,762 seconds, or 1.878 GPU-hours, on the RTX 5060 Ti. The existing v1 baseline
was reused and was not retrained.

## Seed 42 screen

| Configuration | Input | Validation mAP50-95 | Weak-class mean AP50-95 | Continued? |
|---|---:|---:|---:|---|
| weak-640 | 640 | 0.39648 | 0.18342 | Yes |
| weak-no-flip-640 | 640 | 0.39644 | 0.20310 | Yes |
| efficient-512 | 512 | 0.39535 | 0.18004 | No |

The weak-class score is the mean AP50-95 of `crazing` and
`rolled-in_scale`. The top two configurations continued to seeds 43 and 44 as
registered before training.

## Three-seed confirmation

| Configuration | Seeds | Validation mAP50-95 mean ± population std | Weak-class mean ± population std |
|---|---|---:|---:|
| baseline v1 | 42 | 0.42095 (single formal run) | 0.18438 |
| weak-640 | 42/43/44 | 0.40286 ± 0.00458 | 0.18334 ± 0.00486 |
| weak-no-flip-640 | 42/43/44 | 0.39468 ± 0.00195 | 0.18347 ± 0.01501 |

The best candidate, `weak-640`, changed overall validation mAP50-95 by
`-0.01809` and the weak-class mean by `-0.00104` relative to v1. It therefore
passed neither registered promotion route:

- overall mAP50-95 improvement of at least `+0.010`; or
- weak-class improvement of at least `+0.025` with overall degradation no
  worse than `-0.005`.

The project keeps `neu-det-yolo26n-v1`. No v2 test evaluation, ONNX export, or
latency benchmark was run. Machine-readable evidence is in the
[`screen`](screen/screen_ranking.json) and
[`final`](final/experiment_comparison.json) reports; each observation records
the validation-report and checkpoint SHA-256.

This negative result suggests that merely weakening augmentation or removing
vertical flips is insufficient. A future experiment should target the dominant
localization failures or change the detection architecture, while preserving
the same validation-only selection discipline.

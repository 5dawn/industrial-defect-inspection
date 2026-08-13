# v2 registered ablation matrix

The data split, COCO initialization, epoch cap (100), early-stopping patience
(20), deterministic mode, and AMP policy are fixed across runs. The existing
v1 result from `configs/train/yolo26n.yaml` is the baseline and is not retrained.

| Configuration | Input | translate | scale | hsv_v | mosaic | flipud |
|---|---:|---:|---:|---:|---:|---:|
| Baseline v1 | 640 | 0.08 | 0.20 | 0.15 | 0.25 | 0.5 |
| weak-640 | 640 | 0.04 | 0.10 | 0.08 | 0.0 | 0.5 |
| weak-no-flip-640 | 640 | 0.04 | 0.10 | 0.08 | 0.0 | 0.0 |
| efficient-512 | 512 | 0.04 | 0.10 | 0.08 | 0.0 | 0.5 |

Run every candidate with seed 42. Rank only by validation mAP50-95, then use
the registered weak-class/CPU/model-size tie breakers. Confirm the top two with
seeds 43 and 44. Do not inspect test results while selecting a configuration.

A candidate is promoted only when its three-seed mean validation mAP50-95 is
at least 0.010 above v1, or its mean AP50-95 over `crazing` and
`rolled-in_scale` improves by at least 0.025 while overall mAP50-95 falls no
more than 0.005.

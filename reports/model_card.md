# Model Card: neu-det-yolo26n-v1

> Status: formal local evaluation completed on 2026-08-11. Weights are not yet
> published in a GitHub Release.

## Model

- Architecture: Ultralytics YOLO26n detection
- Initialization: COCO pretrained weights
- Input: 640×640, grayscale source copied to three channels by the loader
- Output classes: six NEU-DET defect classes
- License: AGPL-3.0-only under the selected Ultralytics open-source route

## Training

- Dataset: NEU-DET; see `reports/dataset_card.md`
- Split: label-stratified 70/15/15, seed 42
- Configuration: `configs/train/yolo26n.yaml`
- Hardware: NVIDIA GeForce RTX 5060 Ti 16 GB; Intel Core i5-12600KF
- PyTorch / CUDA / Ultralytics: 2.12.1+cu130 / CUDA 13.0 / 8.4.117
- Source state: base commit `69476f7`; the run manifest also records the exact dirty-tree diff hash
- Training duration: 1,144.62 seconds (about 19.1 minutes)
- Epochs: 98 completed; early stopping selected epoch 78
- Batch: Ultralytics AutoBatch selected 22 at 640×640

## Evaluation

| Metric | Validation | Frozen test |
|---|---:|---:|
| mAP50 | 0.733 | 0.776 |
| mAP50-95 | 0.421 | 0.441 |
| Precision @ conf 0.43, IoU 0.5 | 0.777 | 0.775 |
| Recall @ conf 0.43, IoU 0.5 | 0.677 | 0.701 |
| F1 @ conf 0.43, IoU 0.5 | 0.723 | 0.736 |

- Selected confidence threshold: 0.43, selected only on validation micro-F1
- PyTorch CPU p50/p95: 43.38 / 46.82 ms
- ONNX Runtime CPU p50/p95: 22.98 / 26.19 ms
- PyTorch GPU p50/p95: 13.26 / 15.76 ms
- Benchmark protocol: batch 1, 10 warmups, 100 frozen test images, end-to-end wall time
- PT/ONNX comparison: 24/24 detections matched on 20 test images; mean box IoU
  0.999999; maximum confidence delta 0.000015
- Test operating-point errors: 457 TP, 133 FP, 195 FN

The full aggregate report, per-class table, threshold sweep, training curves,
PR curve, and confusion matrix are under `reports/metrics/published` and
`reports/figures/published`. Raw FP/FN images remain local because NEU-DET does
not state a standard redistribution license.

## Intended use and limitations

The model is intended for research, learning, and portfolio demonstration on
images similar to NEU-DET. It has not been calibrated against a real production
line and must not be used as the sole basis for accepting or rejecting products.

Known limitations include a small training set, low-resolution grayscale
imagery, uncertain performance on normal products, and potential sensitivity to
camera, lighting, material, and preprocessing changes.

## Release contents

- `best.pt`
- `model.onnx` (FP32, fixed 640×640 input)
- `evaluation.json`
- SHA-256 checksums
- completed model card and aggregate evaluation report

The source dataset must not be included in the model release.

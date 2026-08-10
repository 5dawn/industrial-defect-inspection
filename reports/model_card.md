# Model Card: neu-det-yolo26n-v1

> Status: template. Complete all pending fields before publishing weights.

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
- Hardware: pending
- PyTorch / CUDA / Ultralytics versions: pending
- Git commit: pending
- Training duration: pending

## Evaluation

| Metric | Validation | Test |
|---|---:|---:|
| mAP50 | pending | pending |
| mAP50-95 | pending | pending |
| Precision | pending | pending |
| Recall | pending | pending |

- Selected confidence threshold: pending
- PyTorch CPU p50/p95: pending
- ONNX CPU p50/p95: pending
- PT/ONNX comparison on 20 test images: pending

Attach the generated per-class table, confusion matrix, and representative
false-positive/false-negative samples to the release.

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
- completed model card

The source dataset must not be included in the model release.


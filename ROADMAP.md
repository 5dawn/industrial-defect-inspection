# Roadmap

## v0.1 — Detection pipeline

- [x] NEU-DET VOC validation and deterministic split
- [x] YOLO-format conversion and annotation previews
- [x] YOLO26n train, validation, test, and ONNX entry points
- [x] Shared `.pt`/`.onnx` inference contract
- [x] FastAPI and Gradio local demo
- [x] Synthetic tests and cross-platform CI
- [x] Train the formal local checkpoint and freeze test metrics
- [x] Publish aggregate metrics, checksums, completed model card, and demo GIF
- [x] Create a source-only v0.1.1 evidence release without dataset-derived weights

## v0.2 — Error-driven model optimization

- [x] Compare PyTorch and ONNX on at least 20 test images.
- [x] Require matching classes and matched-box IoU of at least 0.95.
- [x] Benchmark 100 warmed, batch-one images on named CPU and GPU hardware.
- [x] Quantify validation errors by class, box size, confidence, and IoU.
- [x] Screen the fixed weak-augmentation and 512-pixel experiment matrix.
- [x] Confirm the top configurations over seeds 42, 43, and 44.
- [x] Apply the frozen validation gate; retain v1 because no candidate passed.

## v0.3 — Anomaly localization extension

Use the CC BY 4.0 VisA dataset without mixing its samples or metrics with the
NEU-DET detection benchmark.

- [x] Add task-specific VisA data, model, evaluation, and inference configurations.
- [x] Prepare `candle`, `capsules`, and `pcb1` while preserving official test rows.
- [x] Implement a separate `AnomalyResult`, PatchCore adapter, heatmap, mask, and overlay.
- [x] Calibrate image/pixel thresholds using normal-only validation quantiles.
- [x] Add image/pixel AUROC, Dice, IoU, normal FPR, latency, and RAM reporting.
- [x] Add anomaly API routes and a second Gradio mode without changing detection contracts.
- [x] Run all three real PatchCore fits and publish frozen-test metrics.
- [ ] Create a VisA-attributed checkpoint release and anomaly Demo GIF.

Acceptance requires reproducible preparation, three evaluated categories,
saved qualitative successes and failures, and no VisA files in Git history.

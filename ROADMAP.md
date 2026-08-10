# Roadmap

## v0.1 — Detection pipeline

- [x] NEU-DET VOC validation and deterministic split
- [x] YOLO-format conversion and annotation previews
- [x] YOLO26n train, validation, test, and ONNX entry points
- [x] Shared `.pt`/`.onnx` inference contract
- [x] FastAPI and Gradio local demo
- [x] Synthetic tests and cross-platform CI
- [ ] Train the public checkpoint and freeze test metrics
- [ ] Publish weights, checksums, completed model card, and demo GIF

## v0.2 — Backend verification

- Compare PyTorch and ONNX on at least 20 test images.
- Require matching classes and matched-box IoU of at least 0.95.
- Benchmark 100 warmed, batch-one images on named CPU and GPU hardware.
- Add the backend comparison to `reports/model_card.md`.

## v0.3 — Anomaly localization extension

Use the CC BY 4.0 VisA dataset without mixing its samples or metrics with the
NEU-DET detection benchmark.

1. Add a task-specific VisA data configuration and source record.
2. Start with `candle`, `capsules`, and `pcb1`, preserving the published split.
3. Implement a compact anomaly-localization backend that returns a normalized
   heatmap and binary mask through a separate result schema.
4. Select the mask threshold on validation data only.
5. Report pixel AUROC, Dice, IoU, and p50/p95 latency by category.
6. Add a detection/heatmap mode selector to the existing demo while keeping the
   two evaluation reports independent.

Acceptance requires reproducible preparation, three evaluated categories,
saved qualitative successes and failures, and no VisA files in Git history.


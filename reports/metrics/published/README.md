# Published evaluation evidence

These files contain aggregate, reproducible evidence from the formal local run
completed on 2026-08-11. They do not contain NEU-DET images or annotations.

- `data/dataset_summary.json`: aggregate dataset and split statistics.
- `experiment/experiment_summary.json`: training provenance, validation/test
  metrics, backend parity, latency, and resource measurements.
- `experiment/test_per_class.csv`: per-class test P/R/F1/AP values.
- `../../figures/published/`: data distributions, training curves, threshold
  sweep, PR curve, and normalized confusion matrix.

Metric definitions:

- mAP is collected with confidence 0.001 across the full PR curve.
- Precision, Recall, and F1 are micro-averaged at IoU 0.5 and the validation-
  selected confidence 0.43.
- Latency is batch=1 end-to-end wall time over 100 frozen test images after 10
  warmups on the hardware recorded in `experiment_summary.json`.

The test threshold was not selected or adjusted on the test split. Raw FP/FN
images remain local because the upstream dataset does not state a standard
redistribution license.

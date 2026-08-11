# Dataset Card: NEU-DET

## Summary

NEU-DET is the detection subset of the Northeastern University surface defect
database. Its official page describes 1,800 grayscale images at 200×200 pixels,
with 300 samples for each of six hot-rolled steel defect classes.

Classes used by this project:

1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled-in_scale
6. scratches

Official source:
https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

## Intended use

- Supervised object-detection education and research.
- Reproducible portfolio demonstration.
- Evaluation of lightweight models and local CPU inference.

The dataset is not sufficient evidence for deployment on a production line. In
particular, it is small, low resolution, and does not provide a representative
set of normal production images.

## Preparation

`idi-prepare` pairs image and XML stems, validates Pascal VOC annotations,
checks image/XML dimensions and duplicate content, and converts boxes to YOLO
format. Seed 42 produces a label-stratified 70/15/15 split. Augmentation is
applied by the trainer only to the training split.

The generated `metadata.json` is the source of truth for actual sample and box
counts. The 2026-08-11 preparation run found 1,800 images and 4,186 boxes, split
into 1,260/270/270 images and 2,906/628/652 boxes for train/validation/test.
Three duplicate XML boxes were removed by the deterministic annotation audit.

One byte-identical image pair, `patches_101` and `patches_105`, has different
annotations. The pair is retained in the same split to prevent content leakage,
and the pipeline emits an explicit warning. The source labels are not silently
rewritten. Aggregate-only statistics and plots are published under
`reports/metrics/published/data` and `reports/figures/published/data`.

## Licensing and distribution

The official download page requests citation but does not state a recognized
standard dataset license. This repository therefore:

- does not include images, annotations, or repackaged archives;
- links users to the official source;
- records archive checksums and access date locally;
- requires the original works to be cited in reports using the data.

Users are responsible for reviewing the current upstream terms before use or
distribution. This dataset card is not legal advice.

## Known limitations

- Similar backgrounds or acquisition conditions can inflate random-split scores.
- Class appearances overlap and vary with illumination.
- Grayscale data differs from the RGB data used for common pretrained weights.
- Absence of representative normal images limits false-alarm evaluation.
- Bounding-box quality must be audited after download; it is not assumed perfect.

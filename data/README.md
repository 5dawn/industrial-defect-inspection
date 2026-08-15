# Dataset setup

The repository intentionally does not redistribute NEU-DET or other datasets.

1. Open the [official NEU surface defect database page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm).
2. Download **NEU-DET** (the image-with-annotations archive).
3. Arrange the extracted files as follows:

```text
data/raw/neu_det/
├── images/             # image files; nested folders are accepted
├── annotations_xml/    # Pascal VOC XML files; nested folders are accepted
└── SOURCE.md           # copy SOURCE.template.md and complete it
```

4. Run `idi-prepare --config configs/data/neu_det.yaml`.

If either source directory is missing, the command stops with the expected
paths and points back to this setup step. Do not create empty placeholder image
or annotation files.

The preparation command validates annotations, checks duplicate image content,
creates a deterministic stratified split, converts boxes to YOLO format, and
writes a dataset report. Generated files under `data/processed` are ignored by
Git.

The NEU-DET download page does not state a standard dataset license. Do not
upload the dataset or a repackaged copy to this repository. Review the original
terms and cite the dataset authors when publishing results.

## VisA anomaly localization

VisA is distributed under CC BY 4.0. Download it from the
[official Amazon Science repository](https://github.com/amazon-science/spot-diff)
or its linked AWS Open Data location. Also copy the official
`split_csv/1cls.csv` into the raw directory:

```text
data/raw/visa/
├── candle/Data/Images/{Normal,Anomaly}/
├── candle/Data/Masks/Anomaly/
├── capsules/Data/Images/{Normal,Anomaly}/
├── capsules/Data/Masks/Anomaly/
├── pcb1/Data/Images/{Normal,Anomaly}/
├── pcb1/Data/Masks/Anomaly/
├── split_csv/1cls.csv
└── SOURCE.md
```

The extracted archive may contain the other VisA categories; the v0.3 config
only processes `candle`, `capsules`, and `pcb1`. Run:

```powershell
idi-prepare-visa --config configs/data/visa.yaml
```

The command validates every selected image/mask pair, applies the same
aspect-ratio-preserving 256×256 transform to both, reserves a deterministic
normal-only validation subset, and records hashes. Official test rows remain
unchanged. Generated VisA files are ignored by Git and must not be committed.

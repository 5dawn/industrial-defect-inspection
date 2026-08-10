# Industrial Defect Inspection

[English](../README.md) | 简体中文

这是一个面向计算机视觉实习与开源作品集的工业表面缺陷检测项目。项目使用
PyTorch 和 YOLO26n，在 NEU-DET 六类钢材缺陷数据上提供可复现的数据准备、训练、
验证和单图推理流程，并保留 ONNX、FastAPI 与 Gradio 扩展能力。

> 本项目用于研究与作品集展示，尚未经过真实产线验证，不能作为生产质检或安全关键
> 决策系统。

## 当前状态

工程 MVP 已完成。仓库不会提交数据集、训练权重或本地实验产物；测试集正式指标、
CPU 延迟和 Release 权重将在完成冻结测试评估后发布。在此之前，英文首页中的指标保持
`pending`，2 epochs 冒烟训练结果不视为正式模型成绩。

## 功能

- 检查 Pascal VOC 图片/XML 配对、类别、尺寸、非法框、重复框和重复图片内容。
- 使用固定随机种子完成 70%/15%/15% 训练、验证、测试划分。
- 将 VOC 标注转换为 YOLO 格式，并生成元数据、清单和标注预览。
- 通过 YAML 配置运行 YOLO26n 冒烟训练、正式训练和断点恢复。
- 在验证集或冻结测试集上输出指标、逐类结果、FP/FN 样例和延迟报告。
- 对单张图片输出标注图和结构化 JSON。
- 复用统一推理引擎支持 PyTorch、ONNX、FastAPI 和 Gradio。

## 目录结构

```text
configs/        数据、训练、验证和推理配置
data/           数据下载说明，以及被 Git 忽略的原始/转换数据目录
src/            可安装的 Python 包与命令行入口
tests/          不下载正式数据或权重的合成测试
reports/        数据集卡、模型卡、指标和图表目录
artifacts/      被 Git 忽略的权重、导出模型和预测结果
```

## Windows 安装

项目推荐 Python 3.11：

```powershell
git clone https://github.com/5dawn/industrial-defect-inspection.git
cd industrial-defect-inspection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

先通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装与电脑
CPU 或 CUDA 环境匹配的 PyTorch，再安装项目：

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements-lock-cu130.txt` 只记录一个经过验证的 CUDA 13.0 环境，不是所有电脑的
通用安装入口。

## 准备 NEU-DET

数据集不随仓库分发。请从
[NEU 官方页面](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
下载带检测标注的 NEU-DET，并整理为：

```text
data/raw/neu_det/
|-- images/
|-- annotations_xml/
`-- SOURCE.md
```

复制 `data/SOURCE.template.md` 为 `SOURCE.md`，填写来源、日期和文件哈希，然后运行：

```powershell
idi-prepare --config configs/data/neu_det.yaml
```

准备命令默认拒绝覆盖非空输出目录；确认需要重新生成时显式增加 `--overwrite`。

## 训练

首先运行 CPU 冒烟配置，确认环境、数据加载和 checkpoint 写入正常：

```powershell
idi-train --config configs/train/smoke.yaml
```

冒烟配置仅运行 2 epochs。正式实验使用：

```powershell
idi-train --config configs/train/yolo26n.yaml
```

## 验证

模型、数据集、输出目录、split 和推理参数由 `configs/eval/default.yaml` 管理。可以修改
配置，也可以通过命令行临时覆盖模型路径：

```powershell
idi-evaluate --config configs/eval/default.yaml `
  --model artifacts/runs/smoke/weights/best.pt `
  --split val
```

只有在模型、阈值和输入尺寸全部冻结后，才能将 `--split test` 的结果写入模型卡和
英文首页。

## 单图推理

```powershell
idi-predict --config configs/infer/default.yaml `
  --model artifacts/runs/smoke/weights/best.pt `
  --source path\to\image.jpg
```

结果默认写入 `artifacts/predictions/`，包含标注 JPEG 和结构化 JSON。模型文件或输入
图片不存在时，命令会在耗时推理开始前返回明确错误。

## Web Demo

Web 页面与命令行共用同一个 `InferenceEngine`。使用本地 smoke checkpoint 明确启动
CPU 推理的命令为：

```powershell
idi-web --config configs/infer/default.yaml `
  --model artifacts/runs/smoke/weights/best.pt `
  --device cpu
```

打开 <http://127.0.0.1:7860/demo/>，上传一张 JPEG、PNG 或 WebP 图片，调整置信度后点击
**Run inspection**。页面会显示标注图、缺陷类别、置信度、检测框坐标、预处理/推理/
后处理耗时、设备和模型版本，并允许下载标注图片及 JSON。输出目录由
`configs/infer/default.yaml` 中的 `output_dir` 管理。

Smoke 权重仅用于验证流程，不代表正式模型成绩。模型文件不存在时服务仍会以降级模式
启动，页面显示期望路径和 `--model` 修复命令；`GET /health` 返回 `degraded`，推理请求
返回友好提示。上传限制为 10 MB，损坏文件、伪造图片和不支持的格式会在模型推理前被
拒绝。

## 检查

```powershell
ruff format --check .
ruff check .
pytest -q
```

## 数据与许可

NEU-DET 官方页面未提供清晰的标准数据集许可证，因此仓库只提供来源链接、转换工具和
引用说明，不重新分发原始数据。项目代码采用 AGPL-3.0-only；更多信息见
[`reports/dataset_card.md`](../reports/dataset_card.md) 和
[`reports/model_card.md`](../reports/model_card.md)。

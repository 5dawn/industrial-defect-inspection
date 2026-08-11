"""End-to-end latency, resource, and PT/ONNX parity benchmarks."""

from __future__ import annotations

import argparse
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from industrial_defect_inspection.evaluation.dataset import dataset_images
from industrial_defect_inspection.evaluation.metrics import BBox, box_iou
from industrial_defect_inspection.utils.io import file_record, write_json
from industrial_defect_inspection.utils.runtime import prepare_runtime, resolve_device


class MemorySampler:
    """Sample process RSS while a benchmark is running."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.baseline_bytes = self.process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _parameter_count(model: Any) -> int | None:
    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "parameters"):
        return None
    return sum(parameter.numel() for parameter in core.parameters())


def benchmark_model(
    model_path: Path,
    images: list[Path],
    device: str,
    confidence: float,
    image_size: int,
    count: int = 100,
    warmup: int = 10,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    selected = images[:count]
    if not selected:
        raise ValueError("No benchmark images were provided")
    prepare_runtime(model_path.parent)
    from ultralytics import YOLO

    resolved_device = resolve_device(device)
    model = YOLO(str(model_path))
    for index in range(warmup):
        model.predict(
            str(selected[index % len(selected)]),
            conf=confidence,
            imgsz=image_size,
            device=resolved_device,
            verbose=False,
        )

    cuda_allocated = 0
    cuda_reserved = 0
    try:
        import torch

        use_cuda = resolved_device != "cpu" and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except ImportError:
        torch = None  # type: ignore[assignment]
        use_cuda = False

    wall_times: list[float] = []
    components = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    with MemorySampler() as memory:
        for image_path in selected:
            started = time.perf_counter()
            raw = model.predict(
                str(image_path),
                conf=confidence,
                imgsz=image_size,
                device=resolved_device,
                verbose=False,
            )[0]
            if use_cuda:
                torch.cuda.synchronize()
            wall_times.append((time.perf_counter() - started) * 1000)
            speed = raw.speed or {}
            for name in components:
                components[name].append(max(0.0, float(speed.get(name.removesuffix("_ms"), 0.0))))
    if use_cuda:
        cuda_allocated = int(torch.cuda.max_memory_allocated())
        cuda_reserved = int(torch.cuda.max_memory_reserved())

    mean_wall = statistics.fmean(wall_times)
    return {
        "model": file_record(model_path),
        "device": str(resolved_device),
        "image_size": image_size,
        "confidence": confidence,
        "count": len(selected),
        "warmup": warmup,
        "wall_ms": {
            "mean": mean_wall,
            "p50": statistics.median(wall_times),
            "p95": percentile(wall_times, 0.95),
        },
        "throughput_fps": 1000 / mean_wall if mean_wall else 0.0,
        "components_mean_ms": {
            name: statistics.fmean(values) for name, values in components.items()
        },
        "resources": {
            "baseline_rss_mb": memory.baseline_bytes / 1024**2,
            "peak_rss_mb": memory.peak_bytes / 1024**2,
            "rss_delta_mb": (memory.peak_bytes - memory.baseline_bytes) / 1024**2,
            "cuda_peak_allocated_mb": cuda_allocated / 1024**2,
            "cuda_peak_reserved_mb": cuda_reserved / 1024**2,
            "parameter_count": _parameter_count(model),
        },
    }


def _predictions(raw: Any) -> list[tuple[int, float, BBox]]:
    if raw.boxes is None:
        return []
    return [
        (int(class_id), float(confidence), tuple(float(value) for value in bbox))
        for class_id, confidence, bbox in zip(
            raw.boxes.cls.detach().cpu().tolist(),
            raw.boxes.conf.detach().cpu().tolist(),
            raw.boxes.xyxy.detach().cpu().tolist(),
            strict=True,
        )
    ]


def compare_backends(
    pt_model: Path,
    onnx_model: Path,
    images: list[Path],
    confidence: float,
    image_size: int,
    count: int = 20,
) -> dict[str, Any]:
    prepare_runtime(onnx_model.parent)
    from ultralytics import YOLO

    pt = YOLO(str(pt_model))
    onnx = YOLO(str(onnx_model))
    total_pt = 0
    total_onnx = 0
    matches = 0
    overlaps: list[float] = []
    confidence_deltas: list[float] = []
    for image_path in images[:count]:
        pt_predictions = _predictions(
            pt.predict(
                str(image_path), conf=confidence, imgsz=image_size, device="cpu", verbose=False
            )[0]
        )
        onnx_predictions = _predictions(
            onnx.predict(
                str(image_path), conf=confidence, imgsz=image_size, device="cpu", verbose=False
            )[0]
        )
        total_pt += len(pt_predictions)
        total_onnx += len(onnx_predictions)
        used: set[int] = set()
        for class_id, score, bbox in pt_predictions:
            candidates = [
                (index, box_iou(bbox, candidate_bbox))
                for index, (candidate_class, _, candidate_bbox) in enumerate(onnx_predictions)
                if index not in used and candidate_class == class_id
            ]
            if not candidates:
                continue
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap <= 0:
                continue
            used.add(index)
            matches += 1
            overlaps.append(overlap)
            confidence_deltas.append(abs(score - onnx_predictions[index][1]))
    denominator = max(total_pt, total_onnx, 1)
    result = {
        "count": min(count, len(images)),
        "pt_detections": total_pt,
        "onnx_detections": total_onnx,
        "matched_detections": matches,
        "matched_fraction": matches / denominator,
        "mean_box_iou": statistics.fmean(overlaps) if overlaps else 0.0,
        "minimum_box_iou": min(overlaps) if overlaps else 0.0,
        "mean_confidence_delta": statistics.fmean(confidence_deltas) if confidence_deltas else 0.0,
        "maximum_confidence_delta": max(confidence_deltas) if confidence_deltas else 0.0,
    }
    result["passed"] = bool(
        result["matched_fraction"] >= 0.95
        and result["mean_box_iou"] >= 0.95
        and result["maximum_confidence_delta"] <= 0.05
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PT/ONNX industrial detectors.")
    parser.add_argument("--pt-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--data", default="data/processed/neu_det/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="reports/metrics/published/backend_benchmark.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = dataset_images(Path(args.data), args.split)
    pt_path = Path(args.pt_model)
    onnx_path = Path(args.onnx_model)
    report: dict[str, Any] = {
        "pt_cpu": benchmark_model(
            pt_path, images, "cpu", args.confidence, args.image_size, args.count, args.warmup
        ),
        "onnx_cpu": benchmark_model(
            onnx_path, images, "cpu", args.confidence, args.image_size, args.count, args.warmup
        ),
        "parity": compare_backends(
            pt_path, onnx_path, images, args.confidence, args.image_size, count=20
        ),
    }
    try:
        import torch

        if torch.cuda.is_available():
            report["pt_gpu"] = benchmark_model(
                pt_path, images, "0", args.confidence, args.image_size, args.count, args.warmup
            )
    except ImportError:
        pass
    write_json(Path(args.output), report)
    print(f"Benchmark report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

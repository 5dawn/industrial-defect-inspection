"""Run the combined FastAPI and Gradio local demo."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from industrial_defect_inspection.config import load_inference_config
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.utils.io import write_json
from industrial_defect_inspection.web.api import EngineProtocol, create_app
from industrial_defect_inspection.web.uploads import UploadValidationError, decode_upload


class DemoError(RuntimeError):
    """A user-facing error raised by the web prediction adapter."""


def model_unavailable_message(model_path: Path) -> str:
    """Return actionable guidance for a missing checkpoint."""
    return (
        f"Model is unavailable at {model_path}. Place trained weights at that path "
        "or restart with --model PATH."
    )


def model_status_markdown(engine: EngineProtocol, startup_error: str | None = None) -> str:
    """Build the persistent status banner shown above the demo controls."""
    if startup_error:
        return f"> ⚠️ **Model unavailable:** {startup_error}"
    if not engine.config.model.is_file():
        return f"> ⚠️ **Model unavailable:** {model_unavailable_message(engine.config.model)}"
    state = "loaded" if engine.loaded else "ready for lazy loading"
    return f"> ✅ **Model {state}:** `{engine.config.model}` · device `{engine.device}`"


def run_demo_prediction(
    payload: bytes | None,
    confidence: float,
    engine: EngineProtocol,
    output_dir: Path,
) -> tuple[Any, list[list[Any]], dict[str, Any], str, str, str]:
    """Decode one upload, call the shared engine, and prepare Gradio outputs."""
    if payload is None:
        raise DemoError("Upload a JPEG, PNG, or WebP image before running inspection.")
    try:
        image, original_size, resized = decode_upload(payload)
        result, annotated = engine.predict(image, confidence=confidence)
        result = result.model_copy(
            update={
                "original_image_width": original_size[0],
                "original_image_height": original_size[1],
                "resized": resized,
            }
        )
    except UploadValidationError as exc:
        raise DemoError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise DemoError(model_unavailable_message(engine.config.model)) from exc
    except RuntimeError as exc:
        raise DemoError(f"Model inference is unavailable: {exc}") from exc

    rows = [
        [
            detection.class_name,
            round(detection.confidence, 4),
            *[round(value, 1) for value in detection.bbox_xyxy],
        ]
        for detection in result.detections
    ]
    counts = Counter(detection.class_name for detection in result.detections)
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    image_path = output_dir / f"{identifier}_annotated.jpg"
    json_path = output_dir / f"{identifier}.json"
    annotated.save(image_path, quality=92)
    write_json(json_path, result.model_dump(mode="json"))

    summary = (
        f"**{len(result.detections)} detection(s)** · **total {result.total_ms:.1f} ms**  \n"
        f"preprocess {result.preprocess_ms:.1f} ms · "
        f"inference {result.inference_ms:.1f} ms · "
        f"postprocess {result.postprocess_ms:.1f} ms · "
        f"device `{result.device}` · model `{result.model_version}`"
    )
    if counts:
        summary += "  \n" + ", ".join(f"{name}: {count}" for name, count in counts.items())
    else:
        summary += "  \nNo defects detected at the selected confidence threshold."
    return (
        annotated,
        rows,
        result.model_dump(mode="json"),
        summary,
        str(image_path.resolve()),
        str(json_path.resolve()),
    )


def create_gradio_demo(
    engine: InferenceEngine,
    output_dir: Path,
    startup_error: str | None = None,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to run the web demo") from exc

    def run_prediction(payload: bytes | None, confidence: float):
        try:
            return run_demo_prediction(payload, confidence, engine, output_dir)
        except DemoError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Industrial Defect Inspection") as demo:
        gr.Markdown(
            "# Industrial Defect Inspection\n"
            "Upload a steel-surface image and inspect detected defects. "
            "Research and portfolio demo; not a production quality-control system."
        )
        gr.Markdown(model_status_markdown(engine, startup_error))
        with gr.Row():
            with gr.Column():
                input_image = gr.File(
                    file_count="single",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="binary",
                    label="Upload one image",
                )
                confidence = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=engine.config.confidence,
                    step=0.05,
                    label="Confidence threshold",
                )
                submit = gr.Button("Run inspection", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Annotated result")
                summary = gr.Markdown()
        detections = gr.Dataframe(
            headers=["class", "confidence", "xmin", "ymin", "xmax", "ymax"],
            datatype=["str", "number", "number", "number", "number", "number"],
            interactive=False,
            label="Detections",
        )
        raw_json = gr.JSON(label="Structured result")
        with gr.Row():
            image_download = gr.File(label="Download annotated image")
            json_download = gr.File(label="Download JSON")
        submit.click(
            fn=run_prediction,
            inputs=[input_image, confidence],
            outputs=[
                output_image,
                detections,
                raw_json,
                summary,
                image_download,
                json_download,
            ],
        )
    return demo


def build_application(
    engine: InferenceEngine,
    output_dir: Path,
    startup_error: str | None = None,
):
    import gradio as gr

    api = create_app(engine)
    demo = create_gradio_demo(engine, output_dir, startup_error)
    demo.queue(default_concurrency_limit=1)
    return gr.mount_gradio_app(api, demo, path="/demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local industrial inspection demo.")
    parser.add_argument("--config", default="configs/infer/default.yaml")
    parser.add_argument("--model", help="Override the model path from config.")
    parser.add_argument("--device", help="Override the inference device, for example cpu or 0.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_inference_config(args.config)
    if args.model:
        config = config.model_copy(update={"model": Path(args.model)})
    if args.device:
        config = config.model_copy(update={"device": args.device})
    engine = InferenceEngine(config)

    startup_error: str | None = None
    if not config.model.is_file():
        startup_error = model_unavailable_message(config.model)
    elif not args.no_warmup:
        try:
            engine.warmup()
        except (OSError, RuntimeError, ValueError) as exc:
            startup_error = f"Model warmup failed: {exc}"
    if startup_error:
        print(f"Warning: {startup_error}", file=sys.stderr)

    app = build_application(engine, config.output_dir, startup_error)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to run the web demo") from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

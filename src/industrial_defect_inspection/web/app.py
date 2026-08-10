"""Run the combined FastAPI and Gradio local demo."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from uuid import uuid4

from PIL import Image

from industrial_defect_inspection.config import load_inference_config
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.utils.io import write_json
from industrial_defect_inspection.web.api import create_app


def create_gradio_demo(engine: InferenceEngine, output_dir: Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to run the web demo") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    def run_prediction(image: Image.Image | None, confidence: float):
        if image is None:
            raise gr.Error("Upload an image before running inspection.")
        result, annotated = engine.predict(image, confidence=confidence)
        rows = [
            [
                detection.class_name,
                round(detection.confidence, 4),
                *[round(value, 1) for value in detection.bbox_xyxy],
            ]
            for detection in result.detections
        ]
        counts = Counter(detection.class_name for detection in result.detections)
        identifier = uuid4().hex
        image_path = output_dir / f"{identifier}_annotated.jpg"
        json_path = output_dir / f"{identifier}.json"
        annotated.save(image_path, quality=92)
        write_json(json_path, result.model_dump(mode="json"))
        summary = (
            f"**{len(result.detections)} detection(s)** · "
            f"{result.total_ms:.1f} ms · device `{result.device}` · "
            f"model `{result.model_version}`"
        )
        if counts:
            summary += "  \n" + ", ".join(f"{name}: {count}" for name, count in counts.items())
        return annotated, rows, result.model_dump(mode="json"), summary, image_path, json_path

    with gr.Blocks(title="Industrial Defect Inspection") as demo:
        gr.Markdown(
            "# Industrial Defect Inspection\n"
            "Upload a steel-surface image and inspect detected defects. "
            "Research and portfolio demo; not a production quality-control system."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input image")
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


def build_application(engine: InferenceEngine, output_dir: Path):
    import gradio as gr

    api = create_app(engine)
    demo = create_gradio_demo(engine, output_dir)
    demo.queue(default_concurrency_limit=1)
    return gr.mount_gradio_app(api, demo, path="/demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local industrial inspection demo.")
    parser.add_argument("--config", default="configs/infer/default.yaml")
    parser.add_argument("--model", help="Override the model path from config.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_inference_config(args.config)
    if args.model:
        config = config.model_copy(update={"model": Path(args.model)})
    engine = InferenceEngine(config)
    if not args.no_warmup:
        engine.warmup()
    app = build_application(engine, Path("artifacts/web_outputs"))
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to run the web demo") from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

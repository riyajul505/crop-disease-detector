"""Gradio demo: drag-and-drop a leaf photo, get disease detections + treatment notes.

Local:             python app.py
Hugging Face:      push this repo (with weights) to a Space, SDK = gradio.

Weights resolution order:
  1. WEIGHTS env var
  2. runs/detect/plantvillage_finetune/weights/best.pt  (local training output)
  3. weights/best.pt                                    (checked-in copy for Spaces)
"""

import os
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from treatments import treatment_for

CONF_DEFAULT = 0.4

_REPO = Path(__file__).parent  # anchor lookups to the repo, not the cwd
_WEIGHT_CANDIDATES = [
    os.environ.get("WEIGHTS"),
    _REPO / "runs/detect/plantvillage_finetune/weights/best.pt",
    _REPO / "weights/best.pt",
]
WEIGHTS = next((str(w) for w in _WEIGHT_CANDIDATES if w and Path(w).exists()), None)
if WEIGHTS is None:
    raise SystemExit(
        "No trained weights found. Run train.py first, or set WEIGHTS=path/to/best.pt"
    )

model = YOLO(WEIGHTS)


def detect(image: Image.Image, conf: float):
    if image is None:
        return None, pd.DataFrame()

    result = model.predict(image, conf=conf, verbose=False)[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])  # BGR -> RGB

    rows = []
    for box in result.boxes:
        name = model.names[int(box.cls)]
        rows.append({
            "Detection": name,
            "Confidence": f"{float(box.conf):.0%}",
            "Treatment note": treatment_for(name),
        })
    if not rows:
        rows.append({
            "Detection": "Nothing above threshold",
            "Confidence": "—",
            "Treatment note": "Try lowering the confidence slider, or use a clearer single-leaf photo.",
        })
    return annotated, pd.DataFrame(rows)


with gr.Blocks(title="Crop Disease Detector") as demo:
    gr.Markdown(
        """
        # 🌿 Crop Disease Detector
        YOLOv8n fine-tuned on **PlantVillage** (15 classes — bell pepper, potato, tomato).
        Drop in a leaf photo to get detections with confidence scores and a
        one-line treatment recommendation per finding.
        """
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Leaf photo")
            conf_slider = gr.Slider(0.05, 0.95, value=CONF_DEFAULT, step=0.05,
                                    label="Confidence threshold")
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="Detections")
            table_out = gr.Dataframe(label="Findings & treatment",
                                     headers=["Detection", "Confidence", "Treatment note"],
                                     wrap=True)

    run_btn.click(detect, [image_in, conf_slider], [image_out, table_out])
    image_in.upload(detect, [image_in, conf_slider], [image_out, table_out])

    example_dir = _REPO / "results/sample_inputs"
    if example_dir.is_dir():
        examples = [str(p) for p in sorted(example_dir.glob("*.*"))[:6]]
        if examples:
            gr.Examples(examples=examples, inputs=image_in)

    gr.Markdown(
        "*Treatment notes are general guidance, not a substitute for local "
        "agronomic advice. Model trained on studio-style single-leaf images; "
        "field photos with cluttered backgrounds may need a lower threshold.*"
    )

if __name__ == "__main__":
    demo.launch()

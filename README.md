# 🌿 Crop Disease Detector — YOLOv8 + PlantVillage

A fine-tuned **YOLOv8n** object detection model that identifies **15 classes of crop
diseases** across bell pepper, potato, and tomato from leaf images — deployed as a live
**Gradio** app with confidence scores and per-disease treatment recommendations.

![pipeline](computer_vision_yolov8_pipeline.svg)

## Highlights

- **Real training pipeline** — custom `data.yaml`, scripted 70/20/10 split, and
  auto-generated YOLO labels (PlantVillage ships as a classification dataset;
  `prepare_dataset.py` derives leaf bounding boxes via OpenCV segmentation).
  Dataset: [PlantVillage 15-class subset](https://www.kaggle.com/datasets/emmarex/plantdisease)
  (~20.6K images).
- **Training diagnostics** — loss and mAP curves over epochs (`results/training_curves.png`).
- **Evaluation rigor** — per-class precision/recall/mAP table sorted weakest-first
  (`results/per_class_metrics.csv`) plus a normalized confusion matrix.
- **Applied output** — the demo pairs every detection with a one-line treatment note
  (`treatments.py`).
- **Visual proof** — annotated detections with boxes and confidence scores in
  `results/sample_detections/`.

## Quick start

```bash
pip install -r requirements.txt

# 1. Dataset: convert PlantVillage to YOLO format.
#    Reads a manual download at data/archive/PlantVillage/ by default;
#    use --download to fetch from Kaggle instead (needs ~/.kaggle/kaggle.json).
python prepare_dataset.py                # full ~20.6K images
python prepare_dataset.py --limit 50     # fast smoke run

# 2. Train (≈30 min for 50 epochs on a Colab T4; nano model)
python train.py --device 0               # or --device cpu

# 3. Evaluate: per-class table, curves, confusion matrix, annotated samples
python evaluate.py

# 4. Demo
python app.py                            # http://127.0.0.1:7860
```

## Repository structure

```
crop-disease-detector/
├── prepare_dataset.py      ← Kaggle download + split + bbox label generation
├── train.py                ← fine-tuning script
├── evaluate.py             ← metrics report + curves + confusion matrix
├── app.py                  ← Gradio demo
├── treatments.py           ← 38 one-line treatment recommendations
├── data/
│   └── data.yaml           ← PlantVillage split config (38 classes)
├── configs/
│   └── yolov8_finetune.yaml ← training hyperparameters
├── results/
│   └── sample_detections/  ← annotated output images
├── notebooks/
│   └── EDA_and_training.ipynb ← full documented walkthrough
├── requirements.txt
└── README.md
```

## Tech stack

| Layer      | Library                      | Why |
|------------|------------------------------|-----|
| Model      | `ultralytics` YOLOv8n        | Industry standard 2024–26, fast to fine-tune |
| Dataset    | PlantVillage (Kaggle)        | ~20.6K images, 15 disease classes, well documented |
| Training   | PyTorch + CUDA / Colab T4    | Free GPU, ~30 min for 50 epochs |
| Evaluation | Ultralytics built-in metrics | mAP50, precision, recall, confusion matrix |
| Demo UI    | `gradio`                     | Fastest path to a drag-and-drop CV demo |
| Deployment | Hugging Face Spaces          | Free live demo with GPU inference |

## Results

YOLOv8n fine-tuned for 10 epochs at 320px on CPU. Averages across all 15 classes
on the held-out **test split** (2,076 images the model never saw during training):

| Metric (test split) | Value |
|---------------------|-------|
| mAP50               | 0.992 |
| mAP50-95            | 0.955 |
| Precision           | 0.987 |
| Recall              | 0.979 |

Per-class breakdown in [`results/per_class_metrics.md`](results/per_class_metrics.md).
Weakest class is Tomato — Early blight (recall 0.91); every class clears mAP50 0.98.

## Deploying to Hugging Face Spaces

1. Create a Space (SDK: **Gradio**).
2. Copy `app.py`, `treatments.py`, `requirements.txt`, and your trained weights as
   `weights/best.pt` into the Space repo.
3. Push — the Space builds and serves the public demo URL.

## Notes & limitations

- PlantVillage images are studio-style single leaves on plain backgrounds; field
  photos with clutter will underperform — lower the confidence slider in the demo.
- Bounding boxes are auto-derived (one leaf per image), so this is detection-style
  packaging of a classification dataset; localization quality reflects that.
- Treatment notes are general guidance, not a substitute for local agronomic advice.

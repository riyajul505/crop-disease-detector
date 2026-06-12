"""Fine-tune YOLOv8n on the PlantVillage detection dataset.

Run prepare_dataset.py first to build data/plantvillage_yolo/.

Usage:
    python train.py                          # defaults from configs/yolov8_finetune.yaml
    python train.py --epochs 5 --device cpu  # quick local smoke run
    python train.py --device 0               # Colab/desktop GPU

Outputs land in runs/detect/plantvillage_finetune/ — best.pt weights,
results.csv (per-epoch metrics), and training-batch previews.
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--config", default="configs/yolov8_finetune.yaml")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--device", default=None,
                        help="0 for first GPU, 'cpu' to force CPU; default lets Ultralytics pick")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override config batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override config image size (e.g. 416 for faster CPU training)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Dataloader workers (lower if RAM-constrained)")
    parser.add_argument("--name", default="plantvillage_finetune")
    args = parser.parse_args()

    hyp = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        hyp["epochs"] = args.epochs
    if args.batch is not None:
        hyp["batch"] = args.batch
    if args.imgsz is not None:
        hyp["imgsz"] = args.imgsz
    if args.workers is not None:
        hyp["workers"] = args.workers
    if args.device is not None:
        hyp["device"] = args.device

    model = YOLO(args.weights)
    # project= must be ABSOLUTE: Ultralytics resolves relative paths against its
    # global runs_dir (e.g. C:\Users\<user>\runs), not the working directory,
    # and evaluate.py/app.py then can't find best.pt.
    project = Path("runs/detect").resolve()
    model.train(data=args.data, project=str(project), name=args.name,
                exist_ok=True, **hyp)

    # Validates best.pt on the val split; project= keeps output in the repo.
    metrics = model.val(data=args.data, project=str(project), name="val", exist_ok=True)
    print(f"\nval mAP50:    {metrics.box.map50:.3f}")
    print(f"val mAP50-95: {metrics.box.map:.3f}")
    print(f"val precision: {metrics.box.mp:.3f}  recall: {metrics.box.mr:.3f}")
    print(f"\nWeights: runs/detect/{args.name}/weights/best.pt")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()

"""Evaluation report for the fine-tuned PlantVillage detector.

Produces, under results/:
  - per_class_metrics.csv / .md  — precision, recall, mAP50, mAP50-95 per class
  - training_curves.png          — loss + mAP over epochs (from results.csv)
  - confusion_matrix.png         — copied from the Ultralytics val run
  - sample_detections/           — annotated test images with boxes + confidence

Usage:
    python evaluate.py
    python evaluate.py --weights runs/detect/plantvillage_finetune/weights/best.pt
"""

import argparse
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

RESULTS_DIR = Path("results")


def per_class_table(metrics, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    # metrics.box.ap_class_index lists class ids present in the split,
    # aligned with the per-class p/r/ap arrays.
    for i, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        rows.append({
            "class": names[int(class_id)],
            "precision": round(p, 3),
            "recall": round(r, 3),
            "mAP50": round(ap50, 3),
            "mAP50-95": round(ap, 3),
        })
    df = pd.DataFrame(rows).sort_values("mAP50", ascending=True).reset_index(drop=True)
    return df


def plot_training_curves(run_dir: Path, out_path: Path) -> bool:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv at {csv_path} — skipping curves plot")
        return False
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epoch = df["epoch"]

    for col, label in [("train/box_loss", "box (train)"),
                       ("train/cls_loss", "cls (train)"),
                       ("val/box_loss", "box (val)"),
                       ("val/cls_loss", "cls (val)")]:
        if col in df:
            axes[0].plot(epoch, df[col], label=label,
                         linestyle="--" if "val" in col else "-")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for col, label in [("metrics/mAP50(B)", "mAP50"),
                       ("metrics/mAP50-95(B)", "mAP50-95"),
                       ("metrics/precision(B)", "precision"),
                       ("metrics/recall(B)", "recall")]:
        if col in df:
            axes[1].plot(epoch, df[col], label=label)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("YOLOv8n fine-tuning on PlantVillage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",
                        default="runs/detect/plantvillage_finetune/weights/best.pt")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--samples", type=int, default=12,
                        help="Number of annotated sample images to save")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Eval image size; defaults to the size the model was trained at")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    model = YOLO(args.weights)

    # Evaluate at the training resolution unless overridden — a 320-trained
    # model validated at the 640 default scores worse and runs 4x slower.
    imgsz = args.imgsz or (model.ckpt or {}).get("train_args", {}).get("imgsz") or 640
    print(f"Validating on test split (imgsz={imgsz})...")
    # Absolute project= keeps val output in the repo instead of the global runs_dir.
    metrics = model.val(data=args.data, split="test", imgsz=imgsz, plots=True,
                        project=str(Path("runs/detect").resolve()), name="test_eval",
                        exist_ok=True)

    print(f"\n=== Overall (test split) ===")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"precision: {metrics.box.mp:.3f}")
    print(f"recall:    {metrics.box.mr:.3f}")

    df = per_class_table(metrics, model.names)
    df.to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False)
    (RESULTS_DIR / "per_class_metrics.md").write_text(
        df.to_markdown(index=False), encoding="utf-8")
    print(f"\n=== Per-class metrics (weakest 10 first) ===")
    print(df.head(10).to_string(index=False))
    print(f"\nFull table: {RESULTS_DIR / 'per_class_metrics.csv'}")

    val_dir = Path(metrics.save_dir)
    cm = val_dir / "confusion_matrix_normalized.png"
    if cm.exists():
        shutil.copy2(cm, RESULTS_DIR / "confusion_matrix.png")

    run_dir = Path(args.weights).parent.parent  # weights/best.pt -> run dir
    plot_training_curves(run_dir, RESULTS_DIR / "training_curves.png")

    # Annotated samples: random test images through predict(save=True).
    test_images_dir = Path("data/plantvillage_yolo/images/test")
    if test_images_dir.is_dir():
        pool = list(test_images_dir.glob("*.*"))
        picks = random.Random(0).sample(pool, min(args.samples, len(pool)))
        out_dir = RESULTS_DIR / "sample_detections"
        out_dir.mkdir(exist_ok=True)
        results = model.predict([str(p) for p in picks], conf=args.conf,
                                imgsz=imgsz, verbose=False)
        for r in results:
            annotated = r.plot()  # BGR ndarray with boxes + confidence drawn
            import cv2
            cv2.imwrite(str(out_dir / Path(r.path).name), annotated)
        print(f"Saved {len(results)} annotated samples to {out_dir}")

    print("\nEvaluation complete — see results/")


if __name__ == "__main__":
    main()

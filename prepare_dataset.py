"""Convert the PlantVillage dataset (Kaggle: emmarex/plantdisease) to YOLO format.

PlantVillage is a *classification* dataset (one leaf per image, sorted into
class folders, no bounding boxes). To fine-tune YOLOv8 for detection we
auto-generate one bounding box per image by segmenting the leaf from the
near-uniform studio background, then write YOLO-format labels.

This targets the 15-class subset (bell pepper, potato, tomato; ~20.6K images).
By default it reads a manual download at data/archive/PlantVillage/; pass
--download to fetch from Kaggle via kagglehub instead (needs
~/.kaggle/kaggle.json).

Output layout (consumed by data/data.yaml):

    data/plantvillage_yolo/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Usage:
    python prepare_dataset.py                  # full dataset, 70/20/10 split
    python prepare_dataset.py --limit 50       # 50 images/class smoke run
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

KAGGLE_DATASET = "emmarex/plantdisease"
DEFAULT_SOURCE = Path("data/archive/PlantVillage")
SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42

# PlantVillage folder name -> class id. Order matches data/data.yaml.
FOLDER_TO_ID = {
    "Pepper__bell___Bacterial_spot": 0,
    "Pepper__bell___healthy": 1,
    "Potato___Early_blight": 2,
    "Potato___healthy": 3,
    "Potato___Late_blight": 4,
    "Tomato_Bacterial_spot": 5,
    "Tomato_Early_blight": 6,
    "Tomato_healthy": 7,
    "Tomato_Late_blight": 8,
    "Tomato_Leaf_Mold": 9,
    "Tomato_Septoria_leaf_spot": 10,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 11,
    "Tomato__Target_Spot": 12,
    "Tomato__Tomato_mosaic_virus": 13,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 14,
}


def leaf_bbox(image: np.ndarray) -> tuple[float, float, float, float]:
    """Segment the leaf and return a YOLO-normalized (cx, cy, w, h) box.

    PlantVillage backgrounds are flat gray/black, so Otsu thresholding on
    saturation separates the leaf cleanly in the vast majority of images.
    Falls back to a 96% full-image box when segmentation finds nothing
    plausible (e.g. very desiccated brown leaves on gray).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, bw, bh = cv2.boundingRect(np.vstack(contours))
        # Reject degenerate masks (tiny speck or sliver) — fall through to default.
        if bw * bh >= 0.05 * w * h and bw > 0.1 * w and bh > 0.1 * h:
            pad = 0.02  # slight padding so box edges don't clip the leaf
            x0 = max(0, x - pad * w)
            y0 = max(0, y - pad * h)
            x1 = min(w, x + bw + pad * w)
            y1 = min(h, y + bh + pad * h)
            return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)

    return 0.5, 0.5, 0.96, 0.96


def download() -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    # The archive may nest a PlantVillage/ folder (or even two) — find the
    # level that actually contains the class folders.
    for candidate in [root, *sorted(root.rglob("PlantVillage"))]:
        if candidate.is_dir() and any((candidate / name).is_dir() for name in FOLDER_TO_ID):
            return candidate
    raise SystemExit(f"Could not locate class folders under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="PlantVillage dir containing the class folders")
    parser.add_argument("--download", action="store_true",
                        help="Fetch from Kaggle via kagglehub instead of --source")
    parser.add_argument("--out", type=Path, default=Path("data/plantvillage_yolo"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images per class (for quick smoke runs)")
    args = parser.parse_args()

    if args.download:
        source = download()
    else:
        source = args.source
        if not source.is_dir():
            raise SystemExit(f"{source} not found — place the dataset there, "
                             f"pass --source, or use --download")
    print(f"Source: {source}")

    rng = random.Random(SEED)
    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    skipped = []
    for folder, class_id in FOLDER_TO_ID.items():
        class_dir = source / folder
        if not class_dir.is_dir():
            skipped.append(folder)
            continue
        images = sorted(p for p in class_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(images)
        if args.limit:
            images = images[: args.limit]

        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        split_of = (["train"] * n_train + ["val"] * n_val
                    + ["test"] * (n - n_train - n_val))

        for img_path, split in tqdm(list(zip(images, split_of)),
                                    desc=f"[{class_id:2d}] {folder}", unit="img"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            cx, cy, bw, bh = leaf_bbox(image)
            stem = f"{class_id:02d}_{img_path.stem}"
            shutil.copy2(img_path, args.out / "images" / split / f"{stem}{img_path.suffix.lower()}")
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            counts[split] += 1

    if skipped:
        print(f"\nWARNING: {len(skipped)} class folders not found: {skipped}")

    print("\nDone. Split sizes:", counts)

    # Stamp the absolute dataset path into data.yaml — Ultralytics resolves
    # relative paths against its global datasets_dir, not the yaml location.
    yaml_path = Path("data/data.yaml")
    data_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert len(data_yaml["names"]) == len(FOLDER_TO_ID), "data.yaml out of sync"
    data_yaml["path"] = str(args.out.resolve())
    yaml_path.write_text(yaml.dump(data_yaml, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    print(f"data.yaml path set to {data_yaml['path']}")
    print(f"{len(FOLDER_TO_ID)} classes ready -> train with: python train.py")


if __name__ == "__main__":
    main()

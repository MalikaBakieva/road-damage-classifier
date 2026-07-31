"""Turn raw RDD2022 detection annotations into a patch-classification dataset.

Why crops instead of whole images?
----------------------------------
RDD2022 is a *detection* dataset: a single photo often contains several damage
types at once, so a whole-image multiclass label would be ambiguous and would
inflate apparent accuracy. Cropping each annotated box (plus context) gives a
well-posed single-label classification problem that the brief actually asks for,
while whole-image negatives supply the `not_damaged` class.

Leakage control
---------------
Splitting happens at *image* level, never at crop level. Two crops from the same
photo share lighting, camera and road surface, so putting one in train and one
in test would leak information and overstate performance (Discovery: leakage).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

from ..config import DAMAGE_CLASSES, Config
from ..utils import get_logger, save_json, set_seed
from .rdd_parser import build_annotation_table

ImageFile.LOAD_TRUNCATED_IMAGES = True  # RDD2022 contains a few truncated JPEGs

LOG = get_logger(__name__)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    context: float,
) -> tuple[int, int, int, int]:
    """Grow a box by `context` on each side, clipped to the image."""
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    dx, dy = int(bw * context), int(bh * context)
    return (
        max(0, xmin - dx),
        max(0, ymin - dy),
        min(width, xmax + dx),
        min(height, ymax + dy),
    )


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def sample_negative_box(
    width: int,
    height: int,
    positives: list[tuple[int, int, int, int]],
    rng: random.Random,
    size: int,
    attempts: int = 25,
) -> tuple[int, int, int, int] | None:
    """Sample a road-surface patch that overlaps no annotated damage.

    Negatives are biased to the lower half of the frame, where the road surface
    is; sampling uniformly would produce sky/building crops that make the
    `not_damaged` class trivially separable and inflate the score.
    """
    if width <= size or height <= size:
        return None
    for _ in range(attempts):
        x = rng.randint(0, width - size)
        y = rng.randint(int(height * 0.35), height - size)
        candidate = (x, y, x + size, y + size)
        if not any(boxes_overlap(candidate, p) for p in positives):
            return candidate
    return None


# --------------------------------------------------------------------------
# crop extraction
# --------------------------------------------------------------------------


def extract_crops(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Write cropped patches to <interim>/crops and return the crop index."""
    out_root = Path(cfg.data.interim_dir) / "crops"
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.data.seed)

    records: list[dict] = []
    grouped = df.groupby("image_path", sort=True)
    LOG.info("Extracting crops from %d images ...", len(grouped))

    for n, (image_path, group) in enumerate(grouped, start=1):
        if n % 500 == 0:
            LOG.info("  %d/%d images", n, len(grouped))
        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                width, height = im.size

                positives: list[tuple[int, int, int, int]] = []
                for idx, row in enumerate(group.itertuples()):
                    if row.damage_class is None or not isinstance(row.damage_class, str):
                        continue
                    if row.damage_class not in DAMAGE_CLASSES:
                        continue
                    box = (int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax))
                    if (box[2] - box[0]) < cfg.data.min_box_size or (
                        box[3] - box[1]
                    ) < cfg.data.min_box_size:
                        continue  # too small to be legible after resize
                    positives.append(box)
                    grown = expand_box(box, width, height, cfg.data.crop_context)
                    crop_name = f"{row.image_id}_p{idx}_{row.damage_class}.jpg"
                    dest = out_root / row.damage_class
                    dest.mkdir(parents=True, exist_ok=True)
                    im.crop(grown).save(dest / crop_name, quality=92)
                    records.append(
                        {
                            "crop_path": str(dest / crop_name),
                            "image_id": row.image_id,
                            "country": row.country,
                            "label_multiclass": row.damage_class,
                            "label_binary": "damaged",
                            "box_w": box[2] - box[0],
                            "box_h": box[3] - box[1],
                        }
                    )

                # negatives
                side = max(cfg.data.min_box_size * 3, min(width, height) // 4)
                for k in range(cfg.data.negatives_per_image):
                    neg = sample_negative_box(width, height, positives, rng, side)
                    if neg is None:
                        continue
                    dest = out_root / "not_damaged"
                    dest.mkdir(parents=True, exist_ok=True)
                    image_id = group.iloc[0]["image_id"]
                    crop_name = f"{image_id}_n{k}_not_damaged.jpg"
                    im.crop(neg).save(dest / crop_name, quality=92)
                    records.append(
                        {
                            "crop_path": str(dest / crop_name),
                            "image_id": image_id,
                            "country": group.iloc[0]["country"],
                            "label_multiclass": "not_damaged",
                            "label_binary": "not_damaged",
                            "box_w": neg[2] - neg[0],
                            "box_h": neg[3] - neg[1],
                        }
                    )
        except (OSError, ValueError) as exc:
            LOG.warning("Unreadable image %s (%s) - skipped", image_path, exc)
            continue

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def stratified_group_split(crops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign train/val/test at image level, stratified by the image's
    dominant damage class so rare classes appear in every split."""
    rng = random.Random(cfg.data.seed)

    per_image = (
        crops.groupby("image_id")["label_multiclass"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"label_multiclass": "dominant"})
    )

    assignment: dict[str, str] = {}
    for _dominant, group in per_image.groupby("dominant"):
        ids = sorted(group["image_id"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * cfg.data.test_size))) if n > 2 else 0
        n_val = max(1, int(round(n * cfg.data.val_size))) if n > 2 else 0
        for i, image_id in enumerate(ids):
            if i < n_test:
                assignment[image_id] = "test"
            elif i < n_test + n_val:
                assignment[image_id] = "val"
            else:
                assignment[image_id] = "train"

    crops = crops.copy()
    crops["split"] = crops["image_id"].map(assignment).fillna("train")
    return crops


def write_outputs(crops: pd.DataFrame, cfg: Config) -> dict:
    processed = Path(cfg.data.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    crops.to_csv(processed / "crops.csv", index=False)

    stats = {
        "n_crops": int(len(crops)),
        "n_images": int(crops["image_id"].nunique()),
        "splits": crops["split"].value_counts().to_dict(),
        "multiclass_counts": crops["label_multiclass"].value_counts().to_dict(),
        "binary_counts": crops["label_binary"].value_counts().to_dict(),
        "per_split_multiclass": {
            split: sub["label_multiclass"].value_counts().to_dict()
            for split, sub in crops.groupby("split")
        },
        "countries": crops["country"].value_counts().to_dict(),
    }
    save_json(stats, processed / "dataset_stats.json")
    return stats


def prepare(config_path: str) -> dict:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.data.seed)

    LOG.info("Parsing raw annotations from %s", cfg.data.raw_dir)
    ann = build_annotation_table(cfg.data.raw_dir, cfg.data.countries)
    if ann.empty:
        raise SystemExit(
            f"No annotations found under '{cfg.data.raw_dir}'.\n"
            "Download RDD2022 first (see docs/DATASET.md) or run "
            "`python scripts/make_sample_data.py` for a runnable demo dataset."
        )

    Path(cfg.data.interim_dir).mkdir(parents=True, exist_ok=True)
    ann.to_csv(Path(cfg.data.interim_dir) / "annotations.csv", index=False)

    crops = extract_crops(ann, cfg)
    if crops.empty:
        raise SystemExit("No usable crops were produced - check min_box_size and the raw data.")

    crops = stratified_group_split(crops, cfg)
    stats = write_outputs(crops, cfg)

    LOG.info("Done. %d crops from %d images.", stats["n_crops"], stats["n_images"])
    for split, count in sorted(stats["splits"].items()):
        LOG.info("  %-5s %6d crops", split, count)
    LOG.info("Class balance (multiclass): %s", stats["multiclass_counts"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the road-damage crop dataset.")
    parser.add_argument("--config", default="configs/binary.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()

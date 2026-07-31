"""Generate a small SYNTHETIC RDD2022-shaped dataset.

Purpose
-------
So that `make demo` — prepare -> train -> evaluate -> serve — runs end to end on
any machine with no download and no credentials. It exists to prove the pipeline
is wired correctly and to give the API a checkpoint to load.

IT IS NOT REAL DATA. Metrics obtained on it say nothing about real-world
performance. Reported results in docs/ come from real RDD2022 runs.

Output layout mirrors RDD2022 exactly:

    data/raw/<Country>/train/images/<id>.jpg
    data/raw/<Country>/train/annotations/xmls/<id>.xml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFilter

CODES = ["D00", "D10", "D20", "D40"]
IMG_W, IMG_H = 480, 480


def asphalt(rng: random.Random) -> Image.Image:
    """Grey noisy background that loosely resembles road surface texture."""
    base = rng.randint(95, 145)
    img = Image.new("RGB", (IMG_W, IMG_H), (base, base, base + rng.randint(-4, 4)))
    draw = ImageDraw.Draw(img)
    for _ in range(2500):
        x, y = rng.randint(0, IMG_W - 1), rng.randint(0, IMG_H - 1)
        d = rng.randint(-28, 28)
        v = max(0, min(255, base + d))
        draw.point((x, y), fill=(v, v, v))
    # horizon / sky band so negatives sampled from the top are plausible
    draw.rectangle([0, 0, IMG_W, int(IMG_H * 0.28)], fill=(150, 165, 185))
    if rng.random() < 0.5:  # lane marking
        x = rng.randint(60, IMG_W - 60)
        draw.line([(x, int(IMG_H * 0.3)), (x + rng.randint(-40, 40), IMG_H)],
                  fill=(225, 225, 220), width=rng.randint(4, 9))
    return img.filter(ImageFilter.GaussianBlur(0.4))


def draw_damage(img: Image.Image, code: str, rng: random.Random) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(img)
    dark = (rng.randint(25, 55),) * 3
    y0 = int(IMG_H * 0.35)

    if code == "D00":  # longitudinal crack — mostly vertical
        x = rng.randint(70, IMG_W - 70)
        pts = [(x + rng.randint(-6, 6), y) for y in range(y0, IMG_H - 20, 12)]
        draw.line(pts, fill=dark, width=rng.randint(3, 6))
    elif code == "D10":  # transverse crack — mostly horizontal
        y = rng.randint(y0 + 20, IMG_H - 40)
        pts = [(x, y + rng.randint(-6, 6)) for x in range(30, IMG_W - 30, 12)]
        draw.line(pts, fill=dark, width=rng.randint(3, 6))
    elif code == "D20":  # alligator — dense mesh
        cx, cy = rng.randint(110, IMG_W - 110), rng.randint(y0 + 40, IMG_H - 90)
        for _ in range(16):
            x1 = cx + rng.randint(-70, 70)
            y1 = cy + rng.randint(-55, 55)
            draw.line([(x1, y1), (x1 + rng.randint(-40, 40), y1 + rng.randint(-35, 35))],
                      fill=dark, width=2)
        pts = [(cx - 80, cy - 60, cx + 80, cy + 60)]
        x1, y1, x2, y2 = pts[0]
        return (max(0, x1), max(0, y1), min(IMG_W, x2), min(IMG_H, y2))
    else:  # D40 pothole — dark filled blob
        cx, cy = rng.randint(110, IMG_W - 110), rng.randint(y0 + 50, IMG_H - 80)
        r = rng.randint(28, 55)
        draw.ellipse([cx - r, cy - int(r * 0.7), cx + r, cy + int(r * 0.7)], fill=(20, 20, 20))
        draw.ellipse([cx - r + 6, cy - int(r * 0.7) + 5, cx + r - 6, cy + int(r * 0.7) - 5],
                     outline=(70, 70, 70), width=3)
        return (cx - r - 6, cy - r, cx + r + 6, cy + r)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 18
    return (
        max(0, min(xs) - pad), max(0, min(ys) - pad),
        min(IMG_W, max(xs) + pad), min(IMG_H, max(ys) + pad),
    )


def write_xml(path: Path, filename: str, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(IMG_W)
    ET.SubElement(size, "height").text = str(IMG_H)
    ET.SubElement(size, "depth").text = "3"
    for code, (xmin, ymin, xmax, ymax) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = code
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), (xmin, ymin, xmax, ymax), strict=True):
            ET.SubElement(box, tag).text = str(int(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate(out_dir: str, per_country: int, countries: list[str], seed: int) -> None:
    rng = random.Random(seed)
    root = Path(out_dir)

    for country in countries:
        images_dir = root / country / "train" / "images"
        xml_dir = root / country / "train" / "annotations" / "xmls"
        images_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        for i in range(per_country):
            image_id = f"{country}_{i:05d}"
            img = asphalt(rng)
            boxes: list[tuple[str, tuple[int, int, int, int]]] = []

            # ~25% of images are clean road (true negatives)
            if rng.random() > 0.25:
                # deliberately imbalanced, like the real dataset
                code = rng.choices(CODES, weights=[0.40, 0.28, 0.20, 0.12])[0]
                boxes.append((code, draw_damage(img, code, rng)))
                if rng.random() < 0.25:
                    code2 = rng.choice(CODES)
                    boxes.append((code2, draw_damage(img, code2, rng)))

            if rng.random() < 0.2:  # capture-condition variation
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.4)))

            img.save(images_dir / f"{image_id}.jpg", quality=88)
            write_xml(xml_dir / f"{image_id}.xml", f"{image_id}.jpg", boxes)

        print(f"  {country}: {per_country} images")

    print(f"\nSynthetic dataset written to {root}")
    print("WARNING: synthetic data — for pipeline verification only, not for reporting results.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--per-country", type=int, default=120)
    parser.add_argument("--countries", nargs="+", default=["Japan", "India", "Norway"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.per_country, args.countries, args.seed)


if __name__ == "__main__":
    main()

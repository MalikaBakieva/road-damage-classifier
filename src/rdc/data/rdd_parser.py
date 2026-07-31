"""Parse RDD2022 PASCAL-VOC annotations into a tidy annotation table.

RDD2022 (Arya et al., 2022) ships one XML file per image in VOC format:

    <annotation>
      <filename>Japan_000000.jpg</filename>
      <size><width>600</width><height>600</height><depth>3</depth></size>
      <object>
        <name>D00</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox>
      </object>
      ...
    </annotation>

Images with zero <object> entries are genuine negatives (no visible damage) and
are an important part of the binary scope.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..config import RDD_CODE_TO_CLASS
from ..utils import get_logger

LOG = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_annotation_file(xml_path: Path, image_dir: Path, country: str) -> list[dict]:
    """Return one row per object; images with no objects yield a single row
    with damage_code = None so that negatives are not silently dropped."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:  # corrupt annotation - skip loudly, never crash
        LOG.warning("Skipping unparsable annotation %s (%s)", xml_path.name, exc)
        return []

    root = tree.getroot()
    stem = xml_path.stem
    image_path = _find_image(image_dir, stem)
    if image_path is None:
        LOG.debug("No image found for annotation %s", xml_path.name)
        return []

    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    rows: list[dict] = []
    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip().upper()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = int(float(box.findtext("xmin", "0")))
            ymin = int(float(box.findtext("ymin", "0")))
            xmax = int(float(box.findtext("xmax", "0")))
            ymax = int(float(box.findtext("ymax", "0")))
        except (TypeError, ValueError):
            LOG.warning("Malformed bndbox in %s - skipped", xml_path.name)
            continue

        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box: a known RDD2022 data-quality issue

        rows.append(
            {
                "image_path": str(image_path),
                "image_id": stem,
                "country": country,
                "img_width": width,
                "img_height": height,
                "damage_code": code,
                "damage_class": RDD_CODE_TO_CLASS.get(code),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    if not rows:
        rows.append(
            {
                "image_path": str(image_path),
                "image_id": stem,
                "country": country,
                "img_width": width,
                "img_height": height,
                "damage_code": None,
                "damage_class": None,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None,
            }
        )
    return rows


def _country_dirs(raw_dir: Path, countries: Iterable[str]) -> list[Path]:
    """RDD2022 unpacks to <raw>/<Country>/train/{images,annotations/xmls}.

    We tolerate several common layouts because different mirrors differ.
    """
    found: list[Path] = []
    wanted = {c.lower() for c in countries}
    for child in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not child.is_dir():
            continue
        if wanted and child.name.lower() not in wanted:
            continue
        found.append(child)
    return found


def _locate_pair(country_dir: Path) -> tuple[Path, Path] | None:
    candidates = [
        (country_dir / "train" / "images", country_dir / "train" / "annotations" / "xmls"),
        (country_dir / "images", country_dir / "annotations" / "xmls"),
        (country_dir / "train" / "images", country_dir / "train" / "annotations"),
        (country_dir / "images", country_dir / "annotations"),
    ]
    for image_dir, ann_dir in candidates:
        if image_dir.is_dir() and ann_dir.is_dir():
            return image_dir, ann_dir
    return None


def build_annotation_table(raw_dir: str | Path, countries: Iterable[str]) -> pd.DataFrame:
    """Walk the raw RDD2022 tree and return the full object-level table."""
    raw_dir = Path(raw_dir)
    all_rows: list[dict] = []

    for country_dir in _country_dirs(raw_dir, countries):
        pair = _locate_pair(country_dir)
        if pair is None:
            LOG.warning("Could not locate images/annotations under %s - skipped", country_dir)
            continue
        image_dir, ann_dir = pair
        xmls = sorted(ann_dir.glob("*.xml"))
        LOG.info("%-10s %5d annotation files", country_dir.name, len(xmls))
        for xml_path in xmls:
            all_rows.extend(parse_annotation_file(xml_path, image_dir, country_dir.name))

    df = pd.DataFrame(all_rows)
    if df.empty:
        LOG.warning("No annotations parsed from %s", raw_dir)
        return df

    unknown = df.loc[df["damage_code"].notna() & df["damage_class"].isna(), "damage_code"]
    if len(unknown):
        LOG.info(
            "Dropped %d objects with out-of-scope codes: %s", len(unknown), sorted(unknown.unique())
        )
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Small EDA helper used by the notebook and the prepare script."""
    if df.empty:
        return pd.DataFrame()
    per_class = (
        df.dropna(subset=["damage_class"])
        .groupby(["country", "damage_class"])
        .size()
        .rename("objects")
        .reset_index()
    )
    return per_class.pivot(index="country", columns="damage_class", values="objects").fillna(0)

"""Tests for annotation parsing, crop extraction and splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdc.config import DAMAGE_CLASSES, MULTICLASS_CLASSES, RDD_CODE_TO_CLASS, class_names
from rdc.data.prepare import boxes_overlap, expand_box, sample_negative_box
from rdc.data.rdd_parser import build_annotation_table, parse_annotation_file

# --------------------------------------------------------------------------
# config / taxonomy
# --------------------------------------------------------------------------


def test_class_names_are_stable():
    assert class_names("binary") == ["not_damaged", "damaged"]
    assert class_names("multiclass")[0] == "not_damaged"
    assert set(class_names("multiclass")[1:]) == set(DAMAGE_CLASSES)


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        class_names("regression")


def test_rdd_code_mapping_covers_core_classes():
    for code in ("D00", "D10", "D20", "D40"):
        assert RDD_CODE_TO_CLASS[code] in MULTICLASS_CLASSES


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_expand_box_clips_to_image_bounds():
    assert expand_box((0, 0, 20, 20), 100, 100, 0.5) == (0, 0, 30, 30)
    assert expand_box((90, 90, 100, 100), 100, 100, 1.0) == (80, 80, 100, 100)


def test_boxes_overlap():
    assert boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15))
    assert not boxes_overlap((0, 0, 10, 10), (10, 10, 20, 20))


def test_negative_sampling_avoids_annotated_damage():
    import random

    rng = random.Random(0)
    positives = [(0, 0, 400, 400)]
    for _ in range(20):
        neg = sample_negative_box(480, 480, positives, rng, size=40)
        if neg is not None:
            assert not boxes_overlap(neg, positives[0])


def test_negative_sampling_returns_none_when_image_too_small():
    import random

    assert sample_negative_box(30, 30, [], random.Random(0), size=64) is None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parser_reads_synthetic_tree(raw_dataset: Path):
    df = build_annotation_table(raw_dataset, ["Japan", "India"])
    assert not df.empty
    assert {"image_path", "damage_class", "xmin", "country"} <= set(df.columns)
    assert df["country"].nunique() == 2


def test_parser_keeps_negative_images(raw_dataset: Path):
    df = build_annotation_table(raw_dataset, ["Japan", "India"])
    # images with no <object> appear once with a null damage code
    assert df["damage_code"].isna().any()


def test_parser_survives_corrupt_xml(tmp_path):
    images = tmp_path / "images"
    xmls = tmp_path / "xmls"
    images.mkdir()
    xmls.mkdir()
    (images / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (xmls / "a.xml").write_text("<annotation><unclosed>", encoding="utf-8")

    rows = parse_annotation_file(xmls / "a.xml", images, "Japan")
    assert rows == []  # skipped, not raised


def test_parser_ignores_degenerate_boxes(tmp_path):
    images = tmp_path / "images"
    xmls = tmp_path / "xmls"
    images.mkdir()
    xmls.mkdir()
    (images / "b.jpg").write_bytes(b"\xff\xd8\xff")
    (xmls / "b.xml").write_text(
        "<annotation><size><width>10</width><height>10</height></size>"
        "<object><name>D00</name><bndbox><xmin>5</xmin><ymin>5</ymin>"
        "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    rows = parse_annotation_file(xmls / "b.xml", images, "Japan")
    assert len(rows) == 1 and rows[0]["damage_code"] is None


def test_missing_raw_dir_returns_empty(tmp_path):
    assert build_annotation_table(tmp_path / "nope", ["Japan"]).empty


# --------------------------------------------------------------------------
# crops & splits
# --------------------------------------------------------------------------


def test_crops_are_written_and_indexed(prepared_config):
    import pandas as pd

    crops = pd.read_csv(Path(prepared_config.data.processed_dir) / "crops.csv")
    assert len(crops) > 0
    assert set(crops["label_binary"]) <= {"damaged", "not_damaged"}
    assert set(crops["label_multiclass"]) <= set(MULTICLASS_CLASSES)
    assert all(Path(p).exists() for p in crops["crop_path"].head(20))


def test_no_image_leaks_across_splits(prepared_config):
    """The single most important data test: an image contributes crops to
    exactly one split, otherwise test metrics are optimistically biased."""
    import pandas as pd

    crops = pd.read_csv(Path(prepared_config.data.processed_dir) / "crops.csv")
    per_image_splits = crops.groupby("image_id")["split"].nunique()
    assert (per_image_splits == 1).all()


def test_all_splits_present(prepared_config):
    import pandas as pd

    crops = pd.read_csv(Path(prepared_config.data.processed_dir) / "crops.csv")
    assert {"train", "val", "test"} <= set(crops["split"].unique())


def test_dataset_stats_written(prepared_config):
    from rdc.utils import load_json

    stats = load_json(Path(prepared_config.data.processed_dir) / "dataset_stats.json")
    assert stats["n_crops"] > 0
    assert "multiclass_counts" in stats and "binary_counts" in stats

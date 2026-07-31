"""Shared pytest fixtures.

Every fixture builds its own tiny synthetic dataset in a tmp_path, so the test
suite runs on a clean checkout with no downloads and no network.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (256, 256), (120, 120, 120))


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Valid JPEG magic bytes followed by garbage - the classic bad upload."""
    return b"\xff\xd8\xff\xe0" + b"not actually an image" * 20


@pytest.fixture
def text_file_bytes() -> bytes:
    return b"this is a CSV, not a photograph\n1,2,3\n"


@pytest.fixture
def raw_dataset(tmp_path) -> Path:
    """A miniature RDD2022-shaped raw tree."""
    from make_sample_data import generate

    raw = tmp_path / "raw"
    generate(str(raw), per_country=14, countries=["Japan", "India"], seed=7)
    return raw


@pytest.fixture
def config_factory(tmp_path):
    """Build a Config pointed at tmp_path with fast training settings."""

    def _build(raw_dir: Path, task: str = "binary", **overrides):
        from rdc.config import Config

        payload = {
            "data": {
                "raw_dir": str(raw_dir),
                "interim_dir": str(tmp_path / "interim"),
                "processed_dir": str(tmp_path / "processed"),
                "countries": ["Japan", "India"],
                "image_size": 64,
                "val_size": 0.2,
                "test_size": 0.2,
                "seed": 7,
                "negatives_per_image": 1,
                "min_box_size": 16,
                "num_workers": 0,
            },
            "model": {"backbone": "resnet18", "pretrained": False, "freeze_epochs": 0},
            "train": {
                "task": task,
                "epochs": 1,
                "batch_size": 8,
                "device": "cpu",
                "early_stopping_patience": 1,
                "seed": 7,
            },
            "tracking": {"enabled": False, "tracking_uri": f"file:{tmp_path / 'mlruns'}"},
            "output_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        }
        for section, values in overrides.items():
            payload.setdefault(section, {}).update(values)
        return Config.from_dict(payload)

    return _build


@pytest.fixture
def prepared_config(config_factory, raw_dataset, tmp_path):
    """A config whose crop dataset has already been built."""
    from rdc.data.prepare import extract_crops, stratified_group_split, write_outputs
    from rdc.data.rdd_parser import build_annotation_table

    cfg = config_factory(raw_dataset)
    ann = build_annotation_table(cfg.data.raw_dir, cfg.data.countries)
    crops = extract_crops(ann, cfg)
    crops = stratified_group_split(crops, cfg)
    write_outputs(crops, cfg)
    return cfg


@pytest.fixture
def trained_checkpoint(tmp_path):
    """A tiny randomly-initialised checkpoint - enough to exercise loading,
    inference and the API without spending time on real training."""
    from rdc.config import BINARY_CLASSES
    from rdc.models.factory import build_model, save_checkpoint

    model = build_model("resnet18", num_classes=len(BINARY_CLASSES), pretrained=False)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, model=model, classes=list(BINARY_CLASSES), task="binary",
        backbone="resnet18", image_size=64, metrics={"val": {"macro_f1": 0.0}},
    )
    return path

"""Configuration objects for the GOV-01 road damage classifier.

All runtime configuration is loaded from a YAML file so that every experiment
is reproducible from a single artefact that can be committed to git.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Label scopes
# --------------------------------------------------------------------------
# The brief (Section 3 / Question 1 & 4) requires an explicit, defensible
# category scope. We ship two heads:
#
#   binary     -> damaged / not_damaged      (primary, reliable triage signal)
#   multiclass -> RDD2022 damage taxonomy    (secondary, finer-grained triage)
#
# RDD2022 damage codes used (the four internationally consistent classes):
#   D00 longitudinal crack
#   D10 transverse crack
#   D20 alligator crack
#   D40 pothole
# --------------------------------------------------------------------------

BINARY_CLASSES: list[str] = ["not_damaged", "damaged"]

MULTICLASS_CLASSES: list[str] = [
    "not_damaged",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]

RDD_CODE_TO_CLASS: dict[str, str] = {
    "D00": "longitudinal_crack",
    "D01": "longitudinal_crack",
    "D10": "transverse_crack",
    "D11": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    "D43": "not_damaged",  # crosswalk blur - not a damage class
    "D44": "not_damaged",  # white line blur - not a damage class
    "D50": "not_damaged",  # utility hole - not a damage class
}

DAMAGE_CLASSES: list[str] = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]


def class_names(task: str) -> list[str]:
    """Return the ordered class list for a task ('binary' or 'multiclass')."""
    if task == "binary":
        return list(BINARY_CLASSES)
    if task == "multiclass":
        return list(MULTICLASS_CLASSES)
    raise ValueError(f"Unknown task {task!r}; expected 'binary' or 'multiclass'.")


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    countries: list[str] = field(default_factory=lambda: ["Japan", "India", "Norway"])
    image_size: int = 224
    # Fraction of the *image-level* population (never sample-level) held out.
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    # Negative (not_damaged) crops are sampled from annotation-free regions.
    negatives_per_image: int = 1
    min_box_size: int = 24
    crop_context: float = 0.25  # expand each box by this fraction for context
    num_workers: int = 2


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    # Freeze the backbone for the first N epochs (linear probing warm-up).
    freeze_epochs: int = 1


@dataclass
class TrainConfig:
    task: str = "binary"
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    head_lr_multiplier: float = 10.0
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    # Class imbalance strategy: "none" | "weighted_loss" | "weighted_sampler"
    imbalance_strategy: str = "weighted_loss"
    early_stopping_patience: int = 4
    mixed_precision: bool = False
    grad_clip: float = 1.0
    monitor_metric: str = "macro_f1"
    device: str = "auto"
    seed: int = 42


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    rotation_degrees: float = 8.0
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    random_resized_crop_scale: list[float] = field(default_factory=lambda: [0.7, 1.0])
    gaussian_blur_p: float = 0.15
    random_erasing_p: float = 0.10


@dataclass
class TrackingConfig:
    enabled: bool = True
    experiment_name: str = "gov01-road-damage"
    tracking_uri: str = "file:./mlruns"
    run_name: str | None = None


@dataclass
class InferenceConfig:
    model_path: str = "models/binary/model.pt"
    # Below this max-probability the API returns needs_human_review = True.
    confidence_threshold: float = 0.60
    max_upload_mb: float = 10.0
    allowed_content_types: list[str] = field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    )
    top_k: int = 3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: str = "models"
    reports_dir: str = "reports"

    # ---------------- serialisation helpers ----------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        def build(dc_type, key):
            payload = raw.get(key) or {}
            valid = {f.name for f in dataclasses.fields(dc_type)}
            unknown = set(payload) - valid
            if unknown:
                raise ValueError(f"Unknown keys in '{key}' config section: {sorted(unknown)}")
            return dc_type(**payload)

        return cls(
            data=build(DataConfig, "data"),
            model=build(ModelConfig, "model"),
            train=build(TrainConfig, "train"),
            augmentation=build(AugmentationConfig, "augmentation"),
            tracking=build(TrackingConfig, "tracking"),
            inference=build(InferenceConfig, "inference"),
            output_dir=raw.get("output_dir", "models"),
            reports_dir=raw.get("reports_dir", "reports"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def flat_params(self) -> dict[str, Any]:
        """Flatten config for experiment-tracking parameter logging."""
        flat: dict[str, Any] = {}
        for section, payload in self.to_dict().items():
            if isinstance(payload, dict):
                for key, value in payload.items():
                    flat[f"{section}.{key}"] = value
            else:
                flat[section] = payload
        return flat

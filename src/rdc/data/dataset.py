"""Torch Dataset, transforms and DataLoader factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from ..config import Config, class_names
from ..utils import get_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOG = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------


def build_transforms(cfg: Config, train: bool):
    """Augmentations directly target the failure modes named in the brief
    (Section 7 / Question 3): lighting, angle and image-quality variation."""
    size = cfg.data.image_size
    aug = cfg.augmentation

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    ops = [
        transforms.RandomResizedCrop(
            size, scale=tuple(aug.random_resized_crop_scale), ratio=(0.8, 1.25)
        ),
        transforms.RandomHorizontalFlip(p=aug.horizontal_flip),
        transforms.RandomRotation(degrees=aug.rotation_degrees),
        transforms.ColorJitter(
            brightness=aug.color_jitter_brightness,
            contrast=aug.color_jitter_contrast,
            saturation=aug.color_jitter_saturation,
        ),
    ]
    if aug.gaussian_blur_p > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                p=aug.gaussian_blur_p,
            )
        )
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug.random_erasing_p > 0:
        ops.append(transforms.RandomErasing(p=aug.random_erasing_p, scale=(0.02, 0.12)))
    return transforms.Compose(ops)


def inference_transform(image_size: int = 224):
    """Deterministic transform shared by evaluation and the API, so training
    and serving cannot silently diverge."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


class RoadDamageDataset(Dataset):
    """Crop-level dataset backed by the CSV written by `prepare.py`."""

    def __init__(
        self,
        frame: pd.DataFrame,
        task: str,
        transform=None,
        classes: list[str] | None = None,
        image_size: int = 224,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.transform = transform
        self.image_size = image_size
        self.classes = classes or class_names(task)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.label_column = "label_binary" if task == "binary" else "label_multiclass"

        missing = set(self.frame[self.label_column]) - set(self.class_to_idx)
        if missing:
            raise ValueError(f"Labels not in the {task} class list: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def targets(self) -> np.ndarray:
        return self.frame[self.label_column].map(self.class_to_idx).to_numpy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        try:
            with Image.open(row["crop_path"]) as im:
                image = im.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            # Never crash a training run over one bad file (Requirement 7.3).
            LOG.warning("Unreadable crop %s (%s) - substituting grey patch", row["crop_path"], exc)
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.class_to_idx[row[self.label_column]])


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def class_distribution(frame: pd.DataFrame, task: str) -> dict[str, int]:
    column = "label_binary" if task == "binary" else "label_multiclass"
    return frame[column].value_counts().to_dict()


def compute_class_weights(dataset: RoadDamageDataset) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale (and
    therefore the learning rate) stays comparable across strategies."""
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _sampler(dataset: RoadDamageDataset) -> WeightedRandomSampler:
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    per_sample = (1.0 / counts)[dataset.targets]
    return WeightedRandomSampler(
        weights=torch.tensor(per_sample, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def load_frame(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.processed_dir) / "crops.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m rdc.data.prepare --config <cfg>` first."
        )
    return pd.read_csv(path)


def build_dataloaders(
    cfg: Config,
    frame: pd.DataFrame | None = None,
) -> tuple[dict[str, DataLoader], dict[str, RoadDamageDataset]]:
    frame = load_frame(cfg) if frame is None else frame
    task = cfg.train.task
    classes = class_names(task)

    datasets: dict[str, RoadDamageDataset] = {}
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        if subset.empty:
            LOG.warning("Split '%s' is empty", split)
            continue
        is_train = split == "train"
        datasets[split] = RoadDamageDataset(
            subset,
            task=task,
            transform=build_transforms(cfg, is_train),
            classes=classes,
            image_size=cfg.data.image_size,
        )

    if "train" in datasets:
        use_sampler = cfg.train.imbalance_strategy == "weighted_sampler"
        loaders["train"] = DataLoader(
            datasets["train"],
            batch_size=cfg.train.batch_size,
            shuffle=not use_sampler,
            sampler=_sampler(datasets["train"]) if use_sampler else None,
            num_workers=cfg.data.num_workers,
            pin_memory=False,
            drop_last=False,
        )
    for split in ("val", "test"):
        if split in datasets:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
            )
    return loaders, datasets

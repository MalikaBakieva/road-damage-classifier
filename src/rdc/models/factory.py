"""Model construction, checkpoint save/load.

Transfer learning: an ImageNet-pretrained backbone from `timm` with a fresh
classification head. Road-damage crops are texture-heavy and our labelled set is
small, so training from scratch is not competitive with fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils import get_logger

LOG = get_logger(__name__)

CHECKPOINT_VERSION = 2


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a timm backbone with a fresh head.

    Falls back to torchvision when timm is unavailable, and to random init when
    pretrained weights cannot be downloaded (offline CI), so the pipeline always
    runs end-to-end.
    """
    try:
        import timm

        try:
            model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
        except Exception as exc:  # offline / weights unavailable
            LOG.warning("Pretrained weights unavailable (%s); using random init.", exc)
            model = timm.create_model(
                backbone, pretrained=False, num_classes=num_classes, drop_rate=dropout
            )
        return model
    except ImportError:
        LOG.warning("timm not installed - falling back to torchvision resnet18.")
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model


def classifier_parameter_names(model: nn.Module) -> list[str]:
    """Best-effort identification of head parameters for discriminative LRs."""
    try:
        import timm  # noqa: F401

        head = model.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        return [n for n, p in model.named_parameters() if id(p) in head_ids]
    except Exception:
        return [
            n for n, _ in model.named_parameters() if n.startswith(("fc.", "head.", "classifier."))
        ]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head = set(classifier_parameter_names(model))
    for name, param in model.named_parameters():
        param.requires_grad = True if name in head else trainable


def param_groups(model: nn.Module, lr: float, head_multiplier: float, weight_decay: float):
    head = set(classifier_parameter_names(model))
    backbone_params = [p for n, p in model.named_parameters() if n not in head]
    head_params = [p for n, p in model.named_parameters() if n in head]
    groups = [{"params": backbone_params, "lr": lr, "weight_decay": weight_decay}]
    if head_params:
        groups.append(
            {"params": head_params, "lr": lr * head_multiplier, "weight_decay": weight_decay}
        )
    return groups


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    classes: list[str],
    task: str,
    backbone: str,
    image_size: int,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "classes": list(classes),
        "task": task,
        "backbone": backbone,
        "image_size": image_size,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    LOG.info("Saved checkpoint -> %s", path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. Train one with "
            "`python -m rdc.train --config configs/binary.yaml` or run "
            "`python scripts/train_demo_model.py`."
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in payload or "classes" not in payload:
        raise ValueError(f"'{path}' is not a valid rdc checkpoint.")

    model = build_model(
        backbone=payload.get("backbone", "resnet18"),
        num_classes=len(payload["classes"]),
        pretrained=False,
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta

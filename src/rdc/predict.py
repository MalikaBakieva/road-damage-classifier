"""Inference layer shared by the CLI, the API and the Streamlit demo.

Everything that could differ between training and serving lives here exactly
once: the preprocessing transform, the class list and the confidence policy.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError

from .data.dataset import inference_transform
from .models.factory import load_checkpoint
from .utils import get_logger, resolve_device

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOG = get_logger(__name__)

MAX_PIXELS = 40_000_000  # decompression-bomb guard


class InvalidImageError(ValueError):
    """Raised for unreadable, unsupported or absurdly large inputs."""


def open_image(source: str | Path | bytes | io.BytesIO) -> Image.Image:
    """Decode an image defensively. Requirement 7.3: never crash on bad input."""
    try:
        if isinstance(source, (bytes, bytearray)):
            buffer = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            buffer = source
        else:
            path = Path(source)
            if not path.exists():
                raise InvalidImageError(f"File not found: {path}")
            buffer = io.BytesIO(path.read_bytes())

        if buffer.getbuffer().nbytes == 0:
            raise InvalidImageError("Empty file - no image data.")

        with Image.open(buffer) as probe:
            width, height = probe.size
            fmt = probe.format
            if width * height > MAX_PIXELS:
                raise InvalidImageError(
                    f"Image too large ({width}x{height} px); refusing to decode."
                )
            probe.load()
            image = probe.convert("RGB")
        image.info["source_format"] = fmt
        return image
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Unsupported or corrupt image file: {exc}") from exc


class RoadDamagePredictor:
    """Loads a checkpoint once and serves predictions."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        confidence_threshold: float = 0.6,
        top_k: int = 3,
    ) -> None:
        self.device = resolve_device(device)
        self.model, self.meta = load_checkpoint(model_path, self.device)
        self.classes: list[str] = list(self.meta["classes"])
        self.task: str = self.meta.get("task", "binary")
        self.image_size: int = int(self.meta.get("image_size", 224))
        self.backbone: str = self.meta.get("backbone", "unknown")
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.top_k = max(1, min(int(top_k), len(self.classes)))
        self.transform = inference_transform(self.image_size)
        LOG.info(
            "Predictor ready: task=%s backbone=%s classes=%s device=%s",
            self.task, self.backbone, self.classes, self.device,
        )

    # ---------------- core ----------------

    @torch.no_grad()
    def _forward(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        logits = self.model(batch)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        probs = self._forward([image])[0]
        return self._format(probs, image, time.perf_counter() - started)

    def predict(self, source: str | Path | bytes | io.BytesIO) -> dict[str, Any]:
        return self.predict_image(open_image(source))

    def predict_batch(self, sources: Iterable[str | Path | bytes]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for source in sources:
            try:
                results.append({"input": str(source)[:200], **self.predict(source)})
            except InvalidImageError as exc:
                results.append({"input": str(source)[:200], "error": str(exc)})
        return results

    # ---------------- formatting ----------------

    def _format(self, probs: np.ndarray, image: Image.Image, elapsed: float) -> dict[str, Any]:
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        confidence = float(probs[top_idx])
        label = self.classes[top_idx]

        # Binary triage signal is always derivable, even from the multiclass head.
        is_damaged = label != "not_damaged"
        damage_probability = float(
            sum(p for c, p in zip(self.classes, probs, strict=True) if c != "not_damaged")
        )

        return {
            "task": self.task,
            "predicted_class": label,
            "confidence": round(confidence, 4),
            "is_damaged": bool(is_damaged),
            "damage_probability": round(damage_probability, 4),
            "needs_human_review": bool(confidence < self.confidence_threshold),
            "confidence_threshold": self.confidence_threshold,
            "priority": self._priority(label, damage_probability, confidence),
            "top_k": [
                {"class": self.classes[int(i)], "probability": round(float(probs[int(i)]), 4)}
                for i in order[: self.top_k]
            ],
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs, strict=True)},
            "model": {
                "path": self.model_path,
                "backbone": self.backbone,
                "task": self.task,
                "image_size": self.image_size,
            },
            "image": {
                "width": image.width,
                "height": image.height,
                "format": image.info.get("source_format"),
            },
            "inference_ms": round(elapsed * 1000, 2),
            "disclaimer": (
                "Automated triage support only. This output is not an engineering "
                "safety assessment and must be confirmed by a qualified inspector."
            ),
        }

    @staticmethod
    def _priority(label: str, damage_probability: float, confidence: float) -> str:
        """Map a prediction to an inspection queue bucket.

        Potholes are the most immediate hazard, so they outrank cracks; anything
        the model is unsure about goes to `review` rather than being dropped.
        """
        if confidence < 0.5:
            return "review"
        if label == "pothole":
            return "high"
        if label in ("alligator_crack",):
            return "medium"
        if label in ("longitudinal_crack", "transverse_crack"):
            return "low"
        if label == "damaged":
            return "high" if damage_probability >= 0.85 else "medium"
        return "none"

    # ---------------- introspection ----------------

    def info(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "classes": self.classes,
            "backbone": self.backbone,
            "image_size": self.image_size,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "training_metrics": self.meta.get("metrics", {}),
        }


_PREDICTOR: RoadDamagePredictor | None = None


def get_predictor(
    model_path: str | None = None,
    device: str = "auto",
    confidence_threshold: float = 0.6,
    reload: bool = False,
) -> RoadDamagePredictor:
    """Process-wide singleton so the API loads weights only once."""
    global _PREDICTOR
    if _PREDICTOR is None or reload:
        _PREDICTOR = RoadDamagePredictor(
            model_path or "models/binary/model.pt",
            device=device,
            confidence_threshold=confidence_threshold,
        )
    return _PREDICTOR


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify road images from the CLI.")
    parser.add_argument("images", nargs="+", help="Image paths or a directory")
    parser.add_argument("--model", default="models/binary/model.pt")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    paths: list[str] = []
    for item in args.images:
        p = Path(item)
        if p.is_dir():
            paths += [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in
                      {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        else:
            paths.append(item)

    predictor = RoadDamagePredictor(args.model, confidence_threshold=args.threshold)
    results = predictor.predict_batch(paths)
    text = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {len(results)} results to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

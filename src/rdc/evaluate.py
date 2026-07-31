"""Evaluate a trained checkpoint on the held-out test split.

    python -m rdc.evaluate --config configs/binary.yaml --model models/binary/model.pt

Produces `reports/<task>/`:
    metrics.json          full metric payload
    classification_report.txt
    confusion_matrix.png
    threshold_sweep.csv   coverage/accuracy trade-off for the API threshold
    robustness.json       accuracy under simulated lighting/blur/angle shifts
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .config import Config, class_names
from .data.dataset import build_dataloaders, load_frame
from .metrics import compute_metrics, expected_calibration_error, text_report, threshold_sweep
from .models.factory import load_checkpoint
from .utils import get_logger, resolve_device, save_json, set_seed

LOG = get_logger(__name__)


@torch.no_grad()
def predict_loader(model, loader, device: str):
    model.eval()
    probs, targets = [], []
    for images, y in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not installed - skipping confusion-matrix plot.")
        return

    cm_arr = np.array(cm, dtype=float)
    norm = cm_arr / np.clip(cm_arr.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 1.4 * len(classes) + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm_arr[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def robustness_check(model, cfg: Config, device: str, classes: list[str]) -> dict:
    """Approximate the distribution shifts named in the brief (Question 3/6):
    darker/brighter capture, motion blur, and off-axis angle. Accuracy drop
    under each corruption is reported as a documented limitation, not hidden."""
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from .data.dataset import IMAGENET_MEAN, IMAGENET_STD, RoadDamageDataset

    size = cfg.data.image_size
    base = [transforms.Resize((size, size))]
    tail = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    corruptions = {
        "clean": [],
        "dark": [transforms.ColorJitter(brightness=(0.4, 0.4))],
        "bright": [transforms.ColorJitter(brightness=(1.6, 1.6))],
        "low_contrast": [transforms.ColorJitter(contrast=(0.45, 0.45))],
        "blur": [transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0))],
        "rotate15": [transforms.RandomRotation((15, 15))],
        "downscale": [transforms.Resize((size // 4, size // 4)), transforms.Resize((size, size))],
    }

    frame = load_frame(cfg)
    test_frame = frame[frame["split"] == "test"]
    if test_frame.empty:
        return {}

    results = {}
    for name, ops in corruptions.items():
        tf = transforms.Compose(base + ops + tail)
        ds = RoadDamageDataset(test_frame, cfg.train.task, transform=tf, classes=classes)
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
        probs, y = predict_loader(model, loader, device)
        m = compute_metrics(y, probs.argmax(1), probs, classes)
        results[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        LOG.info("robustness %-12s acc %.4f  macro_f1 %.4f", name, m["accuracy"], m["macro_f1"])
    return results


def evaluate(config_path: str, model_path: str | None = None, split: str = "test") -> dict:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    classes = class_names(cfg.train.task)

    model_path = model_path or str(Path(cfg.output_dir) / cfg.train.task / "model.pt")
    model, meta = load_checkpoint(model_path, device)
    if meta.get("classes") != classes:
        LOG.warning("Checkpoint classes %s differ from config classes %s",
                    meta.get("classes"), classes)
        classes = meta["classes"]

    loaders, _ = build_dataloaders(cfg)
    if split not in loaders:
        raise SystemExit(f"Split '{split}' is empty - nothing to evaluate.")

    probs, y_true = predict_loader(model, loaders[split], device)
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, probs, classes)
    metrics["expected_calibration_error"] = expected_calibration_error(y_true, probs)
    metrics["split"] = split
    metrics["n_samples"] = int(len(y_true))
    metrics["model_path"] = str(model_path)

    sweep = threshold_sweep(y_true, probs)
    metrics["threshold_sweep"] = sweep

    reports = Path(cfg.reports_dir) / cfg.train.task
    reports.mkdir(parents=True, exist_ok=True)

    report_text = text_report(y_true, y_pred, classes)
    (reports / "classification_report.txt").write_text(report_text, encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], classes, reports / "confusion_matrix.png")

    with open(reports / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    robustness = robustness_check(model, cfg, device, classes)
    if robustness:
        save_json(robustness, reports / "robustness.json")
        metrics["robustness"] = robustness

    save_json(metrics, reports / "metrics.json")

    LOG.info("\n%s", report_text)
    LOG.info("macro_f1=%.4f  accuracy=%.4f  ECE=%.4f",
             metrics["macro_f1"], metrics["accuracy"], metrics["expected_calibration_error"])
    LOG.info("Reports written to %s", reports)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/binary.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.config, args.model, args.split)


if __name__ == "__main__":
    main()

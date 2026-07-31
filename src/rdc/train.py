"""Training entrypoint.

    python -m rdc.train --config configs/binary.yaml

Design notes
------------
* Transfer learning with a frozen-backbone warm-up, then full fine-tuning.
* Class imbalance handled by inverse-frequency loss weights (default) or a
  weighted sampler - selectable from config so the choice can be defended
  with numbers rather than assertion.
* Model selection on **validation macro-F1**, never on test. The test split is
  touched exactly once, by `rdc.evaluate`.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config, class_names
from .data.dataset import build_dataloaders, class_distribution, compute_class_weights, load_frame
from .metrics import compute_metrics, text_report
from .models.factory import build_model, param_groups, save_checkpoint, set_backbone_trainable
from .tracking import track
from .utils import count_parameters, get_logger, resolve_device, save_json, set_seed

LOG = get_logger(__name__)


# --------------------------------------------------------------------------
# epoch loops
# --------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: str,
    optimizer=None,
    grad_clip: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch = targets.size(0)
            total_loss += loss.item() * batch
            n += batch
            all_probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 1))
    y_true = np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=int)
    y_pred = probs.argmax(axis=1) if len(probs) else np.zeros((0,), dtype=int)
    return total_loss / max(n, 1), y_true, y_pred, probs


# --------------------------------------------------------------------------
# main training routine
# --------------------------------------------------------------------------


def train(config_path: str, override_task: str | None = None) -> dict:
    cfg = Config.from_yaml(config_path)
    if override_task:
        cfg.train.task = override_task

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    classes = class_names(cfg.train.task)
    LOG.info("Task=%s | device=%s | classes=%s", cfg.train.task, device, classes)

    frame = load_frame(cfg)
    loaders, datasets = build_dataloaders(cfg, frame)
    if "train" not in loaders or "val" not in loaders:
        raise SystemExit("Both a train and a val split are required. Re-run data preparation.")

    LOG.info("train dist: %s", class_distribution(datasets["train"].frame, cfg.train.task))
    LOG.info("val   dist: %s", class_distribution(datasets["val"].frame, cfg.train.task))

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=len(classes),
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)

    # ---- loss with optional class weighting ----
    weight = None
    if cfg.train.imbalance_strategy == "weighted_loss":
        weight = compute_class_weights(datasets["train"]).to(device)
        LOG.info("Class weights: %s", dict(zip(classes, weight.cpu().numpy().round(3), strict=True)))
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.train.lr, cfg.train.head_lr_multiplier, cfg.train.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs, 1)
    )

    best_score, best_state, best_epoch, patience = -1.0, None, -1, 0
    history = []
    reports_dir = Path(cfg.reports_dir) / cfg.train.task
    reports_dir.mkdir(parents=True, exist_ok=True)

    with track(cfg) as tracker:
        tracker.log_params(cfg.flat_params())
        tracker.log_params(
            {
                "n_train": len(datasets["train"]),
                "n_val": len(datasets["val"]),
                "trainable_params": count_parameters(model),
                "device": device,
            }
        )
        tracker.set_tags({"task": cfg.train.task, "backbone": cfg.model.backbone})

        for epoch in range(1, cfg.train.epochs + 1):
            if cfg.model.freeze_epochs:
                set_backbone_trainable(model, trainable=epoch > cfg.model.freeze_epochs)

            t0 = time.time()
            train_loss, ytr, ptr, _ = run_epoch(
                model, loaders["train"], criterion, device, optimizer, cfg.train.grad_clip
            )
            val_loss, yv, pv, probv = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step()

            train_metrics = compute_metrics(ytr, ptr, None, classes)
            val_metrics = compute_metrics(yv, pv, probv, classes)
            score = val_metrics[cfg.train.monitor_metric]

            LOG.info(
                "epoch %02d | train_loss %.4f f1 %.4f | val_loss %.4f f1 %.4f acc %.4f | %.1fs",
                epoch, train_loss, train_metrics["macro_f1"],
                val_loss, val_metrics["macro_f1"], val_metrics["accuracy"], time.time() - t0,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_loss": val_loss,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            tracker.log_metrics(history[-1], step=epoch)

            if score > best_score:
                best_score, best_epoch, patience = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= cfg.train.early_stopping_patience:
                    LOG.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, yv, pv, probv = run_epoch(model, loaders["val"], criterion, device)
        final_val = compute_metrics(yv, pv, probv, classes)
        LOG.info("Best val %s = %.4f (epoch %d)", cfg.train.monitor_metric, best_score, best_epoch)
        LOG.info("\n%s", text_report(yv, pv, classes))

        out_dir = Path(cfg.output_dir) / cfg.train.task
        ckpt = save_checkpoint(
            out_dir / "model.pt",
            model=model,
            classes=classes,
            task=cfg.train.task,
            backbone=cfg.model.backbone,
            image_size=cfg.data.image_size,
            metrics={"val": final_val, "best_epoch": best_epoch},
            extra={"config": cfg.to_dict(), "imbalance_strategy": cfg.train.imbalance_strategy},
        )
        cfg.to_yaml(out_dir / "config_used.yaml")
        save_json({"history": history, "val_metrics": final_val}, reports_dir / "training.json")

        tracker.log_metrics({f"final_val_{k}": v for k, v in final_val.items()
                             if isinstance(v, (int, float))})
        tracker.log_artifact(ckpt)
        tracker.log_artifact(reports_dir / "training.json")

    return {"checkpoint": str(ckpt), "val_metrics": final_val, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the road-damage classifier.")
    parser.add_argument("--config", default="configs/binary.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass"], default=None)
    args = parser.parse_args()
    result = train(args.config, args.task)
    print(json.dumps({"checkpoint": result["checkpoint"],
                      "val_macro_f1": result["val_metrics"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()

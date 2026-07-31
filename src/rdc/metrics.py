"""Evaluation metrics.

Primary metric: **macro-F1**.

Rationale (brief Question 5): the classes are heavily imbalanced and every
damage type matters to the client. Plain accuracy is dominated by the largest
class, and micro-F1 has the same problem. Macro-F1 weights every class equally,
so a model that ignores potholes cannot hide behind good crack performance.
For the binary head we additionally report recall on `damaged`, because a missed
defect is more costly to the department than a false alarm sent to triage.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    classes: list[str],
) -> dict:
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": list(classes),
    }

    if "damaged" in classes:
        idx = classes.index("damaged")
        metrics["damaged_recall"] = float(recall[idx])
        metrics["damaged_precision"] = float(precision[idx])

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass  # a class missing from this split - AUC undefined, not fatal

    return metrics


def text_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0, digits=4,
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over max-probability. Needed because the API exposes a confidence
    threshold - an over-confident model would route nothing to human review."""
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float] | None = None
) -> list[dict]:
    """Coverage/accuracy trade-off for the API confidence threshold: how much
    of the queue can be auto-triaged, and how accurate is that portion?"""
    thresholds = thresholds or [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    rows = []
    for t in thresholds:
        mask = confidences >= t
        coverage = float(mask.mean())
        acc = float((predictions[mask] == y_true[mask]).mean()) if mask.sum() else float("nan")
        rows.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "accuracy_on_covered": acc,
                "n_auto": int(mask.sum()),
                "n_to_human": int((~mask).sum()),
            }
        )
    return rows

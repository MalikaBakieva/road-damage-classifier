"""Tests for the model factory, metrics and the training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rdc.config import BINARY_CLASSES
from rdc.metrics import (
    compute_metrics,
    expected_calibration_error,
    text_report,
    threshold_sweep,
)
from rdc.models.factory import build_model, load_checkpoint, save_checkpoint

# --------------------------------------------------------------------------
# model factory
# --------------------------------------------------------------------------


def test_model_output_shape_matches_class_count():
    model = build_model("resnet18", num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("resnet18", num_classes=2, pretrained=False)
    path = save_checkpoint(
        tmp_path / "m.pt",
        model=model,
        classes=list(BINARY_CLASSES),
        task="binary",
        backbone="resnet18",
        image_size=64,
    )
    loaded, meta = load_checkpoint(path)
    assert meta["classes"] == list(BINARY_CLASSES)
    assert meta["task"] == "binary"

    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_missing_checkpoint_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="rdc.train"):
        load_checkpoint(tmp_path / "absent.pt")


def test_invalid_checkpoint_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"not": "a checkpoint"}, path)
    with pytest.raises(ValueError, match="not a valid rdc checkpoint"):
        load_checkpoint(path)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, None, ["not_damaged", "damaged"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["damaged_recall"] == 1.0


def test_macro_f1_punishes_ignoring_a_minority_class():
    """The reason macro-F1 is the primary metric: a model that always predicts
    the majority class must not look good."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    m = compute_metrics(y_true, y_pred, None, ["not_damaged", "damaged"])
    assert m["accuracy"] == pytest.approx(0.90)
    assert m["macro_f1"] < 0.50
    assert m["damaged_recall"] == 0.0


def test_confusion_matrix_shape():
    classes = ["a", "b", "c"]
    y = np.array([0, 1, 2, 0])
    m = compute_metrics(y, y, None, classes)
    assert np.array(m["confusion_matrix"]).shape == (3, 3)


def test_roc_auc_present_for_binary():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    m = compute_metrics(y, probs.argmax(1), probs, ["not_damaged", "damaged"])
    assert m["roc_auc"] == pytest.approx(1.0)


def test_calibration_error_bounds():
    y = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    ece = expected_calibration_error(y, probs)
    assert 0.0 <= ece <= 1.0


def test_threshold_sweep_coverage_is_monotonic():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2], size=200)
    y = probs.argmax(1)
    sweep = threshold_sweep(y, probs)
    coverages = [row["coverage"] for row in sweep]
    assert coverages == sorted(coverages, reverse=True)
    assert sweep[0]["coverage"] == 1.0


def test_text_report_lists_every_class():
    y = np.array([0, 1])
    report = text_report(y, y, ["not_damaged", "damaged"])
    assert "not_damaged" in report and "damaged" in report


# --------------------------------------------------------------------------
# training loop (fast, CPU, 1 epoch)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_training_produces_a_loadable_checkpoint(prepared_config, tmp_path):
    from rdc.train import train

    config_path = tmp_path / "cfg.yaml"
    prepared_config.to_yaml(config_path)

    result = train(str(config_path))
    model, meta = load_checkpoint(result["checkpoint"])
    assert meta["classes"] == list(BINARY_CLASSES)
    assert 0.0 <= result["val_metrics"]["macro_f1"] <= 1.0
    assert len(result["history"]) >= 1


@pytest.mark.slow
def test_evaluate_writes_reports(prepared_config, tmp_path):
    from rdc.evaluate import evaluate
    from rdc.train import train

    config_path = tmp_path / "cfg.yaml"
    prepared_config.to_yaml(config_path)
    train(str(config_path))

    metrics = evaluate(str(config_path))
    reports = tmp_path / "reports" / "binary"
    assert (reports / "metrics.json").exists()
    assert (reports / "classification_report.txt").exists()
    assert (reports / "threshold_sweep.csv").exists()
    assert metrics["split"] == "test"
    assert "robustness" in metrics

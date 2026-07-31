"""Config loading, validation and reproducibility guarantees."""

from __future__ import annotations

import pytest

from rdc.config import Config, DataConfig
from rdc.utils import set_seed


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.data.image_size == 224
    assert cfg.train.task == "binary"
    assert cfg.train.monitor_metric == "macro_f1"
    # splits must leave a majority for training
    assert cfg.data.val_size + cfg.data.test_size < 0.5


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.train.epochs = 3
    cfg.model.backbone = "efficientnet_b0"
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)

    reloaded = Config.from_yaml(path)
    assert reloaded.train.epochs == 3
    assert reloaded.model.backbone == "efficientnet_b0"
    assert reloaded.to_dict() == cfg.to_dict()


def test_unknown_key_is_rejected():
    """A typo in a config must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="Unknown keys"):
        Config.from_dict({"train": {"epocs": 5}})


def test_flat_params_are_flat():
    flat = Config().flat_params()
    assert "train.epochs" in flat
    assert "data.image_size" in flat
    assert not any(isinstance(v, dict) for v in flat.values())


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    binary = Config.from_yaml(root / "binary.yaml")
    multiclass = Config.from_yaml(root / "multiclass.yaml")
    assert binary.train.task == "binary"
    assert multiclass.train.task == "multiclass"


def test_seeding_is_reproducible():
    import numpy as np
    import torch

    set_seed(123)
    a_np, a_torch = np.random.rand(4), torch.randn(4)
    set_seed(123)
    b_np, b_torch = np.random.rand(4), torch.randn(4)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)


def test_dataconfig_is_a_dataclass_with_defaults():
    assert DataConfig().seed == 42

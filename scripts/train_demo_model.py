"""One-command demo: synthetic data -> prepare -> train -> evaluate.

    python scripts/train_demo_model.py

Produces `models/binary/model.pt` so the API and the Streamlit UI have
something to load immediately after cloning.

The resulting weights are trained on SYNTHETIC images. They demonstrate that
the pipeline works end to end; they are not a road-damage model. Train on real
RDD2022 data (docs/DATASET.md) before quoting any performance number.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rdc.config import Config  # noqa: E402
from rdc.data.prepare import prepare  # noqa: E402
from rdc.evaluate import evaluate  # noqa: E402
from rdc.train import train  # noqa: E402
from rdc.utils import get_logger  # noqa: E402

LOG = get_logger("demo")


def build_demo_config(task: str, epochs: int, workdir: Path) -> Path:
    cfg = Config.from_yaml(ROOT / "configs" / f"{task}.yaml")
    cfg.data.raw_dir = str(workdir / "raw")
    cfg.data.interim_dir = str(workdir / "interim")
    cfg.data.processed_dir = str(workdir / "processed")
    cfg.data.countries = ["Japan", "India", "Norway"]
    cfg.data.image_size = 160
    cfg.data.num_workers = 0
    cfg.model.pretrained = True   # falls back to random init when offline
    cfg.model.freeze_epochs = 0
    cfg.train.epochs = epochs
    cfg.train.batch_size = 32
    cfg.train.device = "cpu"
    cfg.train.early_stopping_patience = epochs
    cfg.tracking.enabled = True
    cfg.tracking.run_name = f"demo-{task}"
    cfg.output_dir = str(ROOT / "models")
    cfg.reports_dir = str(ROOT / "reports")

    path = workdir / f"demo_{task}.yaml"
    cfg.to_yaml(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--per-country", type=int, default=110)
    parser.add_argument("--workdir", default=str(ROOT / "data" / "demo"))
    parser.add_argument("--keep-data", action="store_true", help="Keep the synthetic images")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    LOG.info("[1/4] Generating synthetic RDD2022-shaped data ...")
    from make_sample_data import generate

    generate(str(workdir / "raw"), args.per_country, ["Japan", "India", "Norway"], seed=42)

    config_path = build_demo_config(args.task, args.epochs, workdir)

    LOG.info("[2/4] Preparing crops ...")
    stats = prepare(str(config_path))

    LOG.info("[3/4] Training (%d epochs, CPU) ...", args.epochs)
    result = train(str(config_path))

    LOG.info("[4/4] Evaluating on the held-out test split ...")
    metrics = evaluate(str(config_path))

    if not args.keep_data:
        shutil.rmtree(workdir / "raw", ignore_errors=True)
        shutil.rmtree(workdir / "interim", ignore_errors=True)

    print("\n" + "=" * 68)
    print("DEMO COMPLETE  (synthetic data — not a real-world result)")
    print("=" * 68)
    print(f"crops           : {stats['n_crops']} from {stats['n_images']} images")
    print(f"checkpoint      : {result['checkpoint']}")
    print(f"test macro-F1   : {metrics['macro_f1']:.4f}")
    print(f"test accuracy   : {metrics['accuracy']:.4f}")
    print(f"reports         : reports/{args.task}/")
    print("\nNext:  uvicorn rdc.api.main:app --reload   ->   http://localhost:8000/docs")


if __name__ == "__main__":
    main()

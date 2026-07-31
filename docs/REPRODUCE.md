# Reproducing the Results

> Brief §9: "the prediction/analysis process is reproducible from the submitted
> repository" and "the implementation can be run according to its documentation".

## What makes this reproducible

| Concern | How it is handled |
|---|---|
| Random seeds | `set_seed()` seeds Python, NumPy and PyTorch, and enables deterministic cuDNN. Seeds live in the config (`data.seed`, `train.seed`). |
| Split assignment | Deterministic, image-level, stratified, driven by `data.seed`. The same data and seed always produce the same split. |
| Config drift | The exact config used for a run is saved as `models/<task>/config_used.yaml`, next to the checkpoint. |
| Checkpoint contents | The `.pt` file stores the class list, task, backbone, image size and the full config — not just weights, so a checkpoint is self-describing. |
| Train/serve skew | Training, evaluation and the API all import the same transform from `rdc.data.dataset`. |
| Dependencies | Pinned ranges in `requirements.txt`; exact environment in the Docker image. |
| Environment | `docker/Dockerfile` builds a fixed runtime. |
| Verification | CI runs the full pipeline (data → train → evaluate → serve → health check) on every push. |

## Full reproduction from scratch

```bash
# 1. Environment
git clone <repo-url> road-damage-classifier
cd road-damage-classifier
python -m venv .venv && source .venv/bin/activate
make install

# 2. Data — download RDD2022 into data/raw (see docs/DATASET.md)

# 3. Build the crop dataset (deterministic given the seed)
make prepare CONFIG=configs/binary.yaml

# 4. Train
make train                       # binary
make train-multiclass            # optional

# 5. Evaluate on the held-out test split
make evaluate CONFIG=configs/binary.yaml

# 6. Inspect
make mlflow                      # http://localhost:5000
cat reports/binary/metrics.json
```

## Expected artefacts

```
data/processed/crops.csv              crop index with split assignment
data/processed/dataset_stats.json     class/split/country counts
models/binary/model.pt                checkpoint (weights + metadata)
models/binary/config_used.yaml        exact config for this run
reports/binary/training.json          per-epoch history
reports/binary/metrics.json           test metrics + threshold sweep
reports/binary/classification_report.txt
reports/binary/confusion_matrix.png
reports/binary/robustness.json        accuracy under simulated shifts
mlruns/                               MLflow run history
```

## Verifying a specific prediction

```bash
python -m rdc.predict path/to/image.jpg --model models/binary/model.pt --output result.json
```

The response records the checkpoint path, backbone, task and image size, so any
prediction can be traced back to the run that produced it.

## Determinism caveats

Exact bit-for-bit reproducibility across **different hardware** is not
guaranteed: cuDNN kernel selection, CPU vs GPU floating-point ordering and
library versions all introduce small differences. Same machine plus same
versions plus same seed reproduces results exactly. Across machines, expect
macro-F1 to agree to roughly ±0.01 — use the Docker image if you need tighter
agreement.

Two things that are *always* deterministic regardless of hardware: the split
assignment and the crop extraction. So the evaluation set itself never changes,
which is the part that matters for comparing runs.

## Verifying the reproducibility claims

```bash
make test                        # full suite

# specifically:
pytest tests/test_config.py::test_seeding_is_reproducible
pytest tests/test_data.py::test_no_image_leaks_across_splits
```

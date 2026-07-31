# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this is

GOV-01 capstone deliverable: an ML prototype that classifies municipal road-condition
photographs so inspection reports can be triaged automatically. The client brief is a
*business* requirement — the technical formulation is ours to justify and defend.

**Hard scope rule:** the output is triage *support*, never an engineering safety
assessment. Every API response carries a `disclaimer` field saying so. Do not remove it,
and do not add language implying the model certifies road safety.

## Commands

```bash
make install          # deps (CPU torch)
make demo             # synthetic data -> prepare -> train -> evaluate (~3 min, no download)
make prepare          # build crop dataset from data/raw
make train            # binary model      (configs/binary.yaml)
make train-multiclass # 5-class model     (configs/multiclass.yaml)
make evaluate         # held-out test metrics -> reports/<task>/
make serve            # FastAPI  :8000/docs
make ui               # Streamlit :8501
make test             # full suite
make test-fast        # skip slow training tests
make lint             # ruff check + ruff format --check
make format           # auto-fix
```

## Architecture

```
src/rdc/
  config.py            dataclass config + label taxonomy (single source of truth)
  data/rdd_parser.py   RDD2022 VOC-XML -> tidy annotation table
  data/prepare.py      crop extraction + leak-free image-level splitting
  data/dataset.py      Dataset, transforms, class-imbalance handling
  models/factory.py    timm backbones + self-describing checkpoints
  train.py             training loop, early stopping, MLflow
  evaluate.py          test metrics, confusion matrix, robustness sweep
  metrics.py           macro-F1, ECE, threshold sweep
  predict.py           shared inference layer (CLI / API / UI all use this)
  api/main.py          FastAPI service
app/streamlit_app.py   demo UI
```

## Invariants — do not break these

1. **Splitting is at image level, never crop level.** Two crops from one photo share
   camera, lighting and road surface; splitting at crop level leaks and inflates test
   scores. Enforced by `tests/test_data.py::test_no_image_leaks_across_splits`.

2. **Macro-F1 is the primary metric**, used for early stopping and model selection.
   Accuracy is dominated by the majority class — a model that never predicts `pothole`
   still scores ~90% accuracy. Enforced by
   `tests/test_model.py::test_macro_f1_punishes_ignoring_a_minority_class`.

3. **The test split is read exactly once**, by `rdc.evaluate`. Model selection uses
   validation only. Never tune against test.

4. **Training and serving share one transform** (`rdc.data.dataset.inference_transform`).
   If you change preprocessing, change it there — not in two places.

5. **Invalid input never crashes.** `open_image` raises `InvalidImageError` for corrupt,
   empty, oversized and non-image files; the API maps these to 400/413/415. Covered by
   `tests/test_api.py`.

6. **Checkpoints are self-describing** — they store classes, task, backbone, image size
   and the full config, not just weights. Keep it that way; the API relies on it.

7. **Config drives everything.** New knobs go in `config.py` dataclasses and the YAML,
   not as hardcoded literals. Unknown YAML keys raise (typos must fail loudly).

## Conventions

- Python ≥3.10, `from __future__ import annotations` at the top of every module.
- **ruff only** for both linting and formatting — black was removed deliberately; the
  two disagreed and broke CI. Run `make format` before committing.
- Line length 100.
- Log with the module logger from `rdc.utils.get_logger`, not `print`.
- Data-quality problems are logged and skipped, never raised — one bad file among 47,000
  must not kill a multi-hour job.

## Documentation map

Docs are graded deliverables, not decoration. If you change behaviour, update them.

| File | Contents |
|---|---|
| `docs/DISCOVERY.md` | Brief §5 — dataset choice, leakage risks, fairness (filled) |
| `docs/PROPOSAL.md` | Brief §6 — formulation, baselines, metric rationale (filled) |
| `docs/DATASET.md` | RDD2022 source, licence, every preprocessing decision + why |
| `docs/MODEL_CARD.md` | Intended use, metrics, ethical considerations |
| `docs/LIMITATIONS.md` | Brief §11 answers, known limits, risks, next steps |
| `docs/API.md` | Endpoint reference |
| `docs/REPRODUCE.md` | Reproducibility guarantees and caveats |

## Current state

- Pipeline, metrics, API and all invalid-input paths are verified working.
- `models/` is empty — run `make demo` (synthetic) or train on real RDD2022.
- **The results table in `docs/MODEL_CARD.md` is intentionally blank.** Fill it only
  from a run on real RDD2022 data. Numbers from `make demo` come from synthetic images
  and must never be quoted as model performance.

## Highest-value next steps

1. Download RDD2022 (`docs/DATASET.md`), run `make prepare && make train && make evaluate`,
   fill the model card results table.
2. Compare `efficientnet_b0` and `convnext_tiny` against the `resnet18` reference — one
   config change each, all logged in MLflow.
3. Add temperature scaling on the validation split to improve calibration (the API's
   confidence threshold depends on it).
4. Grouped 5-fold CV for confidence intervals on macro-F1.

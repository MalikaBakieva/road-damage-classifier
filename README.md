# GOV-01 — Road Damage Image Classification

Prototype ML service that classifies municipal road-condition photographs so
inspection reports can be **triaged automatically** instead of reviewed by hand.

> **Scope limit.** The output is decision *support* for prioritising inspections.
> It is **not** an engineering safety assessment and does not replace a qualified
> inspector. This statement is returned with every API response.

**Client:** Municipal Infrastructure Department · **Track:** Field-Based Scenario
· **Brief:** GOV-01

---

## What it does

| | |
|---|---|
| **Input** | One road photograph (JPEG / PNG / WebP / BMP), ≤ 10 MB |
| **Output** | Damage class, confidence, damage probability, triage priority, human-review flag |
| **Primary scope** | `damaged` vs `not_damaged` — the reliable production signal |
| **Secondary scope** | `not_damaged`, `longitudinal_crack`, `transverse_crack`, `alligator_crack`, `pothole` |
| **Primary metric** | **macro-F1** (classes are imbalanced; every damage type matters) |

Example response:

```json
{
  "predicted_class": "pothole",
  "confidence": 0.87,
  "is_damaged": true,
  "damage_probability": 0.94,
  "needs_human_review": false,
  "priority": "high",
  "top_k": [
    {"class": "pothole", "probability": 0.87},
    {"class": "alligator_crack", "probability": 0.07}
  ],
  "inference_ms": 41.2,
  "disclaimer": "Automated triage support only. This output is not an engineering safety assessment ..."
}
```

---

## Quick start

### 1. Run the demo (no dataset download, ~3 minutes on CPU)

```bash
git clone <your-repo-url> road-damage-classifier
cd road-damage-classifier

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install            # or: pip install -r requirements.txt && pip install -e .

make demo               # synthetic data -> prepare -> train -> evaluate
make serve              # http://localhost:8000/docs
```

`make demo` trains on **synthetic** images so that a checkpoint exists and the
service starts. It proves the pipeline is wired correctly — it says nothing about
real-world accuracy. For real numbers, follow step 2.

### 2. Train on real RDD2022 data

```bash
# Download RDD2022 and unpack it to data/raw — see docs/DATASET.md
make prepare CONFIG=configs/binary.yaml     # build crops + leak-free splits
make train                                   # binary model
make evaluate                                # held-out test metrics -> reports/binary/
make train-multiclass                        # optional: 5-class model
```

### 3. Serve it

```bash
make serve                       # FastAPI  -> http://localhost:8000/docs
make ui                          # Streamlit -> http://localhost:8501
make docker-up                   # both, containerised
```

```bash
curl -X POST http://localhost:8000/predict -F "file=@road.jpg"
```

---

## Repository layout

```
├── src/rdc/
│   ├── config.py               dataclass config + label taxonomy
│   ├── data/
│   │   ├── rdd_parser.py       RDD2022 VOC-XML -> tidy table
│   │   ├── prepare.py          crop extraction + leak-free splitting
│   │   └── dataset.py          Dataset, transforms, imbalance handling
│   ├── models/factory.py       timm backbones + checkpoint I/O
│   ├── train.py                training loop, early stopping, MLflow
│   ├── evaluate.py             test metrics, confusion matrix, robustness
│   ├── metrics.py              macro-F1, ECE, threshold sweep
│   ├── predict.py              shared inference layer (CLI / API / UI)
│   └── api/main.py             FastAPI service
├── app/streamlit_app.py        upload demo + triage queue
├── configs/                    binary.yaml · multiclass.yaml
├── scripts/                    synthetic data generator · demo trainer
├── tests/                      pytest suite (data, model, API, bad inputs)
├── docker/ · docker-compose.yml
├── notebooks/01_eda.ipynb
└── docs/
    ├── DISCOVERY.md            brief §5 — Data & Problem Discovery (filled)
    ├── PROPOSAL.md             brief §6 — Technical Proposal (filled)
    ├── DATASET.md              source, licence, download, preprocessing
    ├── MODEL_CARD.md           intended use, metrics, ethical considerations
    ├── LIMITATIONS.md          brief §9/§11 — limitations, risks, next steps
    └── API.md                  endpoint reference
```

---

## How the brief is satisfied

| Brief requirement | Where |
|---|---|
| §5 Data & Problem Discovery | [`docs/DISCOVERY.md`](docs/DISCOVERY.md) |
| §6 Technical Proposal | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |
| §7.1 Accepts an image, returns a classification | `POST /predict` |
| §7.2 Category scope defined and dataset-supported | [`docs/DATASET.md`](docs/DATASET.md), `src/rdc/config.py` |
| §7.3 Handles invalid files without crashing | `src/rdc/predict.py::open_image`, `tests/test_api.py` |
| §7.4 Limitations under lighting / angle / quality documented | `evaluate.py::robustness_check`, [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) |
| §7.5 Not presented as a safety assessment | `disclaimer` field on every response |
| §8 Trained model + documented methodology | `models/`, [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| §8 Evaluation on unseen data | `make evaluate` → `reports/<task>/metrics.json` |
| §8 Inference interface | FastAPI + Streamlit + CLI |
| §8 Reproducible repository | pinned config, seeded splits, Docker, CI |
| §9 Reproducible predictions | `configs/*.yaml` + `models/*/config_used.yaml` |
| §11 Open questions answered | [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) |

---

## Design decisions worth defending

**Crops, not whole images.** RDD2022 is a *detection* dataset — one photo often
contains several damage types, so a whole-image multiclass label is ambiguous.
Each annotated box (plus 25 % context) becomes one classification sample, and
annotation-free road patches become `not_damaged`. This makes the label
well-defined instead of arbitrary.

**Splitting at image level, never crop level.** Two crops from the same photo
share camera, lighting and road surface. Splitting at crop level would leak that
information into the test set and inflate the score. `tests/test_data.py::
test_no_image_leaks_across_splits` enforces this.

**Macro-F1 as the primary metric.** Accuracy is dominated by the largest class:
a model that never predicts `pothole` still scores ~90 % accuracy on this data.
Macro-F1 weights every class equally, so ignoring a class is visibly punished.
For the binary head we also report recall on `damaged`, because a missed defect
costs the department more than a false alarm.

**Binary is primary, multiclass is secondary.** Crack subtypes are visually
similar and the rare classes are small, so multiclass macro-F1 is substantially
lower. The department's actual decision — does this report deserve an inspection?
— only needs the binary signal, so that is what production should rely on.

**Confidence threshold with human review.** Below `RDC_CONFIDENCE_THRESHOLD`
(default 0.60) a prediction is flagged `needs_human_review` rather than acted on.
`reports/<task>/threshold_sweep.csv` shows the coverage/accuracy trade-off so the
department can pick a threshold from evidence.

**Negatives sampled from the road surface.** Negative patches are drawn from the
lower 65 % of the frame, avoiding sky and buildings — otherwise `not_damaged`
would be trivially separable and the reported score meaningless.

---

## Configuration

Everything is driven by YAML (`configs/binary.yaml`, `configs/multiclass.yaml`);
the exact config used for a run is saved next to the checkpoint as
`models/<task>/config_used.yaml`.

Runtime environment variables for the API:

| Variable | Default | Purpose |
|---|---|---|
| `RDC_MODEL_PATH` | `models/binary/model.pt` | Checkpoint to serve |
| `RDC_CONFIDENCE_THRESHOLD` | `0.6` | Below this → `needs_human_review` |
| `RDC_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `RDC_MAX_BATCH` | `16` | Max files per batch request |
| `RDC_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |

---

## Experiment tracking

MLflow logs every run (params, per-epoch metrics, final test metrics, artefacts):

```bash
make train          # writes to ./mlruns
make mlflow         # http://localhost:5000
```

If MLflow is unavailable the tracker degrades to local JSON under
`reports/runs/` — a missing tracking server never breaks training.

---

## Testing

```bash
make test           # everything
make test-fast      # skip the slow training tests
```

The suite covers annotation parsing (including corrupt XML and degenerate
boxes), leak-free splitting, checkpoint round-trips, metric behaviour under
imbalance, and every invalid-input path in the API — corrupt images, wrong
content types, empty files, oversized uploads, and a missing model.

---

## Limitations (short form)

Trained on RDD2022 dashcam-style imagery: daytime, vehicle-mounted, mostly
Japan/India/Norway. Expect degradation on night shots, wet or snow-covered
surfaces, extreme close-ups, and pedestrian phone photos. The model reports
*visible surface damage only* — it cannot judge structural severity, depth or
urgency. Full discussion in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Licence & attribution

Code: MIT (see `LICENSE`).
Dataset: **RDD2022** — Arya, Maeda, Sekimoto et al., released under CC BY-SA 4.0.
Cite the dataset when publishing results; see [`docs/DATASET.md`](docs/DATASET.md).

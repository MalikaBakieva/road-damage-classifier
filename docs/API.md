# API Reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs` · OpenAPI: `/openapi.json`

```bash
make serve                      # local
docker compose up -d api        # containerised
```

---

## `GET /health`

Liveness and model status. Returns `200` even when the model failed to load, so
an orchestrator can distinguish "process dead" from "model missing".

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0", "detail": null }
```

`status` is `"degraded"` and `model_loaded` is `false` when no checkpoint could
be loaded; `detail` then explains why.

---

## `GET /model-info`

```json
{
  "task": "binary",
  "classes": ["not_damaged", "damaged"],
  "backbone": "resnet18",
  "image_size": 224,
  "device": "cpu",
  "confidence_threshold": 0.6,
  "training_metrics": { "val": { "macro_f1": 0.91 }, "best_epoch": 7 }
}
```

Returns `503` if no model is loaded.

---

## `POST /predict`

Classify a single road photograph.

**Request** — `multipart/form-data`, field `file`.
Accepted: JPEG, PNG, WebP, BMP. Max size: `RDC_MAX_UPLOAD_MB` (default 10 MB).

```bash
curl -X POST http://localhost:8000/predict -F "file=@road.jpg"
```

**Response `200`**

```json
{
  "task": "multiclass",
  "predicted_class": "pothole",
  "confidence": 0.8713,
  "is_damaged": true,
  "damage_probability": 0.9402,
  "needs_human_review": false,
  "confidence_threshold": 0.6,
  "priority": "high",
  "top_k": [
    { "class": "pothole", "probability": 0.8713 },
    { "class": "alligator_crack", "probability": 0.0689 },
    { "class": "not_damaged", "probability": 0.0398 }
  ],
  "probabilities": {
    "not_damaged": 0.0398, "longitudinal_crack": 0.0121,
    "transverse_crack": 0.0079, "alligator_crack": 0.0689, "pothole": 0.8713
  },
  "model": { "path": "models/multiclass/model.pt", "backbone": "resnet18",
             "task": "multiclass", "image_size": 224 },
  "image": { "width": 1920, "height": 1080, "format": "JPEG" },
  "inference_ms": 41.2,
  "filename": "road.jpg",
  "disclaimer": "Automated triage support only. This output is not an engineering safety assessment and must be confirmed by a qualified inspector."
}
```

### Field reference

| Field | Meaning |
|---|---|
| `predicted_class` | Highest-probability class |
| `confidence` | Probability of the predicted class (0–1) |
| `is_damaged` | `true` unless the prediction is `not_damaged` |
| `damage_probability` | Total probability mass over all damage classes — the binary triage signal, available even from the multiclass head |
| `needs_human_review` | `true` when `confidence < confidence_threshold` |
| `priority` | Triage bucket — see below |
| `top_k` | Highest-probability classes, descending |
| `inference_ms` | Model forward-pass time (excludes network) |
| `disclaimer` | Scope statement; always present |

### Priority mapping

| Value | When | Suggested action |
|---|---|---|
| `high` | Pothole, or `damaged` with probability ≥ 0.85 | Inspect promptly |
| `medium` | Alligator cracking, or `damaged` below 0.85 | Schedule inspection |
| `low` | Longitudinal or transverse crack | Monitor |
| `review` | Confidence < 0.5 | Route to a human reviewer |
| `none` | `not_damaged` | No action |

Potholes rank highest because they are the most immediate hazard to road users.

### Errors

| Status | Cause | Example detail |
|---|---|---|
| `400` | Corrupt, unreadable or empty image | `Unsupported or corrupt image file: cannot identify image file` |
| `413` | File exceeds the size limit | `File is 14.2 MB; the limit is 10.0 MB.` |
| `415` | Unsupported content type | `Unsupported content type 'text/csv'.` |
| `422` | `file` field missing | FastAPI validation error |
| `503` | No model loaded | `Model not loaded: ... Train a model or set RDC_MODEL_PATH ...` |

Every error returns JSON with an actionable message — never a stack trace.

---

## `POST /predict-batch`

Classify up to `RDC_MAX_BATCH` images (default 16). **Per-item** error handling:
one bad file does not fail the request.

```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "files=@a.jpg" -F "files=@b.jpg" -F "files=@broken.jpg"
```

```json
{
  "count": 3, "succeeded": 2, "failed": 1,
  "items": [
    { "filename": "a.jpg", "ok": true, "result": { "...": "..." } },
    { "filename": "b.jpg", "ok": true, "result": { "...": "..." } },
    { "filename": "broken.jpg", "ok": false,
      "error": "Unsupported or corrupt image file: cannot identify image file" }
  ]
}
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RDC_MODEL_PATH` | `models/binary/model.pt` | Checkpoint to serve |
| `RDC_CONFIDENCE_THRESHOLD` | `0.6` | Below this → `needs_human_review` |
| `RDC_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `RDC_MAX_BATCH` | `16` | Max files per batch request |
| `RDC_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `RDC_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

Serve the multiclass model instead:

```bash
RDC_MODEL_PATH=models/multiclass/model.pt make serve
```

---

## Python client

```python
import requests

with open("road.jpg", "rb") as fh:
    r = requests.post("http://localhost:8000/predict",
                      files={"file": ("road.jpg", fh, "image/jpeg")}, timeout=30)
r.raise_for_status()
result = r.json()

if result["needs_human_review"]:
    queue_for_human(result)
else:
    route(result["priority"])
```

Or skip HTTP entirely:

```python
from rdc.predict import RoadDamagePredictor

predictor = RoadDamagePredictor("models/binary/model.pt")
print(predictor.predict("road.jpg"))
```

---

## Notes for production

Out of scope for this prototype, but required before real deployment:
authentication, rate limiting, structured audit logging, request/prediction
persistence for a feedback loop, and drift monitoring on the rate of
low-confidence predictions.

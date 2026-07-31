"""FastAPI inference service for the GOV-01 road damage classifier.

    uvicorn rdc.api.main:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
RDC_MODEL_PATH           checkpoint to serve      (default models/binary/model.pt)
RDC_CONFIDENCE_THRESHOLD below this -> needs_human_review (default 0.6)
RDC_MAX_UPLOAD_MB        upload size limit        (default 10)
RDC_DEVICE               auto | cpu | cuda        (default auto)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..predict import InvalidImageError, RoadDamagePredictor, open_image
from ..utils import get_logger
from .schemas import (
    BatchItem,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

LOG = get_logger("rdc.api")
VERSION = "1.0.0"

MODEL_PATH = os.getenv("RDC_MODEL_PATH", "models/binary/model.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("RDC_CONFIDENCE_THRESHOLD", "0.6"))
MAX_UPLOAD_MB = float(os.getenv("RDC_MAX_UPLOAD_MB", "10"))
DEVICE = os.getenv("RDC_DEVICE", "auto")
MAX_BATCH = int(os.getenv("RDC_MAX_BATCH", "16"))

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream",  # some clients send this for valid images
}

_state: dict = {"predictor": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load weights once at startup. A load failure must not kill the process:
    /health then reports degraded so an orchestrator can act on it."""
    try:
        _state["predictor"] = RoadDamagePredictor(
            MODEL_PATH, device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD
        )
        LOG.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:
        _state["load_error"] = str(exc)
        LOG.error("Model could not be loaded: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="GOV-01 Road Damage Classification API",
    description=(
        "Automated triage support for municipal road-condition photographs. "
        "Returns a damage classification, a confidence score and a suggested "
        "inspection priority. **Not** an engineering safety assessment."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RDC_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _require_predictor() -> RoadDamagePredictor:
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model not loaded: {_state.get('load_error')}. "
                "Train a model or set RDC_MODEL_PATH to a valid checkpoint."
            ),
        )
    return predictor


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Send one of: {sorted(ALLOWED_CONTENT_TYPES - {'application/octet-stream'})}."
            ),
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    size_mb = len(payload) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,  # Content Too Large (name differs across Starlette versions)
            detail=f"File is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.",
        )
    return payload


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "GOV-01 Road Damage Classification API",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/predict-batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    loaded = _state.get("predictor") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=VERSION,
        detail=None if loaded else _state.get("load_error"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info():
    return ModelInfoResponse(**_require_predictor().info())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Corrupt or unreadable image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    },
)
async def predict(file: UploadFile = File(..., description="Road-condition photograph")):
    """Classify a single road image."""
    predictor = _require_predictor()
    payload = await _read_upload(file)
    try:
        image = open_image(payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = predictor.predict_image(image)
    result["filename"] = file.filename
    return result


@app.post("/predict-batch", response_model=BatchResponse, tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify up to RDC_MAX_BATCH images. Individual failures are reported
    per item so one bad file cannot fail the whole request."""
    predictor = _require_predictor()
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400, detail=f"Too many files ({len(files)}); the limit is {MAX_BATCH}."
        )

    items: list[BatchItem] = []
    for file in files:
        name = file.filename or "unnamed"
        try:
            payload = await _read_upload(file)
            image = open_image(payload)
            result = predictor.predict_image(image)
            result["filename"] = name
            items.append(BatchItem(filename=name, ok=True, result=PredictionResponse(**result)))
        except HTTPException as exc:
            items.append(BatchItem(filename=name, ok=False, error=str(exc.detail)))
        except InvalidImageError as exc:
            items.append(BatchItem(filename=name, ok=False, error=str(exc)))
        except Exception as exc:  # defence in depth
            LOG.exception("Unexpected failure on %s", name)
            items.append(BatchItem(filename=name, ok=False, error=f"Internal error: {exc}"))

    succeeded = sum(1 for i in items if i.ok)
    return BatchResponse(
        count=len(items), succeeded=succeeded, failed=len(items) - succeeded, items=items
    )


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="invalid_image",
            detail=str(exc),
            hint="Send a JPEG, PNG, WebP or BMP road photograph.",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    LOG.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="The request could not be processed.",
            hint="Check the service logs; the request was not retried.",
        ).model_dump(),
    )

"""Pydantic response models for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    class_name: str = Field(alias="class")
    probability: float

    model_config = {"populate_by_name": True}


class ModelInfo(BaseModel):
    path: str
    backbone: str
    task: str
    image_size: int


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str | None = None


class PredictionResponse(BaseModel):
    task: str
    predicted_class: str
    confidence: float
    is_damaged: bool
    damage_probability: float
    needs_human_review: bool
    confidence_threshold: float
    priority: str = Field(description="Triage bucket: high | medium | low | review | none")
    top_k: list[ClassProbability]
    probabilities: dict[str, float]
    model: ModelInfo
    image: ImageInfo
    inference_ms: float
    disclaimer: str
    filename: str | None = None

    model_config = {"protected_namespaces": ()}


class BatchItem(BaseModel):
    filename: str
    ok: bool
    result: PredictionResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    count: int
    succeeded: int
    failed: int
    items: list[BatchItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    detail: str | None = None

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    task: str
    classes: list[str]
    backbone: str
    image_size: int
    device: str
    confidence_threshold: float
    training_metrics: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None

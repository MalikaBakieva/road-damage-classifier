"""API and predictor tests, with emphasis on invalid input handling.

Functional requirement 7.3 — "must handle invalid files or unsupported inputs
without crashing" — is the main thing under test here.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from rdc.predict import InvalidImageError, RoadDamagePredictor, open_image

# --------------------------------------------------------------------------
# image decoding
# --------------------------------------------------------------------------


def test_open_valid_bytes(sample_image_bytes):
    image = open_image(sample_image_bytes)
    assert image.mode == "RGB"
    assert image.size == (256, 256)


def test_open_corrupt_bytes_raises_invalid_image(corrupt_image_bytes):
    with pytest.raises(InvalidImageError):
        open_image(corrupt_image_bytes)


def test_open_text_file_raises_invalid_image(text_file_bytes):
    with pytest.raises(InvalidImageError):
        open_image(text_file_bytes)


def test_open_empty_bytes_raises_invalid_image():
    with pytest.raises(InvalidImageError, match="Empty file"):
        open_image(b"")


def test_open_missing_path_raises_invalid_image(tmp_path):
    with pytest.raises(InvalidImageError, match="File not found"):
        open_image(tmp_path / "does_not_exist.jpg")


def test_greyscale_and_rgba_are_converted(tmp_path):
    for mode, color in (("L", 128), ("RGBA", (10, 20, 30, 255))):
        path = tmp_path / f"{mode}.png"
        Image.new(mode, (64, 64), color).save(path)
        assert open_image(path).mode == "RGB"


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------


def test_prediction_payload_is_complete(trained_checkpoint, sample_image_bytes):
    predictor = RoadDamagePredictor(trained_checkpoint, device="cpu")
    result = predictor.predict(sample_image_bytes)

    for key in (
        "predicted_class",
        "confidence",
        "is_damaged",
        "damage_probability",
        "needs_human_review",
        "priority",
        "top_k",
        "probabilities",
        "model",
        "image",
        "inference_ms",
        "disclaimer",
    ):
        assert key in result, f"missing key: {key}"

    assert result["predicted_class"] in predictor.classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_disclaimer_states_it_is_not_a_safety_assessment(trained_checkpoint, sample_image_bytes):
    """Functional requirement 7.5 - the scope limit must travel with the output."""
    predictor = RoadDamagePredictor(trained_checkpoint, device="cpu")
    result = predictor.predict(sample_image_bytes)
    assert "not an engineering safety assessment" in result["disclaimer"].lower()


def test_low_confidence_is_routed_to_human_review(trained_checkpoint, sample_image_bytes):
    predictor = RoadDamagePredictor(trained_checkpoint, device="cpu", confidence_threshold=0.999)
    assert predictor.predict(sample_image_bytes)["needs_human_review"] is True


def test_high_threshold_zero_means_no_review(trained_checkpoint, sample_image_bytes):
    predictor = RoadDamagePredictor(trained_checkpoint, device="cpu", confidence_threshold=0.0)
    assert predictor.predict(sample_image_bytes)["needs_human_review"] is False


def test_batch_reports_per_item_errors(trained_checkpoint, sample_image_bytes, tmp_path):
    good = tmp_path / "good.jpg"
    good.write_bytes(sample_image_bytes)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")

    predictor = RoadDamagePredictor(trained_checkpoint, device="cpu")
    results = predictor.predict_batch([good, bad])
    assert len(results) == 2
    assert "error" not in results[0]
    assert "error" in results[1]  # one bad file must not sink the batch


def test_predictor_info(trained_checkpoint):
    info = RoadDamagePredictor(trained_checkpoint, device="cpu").info()
    assert info["task"] == "binary"
    assert info["classes"] == ["not_damaged", "damaged"]


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


@pytest.fixture
def client(trained_checkpoint, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RDC_MODEL_PATH", str(trained_checkpoint))

    import importlib

    from rdc.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["classes"] == ["not_damaged", "damaged"]
    assert body["task"] == "binary"


def test_predict_returns_classification(client, sample_image_bytes):
    response = client.post(
        "/predict", files={"file": ("road.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "road.jpg"
    assert body["predicted_class"] in ("damaged", "not_damaged")
    assert "X-Process-Time-ms" in response.headers


def test_predict_rejects_corrupt_image(client, corrupt_image_bytes):
    response = client.post(
        "/predict", files={"file": ("broken.jpg", corrupt_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert (
        "corrupt" in response.json()["detail"].lower()
        or "unsupported" in response.json()["detail"].lower()
    )


def test_predict_rejects_unsupported_content_type(client, text_file_bytes):
    response = client.post("/predict", files={"file": ("data.csv", text_file_bytes, "text/csv")})
    assert response.status_code == 415


def test_predict_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_predict_rejects_oversized_file(client, monkeypatch):
    from rdc.api import main as api_main

    monkeypatch.setattr(api_main, "MAX_UPLOAD_MB", 0.001)
    big = io.BytesIO()
    Image.new("RGB", (900, 900), (200, 30, 30)).save(big, format="JPEG", quality=95)
    response = client.post("/predict", files={"file": ("big.jpg", big.getvalue(), "image/jpeg")})
    assert response.status_code == 413


def test_predict_missing_file_field(client):
    assert client.post("/predict").status_code == 422


def test_batch_mixes_success_and_failure(client, sample_image_bytes, corrupt_image_bytes):
    response = client.post(
        "/predict-batch",
        files=[
            ("files", ("ok.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("bad.jpg", corrupt_image_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["succeeded"] == 1
    assert body["failed"] == 1


def test_batch_size_limit(client, sample_image_bytes, monkeypatch):
    from rdc.api import main as api_main

    monkeypatch.setattr(api_main, "MAX_BATCH", 1)
    response = client.post(
        "/predict-batch",
        files=[
            ("files", ("a.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("b.jpg", sample_image_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 400


def test_service_degrades_gracefully_without_a_model(monkeypatch, tmp_path):
    """No checkpoint must yield 503 with an actionable message - not a crash."""
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("RDC_MODEL_PATH", str(tmp_path / "missing.pt"))
    from rdc.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False

        response = client.post("/predict", files={"file": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")})
        assert response.status_code == 503

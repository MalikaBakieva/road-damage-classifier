"""Streamlit demo UI for the GOV-01 road damage classifier.

    streamlit run app/streamlit_app.py

Talks to the FastAPI service when RDC_API_URL is set and reachable; otherwise
loads the checkpoint in-process so the demo works standalone.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API_URL = os.getenv("RDC_API_URL", "").rstrip("/")
MODEL_PATH = os.getenv("RDC_MODEL_PATH", str(ROOT / "models" / "binary" / "model.pt"))

PRIORITY_STYLE = {
    "high": ("🔴", "Inspect promptly"),
    "medium": ("🟠", "Schedule inspection"),
    "low": ("🟡", "Monitor"),
    "review": ("⚪", "Low confidence - route to a human reviewer"),
    "none": ("🟢", "No action"),
}

st.set_page_config(page_title="Road Damage Triage — GOV-01", page_icon="🛣️", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load_local_predictor(model_path: str, threshold: float):
    from rdc.predict import RoadDamagePredictor

    return RoadDamagePredictor(model_path, confidence_threshold=threshold)


def call_api(payload: bytes, filename: str, api_url: str):
    import requests

    response = requests.post(
        f"{api_url}/predict",
        files={"file": (filename, payload, "image/jpeg")},
        timeout=30,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.60, 0.05,
    help="Predictions below this confidence are flagged for human review.",
)
backend = st.sidebar.radio(
    "Inference backend",
    ["REST API", "Local model"] if API_URL else ["Local model", "REST API"],
)
if backend == "REST API":
    API_URL = st.sidebar.text_input("API URL", API_URL or "http://localhost:8000").rstrip("/")
else:
    MODEL_PATH = st.sidebar.text_input("Checkpoint path", MODEL_PATH)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Scope note.** Output is triage support for prioritising inspections. "
    "It is not an engineering safety assessment and does not replace a "
    "qualified inspector's judgement."
)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

st.title("🛣️ Road Damage Triage")
st.caption("GOV-01 · Municipal Infrastructure Department · prototype")

uploaded = st.file_uploader(
    "Upload road-condition photographs",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info(
        "Upload one or more road photographs to get a damage classification, a "
        "confidence score and a suggested inspection priority."
    )
    st.stop()

rows = []
for file in uploaded:
    payload = file.getvalue()
    col_image, col_result = st.columns([1, 1.4])

    with col_image:
        st.image(payload, caption=file.name, use_container_width=True)

    with col_result:
        try:
            if backend == "REST API":
                result = call_api(payload, file.name, API_URL)
            else:
                predictor = load_local_predictor(MODEL_PATH, threshold)
                predictor.confidence_threshold = threshold
                result = predictor.predict(payload)
        except Exception as exc:
            st.error(f"Could not process **{file.name}** — {exc}")
            rows.append({"file": file.name, "prediction": "ERROR", "confidence": None,
                         "priority": None, "review": None})
            continue

        icon, action = PRIORITY_STYLE.get(result["priority"], ("⚪", ""))
        st.subheader(f"{icon} {result['predicted_class'].replace('_', ' ').title()}")
        st.caption(action)

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{result['confidence']:.1%}")
        m2.metric("Damage probability", f"{result['damage_probability']:.1%}")
        m3.metric("Latency", f"{result['inference_ms']:.0f} ms")

        if result["needs_human_review"]:
            st.warning("Below the confidence threshold — routed to human review.")

        st.bar_chart(pd.Series(result["probabilities"], name="probability"))

        with st.expander("Raw response"):
            st.json(result)

        rows.append(
            {
                "file": file.name,
                "prediction": result["predicted_class"],
                "confidence": round(result["confidence"], 3),
                "priority": result["priority"],
                "review": result["needs_human_review"],
            }
        )
    st.divider()

if rows:
    st.subheader("Triage queue")
    frame = pd.DataFrame(rows)
    order = {"high": 0, "medium": 1, "review": 2, "low": 3, "none": 4}
    frame = frame.sort_values("priority", key=lambda s: s.map(order).fillna(5))
    st.dataframe(frame, use_container_width=True, hide_index=True)

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    st.download_button(
        "Download queue as CSV", buffer.getvalue(), "triage_queue.csv", "text/csv"
    )

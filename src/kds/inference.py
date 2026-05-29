"""Model loading and phase prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from kds.config import FEATURE_ORDER


@st.cache_resource(show_spinner=False)
def load_model(path: str | Path):
    """Load the trained Random Forest model once and cache it."""
    path = Path(path)
    if not path.exists():
        return None, f"Model file '{path}' not found. Place it in the models/ folder."
    try:
        checkpoint = joblib.load(path)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        else:
            model = checkpoint
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def predict_phase(model, features: dict):
    """Run the Random Forest on the ordered feature vector."""
    vector = np.array(
        [[features.get(name, 0.0) for name in FEATURE_ORDER]], dtype=float
    )
    label = model.predict(vector)[0]

    confidence = None
    proba_df = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        confidence = float(np.max(proba))
        proba_df = (
            pd.DataFrame(
                {"Phase": [str(c) for c in classes], "Probability": proba}
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
    return str(label), confidence, proba_df

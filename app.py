"""
Meiosis Phase Classifier — Streamlit entry point.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (see README.md)
"""

import sys
from pathlib import Path

# Allow imports from src/ when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from kds.config import MODEL_PATH
from kds.features import extract_features
from kds.inference import load_model, predict_phase
from kds.preprocessing import preprocess_image
from kds.segmentation import segment_and_detect

st.set_page_config(
    page_title="Meiosis Phase Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Meiosis Phase Classifier")
    st.caption(
        "Upload a microscopy image to segment chromatin, detect spindle-like "
        "lines, and predict the meiotic phase with a Random Forest model."
    )

    model, model_err = load_model(MODEL_PATH)

    with st.sidebar:
        st.header("Upload & Settings")
        uploaded = st.file_uploader(
            "Microscopy image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Supported: JPG, PNG, TIF/TIFF",
        )
        st.subheader("Preprocessing")
        clip_limit = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5)
        tile_grid = st.slider("CLAHE tile grid", 4, 16, 8, 1)
        denoise = st.slider("Denoise strength", 1, 30, 10, 1)
        st.subheader("Segmentation")
        min_area = st.slider("Min chromatin area (px)", 5, 200, 30, 5)

        if model_err:
            st.warning(model_err)
        else:
            st.success("Model loaded")

    if uploaded is None:
        st.info("Upload a microscopy image from the sidebar to begin.")
        return

    img_bytes = uploaded.getvalue()

    try:
        gray, processed = preprocess_image(img_bytes, clip_limit, tile_grid, denoise)
        overlay, contours, lines = segment_and_detect(processed, min_area)
        features = extract_features(processed, contours, lines)
    except Exception as exc:
        st.error(f"Image processing failed: {exc}")
        return

    st.subheader("Image Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original (grayscale)**")
        st.image(gray, use_container_width=True, clamp=True)
    with col2:
        st.markdown("**Processed — chromatin (green) & spindle lines (red)**")
        st.image(overlay, use_container_width=True)

    st.subheader("Phase Prediction")
    if model is None:
        st.warning(
            "No model available — showing extracted features only. "
            "Add your `.joblib` file to the models/ folder."
        )
    else:
        try:
            label, confidence, proba_df = predict_phase(model, features)
            m1, m2 = st.columns(2)
            m1.metric("Predicted Phase", label)
            if confidence is not None:
                m2.metric("Confidence", f"{confidence * 100:.1f}%")
            st.success(f"This image is classified as **{label}**.")
            if proba_df is not None:
                st.markdown("**Class probabilities**")
                st.bar_chart(proba_df.set_index("Phase"))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with st.expander("Extracted Spatial Features", expanded=False):
        feat_df = pd.DataFrame(
            {
                "Feature": list(features.keys()),
                "Value": [round(v, 4) for v in features.values()],
            }
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download features (CSV)",
            feat_df.to_csv(index=False).encode("utf-8"),
            file_name="meiosis_features.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

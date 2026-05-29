"""Image preprocessing: background correction, CLAHE, denoising."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def preprocess_image(
    img_bytes: bytes,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    denoise_strength: int = 10,
):
    """Decode bytes -> grayscale -> background correction -> CLAHE -> denoise."""
    file_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(gray, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(corrected)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return gray, denoised

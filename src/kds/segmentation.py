"""Chromatin segmentation and spindle-like line detection."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def segment_and_detect(processed: np.ndarray, min_area: int = 30):
    """Segment chromatin blobs and detect spindle-like lines."""
    _, thresh = cv2.threshold(
        processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    edges = cv2.Canny(processed, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=8
    )

    overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb, contours, lines

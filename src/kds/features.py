"""Spatial feature extraction from segmented chromatin and spindle lines."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def extract_features(processed: np.ndarray, _contours, _lines) -> dict:
    """Compute spatial features; keys should align with the trained model."""
    h, w = processed.shape[:2]
    img_area = float(h * w)

    areas = np.array([cv2.contourArea(c) for c in _contours], dtype=float)
    centroids = []
    eccentricities = []
    for c in _contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        if len(c) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(c)
            major, minor = max(MA, ma), max(min(MA, ma), 1e-6)
            eccentricities.append(np.sqrt(max(1 - (minor / major) ** 2, 0)))

    centroids = np.array(centroids) if centroids else np.zeros((0, 2))
    centroid_spread = (
        float(np.mean(np.std(centroids, axis=0))) if len(centroids) else 0.0
    )

    if _lines is not None and len(_lines):
        seg = _lines.reshape(-1, 4).astype(float)
        lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        angles = np.degrees(
            np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])
        )
    else:
        lengths = np.array([])
        angles = np.array([])

    gx = cv2.Sobel(processed, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(processed, cv2.CV_32F, 0, 1)
    grad_mag = np.hypot(gx, gy)
    texture_contrast = float(np.var(grad_mag))
    texture_homogeneity = float(1.0 / (1.0 + np.mean(grad_mag)))

    return {
        "num_chromatin_objects": float(len(_contours)),
        "total_chromatin_area": float(areas.sum()) if areas.size else 0.0,
        "mean_chromatin_area": float(areas.mean()) if areas.size else 0.0,
        "chromatin_area_std": float(areas.std()) if areas.size else 0.0,
        "max_chromatin_area": float(areas.max()) if areas.size else 0.0,
        "chromatin_density": float(areas.sum() / img_area) if areas.size else 0.0,
        "centroid_spread": centroid_spread,
        "mean_eccentricity": float(np.mean(eccentricities)) if eccentricities else 0.0,
        "num_spindle_lines": float(len(lengths)),
        "mean_spindle_length": float(lengths.mean()) if lengths.size else 0.0,
        "spindle_length_std": float(lengths.std()) if lengths.size else 0.0,
        "mean_spindle_angle": float(angles.mean()) if angles.size else 0.0,
        "spindle_angle_std": float(angles.std()) if angles.size else 0.0,
        "texture_contrast": texture_contrast,
        "texture_homogeneity": texture_homogeneity,
    }

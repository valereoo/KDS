"""Spatial feature extraction from segmented chromatin and spindle lines."""

import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

def safe_divide(a, b, default=0.0):
    """Membantu pembagian aman agar tidak terjadi error ZeroDivisionError."""
    return float(a / b) if abs(b) > 1e-12 else float(default)

@st.cache_data(show_spinner=False)
def extract_features(processed: np.ndarray, _contours, _lines) -> dict:
    """Compute spatial features matching the Random Forest model's training pipeline."""
    h, w = processed.shape[:2]
    image_area = float(h * w)
    diag = float(np.hypot(h, w))
    center_image = np.array([w / 2, h / 2], dtype=float)

    areas = []
    perimeters = []
    circularities = []
    centroids = []
    points_xy = []

    # 1. Contour Measurements (Kepadatan, Jumlah, & Area)
    for c in _contours:
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        if area <= 0:
            continue

        moments = cv2.moments(c)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            pts = c.reshape(-1, 2)
            cx, cy = pts.mean(axis=0).astype(float)

        circularity = safe_divide(4 * math.pi * area, perimeter ** 2, 0.0)

        areas.append(area)
        perimeters.append(perimeter)
        circularities.append(circularity)
        centroids.append([cx, cy])
        points_xy.extend(c.reshape(-1, 2))

    areas_arr = np.array(areas, dtype=float)
    circs_arr = np.array(circularities, dtype=float)
    pts_xy = np.array(points_xy, dtype=float)

    features = {}
    features_found = len(areas_arr) > 0

    features["object_count"] = float(len(areas_arr))
    features["chromatin_area_ratio"] = float(areas_arr.sum() / image_area) if features_found else 0.0
    features["mean_object_area_norm"] = float(areas_arr.mean() / image_area) if features_found else 0.0
    features["std_object_area_norm"] = float(areas_arr.std() / image_area) if features_found else 0.0
    features["max_object_area_norm"] = float(areas_arr.max() / image_area) if features_found else 0.0
    features["mean_circularity"] = float(circs_arr.mean()) if features_found else 0.0
    features["std_circularity"] = float(circs_arr.std()) if features_found else 0.0

    # 2. PCA Spatial Features (Orientasi & Bentuk Kromatin)
    if pts_xy.shape[0] < 3:
        features.update({
            "chromatin_centroid_x_norm": 0.5,
            "chromatin_centroid_y_norm": 0.5,
            "centroid_distance_to_center_norm": 0.0,
            "pca_axis_ratio": 1.0,
            "pca_major_span_norm": 0.0,
            "pca_minor_span_norm": 0.0,
            "pca_angle_rad": 0.0,
            "band_score": 0.0,
        })
    else:
        centroid = pts_xy.mean(axis=0)
        centered = pts_xy - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Urutkan eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major_vec = eigvecs[:, 0]
        minor_vec = eigvecs[:, 1]
        major_proj = centered @ major_vec
        minor_proj = centered @ minor_vec

        major_span = float(np.percentile(major_proj, 99) - np.percentile(major_proj, 1))
        minor_span = float(np.percentile(minor_proj, 99) - np.percentile(minor_proj, 1))
        axis_ratio = float(np.sqrt((eigvals[0] + 1e-8) / (eigvals[1] + 1e-8)))
        pca_angle = float(np.arctan2(major_vec[1], major_vec[0]))

        centroid_distance = float(np.linalg.norm(centroid - center_image) / diag)

        elongation_score = float(np.clip((axis_ratio - 1.0) / 5.0, 0, 1))
        thinness_score = float(1.0 / (1.0 + 5.0 * (minor_span / (diag + 1e-8))))
        centrality_score = float(np.exp(-5.0 * centroid_distance))
        band_score = float(elongation_score * thinness_score * centrality_score)

        features.update({
            "chromatin_centroid_x_norm": float(centroid[0] / w),
            "chromatin_centroid_y_norm": float(centroid[1] / h),
            "centroid_distance_to_center_norm": centroid_distance,
            "pca_axis_ratio": axis_ratio,
            "pca_major_span_norm": float(major_span / diag),
            "pca_minor_span_norm": float(minor_span / diag),
            "pca_angle_rad": pca_angle,
            "band_score": band_score,
        })

    # 3. Two-Cluster Features (Pemisahan/Kutub menggunakan KMeans)
    if pts_xy.shape[0] < 6:
        features.update({
            "cluster_separation_norm": 0.0,
            "cluster_balance": 0.0,
            "cluster_compactness_norm": 0.0,
        })
    else:
        points_sample = pts_xy
        # Lakukan sampling agar performa web tetap cepat jika objek terlalu banyak
        if points_sample.shape[0] > 1500:
            rng = np.random.default_rng(42)
            idx = rng.choice(points_sample.shape[0], size=1500, replace=False)
            points_sample = points_sample[idx]

        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points_sample)
            centers = kmeans.cluster_centers_
            counts = np.bincount(labels, minlength=2).astype(float)

            separation = float(np.linalg.norm(centers[0] - centers[1]) / diag)
            balance = float(counts.min() / (counts.max() + 1e-8))
            distances = np.linalg.norm(points_sample - centers[labels], axis=1)
            compactness = float(distances.mean() / diag)

            features.update({
                "cluster_separation_norm": separation,
                "cluster_balance": balance,
                "cluster_compactness_norm": compactness,
            })
        except Exception:
            features.update({
                "cluster_separation_norm": 0.0,
                "cluster_balance": 0.0,
                "cluster_compactness_norm": 0.0,
            })

    # 4. Spindle Lines Features (Garis-garis Spindel)
    if _lines is not None and len(_lines) > 0:
        lines_reshaped = _lines.reshape(-1, 4).astype(float)
        lengths = np.hypot(lines_reshaped[:, 2] - lines_reshaped[:, 0], lines_reshaped[:, 3] - lines_reshaped[:, 1])
        angles = np.arctan2(lines_reshaped[:, 3] - lines_reshaped[:, 1], lines_reshaped[:, 2] - lines_reshaped[:, 0])

        weights = lengths / (lengths.sum() + 1e-8)
        c = float(np.sum(weights * np.cos(2 * angles)))
        s = float(np.sum(weights * np.sin(2 * angles)))
        coherence = float(np.sqrt(c ** 2 + s ** 2))
        dominant_angle = float(0.5 * np.arctan2(s, c))

        features.update({
            "spindle_line_count": float(len(lengths)),
            "spindle_total_length_norm": float(lengths.sum() / diag),
            "spindle_mean_length_norm": float(lengths.mean() / diag),
            "spindle_angle_coherence": coherence,
            "spindle_dominant_angle_rad": dominant_angle,
        })
    else:
        features.update({
            "spindle_line_count": 0.0,
            "spindle_total_length_norm": 0.0,
            "spindle_mean_length_norm": 0.0,
            "spindle_angle_coherence": 0.0,
            "spindle_dominant_angle_rad": 0.0,
        })

    # 5. Angle Alignment (Keselarasan Orientasi)
    pca_ang = features.get("pca_angle_rad", 0.0)
    spindle_ang = features.get("spindle_dominant_angle_rad", 0.0)

    diff = abs(pca_ang - spindle_ang)
    diff = min(diff, math.pi - diff) if diff <= math.pi else diff % math.pi

    features.update({
        "axis_line_alignment": float(abs(math.cos(diff))),
        "axis_line_perpendicularity": float(abs(math.sin(diff))),
    })

    # 6. Mengisi Nilai Kosong (Menghindari fitur bernilai Null / Error)
    # Fitur preprocessing diset ke 0.0 karena tidak dihitung ulang untuk hemat waktu komputasi UI
    dummy_keys = [
        "file_size_bytes", "edge_density", "candidate_line_edge_density",
        "gray_mean", "gray_std", "clahe_mean", "clahe_std",
        "background_corrected_mean", "background_corrected_std",
        "sharpened_mean", "sharpened_std", "contrast_gain_clahe",
        "contrast_gain_background_corrected"
    ]
    for key in dummy_keys:
        features[key] = 0.0

    return features
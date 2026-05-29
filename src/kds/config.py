"""Application configuration and paths."""

import os
from pathlib import Path

# Project root (KDS/) — two levels up from this file: src/kds/config.py
ROOT_DIR = Path(__file__).resolve().parents[2]

_default_model = ROOT_DIR / "models" / "meiosis_phase_opencv_random_forest.joblib"
MODEL_PATH = Path(os.environ.get("KDS_MODEL_PATH", str(_default_model)))

FEATURE_ORDER = [
    "file_size_bytes",
    "chromatin_area_ratio",
    "edge_density",
    "candidate_line_edge_density",
    "gray_mean",
    "gray_std",
    "clahe_mean",
    "clahe_std",
    "background_corrected_mean",
    "background_corrected_std",
    "sharpened_mean",
    "sharpened_std",
    "contrast_gain_clahe",
    "contrast_gain_background_corrected",
    "object_count",
    "mean_object_area_norm",
    "std_object_area_norm",
    "max_object_area_norm",
    "mean_circularity",
    "std_circularity",
    "chromatin_centroid_x_norm",
    "chromatin_centroid_y_norm",
    "centroid_distance_to_center_norm",
    "pca_axis_ratio",
    "pca_major_span_norm",
    "pca_minor_span_norm",
    "pca_angle_rad",
    "band_score",
    "cluster_separation_norm",
    "cluster_balance",
    "cluster_compactness_norm",
    "spindle_line_count",
    "spindle_total_length_norm",
    "spindle_mean_length_norm",
    "spindle_angle_coherence",
    "spindle_dominant_angle_rad",
    "axis_line_alignment",
    "axis_line_perpendicularity",
]

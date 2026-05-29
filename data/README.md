# Data

Put **all CSV files** used for training or evaluation here.

Your notebook in `notebooks/` should read and write CSVs in this folder. They are **gitignored** by default (many are large); use Git LFS if you need them on GitHub.

| File | Purpose |
|------|---------|
| `opencv_spatial_features.csv` | Extracted features for training |
| `manual_phase_labels_template.csv` | Phase labels |
| `predictions_test.csv` | Test-set predictions |
| `feature_importance.csv` | Model feature importance |
| `permutation_importance.csv` | Permutation importance |

These files are **not** required to run the deployed Streamlit app—only the `.joblib` in `models/` is.

# Meiosis Phase Classifier (KDS)

Classify meiotic phases from microscopy images using OpenCV preprocessing, spatial feature extraction, and a Random Forest model.

**Repository:** [github.com/valereoo/KDS](https://github.com/valereoo/KDS)

---

## How to run this repo

| Goal | Where | How |
|------|--------|-----|
| **Train / regenerate model** | Google Colab | [Open notebook in Colab](#1-train-the-model-google-colab-recommended) |
| **Use the classifier (UI)** | Local or Streamlit Cloud | [Run the Streamlit app](#2-run-the-web-app-streamlit) |

**Live app:** [kdskelompok17.streamlit.app](https://kdskelompok17.streamlit.app/)

---

## Project structure

```
KDS/
├── notebooks/             # Training notebook (run on Colab)
├── data/                  # CSV inputs/outputs (gitignored if large)
├── models/                # Trained .joblib for the web app
├── app.py                 # Streamlit entry point
├── src/kds/               # App logic used by Streamlit
└── requirements.txt
```

---

## 1. Train the model (Google Colab) — recommended

The training pipeline lives in:

`notebooks/Kelompok17_K04_IF3211_18223010_18223016_18223030_18223068_klasifikasi_fase_meiosis_opencv.ipynb`

It downloads the Kaggle dataset [`hasan1101/mitosis-detection`](https://www.kaggle.com/datasets/hasan1101/mitosis-detection), extracts OpenCV features, trains a Random Forest, and writes artifacts to `outputs_meiosis_opencv/`.

### Open in Colab (one click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valereoo/KDS/blob/main/notebooks/Kelompok17_K04_IF3211_18223010_18223016_18223030_18223068_klasifikasi_fase_meiosis_opencv.ipynb)

### Or clone inside a new Colab notebook

Run this in the **first cell** of Colab, then open the notebook from the file browser (`notebooks/` → your `.ipynb`):

```python
!git clone https://github.com/valereoo/KDS.git
%cd KDS/notebooks
```

### Colab setup checklist

1. **Runtime** → Change runtime type → **Python 3** (GPU not required).
2. Run cells **top to bottom** from the training notebook.
3. **Kaggle data** — the notebook uses `kagglehub` to download the dataset automatically. If download fails:
   - Create a [Kaggle API token](https://www.kaggle.com/settings) (`kaggle.json`).
   - In Colab: **Upload** `kaggle.json`, then run:
     ```python
     !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     ```
   - Re-run the dataset download cell.
4. **Optional manual labels** — place `manual_phase_labels.csv` in the notebook working directory (`notebooks/` when using `%cd KDS/notebooks`), or use the template the notebook generates.

### After training on Colab

The notebook saves files under `outputs_meiosis_opencv/`. Copy the important ones into the repo layout so the Streamlit app can use them:

| From (Colab) | Copy to (repo) |
|--------------|----------------|
| `outputs_meiosis_opencv/meiosis_phase_opencv_random_forest.joblib` | `models/` |
| `outputs_meiosis_opencv/opencv_spatial_features.csv` | `data/` |
| `outputs_meiosis_opencv/manual_phase_labels_template.csv` | `data/` |
| `outputs_meiosis_opencv/predictions_test.csv` | `data/` |
| `outputs_meiosis_opencv/feature_importance.csv` | `data/` |
| `outputs_meiosis_opencv/permutation_importance.csv` | `data/` |

**Download from Colab** (run in a new cell after training):

```python
from google.colab import files
from pathlib import Path
import shutil

out = Path("outputs_meiosis_opencv")
zip_path = "/content/kds_artifacts.zip"
shutil.make_archive("/content/kds_artifacts", "zip", out)
files.download(zip_path)
```

Unzip locally into `models/` and `data/` as in the table above.

**Commit the model to GitHub** (use Git LFS for the `.joblib`):

```bash
git lfs install
git lfs track "*.joblib"
git add models/meiosis_phase_opencv_random_forest.joblib
git commit -m "Add trained model"
git push
```

---

## 2. Run the web app (Streamlit)

The interactive classifier is `app.py`. It needs the trained model in `models/meiosis_phase_opencv_random_forest.joblib` (from Colab or a previous local run).

### Local

```bash
git clone https://github.com/valereoo/KDS.git
cd KDS

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### Streamlit Community Cloud (deploy)

1. Push this repo to GitHub (include the model via Git LFS, or upload it after first Colab run).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. **New app** → repository `valereoo/KDS` → **Main file:** `app.py` → Deploy.

Deployed app: [https://kdskelompok17.streamlit.app/](https://kdskelompok17.streamlit.app/)

---

## 3. Clone and push to GitHub

```bash
git clone https://github.com/valereoo/KDS.git
cd KDS

git lfs install
git lfs track "*.joblib"

git add .
git commit -m "Your message"
git push origin main
```

- `env/`, `.venv/`, and large `*.csv` files in `data/` are gitignored by default.
- Track `.joblib` with **Git LFS** (see `models/README.md`).

---

## Where to put files

| File | Folder |
|------|--------|
| Training notebook (`.ipynb`) | `notebooks/` |
| CSV datasets & exports | `data/` |
| Trained model (`.joblib`) | `models/` |
| Virtual environment | `env/` or `.venv/` (local only, not committed) |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `KDS_MODEL_PATH` | Override path to the `.joblib` model |

---

## License

Add your license here (e.g. MIT).

# Model artifacts

Place the trained model here:

```
models/meiosis_phase_opencv_random_forest.joblib
```

The file is ~70 MB. For GitHub, use **Git LFS**:

```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes models/meiosis_phase_opencv_random_forest.joblib
```

Streamlit Community Cloud will download LFS files automatically when you deploy.

Override the path locally with:

```bash
set KDS_MODEL_PATH=C:\path\to\model.joblib
```

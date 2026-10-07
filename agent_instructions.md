# Agent Instructions – Salary Prediction Project

## Overview
This project is a binary classification ML web app predicting whether an individual earns ≤50K or >50K annually, using the UCI Adult Census Income dataset.

## Dataset
- **File**: `data/dataset.csv`
- **Source**: UCI Adult Census Income (48,842 rows × 15 columns)
- **Target**: `income` column → `<=50K` (0) or `>50K` (1)
- **Missing values**: Originally encoded as `?` — replaced with NaN during preprocessing

## Key Files
| File | Role |
|------|------|
| `train_model.py` | Load data → preprocess → train models → save `models/model.pkl` |
| `app.py` | Streamlit UI — load model, collect inputs, display prediction |
| `models/model.pkl` | Joblib artifact containing pipeline + label encoder + metadata |
| `data/dataset.csv` | Raw dataset (do not modify) |

## How to Re-train
```bash
python train_model.py
```

## How to Add a New Model
In `train_model.py`, add an entry to the `candidates` dict:
```python
candidates["XGBoost"] = XGBClassifier(n_estimators=200, random_state=42)
```
Then re-run `python train_model.py`.

## Environment Variables
This project currently requires no API keys. See `.env.example` for the template.

## Deployment Notes
- All paths are **relative** — safe for Streamlit Community Cloud / Render.
- `model.pkl` must exist before launching `app.py`. Run `train_model.py` once first.
- To auto-generate `model.pkl` on cloud deploy, add a `packages.txt` build step or commit the file.

## Coding Conventions
- Python 3.9+
- PEP 8 style
- No absolute paths
- No Jupyter Notebooks
- No Flask / Django

"""
train_model.py
--------------
Loads the Adult Census Income dataset, preprocesses it, trains and evaluates
multiple classification models, picks the best one, and saves the full
sklearn Pipeline + label encoder to models/model.pkl.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

print("=" * 60)
print("  Adult Census Income – Salary Prediction")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. Basic exploration
# ─────────────────────────────────────────────
TARGET = "income"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"\nTarget column  : {TARGET}")
print(f"Feature columns: {FEATURES}")
print(f"\nTarget distribution:\n{df[TARGET].value_counts()}")

# ─────────────────────────────────────────────
# 3. Replace '?' with NaN
# ─────────────────────────────────────────────
df.replace("?", np.nan, inplace=True)

# Strip whitespace from string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# ─────────────────────────────────────────────
# 4. Encode target
# ─────────────────────────────────────────────
le_target = LabelEncoder()
y = le_target.fit_transform(df[TARGET])
# Store mapping: 0 -> <=50K, 1 -> >50K  (LabelEncoder sorts alphabetically)
print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

X = df[FEATURES].copy()

# ─────────────────────────────────────────────
# 5. Identify numeric vs categorical columns
# ─────────────────────────────────────────────
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumeric features   : {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# ─────────────────────────────────────────────
# 6. Build preprocessing pipeline
# ─────────────────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

# ─────────────────────────────────────────────
# 7. Train / test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]:,} | Test set: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 8. Define candidate models
# ─────────────────────────────────────────────
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n─── Cross-validation (5-fold ROC-AUC) ───")
cv_scores = {}
for name, clf in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_scores[name] = scores.mean()
    print(f"  {name:28s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 9. Select best model
# ─────────────────────────────────────────────
best_name = max(cv_scores, key=cv_scores.get)
best_clf = candidates[best_name]
print(f"\n[BEST] Best model: {best_name}  (CV ROC-AUC = {cv_scores[best_name]:.4f})")

# ─────────────────────────────────────────────
# 10. Final training on full train set
# ─────────────────────────────────────────────
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_clf),
])

best_pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 11. Evaluate on hold-out test set
# ─────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n─── Test-set evaluation ───")
print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=le_target.classes_)}")

# ─────────────────────────────────────────────
# 12. Save pipeline + metadata
# ─────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

artifact = {
    "pipeline":       best_pipeline,
    "label_encoder":  le_target,
    "feature_names":  FEATURES,
    "numeric_cols":   numeric_cols,
    "categorical_cols": categorical_cols,
    "best_model_name": best_name,
    "test_accuracy":  acc,
    "test_roc_auc":   auc,
    # Store unique values for categorical dropdowns in Streamlit
    "cat_unique_values": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in categorical_cols
    },
    # Store numeric ranges for sliders in Streamlit
    "num_ranges": {
        col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
        for col in numeric_cols
    },
}

joblib.dump(artifact, MODEL_PATH)
print(f"\n[SAVED] Model artifact saved to: {MODEL_PATH}")
print("=" * 60)
print("  Training complete!")
print("=" * 60)

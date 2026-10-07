"""
app.py
------
Streamlit web application for Adult Census Income (Salary) Prediction.
Loads the trained ML pipeline from models/model.pkl and lets users
enter feature values to receive a salary prediction (<=50K or >50K).
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor – Adult Census Income",
    page_icon="💼",
    layout="wide",
)

# ─────────────────────────────────────────────
# Load model artifact
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "model.pkl")

@st.cache_resource(show_spinner="Loading ML model…")
def load_artifact(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

artifact = load_artifact(MODEL_PATH)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("💼 Salary Prediction App")
st.markdown(
    """
    Predict whether an individual earns **≤ 50K** or **> 50K** per year  
    based on census information. Enter the details below and click **Predict**.
    """
)

if artifact is None:
    st.error(
        "⚠️ Model not found. Please run `python train_model.py` first to train and save the model.",
        icon="🚨",
    )
    st.stop()

# ─────────────────────────────────────────────
# Sidebar – model info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 Model Information")
    st.metric("Best Model", artifact["best_model_name"])
    st.metric("Test Accuracy", f"{artifact['test_accuracy']:.2%}")
    st.metric("Test ROC-AUC", f"{artifact['test_roc_auc']:.4f}")
    st.divider()
    st.caption(
        "Model trained on the [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) dataset."
    )

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
st.subheader("📋 Enter Individual Details")

num_ranges  = artifact["num_ranges"]
cat_uniques = artifact["cat_unique_values"]
numeric_cols    = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]
feature_names   = artifact["feature_names"]

# Build a two-column form
col1, col2 = st.columns(2)
user_input = {}

def _col_for(idx):
    return col1 if idx % 2 == 0 else col2

# ── Numeric inputs ──────────────────────────
for i, col in enumerate(numeric_cols):
    rng = num_ranges[col]
    label = col.replace("-", " ").replace("_", " ").title()
    with _col_for(i):
        if col in ("age", "educational-num", "hours-per-week"):
            user_input[col] = st.slider(
                label,
                min_value=int(rng["min"]),
                max_value=int(rng["max"]),
                value=int(rng["mean"]),
                step=1,
            )
        else:
            user_input[col] = st.number_input(
                label,
                min_value=float(rng["min"]),
                max_value=float(rng["max"]),
                value=float(rng["mean"]),
                step=1.0,
                format="%.0f",
            )

# ── Categorical inputs ───────────────────────
offset = len(numeric_cols)
for i, col in enumerate(categorical_cols):
    options = cat_uniques[col]
    label = col.replace("-", " ").replace("_", " ").title()
    with _col_for(offset + i):
        user_input[col] = st.selectbox(label, options=options)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Predict Salary", type="primary", use_container_width=True)

if predict_btn:
    # Build input DataFrame in the exact feature order the pipeline was trained on
    input_df = pd.DataFrame([{col: user_input[col] for col in feature_names}])

    pipeline = artifact["pipeline"]
    le       = artifact["label_encoder"]

    pred_encoded = pipeline.predict(input_df)[0]
    pred_proba   = pipeline.predict_proba(input_df)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]

    # Class probabilities
    prob_dict = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(pred_proba)}

    st.subheader("📊 Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred_label == ">50K":
            st.success(f"### Income: **{pred_label}**", icon="✅")
            st.markdown("This individual is predicted to earn **more than $50K/year**.")
        else:
            st.warning(f"### Income: **{pred_label}**", icon="⚠️")
            st.markdown("This individual is predicted to earn **$50K/year or less**.")

    with res_col2:
        st.markdown("**Prediction Confidence**")
        for label, prob in sorted(prob_dict.items(), reverse=True):
            icon = "🟢" if label == ">50K" else "🔴"
            st.progress(prob, text=f"{icon} {label}: {prob:.1%}")

    # Show input summary
    with st.expander("📝 Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Built with ❤️ using Python · scikit-learn · Streamlit  |  "
    "Dataset: UCI Adult Census Income"
)

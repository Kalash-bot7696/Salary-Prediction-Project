# 💼 Salary Prediction – Adult Census Income

A Machine Learning web application that predicts whether an individual earns **≤ 50K** or **> 50K** per year based on census data. Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📂 Project Structure

```
project/
├── app.py                  # Streamlit web application
├── train_model.py          # Data preprocessing + model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── agent_instructions.md   # Agent/developer instructions
├── .env.example            # Environment variable template
├── data/
│   └── dataset.csv         # Adult Census Income dataset
└── models/
    └── model.pkl           # Saved trained ML pipeline
```

---

## 🚀 Local Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/salary-prediction.git
cd salary-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```
This will:
- Load and preprocess `data/dataset.csv`
- Train and cross-validate four classification models
- Select the best model by ROC-AUC
- Save the trained pipeline to `models/model.pkl`

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the entire project (including `data/dataset.csv`) to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo, branch (`main`), and set the main file path to `app.py`.
4. Before the first deployment, make sure `models/model.pkl` is committed **or** add a build step.

> **Tip:** To auto-train on first deploy, add a Streamlit `on_start` hook or include `model.pkl` directly in the repo (it is small enough for git).

---

## 🤖 ML Details

| Item | Value |
|------|-------|
| Dataset | UCI Adult Census Income |
| Problem | Binary Classification |
| Target | `income` (≤50K / >50K) |
| Models compared | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Selection criterion | 5-fold cross-validated ROC-AUC |
| Preprocessing | Median imputation (numeric), Mode imputation + OHE (categorical), StandardScaler |

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt).

---

## 📄 License

MIT

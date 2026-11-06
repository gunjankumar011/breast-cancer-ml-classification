import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Cancer Classifier Lab", page_icon="🩺", layout="wide")

# model name -> pickle file
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_nb.pkl",
    "SVM": "svm.pkl",
}
# these were trained on scaled data, the other two weren't
NEEDS_SCALING = ["Logistic Regression", "KNN", "SVM"]


@st.cache_resource
def load_everything():
    models = {name: joblib.load("models/" + f) for name, f in MODEL_FILES.items()}
    scaler = joblib.load("models/scaler.pkl")
    cols = joblib.load("models/feature_names.pkl")
    return models, scaler, cols


def get_scores(model, X):
    # svm might be saved without probability=True, so fall back
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_model(name, X_raw):
    model = models[name]
    X = scaler.transform(X_raw) if name in NEEDS_SCALING else X_raw
    return model.predict(X), get_scores(model, X)


def calc_metrics(y, pred, score):
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def read_upload(file):
    """returns (df, y, X) or stops the app with an error message"""
    df = pd.read_csv(file)

    if "diagnosis" not in df.columns:
        st.error("No 'diagnosis' column found in the CSV.")
        st.stop()

    y = df["diagnosis"].astype(str).str.strip().str.upper().map(
        {"B": 0, "M": 1, "0": 0, "1": 1}
    )
    if y.isnull().any():
        st.error("diagnosis should only have B, M, 0 or 1.")
        st.stop()

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"These feature columns are missing: {missing}")
        st.stop()

    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        st.error("Some feature values are empty or not numbers.")
        st.stop()

    return df, y, X


# ---------- sidebar ----------
try:
    models, scaler, feature_names = load_everything()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

with st.sidebar:
    st.title("🩺 Control Panel")
    st.caption("Breast Cancer (Diagnostic)")
    st.divider()

    file = st.file_uploader("1. Upload test CSV", type=["csv"])
    mode = st.radio("2. What do you want to do?", ["Single model", "Compare all models"])

    chosen = None
    if mode == "Single model":
        chosen = st.selectbox("3. Pick a model", list(models.keys()))

    go = st.button("Run", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"{len(models)} models loaded ✅")

# ---------- main page ----------
st.title("Classifier Comparison Dashboard")

if file is None:
    st.info("👈 Upload a test CSV from the sidebar to begin. It needs the feature columns plus a `diagnosis` column.")
    st.stop()

try:
    df, y_true, X_test = read_upload(file)
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.stop()

tab_data, tab_result, tab_cm = st.tabs(["📄 Data", "📊 Results", "🧮 Confusion matrix"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Benign", int((y_true == 0).sum()))
    c3.metric("Malignant", int((y_true == 1).sum()))
    st.dataframe(df.head(10), use_container_width=True)

if not go:
    with tab_result:
        st.write("Press **Run** in the sidebar.")
    with tab_cm:
        st.write("Nothing to show yet.")
    st.stop()

names = [chosen] if mode == "Single model" else list(models.keys())
all_metrics, all_preds = {}, {}

with st.spinner("Crunching predictions..."):
    for n in names:
        pred, score = run_model(n, X_test)
        all_metrics[n] = calc_metrics(y_true, pred, score)
        all_preds[n] = pred

with tab_result:
    if mode == "Single model":
        m = all_metrics[chosen]
        st.subheader(chosen)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, f"{v:.3f}")

        st.text(classification_report(
            y_true, all_preds[chosen],
            target_names=["Benign", "Malignant"], zero_division=0,
        ))
    else:
        table = pd.DataFrame(all_metrics).T.round(4)
        best = table["F1"].idxmax()
        st.success(f"Best F1 score: **{best}** ({table.loc[best, 'F1']:.4f})")
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table[["Accuracy", "Precision", "Recall", "F1"]])

with tab_cm:
    # one matrix per model, two per row
    for i in range(0, len(names), 2):
        row = st.columns(2)
        for col, n in zip(row, names[i:i + 2]):
            cm = confusion_matrix(y_true, all_preds[n])
            col.markdown(f"**{n}**")
            col.dataframe(pd.DataFrame(
                cm,
                index=["Actual Benign", "Actual Malignant"],
                columns=["Pred Benign", "Pred Malignant"],
            ))
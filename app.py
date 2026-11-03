
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------
# 1. Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="ML Classifier Comparison",
    page_icon="",
    layout="wide"
)

st.title("Machine Learning Classifier Comparison")
st.subheader("Breast Cancer Wisconsin Diagnostic Dataset")

st.write(
    "This application evaluates five pre-trained "
    "machine learning classification models."
)
 
# 2. Load Saved Models
 

@st.cache_resource
def load_models():

    log_model = joblib.load("models/logistic_regression.pkl")
    dt_model = joblib.load("models/decision_tree.pkl")
    knn_model = joblib.load("models/knn.pkl")
    nb_model = joblib.load("models/gaussian_nb.pkl")
    svm_model = joblib.load("models/svm.pkl")

    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    models = {
        "Logistic Regression": log_model,
        "Decision Tree": dt_model,
        "KNN": knn_model,
        "Gaussian Naive Bayes": nb_model,
        "SVM": svm_model
    }

    return models, scaler, feature_names


try:
    models, scaler, feature_names = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

 
# 3. Upload Test Dataset
 

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing features and diagnosis",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")
        st.dataframe(test_df.head())

        # Check target column
        if "diagnosis" not in test_df.columns:
            st.error("CSV must contain diagnosis column.")
            st.stop()

        # Convert diagnosis into numeric labels
        y_true = (
            test_df["diagnosis"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "B": 0,
                "M": 1,
                "0": 0,
                "1": 1
            })
        )

        if y_true.isnull().any():
            st.error("Invalid diagnosis values. Use B, M, 0 or 1.")
            st.stop()

        # Check required features
        missing_features = [
            col for col in feature_names
            if col not in test_df.columns
        ]

        if missing_features:
            st.error(f"Missing features: {missing_features}")
            st.stop()

        # Arrange features in training order
        X_test = test_df[feature_names].copy()

        # Check numeric values
        for col in feature_names:
            X_test[col] = pd.to_numeric(
                X_test[col],
                errors="coerce"
            )

        if X_test.isnull().any().any():
            st.error("Features contain missing or non-numeric values.")
            st.stop()

        st.success(
            f"Dataset uploaded successfully! Total records: {len(X_test)}"
        )
 
        # 4. Select Classifier
      

        st.header("Select Machine Learning Model")

        model_name = st.selectbox(
            "Choose Classifier",
            list(models.keys())
        )

    
        # 5. Evaluate Model
      
        if st.button("Evaluate Model"):

            model = models[model_name]

            # Scale only required models
            if model_name in [
                "Logistic Regression",
                "KNN",
                "SVM"
            ]:
                X_input = scaler.transform(X_test)

            else:
                X_input = X_test

            # Predict using saved model
            y_pred = model.predict(X_input)

            y_prob = model.predict_proba(X_input)[:, 1]

            # Calculate metrics
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_prob),
                "Precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_true, y_pred)
            }

         
            # 6. Display Results
           

            st.header(f"{model_name} Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC", f"{metrics['AUC']:.4f}")
            col3.metric("Precision", f"{metrics['Precision']:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            col6.metric("MCC", f"{metrics['MCC']:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)

            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual Benign", "Actual Malignant"],
                    columns=["Predicted Benign", "Predicted Malignant"]
                )
            )

            # Classification Report
            st.subheader("Classification Report")

            st.text(
                classification_report(
                    y_true,
                    y_pred,
                    target_names=["Benign", "Malignant"],
                    zero_division=0
                )
            )

    except Exception as e:
        st.error(f"Unable to evaluate dataset: {e}")

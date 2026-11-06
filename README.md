# Breast Cancer Classification Using Machine Learning

## Project Overview

This project focuses on developing and evaluating multiple Machine Learning classification models to predict whether a breast cancer tumour is Benign or Malignant.

The project implements different classification algorithms, compares their performance, and provides an interactive web application using Streamlit to demonstrate model predictions.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Pickle
- Git & GitHub

## Machine Learning Algorithms

The following classification algorithms are implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Project Structure
 
ML Assignment 2/
│
├── app.py
├── breast-cancer.csv
├── test_data.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── feature_names.pkl
│   ├── gaussian_nb.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── svm.pkl
│
└── README.md
 

## Features

- Breast cancer classification using multiple ML algorithms.
- Model training and evaluation.
- Comparison of different classification algorithms.
- Pre-trained models saved using Pickle.
- Feature scaling for improved model performance.
- Interactive Streamlit web application.
- Prediction using trained machine learning models.

## Installation and Setup

### 1. Clone the Repository
 
git clone <https://github.com/gunjankumar011/breast-cancer-ml-classification>
 
### 2. Navigate to the Project Directory
 
cd "ML Assignment 2"
 

### 3. Install Dependencies
 
pip install numpy pandas scikit-learn matplotlib streamlit
 

### 4. Run the Application
 
streamlit run app.py
 

### 5. Open in Browser

The application will be available at:
 
http://localhost:8501
 

## Model Files

The trained machine learning models are stored in the `models` directory in Pickle (`.pkl`) format.

These files allow the application to load pre-trained models without requiring model training every time the application starts.

## Dataset

The project uses a breast cancer dataset for binary classification.

The objective is to classify tumour observations into two categories:

Benign or Malignant

## Learning Outcomes

- Understanding supervised machine learning.
- Implementing different classification algorithms.
- Understanding feature scaling and preprocessing.
- Evaluating and comparing classification models.
- Saving and loading trained ML models.
- Deploying a machine learning application using Streamlit.

## Author

**Gunjan Kumar**

M.Tech – Artificial Intelligence and Machine Learning  
BITS Pilani (2026AK05080)

---

*This project is developed as part of the Machine Learning coursework at BITS Pilani.*
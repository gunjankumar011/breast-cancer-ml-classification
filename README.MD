# Machine Learning Classification App

## Project Overview

This project is a Python-based Machine Learning application developed to predict breast cancer diagnosis using multiple classification algorithms.

The application uses pre-trained machine learning models to evaluate input data and generate predictions. It also provides model performance information to help compare different classification algorithms.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pickle
- Streamlit (if used as the application interface)

## Machine Learning Algorithms

The project includes the following classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes

## Project Structure
 
ML Assignment 2/
│
├── app.py
├── breast-cancer.csv
├── test_data.csv
│
└── models/
    ├── decision_tree.pkl
    ├── feature_names.pkl
    ├── gaussian_nb.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── scaler.pkl
    └── svm.pkl
```

## Installation and Setup

### 1. Clone the Repository
 
git clone <https://github.com/gunjankumar011/breast-cancer-ml-classification>
 

### 2. Navigate to Project Directory
 
cd "ML Assignment 2"
 

### 3. Install Dependencies
 
pip install numpy pandas scikit-learn matplotlib streamlit
 

### 4. Run the Application

If the application uses Streamlit:
 
streamlit run app.py
 

Otherwise, for a standard Python application:
 
python app.py
 

## Dataset

The application uses a Breast Cancer dataset containing medical features used for classification.

The target variable represents:

- **0:** Benign
- **1:** Malignant

## Model Files

Pre-trained models are stored in the `models` directory. These files are loaded by the application to generate predictions without retraining the models every time.

The project also includes a scaler and feature names to maintain consistency during prediction.

## Key Features

- Multiple Machine Learning classification algorithms.
- Pre-trained model integration.
- Breast cancer diagnosis prediction.
- Model performance evaluation.
- Data preprocessing and feature scaling.
- Comparison of classification models.

## Future Improvements

- Add an interactive prediction dashboard.
- Display confusion matrix and classification reports.
- Add accuracy comparison charts.
- Deploy the application to a cloud platform.
- Improve user experience and visualization.

## Author

Developed as part of the Machine Learning coursework and practical assignments for M.Tech Artificial Intelligence and Machine Learning by Gunjan Kumar.

## License

This project is intended for educational and academic purposes.
# 🏥 Multi-Disease Prediction System

**Author:** Bhardwaja  
**Programme:** MSc Artificial Intelligence & Business Intelligence — University of Leicester  
**Project Type:** Machine Learning · Healthcare AI · Portfolio Project

---

## 📌 Overview

This project predicts three diseases from patient diagnostic data using trained ML models, deployed as an interactive web app using Streamlit.

| Disease | Dataset | Best Model | Features |
|---|---|---|---|
| Chronic Kidney Disease | UCI CKD (400 samples) | Random Forest | 24 |
| Diabetes | Pima Indians Extended (1543 samples) | Random Forest | 8 |
| Breast Cancer | UCI Wisconsin (569 samples) | SVM (RBF) | 30 |

---

## 📁 Project Structure

```
multi-disease-prediction/
│
├── notebooks/
│   ├── kidney_disease_prediction.ipynb    ← Full EDA + 3 models + saved model
│   ├── diabetes_prediction.ipynb          ← Full EDA + 4 models + saved model
│   └── breast_cancer_prediction.ipynb     ← Full EDA + 4 models + saved model
│
├── data/
│   ├── kidney_disease.csv
│   ├── diabetes_disease.csv
│   └── breast_cancer_data.csv
│
├── models/                                ← Generated after running notebooks
│   ├── kidney_model.pkl
│   ├── kidney_imputer.pkl
│   ├── diabetes_model.pkl
│   ├── diabetes_imputer.pkl
│   ├── breast_cancer_model.pkl
│   └── breast_cancer_scaler.pkl
│
├── app.py                                 ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train models (run each notebook)
Open and run all cells in:
- `notebooks/kidney_disease_prediction.ipynb`
- `notebooks/diabetes_prediction.ipynb`
- `notebooks/breast_cancer_prediction.ipynb`

This generates the `.pkl` model files.

### Step 3 — Launch the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🧠 ML Techniques Used

- **Preprocessing:** Median/Mean Imputation, StandardScaler, domain-specific encoding
- **Models:** KNN, Random Forest, Logistic Regression, Naive Bayes, SVM, Gradient Boosting
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
- **Cross-Validation:** 5-fold CV for robust generalisation estimates
- **No Data Leakage:** Imputer and scaler fitted only on training data

---

## ⚠️ Disclaimer

This tool is for **educational and portfolio purposes only**.  
It is NOT a substitute for professional medical advice or clinical diagnosis.

---

## 🔗 Technologies

`Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Streamlit` · `joblib`

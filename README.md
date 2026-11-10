
# OncoVision AI – Breast Cancer Decision Support Prototype

An end-to-end machine learning web application prototype for exploring breast cancer classification using the Wisconsin Breast Cancer Diagnostic Dataset.

> **Research Prototype Disclaimer**
>
> This application is for educational and research demonstration only. It is not a medical device and must not be used to diagnose, screen, triage, or make treatment decisions for patients. Model predictions are not clinical diagnoses. Real medical decisions must be made by qualified healthcare professionals using appropriate clinical examinations and validated diagnostic procedures.

---

## Dataset

This project uses the Wisconsin Breast Cancer Diagnostic (WDBC) dataset.

- 569 observations
- 30 numerical features
- Two original dataset classes: benign and malignant

The features describe ten nuclear characteristics:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each characteristic has three measurements:

- Mean
- Standard error
- Worst value

## Machine Learning Pipeline

The project uses a scikit-learn pipeline containing:

1. Missing-value imputation
2. ANOVA-based feature selection using `SelectKBest` and `f_classif`
3. Feature scaling using `StandardScaler`
4. Logistic Regression classification

The training script evaluates the model using a stratified train/test split and calculates metrics such as accuracy, precision, recall, F1-score, ROC-AUC, log loss, Brier score, and confusion matrix.

Reported performance depends on the exact dataset, split, and trained model artifacts. These results are experimental and do not establish clinical validity or suitability for patient care.

---

## Project Structure

```text
OncoVision-AI/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── model_service.py
│   ├── train.py
│   └── data_utils.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── breast_cancer_model.joblib
│   ├── model_metadata.json
│   ├── selected_features.json
│   └── feature_stats.json
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   └── BrestCancer_Diagnosis.ipynb
├── tests/
│   └── test_api.py
├── .gitignore
├── Procfile
├── render.yaml
├── requirements.txt
├── run.py
└── README.md
```

---

## Requirements

- Python 3.11
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Ensure the trained model artifacts are available in the `models/` directory.

Start the application:

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run Tests

```bash
python -m unittest tests/test_api.py
```

---

## Application Features

### 1. Dual Input Modes

- Quick mode: Enter the selected top 10 model features.
- Full mode: Enter all 30 dataset features.

In Quick mode, missing model inputs are handled by the configured preprocessing pipeline.

### 2. Example Preset Cases

The application includes three illustrative examples:

- Benign-labeled example
- Malignant-labeled example
- Intermediate-probability example

These are demonstration inputs, not real patient records or validated clinical cases.

### 3. Model Explanation

The interface compares entered measurements with benign and malignant dataset reference statistics.

These comparisons are descriptive explanations, not validated causal explanations or formal SHAP feature attributions.

### 4. Batch Prediction

Upload a CSV file containing multiple feature records or use the provided demonstration dataset.

The application displays model predictions and probability-based categories for each record.

These categories are for software demonstration only and must not be interpreted as clinical triage decisions.

### 5. Printable Summary

The application provides a printable summary of model output for demonstration and educational purposes.

It is not a medical report or a substitute for a professional diagnostic report.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Application and model readiness |
| GET | `/api/metadata` | Model metadata |
| GET | `/api/feature-stats` | Feature reference statistics |
| GET | `/api/presets` | Demonstration preset cases |
| POST | `/api/predict` | Single-record prediction |
| POST | `/api/predict-batch` | Batch predictions |
| GET | `/api/sample-csv` | Download sample CSV |
| POST | `/api/retrain` | Disabled in the deployed application |

---

## Deployment

This project is configured for deployment as a Python Flask web service on Render.

The application uses Gunicorn as its production server.

Start command:

```bash
gunicorn --bind 0.0.0.0:$PORT run:app
```

The trained model artifacts must be available in the repository before deployment.

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML
- CSS
- JavaScript
- Gunicorn
- Render

---

## Project Purpose

OncoVision AI demonstrates how machine learning, data preprocessing, model evaluation, and web application development can be combined into an educational decision-support prototype.

**This project is not intended for real-world medical diagnosis or patient management.**

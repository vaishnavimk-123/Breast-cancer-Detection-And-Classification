# OncoVision AI: Breast Cancer Diagnostic Decision Support System (CDSS)

An end-to-end, clinically-informed machine learning web application prototype for Fine Needle Aspirate (FNA) breast cytology analysis, risk stratification, and patient triage.

> **CLINICAL NOTICE & DISCLAIMER:**
> This software is a research prototype designed to demonstrate machine learning-assisted clinical decision support. It is not approved as a primary diagnostic device. All AI assessments must be correlated with clinical examination, radiological imaging (BI-RADS), and histopathological evaluation by a certified pathologist.

---

## 🔬 Clinical Overview & Dataset

The system is trained on the **Wisconsin Breast Cancer Diagnostic Dataset (WDBC)** collected by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian from the University of Wisconsin Hospitals, Madison.

The dataset comprises **569 patient FNA biopsies** characterized across **30 nuclear morphometric attributes**:
- **10 Core Nuclear Properties**: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension.
- **3 Metric Perspectives**: Mean values, Standard Error (SE) variances, and Worst/Extreme measurements.

### Top 10 High-Impact Diagnostic Biomarkers (ANOVA F-Score Ranking)
Using **SelectKBest with ANOVA F-test (`f_classif`)**, the most decisive morphological factors distinguishing benign from malignant lesions were identified:
1. `concave points_worst` (F-score: 964.4)
2. `perimeter_worst` (F-score: 897.9)
3. `concave points_mean` (F-score: 861.7)
4. `radius_worst` (F-score: 860.8)
5. `perimeter_mean` (F-score: 697.2)
6. `area_worst` (F-score: 661.6)
7. `radius_mean` (F-score: 647.0)
8. `area_mean` (F-score: 573.1)
9. `concavity_mean` (F-score: 533.8)
10. `concavity_worst` (F-score: 436.7)

---

## 📊 Validated Model Performance Metrics

Evaluated on an **80/20 stratified holdout test split** ($N = 114$ patients):

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Testing Accuracy** | **95.61%** | Overall correct classification rate |
| **ROC-AUC** | **0.9974** | Near-perfect discriminative probability ranking |
| **Sensitivity (Recall)** | **90.48%** | Minimized false negatives to prevent missed malignancies |
| **Specificity** | **98.61%** | Low false positives (FP = 1) to avoid unnecessary biopsies |
| **Precision (PPV)** | **97.44%** | Reliability of positive malignant predictions |
| **F1-Score** | **93.83%** | Harmonic balance between sensitivity and precision |
| **Brier Score** | **0.0381** | Well-calibrated probabilistic output |

### Holdout Confusion Matrix ($N = 114$)
| | Predicted Benign (0) | Predicted Malignant (1) |
| :--- | :---: | :---: |
| **Actual Benign (B)** | **71 (True Negative)** | **1 (False Positive)** |
| **Actual Malignant (M)** | **4 (False Negative)** | **38 (True Positive)** |

---

## 🏛️ Project Directory Structure

```text
Creast_cancer/
├── backend/
│   ├── __init__.py
│   ├── app.py                 # Flask REST API server (routes, CORS, batch processing)
│   ├── model_service.py       # Inference engine, XAI feature attribution, validation
│   ├── train.py               # Reproducible pipeline training & metrics calculation
│   └── data_utils.py          # Clinical feature definitions, normal ranges, presets
├── models/
│   ├── breast_cancer_model.joblib  # Serialized scikit-learn pipeline
│   ├── model_metadata.json    # Version-locked accuracy, AUC, confusion matrix
│   ├── selected_features.json # Top 10 ANOVA biomarkers
│   └── feature_stats.json     # Population percentiles and reference baselines
├── frontend/
│   ├── templates/
│   │   └── index.html         # Responsive, medical-grade UI
│   └── static/
│       ├── css/style.css      # Clinical styling, print stylesheets
│       └── js/app.js          # Interactive controllers, XAI charts, batch triage
├── data/
│   └── breast_cancer.csv      # Clean Wisconsin diagnostic dataset
├── notebooks/
│   └── BrestCancer_Diagnosis.ipynb # Original research notebook
├── tests/
│   └── test_api.py            # Comprehensive unit and integration test suite
├── run.py                     # Unified single-command application launcher
├── requirements.txt           # Project dependencies
└── README.md                  # Clinical and technical documentation
```

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.10+ (Tested on Python 3.13)
- Required libraries listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Launch the Application
Run the unified launcher:
```bash
python run.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### 3. Run Automated Tests
Execute the automated test suite covering model loading, prediction logic, edge cases, and REST endpoints:
```bash
python -m unittest tests/test_api.py
```

---

## 🩺 Clinical Features & Workflow

### 1. Dual Diagnostic Input Modes
- **Top 10 High-Impact Biomarkers (Quick Mode)**: Optimized for fast entry using the 10 ANOVA-selected features. Secondary features are imputed with Wisconsin population medians.
- **Full 30-Parameter FNA Cytology Panel**: Grouped into collapsible tabs (*Means*, *Standard Errors*, *Worst Values*) with unit specifications, physiological ranges, and anomaly alerts.

### 2. One-Click Clinical Presets
- 🟢 **Case 1: Typical Benign Lesion (Fibroadenoma)** — Demonstrates low risk output (< 25% malignancy).
- 🔴 **Case 2: Invasive Ductal Carcinoma (High Grade)** — Demonstrates high risk output (> 90% malignancy).
- 🟡 **Case 3: Borderline / Atypical Cell Feature** — Demonstrates the AI uncertainty alert for gray-zone cases near the 50% decision boundary.

### 3. Explainable AI (XAI) Biomarker Attribution
- Displays individual patient measurements compared side-by-side against **Benign Cohort Medians** and **Malignant Cohort Medians**.
- Visual spread bars show the relative percentile within the known population.
- Identifies specific cytological drivers (e.g. "Severe Concavity", "Nuclear Hypertrophy").

### 4. Population Batch Triage Screening
- Upload a multi-patient CSV or click **"Load Wisconsin Demo Batch"**.
- Automated triage highlights:
  - High Risk patients needing immediate oncology consultation.
  - Borderline cases requiring repeat biopsy or pathologist review.
  - Low Risk routine screening cases.

### 5. Printable Clinical Record Export
- Click **"Print Clinical Summary"** to generate a clean, header-formatted medical summary ready for physical inclusion in patient charts or PDF export.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and model readiness |
| `GET` | `/api/metadata` | Model performance metrics, confusion matrix, feature list |
| `GET` | `/api/feature-stats` | Reference distribution baselines for all 30 features |
| `GET` | `/api/presets` | Preset clinical test cases |
| `POST` | `/api/predict` | Single patient inference, risk assessment, and XAI breakdown |
| `POST` | `/api/predict-batch` | Batch CSV screening or array of patient records |
| `GET` | `/api/sample-csv` | Downloadable CSV template for batch screening |
| `POST` | `/api/retrain` | Triggers retraining and live reload of model artifacts |


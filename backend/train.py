"""
Breast Cancer Diagnostic Model Training Pipeline
Reproducible training script with comprehensive clinical performance metrics.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_and_clean_data(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(ROOT_DIR, "data", "breast_cancer.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().copy()
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df = df.dropna(subset=["diagnosis"]).copy()

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"B": 0, "M": 1})
    )
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    X = df.drop(columns=["diagnosis"]).select_dtypes(include=np.number).dropna(axis=1, how="all")
    y = df["diagnosis"]

    return df, X, y


def train_and_evaluate(csv_path="data/breast_cancer.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    df, X, y = load_and_clean_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    k_features = min(10, X_train.shape[1])

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_test_prob))
    loss = float(log_loss(y_test, y_test_prob))
    brier = float(brier_score_loss(y_test, y_test_prob))

    cm = confusion_matrix(y_test, y_test_pred)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity & Sensitivity
    sensitivity = recall  # TP / (TP + FN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Selected features
    selector = pipeline.named_steps["feature_selection"]
    selected_mask = selector.get_support()
    all_feature_names = list(X_train.columns)
    selected_features = [f for f, s in zip(all_feature_names, selected_mask) if s]

    scores = selector.scores_
    feature_importance_list = []
    for feat, score in zip(all_feature_names, scores):
        feature_importance_list.append({
            "feature": feat,
            "f_score": float(score) if not np.isnan(score) else 0.0,
            "selected": bool(feat in selected_features)
        })
    feature_importance_list.sort(key=lambda x: x["f_score"], reverse=True)

    # Calculate distributions for each feature
    feature_stats = {}
    for feat in all_feature_names:
        benign_vals = df[df["diagnosis"] == 0][feat]
        malignant_vals = df[df["diagnosis"] == 1][feat]
        feature_stats[feat] = {
            "min": float(df[feat].min()),
            "max": float(df[feat].max()),
            "mean": float(df[feat].mean()),
            "median": float(df[feat].median()),
            "std": float(df[feat].std()),
            "benign_mean": float(benign_vals.mean()),
            "benign_median": float(benign_vals.median()),
            "benign_std": float(benign_vals.std()),
            "benign_q25": float(benign_vals.quantile(0.25)),
            "benign_q75": float(benign_vals.quantile(0.75)),
            "malignant_mean": float(malignant_vals.mean()),
            "malignant_median": float(malignant_vals.median()),
            "malignant_std": float(malignant_vals.std()),
            "malignant_q25": float(malignant_vals.quantile(0.25)),
            "malignant_q75": float(malignant_vals.quantile(0.75)),
        }

    # Model metadata
    metadata = {
        "model_name": "ANOVA SelectKBest + Logistic Regression Pipeline",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "metrics": {
            "accuracy": test_acc,
            "train_accuracy": train_acc,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "roc_auc": auc,
            "log_loss": loss,
            "brier_score": brier,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        },
        "all_features": all_feature_names,
        "selected_features": selected_features,
        "feature_ranking": feature_importance_list[:15]
    }

    # Save artifacts
    model_path = os.path.join(output_dir, "breast_cancer_model.joblib")
    joblib.dump(pipeline, model_path)

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    features_path = os.path.join(output_dir, "selected_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)

    stats_path = os.path.join(output_dir, "feature_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)

    # Also keep root files synchronized so legacy callers work
    try:
        with open("breast_cancer_model.pkl", "wb") as f:
            pickle.dump(pipeline, f, protocol=4)
        with open("selected_features.pkl", "wb") as f:
            pickle.dump(selected_features, f, protocol=4)
    except Exception as e:
        print(f"Notice: Could not write legacy root pkl files: {e}")

    print("==========================================")
    print("Model Training & Evaluation Successful!")
    print(f"Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:    {test_acc * 100:.2f}%")
    print(f"Sensitivity (Recall):{sensitivity * 100:.2f}%")
    print(f"Specificity:         {specificity * 100:.2f}%")
    print(f"Precision:           {precision * 100:.2f}%")
    print(f"F1-Score:            {f1 * 100:.2f}%")
    print(f"ROC-AUC:             {auc:.4f}")
    print(f"Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("Selected Features (Top 10):", selected_features)
    print(f"Saved artifacts to {output_dir}/")
    print("==========================================")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()


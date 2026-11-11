"""
Model Service: High-performance inference engine, input validation,
clinical risk assessment, and explainable AI (XAI) feature attribution.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from backend.data_utils import FEATURE_DEFINITIONS, CLINICAL_PRESET_CASES, classify_risk
except ImportError:
    from data_utils import FEATURE_DEFINITIONS, CLINICAL_PRESET_CASES, classify_risk


class BreastCancerModelService:
    def __init__(self, model_dir=None):
        if model_dir is None:
            self.model_dir = os.path.join(ROOT_DIR, "models")
        elif not os.path.isabs(model_dir):
            cand = os.path.join(ROOT_DIR, model_dir)
            self.model_dir = cand if os.path.exists(cand) else os.path.abspath(model_dir)
        else:
            self.model_dir = model_dir

        self.pipeline = None
        self.metadata = {}
        self.selected_features = []
        self.feature_stats = {}
        self.all_features = []
        self._load_artifacts()

    def _patch_model_compatibility(self):
        """Ensures the loaded model step has all attributes expected by the runtime's scikit-learn version."""
        if self.pipeline is not None and hasattr(self.pipeline, "named_steps"):
            if "model" in self.pipeline.named_steps:
                model_step = self.pipeline.named_steps["model"]
                if not hasattr(model_step, "multi_class"):
                    model_step.multi_class = "auto"

    def _train_fallback(self):
        """Train and persist model artifacts on-the-fly if missing or version-incompatible."""
        try:
            try:
                from backend.train import train_and_evaluate
            except ImportError:
                from train import train_and_evaluate
            csv_path = os.path.join(ROOT_DIR, "data", "breast_cancer.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(ROOT_DIR, "breast_cancer.csv")
            return train_and_evaluate(csv_path=csv_path, output_dir=self.model_dir)
        except Exception as e:
            print(f"Warning: Auto-training fallback encountered error: {e}")
            return None

    def _load_artifacts(self):
        os.makedirs(self.model_dir, exist_ok=True)

        # Support both joblib and pickle artifacts, with fallback to root dir
        model_path = os.path.join(self.model_dir, "breast_cancer_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, "breast_cancer_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "breast_cancer_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "breast_cancer_model.pkl")

        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        features_path = os.path.join(self.model_dir, "selected_features.json")
        stats_path = os.path.join(self.model_dir, "feature_stats.json")

        # Train on-the-fly if essential artifacts are missing
        if not os.path.exists(model_path) or not os.path.exists(metadata_path) or not os.path.exists(stats_path):
            print("[+] Model artifacts missing or incomplete. Training pipeline on-the-fly...")
            self._train_fallback()
            model_path = os.path.join(self.model_dir, "breast_cancer_model.joblib")

        # Load pipeline using joblib
        try:
            self.pipeline = joblib.load(model_path)
            self._patch_model_compatibility()
        except Exception as load_err:
            print(f"Notice: Failed to load artifact ({load_err}). Retraining model in current environment...")
            self._train_fallback()
            model_path = os.path.join(self.model_dir, "breast_cancer_model.joblib")
            self.pipeline = joblib.load(model_path)
            self._patch_model_compatibility()

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        
        if os.path.exists(features_path):
            with open(features_path, "r", encoding="utf-8") as f:
                self.selected_features = json.load(f)
        elif os.path.exists(os.path.join(ROOT_DIR, "selected_features.pkl")):
            import pickle
            with open(os.path.join(ROOT_DIR, "selected_features.pkl"), "rb") as f:
                self.selected_features = pickle.load(f)
        else:
            self.selected_features = self.metadata.get("selected_features", [])

        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                self.feature_stats = json.load(f)

        self.all_features = self.metadata.get("all_features", list(FEATURE_DEFINITIONS.keys()))

    def get_metadata(self):
        return {
            "metadata": self.metadata,
            "feature_definitions": FEATURE_DEFINITIONS,
            "selected_features": self.selected_features,
            "preset_cases": CLINICAL_PRESET_CASES
        }

    def get_feature_stats(self):
        return self.feature_stats

    def _prepare_feature_row(self, input_dict):
        """
        Ensures all 30 features expected by the pipeline exist.
        If a feature is omitted (e.g. from 10-biomarker quick mode),
        it is imputed with the baseline population median.
        """
        row = {}
        missing_count = 0
        validation_warnings = []

        for feat in self.all_features:
            val = input_dict.get(feat)
            if val is None or val == "" or (isinstance(val, (int, float)) and np.isnan(val)):
                row[feat] = self.feature_stats.get(feat, {}).get("median", 0.0)
                missing_count += 1
            else:
                try:
                    fval = float(val)
                    row[feat] = fval

                    if feat in self.feature_stats:
                        fmin = self.feature_stats[feat]["min"]
                        fmax = self.feature_stats[feat]["max"]
                        if fval < 0:
                            validation_warnings.append(f"{feat}: Negative value ({fval}) is physiologically invalid.")
                        elif fval > fmax * 2.5:
                            validation_warnings.append(f"{feat}: Value {fval} exceeds extreme cohort maximum ({fmax}).")
                except (ValueError, TypeError):
                    row[feat] = self.feature_stats.get(feat, {}).get("median", 0.0)
                    missing_count += 1

        df_row = pd.DataFrame([row], columns=self.all_features)
        return df_row, missing_count, validation_warnings

    def explain_features(self, input_row):
        """
        Provides Explainable AI (XAI) insights:
        Compares input values against benign and malignant cohort statistics.
        """
        analysis = []
        for feat in self.selected_features:
            val = float(input_row.get(feat, 0.0))
            stats = self.feature_stats.get(feat, {})
            b_med = stats.get("benign_median", 0.0)
            m_med = stats.get("malignant_median", 0.0)
            f_min = stats.get("min", 0.0)
            f_max = stats.get("max", 1.0)

            spread = (f_max - f_min) if (f_max - f_min) > 0 else 1.0
            norm_val = max(0.0, min(100.0, ((val - f_min) / spread) * 100))
            norm_b = max(0.0, min(100.0, ((b_med - f_min) / spread) * 100))
            norm_m = max(0.0, min(100.0, ((m_med - f_min) / spread) * 100))

            dist_to_b = abs(val - b_med)
            dist_to_m = abs(val - m_med)
            leans_towards = "Malignant Pattern" if dist_to_m < dist_to_b else "Benign Pattern"

            meta = FEATURE_DEFINITIONS.get(feat, {})

            analysis.append({
                "feature": feat,
                "label": meta.get("label", feat),
                "unit": meta.get("unit", ""),
                "patient_value": round(val, 4),
                "benign_median": round(b_med, 4),
                "malignant_median": round(m_med, 4),
                "cohort_min": round(f_min, 4),
                "cohort_max": round(f_max, 4),
                "norm_value": round(norm_val, 1),
                "norm_benign": round(norm_b, 1),
                "norm_malignant": round(norm_m, 1),
                "leans_towards": leans_towards,
                "is_malignant_lean": bool(dist_to_m < dist_to_b),
                "description": meta.get("description", ""),
                "clinical_significance": meta.get("clinical_significance", "")
            })

        analysis.sort(key=lambda x: (not x["is_malignant_lean"], -x["norm_value"]))
        return analysis

    def predict_single(self, input_dict, patient_id=None):
        """
        Perform clinical prediction on single patient record.
        """
        df_row, missing_count, warnings = self._prepare_feature_row(input_dict)

        # Ensure compatibility across scikit-learn versions
        self._patch_model_compatibility()
        try:
            prob_array = self.pipeline.predict_proba(df_row)[0]
            prediction_class = int(self.pipeline.predict(df_row)[0])
        except AttributeError as attr_err:
            if "multi_class" in str(attr_err):
                if hasattr(self.pipeline, "named_steps") and "model" in self.pipeline.named_steps:
                    self.pipeline.named_steps["model"].multi_class = "auto"
                prob_array = self.pipeline.predict_proba(df_row)[0]
                prediction_class = int(self.pipeline.predict(df_row)[0])
            else:
                raise

        benign_prob = float(prob_array[0])
        malignant_prob = float(prob_array[1])

        diagnosis_label = "Malignant" if prediction_class == 1 else "Benign"
        risk_info = classify_risk(malignant_prob)

        explanation = self.explain_features(df_row.iloc[0].to_dict())

        malignant_drivers = [e["label"] for e in explanation if e["is_malignant_lean"]]
        benign_drivers = [e["label"] for e in explanation if not e["is_malignant_lean"]]

        response = {
            "case_id": patient_id or f"PT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diagnosis": diagnosis_label,
            "prediction_code": prediction_class,
            "probabilities": {
                "benign": round(benign_prob, 4),
                "malignant": round(malignant_prob, 4),
                "benign_percentage": round(benign_prob * 100, 2),
                "malignant_percentage": round(malignant_prob * 100, 2),
            },
            "risk_assessment": risk_info,
            "explainability": {
                "key_biomarkers": explanation,
                "malignant_indicators": malignant_drivers,
                "benign_indicators": benign_drivers,
            },
            "data_quality": {
                "features_provided": len(self.all_features) - missing_count,
                "features_imputed": missing_count,
                "validation_warnings": warnings
            }
        }

        return response

    def predict_batch(self, records_list):
        results = []
        total = len(records_list)
        malignant_count = 0
        benign_count = 0
        indeterminate_count = 0

        for i, record in enumerate(records_list):
            pid = record.get("id") or record.get("patient_id") or f"CASE-{i+1:04d}"
            pred_res = self.predict_single(record, patient_id=str(pid))
            
            if pred_res["diagnosis"] == "Malignant":
                malignant_count += 1
            else:
                benign_count += 1

            if pred_res["risk_assessment"]["is_borderline"]:
                indeterminate_count += 1

            results.append({
                "patient_id": str(pid),
                "diagnosis": pred_res["diagnosis"],
                "malignancy_percentage": pred_res["probabilities"]["malignant_percentage"],
                "risk_category": pred_res["risk_assessment"]["risk_category"],
                "badge": pred_res["risk_assessment"]["badge"],
                "alert_level": pred_res["risk_assessment"]["alert_level"],
                "requires_urgent_review": pred_res["risk_assessment"]["requires_urgent_review"],
                "is_borderline": pred_res["risk_assessment"]["is_borderline"],
            })

        return {
            "total_screened": total,
            "summary": {
                "benign_count": benign_count,
                "benign_percentage": round((benign_count / total * 100), 1) if total > 0 else 0,
                "malignant_count": malignant_count,
                "malignant_percentage": round((malignant_count / total * 100), 1) if total > 0 else 0,
                "indeterminate_count": indeterminate_count,
                "indeterminate_percentage": round((indeterminate_count / total * 100), 1) if total > 0 else 0,
            },
            "patients": results
        }


# Global singleton instance
_model_service_instance = None

def get_model_service():
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = BreastCancerModelService()
    return _model_service_instance


"""
Model Service: Model inference, input validation,
risk assessment, and cohort-based feature explanations.

OncoVision AI - Research Prototype
"""

import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

from backend.data_utils import (
    FEATURE_DEFINITIONS,
    CLINICAL_PRESET_CASES,
    classify_risk,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BreastCancerModelService:

    def __init__(self, model_dir=None):

        if model_dir is None:
            self.model_dir = os.path.join(ROOT_DIR, "models")

        elif not os.path.isabs(model_dir):
            candidate = os.path.join(ROOT_DIR, model_dir)
            self.model_dir = (
                candidate
                if os.path.exists(candidate)
                else os.path.abspath(model_dir)
            )

        else:
            self.model_dir = model_dir

        self.pipeline = None
        self.metadata = {}
        self.selected_features = []
        self.feature_stats = {}
        self.all_features = []

        self._load_artifacts()

    # --------------------------------------------------
    # LOAD MODEL AND SUPPORTING FILES
    # --------------------------------------------------

    def _load_artifacts(self):

        possible_model_paths = [
            os.path.join(
                self.model_dir,
                "breast_cancer_model.joblib"
            ),
            os.path.join(
                self.model_dir,
                "breast_cancer_model.pkl"
            ),
            os.path.join(
                ROOT_DIR,
                "breast_cancer_model.joblib"
            ),
            os.path.join(
                ROOT_DIR,
                "breast_cancer_model.pkl"
            ),
        ]

        model_path = next(
            (
                path for path in possible_model_paths
                if os.path.isfile(path)
            ),
            None
        )

        if model_path is None:
            raise FileNotFoundError(
                "Trained model not found. Expected "
                "'breast_cancer_model.joblib' inside the models folder. "
                "Run the training script first."
            )

        metadata_path = os.path.join(
            self.model_dir,
            "model_metadata.json"
        )

        features_path = os.path.join(
            self.model_dir,
            "selected_features.json"
        )

        stats_path = os.path.join(
            self.model_dir,
            "feature_stats.json"
        )

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Model metadata file not found: {metadata_path}"
            )

        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"Feature statistics file not found: {stats_path}"
            )

        try:
            self.pipeline = joblib.load(model_path)

        except Exception as error:
            raise RuntimeError(
                f"Could not load trained model: {error}"
            ) from error

        if not hasattr(self.pipeline, "predict"):
            raise RuntimeError(
                "Loaded model does not support prediction."
            )

        if not hasattr(self.pipeline, "predict_proba"):
            raise RuntimeError(
                "Loaded model does not support probability prediction."
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

            with open(stats_path, "r", encoding="utf-8") as file:
                self.feature_stats = json.load(file)

        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"Could not read model metadata or feature statistics: {error}"
            ) from error

        if not isinstance(self.metadata, dict):
            raise RuntimeError("Model metadata must contain a JSON object.")

        if not isinstance(self.feature_stats, dict):
            raise RuntimeError("Feature statistics must contain a JSON object.")

        # Load selected features.
        if os.path.isfile(features_path):

            try:
                with open(features_path, "r", encoding="utf-8") as file:
                    self.selected_features = json.load(file)

            except (json.JSONDecodeError, OSError) as error:
                raise RuntimeError(
                    f"Could not read selected features: {error}"
                ) from error

        else:
            # Backward compatibility with the previous pickle format.
            legacy_features_path = os.path.join(
                ROOT_DIR,
                "selected_features.pkl"
            )

            if os.path.isfile(legacy_features_path):

                try:
                    self.selected_features = joblib.load(
                        legacy_features_path
                    )

                except Exception as error:
                    raise RuntimeError(
                        f"Could not load legacy selected features: {error}"
                    ) from error

            else:
                self.selected_features = self.metadata.get(
                    "selected_features",
                    []
                )

        if not isinstance(self.selected_features, list):
            raise RuntimeError(
                "Selected features must be stored as a list."
            )

        # Determine the complete feature list.
        self.all_features = self.metadata.get("all_features")

        if not self.all_features:
            self.all_features = getattr(
                self.pipeline,
                "feature_names_in_",
                list(FEATURE_DEFINITIONS.keys())
            )

        self.all_features = list(self.all_features)

        if not self.all_features:
            raise RuntimeError(
                "No input features were found in the model metadata."
            )

        if not self.selected_features:
            raise RuntimeError(
                "No selected features were found."
            )

        invalid_selected_features = [
            feature
            for feature in self.selected_features
            if feature not in self.all_features
        ]

        if invalid_selected_features:
            raise RuntimeError(
                "Selected features do not match the model input features: "
                + ", ".join(invalid_selected_features)
            )

        print(f"Model loaded successfully: {model_path}")
        print(f"Total input features: {len(self.all_features)}")
        print(f"Selected features: {len(self.selected_features)}")

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    def get_metadata(self):

        return {
            "metadata": self.metadata,
            "feature_definitions": FEATURE_DEFINITIONS,
            "selected_features": self.selected_features,
            "preset_cases": CLINICAL_PRESET_CASES
        }

    def get_feature_stats(self):

        return self.feature_stats

    # --------------------------------------------------
    # INPUT VALIDATION AND PREPARATION
    # --------------------------------------------------

    def _prepare_feature_row(self, input_dict):

        """
        Validate incoming feature values and prepare a DataFrame.

        Missing or invalid values are represented as NaN so that
        the trained sklearn pipeline can apply its own imputer.
        """

        if not isinstance(input_dict, dict):
            raise ValueError(
                "Input features must be provided as a JSON object."
            )

        row = {}
        missing_count = 0
        validation_warnings = []

        for feature in self.all_features:

            value = input_dict.get(feature)

            # Handle missing values.
            if value is None or (
                isinstance(value, str) and not value.strip()
            ):

                row[feature] = np.nan
                missing_count += 1
                continue

            # Reject boolean values as numeric measurements.
            if isinstance(value, bool):

                row[feature] = np.nan
                missing_count += 1

                validation_warnings.append(
                    f"{feature}: Boolean value is not a valid numeric measurement."
                )

                continue

            try:

                numeric_value = float(value)

                if not np.isfinite(numeric_value):
                    raise ValueError("Value must be finite.")

                row[feature] = numeric_value

                stats = self.feature_stats.get(feature, {})

                minimum = stats.get("min")
                maximum = stats.get("max")

                # Negative measurements are flagged.
                if numeric_value < 0:

                    validation_warnings.append(
                        f"{feature}: Negative value detected."
                    )

                # Flag values far outside the training cohort range.
                elif maximum is not None:

                    try:
                        maximum = float(maximum)

                        if (
                            np.isfinite(maximum)
                            and maximum > 0
                            and numeric_value > maximum * 2.5
                        ):

                            validation_warnings.append(
                                f"{feature}: Value exceeds the expected cohort range."
                            )

                    except (ValueError, TypeError):
                        pass

            except (ValueError, TypeError, OverflowError):

                row[feature] = np.nan
                missing_count += 1

                validation_warnings.append(
                    f"{feature}: Invalid numeric value; model imputation will be used."
                )

        # Preserve the exact feature order expected by the pipeline.
        feature_dataframe = pd.DataFrame(
            [row],
            columns=self.all_features
        )

        return (
            feature_dataframe,
            missing_count,
            validation_warnings
        )

    # --------------------------------------------------
    # FEATURE EXPLANATIONS
    # --------------------------------------------------

    def explain_features(self, input_row):

        """
        Compare input values with benign and malignant cohort medians.

        IMPORTANT:
        These are descriptive cohort comparisons, not SHAP values
        or direct explanations of the model's internal decision.
        """

        analysis = []

        for feature in self.selected_features:

            stats = self.feature_stats.get(feature, {})

            value = input_row.get(feature, np.nan)

            try:

                value = float(value)

                if not np.isfinite(value):
                    raise ValueError

            except (ValueError, TypeError, OverflowError):

                value = stats.get("median", 0.0)

            benign_median = stats.get("benign_median", 0.0)
            malignant_median = stats.get("malignant_median", 0.0)

            feature_min = stats.get("min", 0.0)
            feature_max = stats.get("max", 1.0)

            # Ensure statistics are numeric.
            try:
                benign_median = float(benign_median)
                malignant_median = float(malignant_median)
                feature_min = float(feature_min)
                feature_max = float(feature_max)

            except (ValueError, TypeError, OverflowError):

                benign_median = 0.0
                malignant_median = 0.0
                feature_min = 0.0
                feature_max = 1.0

            spread = (
                feature_max - feature_min
                if feature_max > feature_min
                else 1.0
            )

            # Normalize values to a 0-100 display scale.
            normalized_value = max(
                0.0,
                min(
                    100.0,
                    ((value - feature_min) / spread) * 100
                )
            )

            normalized_benign = max(
                0.0,
                min(
                    100.0,
                    ((benign_median - feature_min) / spread) * 100
                )
            )

            normalized_malignant = max(
                0.0,
                min(
                    100.0,
                    ((malignant_median - feature_min) / spread) * 100
                )
            )

            distance_to_benign = abs(
                value - benign_median
            )

            distance_to_malignant = abs(
                value - malignant_median
            )

            malignant_lean = (
                distance_to_malignant < distance_to_benign
            )

            leans_towards = (
                "Malignant Pattern"
                if malignant_lean
                else "Benign Pattern"
            )

            feature_metadata = FEATURE_DEFINITIONS.get(
                feature,
                {}
            )

            analysis.append({

                "feature": feature,

                "label": feature_metadata.get(
                    "label",
                    feature
                ),

                "unit": feature_metadata.get(
                    "unit",
                    ""
                ),

                "patient_value": round(value, 4),

                "benign_median": round(
                    benign_median,
                    4
                ),

                "malignant_median": round(
                    malignant_median,
                    4
                ),

                "cohort_min": round(
                    feature_min,
                    4
                ),

                "cohort_max": round(
                    feature_max,
                    4
                ),

                "norm_value": round(
                    normalized_value,
                    1
                ),

                "norm_benign": round(
                    normalized_benign,
                    1
                ),

                "norm_malignant": round(
                    normalized_malignant,
                    1
                ),

                "leans_towards": leans_towards,

                "is_malignant_lean": bool(
                    malignant_lean
                ),

                "description": feature_metadata.get(
                    "description",
                    ""
                ),

                "clinical_significance": feature_metadata.get(
                    "clinical_significance",
                    ""
                )

            })

        # Keep the existing frontend sorting behaviour.
        analysis.sort(
            key=lambda item: (
                not item["is_malignant_lean"],
                -item["norm_value"]
            )
        )

        return analysis

    # --------------------------------------------------
    # SINGLE PREDICTION
    # --------------------------------------------------

    def predict_single(self, input_dict, patient_id=None):

        """
        Run model inference for one record.

        This output is a research-model result, not a clinical diagnosis.
        """

        if self.pipeline is None:
            raise RuntimeError("The prediction model is not loaded.")

        feature_dataframe, missing_count, warnings = (
            self._prepare_feature_row(input_dict)
        )

        # Obtain model probabilities.
        probability_array = self.pipeline.predict_proba(
            feature_dataframe
        )[0]

        model_classes = list(
            getattr(self.pipeline, "classes_", [0, 1])
        )

        if 0 not in model_classes or 1 not in model_classes:
            raise RuntimeError(
                "The trained model must contain both classes 0 and 1."
            )

        benign_index = model_classes.index(0)
        malignant_index = model_classes.index(1)

        benign_probability = float(
            probability_array[benign_index]
        )

        malignant_probability = float(
            probability_array[malignant_index]
        )

        prediction_class = int(
            self.pipeline.predict(feature_dataframe)[0]
        )

        diagnosis_label = (
            "Malignant"
            if prediction_class == 1
            else "Benign"
        )

        risk_info = classify_risk(
            malignant_probability
        )

        # Generate descriptive feature comparisons.
        explanation = self.explain_features(
            feature_dataframe.iloc[0].to_dict()
        )

        malignant_drivers = [
            item["label"]
            for item in explanation
            if item["is_malignant_lean"]
        ]

        benign_drivers = [
            item["label"]
            for item in explanation
            if not item["is_malignant_lean"]
        ]

        response = {

            "case_id": (
                str(patient_id)
                if patient_id
                else f"PT-{uuid.uuid4().hex[:8].upper()}"
            ),

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "diagnosis": diagnosis_label,

            "prediction_code": prediction_class,

            "probabilities": {

                "benign": round(
                    benign_probability,
                    4
                ),

                "malignant": round(
                    malignant_probability,
                    4
                ),

                "benign_percentage": round(
                    benign_probability * 100,
                    2
                ),

                "malignant_percentage": round(
                    malignant_probability * 100,
                    2
                )

            },

            "risk_assessment": risk_info,

            "explainability": {

                "key_biomarkers": explanation,

                "malignant_indicators": malignant_drivers,

                "benign_indicators": benign_drivers

            },

            "data_quality": {

                "features_provided": (
                    len(self.all_features) - missing_count
                ),

                "features_imputed": missing_count,

                "validation_warnings": warnings

            }

        }

        return response

    # --------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------

    def predict_batch(self, records_list):

        """
        Process multiple records supplied as a list of dictionaries.
        """

        if not isinstance(records_list, list):
            raise ValueError(
                "Batch input must be a list of records."
            )

        if not records_list:
            raise ValueError(
                "Batch input cannot be empty."
            )

        results = []

        total = len(records_list)

        malignant_count = 0
        benign_count = 0
        indeterminate_count = 0

        for index, record in enumerate(records_list):

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record {index + 1} must be a JSON object."
                )

            patient_id = (
                record.get("id")
                or record.get("patient_id")
                or f"CASE-{index + 1:04d}"
            )

            prediction_result = self.predict_single(
                record,
                patient_id=str(patient_id)
            )

            if prediction_result["diagnosis"] == "Malignant":
                malignant_count += 1

            else:
                benign_count += 1

            risk_assessment = prediction_result[
                "risk_assessment"
            ]

            if risk_assessment.get("is_borderline", False):
                indeterminate_count += 1

            results.append({

                "patient_id": str(patient_id),

                "diagnosis": prediction_result["diagnosis"],

                "malignancy_percentage": (
                    prediction_result["probabilities"][
                        "malignant_percentage"
                    ]
                ),

                "risk_category": risk_assessment.get(
                    "risk_category"
                ),

                "badge": risk_assessment.get(
                    "badge"
                ),

                "alert_level": risk_assessment.get(
                    "alert_level"
                ),

                "requires_urgent_review": risk_assessment.get(
                    "requires_urgent_review"
                ),

                "is_borderline": risk_assessment.get(
                    "is_borderline",
                    False
                )

            })

        return {

            "total_screened": total,

            "summary": {

                "benign_count": benign_count,

                "benign_percentage": round(
                    benign_count / total * 100,
                    1
                ),

                "malignant_count": malignant_count,

                "malignant_percentage": round(
                    malignant_count / total * 100,
                    1
                ),

                "indeterminate_count": indeterminate_count,

                "indeterminate_percentage": round(
                    indeterminate_count / total * 100,
                    1
                )

            },

            "patients": results

        }


# --------------------------------------------------
# GLOBAL MODEL SERVICE INSTANCE
# --------------------------------------------------

_model_service_instance = None


def get_model_service():

    global _model_service_instance

    if _model_service_instance is None:

        _model_service_instance = BreastCancerModelService()

    return _model_service_instance

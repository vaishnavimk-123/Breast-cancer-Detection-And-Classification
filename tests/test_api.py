"""
Unit & Integration Tests for OncoVision AI Decision Support System
"""

import os
import sys
import unittest
import json

# Add root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app
from backend.model_service import BreastCancerModelService
from backend.data_utils import CLINICAL_PRESET_CASES


class TestBreastCancerCDSS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = BreastCancerModelService(model_dir=os.path.join(ROOT_DIR, "models"))
        cls.client = app.test_client()

    def test_01_model_loading_and_metadata(self):
        self.assertIsNotNone(self.service.pipeline)
        meta = self.service.get_metadata()
        self.assertIn("metadata", meta)
        self.assertIn("metrics", meta["metadata"])
        # Accuracy should be > 90%
        self.assertGreater(meta["metadata"]["metrics"]["accuracy"], 0.90)
        self.assertEqual(len(self.service.selected_features), 10)

    def test_02_predict_benign_case(self):
        case_benign = CLINICAL_PRESET_CASES[0]
        res = self.service.predict_single(case_benign["data"], patient_id=case_benign["id"])
        self.assertEqual(res["diagnosis"], "Benign")
        self.assertLess(res["probabilities"]["malignant"], 0.35)
        self.assertEqual(res["risk_assessment"]["badge"], "success")

    def test_03_predict_malignant_case(self):
        case_malignant = CLINICAL_PRESET_CASES[1]
        res = self.service.predict_single(case_malignant["data"], patient_id=case_malignant["id"])
        self.assertEqual(res["diagnosis"], "Malignant")
        self.assertGreater(res["probabilities"]["malignant"], 0.70)
        self.assertEqual(res["risk_assessment"]["badge"], "danger")

    def test_04_predict_quick_10_features_only(self):
        """Test partial inputs where only the top 10 biomarkers are provided."""
        case_benign = CLINICAL_PRESET_CASES[0]
        quick_10_data = {k: case_benign["data"][k] for k in self.service.selected_features}
        res = self.service.predict_single(quick_10_data, patient_id="QUICK-10-TEST")
        self.assertEqual(res["diagnosis"], "Benign")
        self.assertEqual(res["data_quality"]["features_imputed"], 20)

    def test_05_api_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_06_api_predict_endpoint(self):
        case_malignant = CLINICAL_PRESET_CASES[1]
        response = self.client.post(
            "/api/predict",
            data=json.dumps({"patient_id": "TEST-API-01", "features": case_malignant["data"]}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["diagnosis"], "Malignant")
        self.assertIn("risk_assessment", data)
        self.assertIn("explainability", data)

    def test_07_api_batch_predict(self):
        batch_payload = [
            {"id": "BATCH-P1", **CLINICAL_PRESET_CASES[0]["data"]},
            {"id": "BATCH-P2", **CLINICAL_PRESET_CASES[1]["data"]},
        ]
        response = self.client.post(
            "/api/predict-batch",
            data=json.dumps(batch_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["total_screened"], 2)
        self.assertEqual(len(data["patients"]), 2)
        self.assertEqual(data["summary"]["benign_count"], 1)
        self.assertEqual(data["summary"]["malignant_count"], 1)

    def test_08_static_assets_and_frontend(self):
        # Frontend home page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"OncoVision", res.data)
        res.close()

        # Static CSS
        res_css = self.client.get("/static/css/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn(b"clinical-card", res_css.data)
        res_css.close()

        # Static JS
        res_js = self.client.get("/static/js/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn(b"appState", res_js.data)
        res_js.close()

    def test_09_api_metadata_and_presets(self):
        res_meta = self.client.get("/api/metadata")
        self.assertEqual(res_meta.status_code, 200)
        meta = res_meta.get_json()
        self.assertIn("feature_definitions", meta)

        res_presets = self.client.get("/api/presets")
        self.assertEqual(res_presets.status_code, 200)
        self.assertGreaterEqual(len(res_presets.get_json()["presets"]), 3)

    def test_10_api_sample_csv(self):
        res = self.client.get("/api/sample-csv")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn(b"radius_mean", res.data)

    def test_11_invalid_input_handling(self):
        # Empty payload
        res = self.client.post("/api/predict", data="", content_type="application/json")
        self.assertEqual(res.status_code, 400)

        # Out of range input validation warning check
        res_warn = self.service.predict_single({"radius_mean": -5.0, "area_worst": 999999.0})
        self.assertGreater(len(res_warn["data_quality"]["validation_warnings"]), 0)

    def test_12_root_pkl_files(self):
        import pickle
        import joblib
        with open(os.path.join(ROOT_DIR, "breast_cancer_model.pkl"), "rb") as f:
            m = pickle.load(f)
        self.assertIsNotNone(m)

        with open(os.path.join(ROOT_DIR, "selected_features.pkl"), "rb") as f:
            sf = pickle.load(f)
        self.assertEqual(len(sf), 10)


if __name__ == "__main__":
    unittest.main()


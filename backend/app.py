
"""
Flask REST API and Clinical Decision Support Backend Server
OncoVision AI - Research Prototype
"""

import os
import io
import pandas as pd

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

from backend.model_service import get_model_service
from backend.data_utils import CLINICAL_PRESET_CASES

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "frontend"),
    static_folder=os.path.join(ROOT_DIR, "frontend"),
    static_url_path="/"
)

CORS(app)

# Initialize model service
try:
    model_service = get_model_service()
except Exception as e:
    print(f"Model initialization warning: {e}")
    model_service = None


def get_ready_model_service():
    """Return the loaded model service or None if unavailable."""
    global model_service

    try:
        if model_service is None:
            model_service = get_model_service()

        if (
            model_service is None
            or getattr(model_service, "pipeline", None) is None
        ):
            return None

        return model_service

    except Exception as e:
        print(f"Model loading error: {e}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    service = get_ready_model_service()
    is_ready = service is not None

    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "OncoVision AI - Breast Cancer Research Prototype",
        "model_loaded": is_ready,
        "api_version": "2.0.0"
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    service = get_ready_model_service()

    if service is None:
        return jsonify({"error": "Model is not available"}), 503

    return jsonify(service.get_metadata())


@app.route("/api/feature-stats", methods=["GET"])
def get_feature_stats():
    service = get_ready_model_service()

    if service is None:
        return jsonify({"error": "Model is not available"}), 503

    return jsonify(service.get_feature_stats())


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify({
        "presets": CLINICAL_PRESET_CASES
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    service = get_ready_model_service()

    if service is None:
        return jsonify({"error": "Model is not available"}), 503

    data = request.get_json(force=True, silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Invalid JSON body provided"}), 400

    patient_id = data.get("patient_id") or data.get("case_id")
    features = data.get("features", data)

    try:
        result = service.predict_single(
            features,
            patient_id=patient_id
        )
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    service = get_ready_model_service()

    if service is None:
        return jsonify({"error": "Model is not available"}), 503

    records = []

    # CSV file upload
    if "file" in request.files:
        uploaded_file = request.files["file"]

        if not uploaded_file.filename:
            return jsonify({"error": "Uploaded file has no filename"}), 400

        try:
            df = pd.read_csv(uploaded_file)
            df = df.drop(columns=["Unnamed: 32"], errors="ignore")
            records = df.to_dict(orient="records")

        except Exception as e:
            print(f"CSV parsing error: {e}")
            return jsonify({"error": "Failed to parse uploaded CSV"}), 400

    else:
        # JSON records
        data = request.get_json(force=True, silent=True)

        if isinstance(data, list):
            records = data

        elif isinstance(data, dict) and "records" in data:
            records = data["records"]

        elif isinstance(data, dict) and "patients" in data:
            records = data["patients"]

        else:
            return jsonify({
                "error": "JSON body must be an array or contain records or patients"
            }), 400

    if not isinstance(records, list) or not records:
        return jsonify({"error": "No valid records found"}), 400

    if len(records) > 1000:
        return jsonify({
            "error": "Batch size limit exceeded (maximum 1000 records)"
        }), 400

    try:
        batch_results = service.predict_batch(records)
        return jsonify(batch_results)

    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({"error": "Batch prediction failed"}), 500


@app.route("/api/sample-csv", methods=["GET"])
def download_sample_csv():
    """Download a sample CSV for testing the batch screening interface."""

    csv_path = os.path.join(ROOT_DIR, "data", "breast_cancer.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": "Sample dataset is unavailable"}), 404

    try:
        df = pd.read_csv(csv_path)

        benign_sample = df[df["diagnosis"] == "B"].head(3)
        malignant_sample = df[df["diagnosis"] == "M"].head(2)

        sample_df = pd.concat(
            [benign_sample, malignant_sample]
        ).sample(frac=1.0, random_state=42)

        sample_df = sample_df.drop(
            columns=["Unnamed: 32"],
            errors="ignore"
        )

        output = io.StringIO()
        sample_df.to_csv(output, index=False)

        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)

        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sample_breast_cancer_batch.csv"
        )

    except Exception as e:
        print(f"Sample CSV error: {e}")
        return jsonify({"error": "Could not generate sample CSV"}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain_endpoint():
    """Public retraining is disabled for deployment safety."""

    return jsonify({
        "error": "Model retraining is disabled on the public demo."
    }), 403


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found"}), 404

    return render_template("index.html"), 200


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "An internal server error occurred"
    }), 500


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method not allowed"
    }), 405


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in (
        "1", "true", "yes"
    )

    app.run(
        host=host,
        port=port,
        debug=debug
    )

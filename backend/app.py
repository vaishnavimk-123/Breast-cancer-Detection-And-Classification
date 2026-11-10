"""
Flask REST API and Clinical Decision Support Backend Server
"""

import os
import io
import json
import csv
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

from backend.model_service import get_model_service
from backend.train import train_and_evaluate
from backend.data_utils import CLINICAL_PRESET_CASES, FEATURE_DEFINITIONS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "frontend"),
    static_folder=os.path.join(ROOT_DIR, "frontend"),
    static_url_path="/"
)
CORS(app)

# Ensure model service is initialized
try:
    model_service = get_model_service()
except Exception as e:
    print(f"Warning: Model service lazy loading pending or model needs training: {e}")
    model_service = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    global model_service
    is_ready = model_service is not None and model_service.pipeline is not None
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "OncoVision AI - Breast Cancer Clinical Decision Support Prototype",
        "model_loaded": is_ready,
        "api_version": "2.0.0"
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    global model_service
    if not model_service:
        model_service = get_model_service()
    return jsonify(model_service.get_metadata())


@app.route("/api/feature-stats", methods=["GET"])
def get_feature_stats():
    global model_service
    if not model_service:
        model_service = get_model_service()
    return jsonify(model_service.get_feature_stats())


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify({
        "presets": CLINICAL_PRESET_CASES
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    global model_service
    if not model_service:
        model_service = get_model_service()

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body provided"}), 400

    patient_id = data.get("patient_id") or data.get("case_id")
    features = data.get("features", data)

    try:
        result = model_service.predict_single(features, patient_id=patient_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    global model_service
    if not model_service:
        model_service = get_model_service()

    records = []

    # Check if multipart file upload
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            return jsonify({"error": "Uploaded file has no filename"}), 400
        
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            # Drop unnecessary columns if present
            df = df.drop(columns=["Unnamed: 32"], errors="ignore")
            records = df.to_dict(orient="records")
        except Exception as e:
            return jsonify({"error": f"Failed to parse uploaded CSV: {str(e)}"}), 400
    else:
        # JSON list in request body
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No CSV file or JSON records array provided"}), 400
        
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "records" in data:
            records = data["records"]
        elif isinstance(data, dict) and "patients" in data:
            records = data["patients"]
        else:
            return jsonify({"error": "JSON body must be an array of patient records or contain 'records' key"}), 400

    if not records:
        return jsonify({"error": "No records found in batch submission"}), 400

    if len(records) > 1000:
        return jsonify({"error": "Batch size limit exceeded (maximum 1000 records per submission)"}), 400

    try:
        batch_results = model_service.predict_batch(records)
        return jsonify(batch_results)
    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500


@app.route("/api/sample-csv", methods=["GET"])
def download_sample_csv():
    """
    Returns a downloadable CSV containing 5 representative anonymized Wisconsin dataset rows
    formatted for instant testing of the Batch Screening module.
    """
    csv_path = os.path.join(ROOT_DIR, "data", "breast_cancer.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(ROOT_DIR, "breast_cancer.csv")

    try:
        df = pd.read_csv(csv_path)
        # Select 5 sample rows: 3 benign, 2 malignant
        benign_sample = df[df["diagnosis"] == "B"].head(3)
        malignant_sample = df[df["diagnosis"] == "M"].head(2)
        sample_df = pd.concat([benign_sample, malignant_sample]).sample(frac=1.0, random_state=42)
        sample_df = sample_df.drop(columns=["Unnamed: 32"], errors="ignore")

        output = io.StringIO()
        sample_df.to_csv(output, index=False)
        output.seek(0)

        mem = io.BytesIO()
        mem.write(output.getvalue().encode("utf-8"))
        mem.seek(0)

        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sample_breast_cancer_batch.csv"
        )
    except Exception as e:
        return jsonify({"error": f"Could not generate sample CSV: {str(e)}"}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain_endpoint():
    """
    Triggers model re-training and live service reload.
    """
    global model_service
    try:
        csv_path = os.path.join(ROOT_DIR, "data", "breast_cancer.csv")
        models_dir = os.path.join(ROOT_DIR, "models")
        meta = train_and_evaluate(csv_path=csv_path, output_dir=models_dir)
        # Reload service
        model_service = get_model_service()
        model_service._load_artifacts()
        return jsonify({
            "message": "Model retrained and reloaded successfully!",
            "metadata": meta
        })
    except Exception as e:
        return jsonify({"error": f"Re-training failed: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found"}), 404
    return render_template("index.html"), 200


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "An internal server error occurred"}), 500


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)


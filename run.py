"""
Unified Application Launcher for Breast Cancer Clinical Decision Support System (CDSS)
"""

import os
import sys

# Ensure workspace root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.train import train_and_evaluate
from backend.app import app


def main():
    print("=" * 65)
    print("  ONCOVISION AI: Breast Cancer Diagnostic Decision Support System")
    print("  Clinical AI Research Prototype")
    print("=" * 65)

    # Verify model artifact exists
    model_path = os.path.join(ROOT_DIR, "models", "breast_cancer_model.joblib")
    if not os.path.exists(model_path):
        print("\n[+] Model artifact not detected. Initiating initial training pipeline...")
        train_and_evaluate(
            csv_path=os.path.join(ROOT_DIR, "data", "breast_cancer.csv"),
            output_dir=os.path.join(ROOT_DIR, "models")
        )
    else:
        print("\n[+] Verified trained model artifact: models/breast_cancer_model.joblib")

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print(f"\n[+] Starting Clinical Web Application Server...")
    print(f"    -> Access URL: http://{host}:{port}")
    print(f"    -> Health API: http://{host}:{port}/health")
    print(f"    -> Metadata:   http://{host}:{port}/api/metadata")
    print("\nPress Ctrl+C to terminate server.\n" + "=" * 65 + "\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()



"""
Unified Application Launcher for OncoVision AI.

Breast Cancer Clinical Decision Support System
Research Prototype
"""

import os
import sys

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# --------------------------------------------------
# IMPORT FLASK APPLICATION
# --------------------------------------------------

from backend.app import app


# --------------------------------------------------
# APPLICATION LAUNCHER
# --------------------------------------------------

def main():

    print("=" * 65)

    print(
        "  ONCOVISION AI: Breast Cancer "
        "Decision Support System"
    )

    print("  Clinical AI Research Prototype")

    print("=" * 65)

    # Verify that the trained model exists.
    model_path = os.path.join(
        ROOT_DIR,
        "models",
        "breast_cancer_model.joblib"
    )

    if not os.path.isfile(model_path):

        print(
            "\n[ERROR] Trained model artifact not found."
        )

        print(
            "Expected location: models/breast_cancer_model.joblib"
        )

        print(
            "Train the model locally and commit the generated "
            "model artifacts before starting the application."
        )

        sys.exit(1)

    print(
        "\n[+] Verified trained model artifact:"
    )

    print(
        "    models/breast_cancer_model.joblib"
    )

    # Server configuration.
    host = os.environ.get(
        "HOST",
        "0.0.0.0"
    )

    port = int(
        os.environ.get("PORT", "5000")
    )

    debug = (
        os.environ.get("FLASK_DEBUG", "0").lower()
        in ("1", "true", "yes")
    )

    print(
        "\n[+] Starting OncoVision AI application..."
    )

    print(
        f"    -> Host: {host}"
    )

    print(
        f"    -> Port: {port}"
    )

    print(
        f"    -> Health API: http://localhost:{port}/health"
    )

    print(
        f"    -> Metadata API: http://localhost:{port}/api/metadata"
    )

    print(
        "\nPress Ctrl+C to terminate the server."
    )

    print("=" * 65)

    app.run(
        host=host,
        port=port,
        debug=debug
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()

"""Builds calibrated versions of the Random Forest.

Calibration corrects predicted probabilities so they line up with reality.
Builds both sigmoid and isotonic variants, and evaluate_model.py picks whichever
one actually scores better.

Usage: python -m src.calibrate_model --timestamp 20261003_120000
"""

import argparse
import os

import mlflow
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV

from src.data import load_split
from src.train_model import MODELS_DIR, build_random_forest

CALIBRATION_METHODS = ("sigmoid", "isotonic")


def build_calibrated(method):
    return CalibratedClassifierCV(
        estimator=build_random_forest(),
        method=method,
        cv=5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    print(f"[calibrate] training rows : {X_train.shape[0]}")
    print(f"[calibrate] methods       : {', '.join(CALIBRATION_METHODS)}")

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("breast_cancer_ci_ct")

    with mlflow.start_run(run_name=f"calibrate_{args.timestamp}"):
        mlflow.log_param("calibration_methods", ", ".join(CALIBRATION_METHODS))
        mlflow.log_param("calibration_cv_folds", 5)

        for method in CALIBRATION_METHODS:
            model = build_calibrated(method)
            model.fit(X_train, y_train)

            out_path = os.path.join(MODELS_DIR, f"model_{args.timestamp}_rf_{method}.joblib")
            dump(model, out_path)
            print(f"[calibrate] saved -> {out_path}")

    print("[calibrate] done")


if __name__ == "__main__":
    main()

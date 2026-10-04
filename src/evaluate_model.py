"""Scores every model on the test set, picks a champion, and checks the gate.

The champion is whichever model has the lowest Brier score. If that model's F1
is below --min-f1 the script exits with code 1, which fails the workflow step
and stops the pipeline before the model gets committed.

Usage: python -m src.evaluate_model --timestamp 20261003_120000 --min-f1 0.92
"""

import argparse
import json
import os
import shutil
import sys

import mlflow
from joblib import load
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from src.data import load_split
from src.train_model import MODELS_DIR

METRICS_DIR = "metrics"
SELECTION_METRIC = "brier_score"


def passes_gate(f1, threshold):
    return f1 >= threshold


def score_model(path, X_test, y_test):
    model = load(path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        # Brier is the mean squared error of the predicted probabilities,
        # so lower is better.
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--min-f1", type=float, default=0.92)
    args = parser.parse_args()
    timestamp = args.timestamp

    os.makedirs(METRICS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    candidates = {
        "rf_uncalibrated": f"model_{timestamp}_rf_uncalibrated.joblib",
        "rf_sigmoid": f"model_{timestamp}_rf_sigmoid.joblib",
        "rf_isotonic": f"model_{timestamp}_rf_isotonic.joblib",
        "logreg_baseline": f"model_{timestamp}_logreg_baseline.joblib",
    }

    results = {}
    for name, filename in candidates.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"[evaluate] missing model file: {path}")
            sys.exit(1)
        results[name] = score_model(path, X_test, y_test)
        s = results[name]
        print(
            f"[evaluate] {name:18} f1={s['f1']:.4f}  "
            f"auc={s['roc_auc']:.4f}  brier={s['brier_score']:.4f}"
        )

    # Did calibration actually help the forest?
    uncal = results["rf_uncalibrated"][SELECTION_METRIC]
    best_cal_name = min(("rf_sigmoid", "rf_isotonic"), key=lambda n: results[n][SELECTION_METRIC])
    best_cal = results[best_cal_name][SELECTION_METRIC]
    calibration_effect = {
        "rf_brier_uncalibrated": uncal,
        "best_calibrated_variant": best_cal_name,
        "rf_brier_calibrated": best_cal,
        "improvement": round(uncal - best_cal, 4),
        "calibration_helped": bool(best_cal < uncal),
    }
    verdict = "helped" if calibration_effect["calibration_helped"] else "did not help"
    print(f"[evaluate] calibration {verdict}: {uncal:.4f} -> {best_cal:.4f} ({best_cal_name})")

    champion = min(results, key=lambda n: results[n][SELECTION_METRIC])
    print(f"[evaluate] champion: {champion} (brier {results[champion][SELECTION_METRIC]})")

    # Copy the winner to a fixed name so later steps don't have to guess.
    shutil.copyfile(
        os.path.join(MODELS_DIR, candidates[champion]),
        os.path.join(MODELS_DIR, f"model_{timestamp}_champion.joblib"),
    )

    payload = {
        "timestamp": timestamp,
        "test_rows": int(X_test.shape[0]),
        "selection": {
            "metric": SELECTION_METRIC,
            "champion": champion,
            "champion_file": f"model_{timestamp}_champion.joblib",
        },
        "gate": {"metric": "f1", "threshold": args.min_f1, "applied_to": champion},
        "calibration_effect": calibration_effect,
        "models": results,
    }

    metrics_path = os.path.join(METRICS_DIR, f"{timestamp}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(payload, fh, indent=4)
    print(f"[evaluate] wrote -> {metrics_path}")

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("breast_cancer_ci_ct")
    with mlflow.start_run(run_name=f"evaluate_{timestamp}"):
        for model_name, scores in results.items():
            for metric_name, value in scores.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
        mlflow.log_param("champion", champion)
        mlflow.log_metric("calibration_improvement", calibration_effect["improvement"])

    champion_f1 = results[champion]["f1"]
    if not passes_gate(champion_f1, args.min_f1):
        print(
            f"\n[evaluate] quality gate failed: {champion} F1 = {champion_f1}, "
            f"needs {args.min_f1}. Model not committed."
        )
        sys.exit(1)

    print(f"\n[evaluate] quality gate passed: {champion} F1 = {champion_f1}")


if __name__ == "__main__":
    main()

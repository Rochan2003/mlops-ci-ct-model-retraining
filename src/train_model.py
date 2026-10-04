"""Trains the models and saves them with a version timestamp.

Usage: python -m src.train_model --timestamp 20261003_120000
"""

import argparse
import os

import mlflow
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE, dataset_summary, load_split

MODELS_DIR = "models"


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def build_logistic_baseline():
    # Features range from about 0.05 to 4000, so this needs scaling. The
    # Pipeline keeps the scaler bundled with the model.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    print(f"[train] timestamp     : {timestamp}")
    print(f"[train] training rows : {X_train.shape[0]}")
    print(f"[train] test rows     : {X_test.shape[0]}")

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("breast_cancer_ci_ct")

    with mlflow.start_run(run_name=f"train_{timestamp}"):
        mlflow.log_params(dataset_summary())
        mlflow.log_param("timestamp", timestamp)

        forest = build_random_forest()
        forest.fit(X_train, y_train)
        forest_path = os.path.join(MODELS_DIR, f"model_{timestamp}_rf_uncalibrated.joblib")
        dump(forest, forest_path)
        print(f"[train] saved -> {forest_path}")

        baseline = build_logistic_baseline()
        baseline.fit(X_train, y_train)
        baseline_path = os.path.join(MODELS_DIR, f"model_{timestamp}_logreg_baseline.joblib")
        dump(baseline, baseline_path)
        print(f"[train] saved -> {baseline_path}")

        # Only training scores here. The real evaluation is in evaluate_model.py
        # on the held-out test set.
        mlflow.log_metric("rf_train_accuracy", forest.score(X_train, y_train))
        mlflow.log_metric("logreg_train_accuracy", baseline.score(X_train, y_train))

    print("[train] done")


if __name__ == "__main__":
    main()

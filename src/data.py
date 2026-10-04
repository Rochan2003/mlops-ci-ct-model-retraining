"""Loads the dataset and makes the train/test split."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_split():
    """Return X_train, X_test, y_train, y_test.

    Both train_model.py and evaluate_model.py call this so they always work
    with the same split.
    """
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    # stratify keeps the malignant/benign ratio the same in both halves
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def dataset_summary():
    """Dataset facts, logged to MLflow with each run."""
    dataset = load_breast_cancer()
    return {
        "dataset_name": "breast_cancer",
        "n_samples": int(dataset.data.shape[0]),
        "n_features": int(dataset.data.shape[1]),
        "class_names": ", ".join(dataset.target_names),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

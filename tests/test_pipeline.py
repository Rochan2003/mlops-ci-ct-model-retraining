"""Tests that run in CI before any training happens."""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from src.calibrate_model import CALIBRATION_METHODS, build_calibrated
from src.data import RANDOM_STATE, load_split
from src.evaluate_model import passes_gate
from src.train_model import build_logistic_baseline, build_random_forest


def test_split_is_reproducible():
    a_train, a_test, ay_train, ay_test = load_split()
    b_train, b_test, by_train, by_test = load_split()

    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_test, b_test)
    np.testing.assert_array_equal(ay_train, by_train)
    np.testing.assert_array_equal(ay_test, by_test)


def test_train_and_test_do_not_overlap():
    X_train, X_test, _, _ = load_split()

    train_rows = {row.tobytes() for row in X_train}
    test_rows = {row.tobytes() for row in X_test}

    assert train_rows.isdisjoint(test_rows)


def test_split_sizes():
    X_train, X_test, y_train, y_test = load_split()

    assert X_train.shape[0] + X_test.shape[0] == 569
    assert X_train.shape[1] == X_test.shape[1] == 30
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_stratification_preserved():
    _, _, y_train, y_test = load_split()

    assert abs(y_train.mean() - y_test.mean()) < 0.02


@pytest.mark.parametrize("builder", [build_random_forest, build_logistic_baseline])
def test_models_produce_valid_probabilities(builder):
    X_train, X_test, y_train, _ = load_split()

    model = builder()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (X_test.shape[0], 2)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrated_models_build_and_predict(method):
    X_train, X_test, y_train, _ = load_split()

    model = build_calibrated(method)
    model.fit(X_train, y_train)

    assert model.predict(X_test).shape == (X_test.shape[0],)


def test_calibration_improves_a_miscalibrated_model():
    """Sanity check on the calibration code.

    Calibration doesn't improve Brier score on my actual model, so I wanted to
    be sure that was a property of the data and not a bug. A deliberately
    under-fitted forest is genuinely miscalibrated, and calibration should fix
    that. If this ever fails, the calibration code is wrong.
    """
    X_train, X_test, y_train, y_test = load_split()

    def shallow_forest():
        return RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=RANDOM_STATE, n_jobs=1
        )

    raw = shallow_forest().fit(X_train, y_train)
    raw_brier = brier_score_loss(y_test, raw.predict_proba(X_test)[:, 1])

    calibrated = CalibratedClassifierCV(
        estimator=shallow_forest(), method="sigmoid", cv=5
    ).fit(X_train, y_train)
    calibrated_brier = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1])

    assert calibrated_brier < raw_brier


def test_gate_accepts_good_model():
    assert passes_gate(f1=0.96, threshold=0.92)


def test_gate_rejects_bad_model():
    assert not passes_gate(f1=0.80, threshold=0.92)


def test_gate_is_inclusive_at_threshold():
    assert passes_gate(f1=0.92, threshold=0.92)

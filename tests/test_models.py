"""Tests for the churn classification models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from churn.data import clean, load_raw
from churn.features import build_features
from churn.models import (
    evaluate,
    feature_importance,
    split_data,
    train_baseline,
    train_logistic,
    train_random_forest,
)


@pytest.fixture(scope="module")
def features_and_target():
    df = clean(load_raw())
    X = build_features(df)
    y = df["Churn"]
    return X, y


@pytest.fixture(scope="module")
def data(features_and_target):
    X, y = features_and_target
    return split_data(X, y)


def test_split_is_stratified(data):
    X_train, X_test, y_train, y_test = data
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.01


def test_split_sizes(data):
    X_train, X_test, _, _ = data
    assert len(X_test) == pytest.approx(len(X_train) / 4, rel=0.05)


def test_baseline_predicts_majority_class(data):
    X_train, X_test, y_train, y_test = data
    model = train_baseline(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert metrics["recall"] == 0.0
    assert metrics["accuracy"] == pytest.approx(1 - y_test.mean(), abs=0.01)


def test_logistic_beats_baseline_on_recall(data):
    X_train, X_test, y_train, y_test = data
    baseline = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    logistic = evaluate(train_logistic(X_train, y_train), X_test, y_test)
    assert logistic["recall"] > baseline["recall"]


def test_logistic_roc_auc_is_reasonable(data):
    X_train, X_test, y_train, y_test = data
    model = train_logistic(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert 0.7 < metrics["roc_auc"] < 0.95


def test_logistic_coefficients_are_finite(data):
    X_train, X_test, y_train, y_test = data
    model = train_logistic(X_train, y_train)
    coefs = model.named_steps["logisticregression"].coef_
    assert np.all(np.isfinite(coefs))


def test_random_forest_beats_baseline_on_f1(data):
    X_train, X_test, y_train, y_test = data
    baseline = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    forest = evaluate(train_random_forest(X_train, y_train), X_test, y_test)
    assert forest["f1"] > baseline["f1"]


def test_random_forest_is_reproducible(data):
    X_train, X_test, y_train, y_test = data
    first = evaluate(train_random_forest(X_train, y_train), X_test, y_test)
    second = evaluate(train_random_forest(X_train, y_train), X_test, y_test)
    assert first == second


def test_feature_importance_sums_to_one(data):
    X_train, X_test, y_train, y_test = data
    model = train_random_forest(X_train, y_train)
    importances = feature_importance(model, X_train.columns)
    assert importances.sum() == pytest.approx(1.0, abs=1e-6)
    assert importances.iloc[0] >= importances.iloc[-1]


def test_feature_importance_is_sorted_descending(data):
    X_train, X_test, y_train, y_test = data
    model = train_random_forest(X_train, y_train)
    importances = feature_importance(model, X_train.columns)
    assert list(importances) == sorted(importances, reverse=True)


def test_tenure_or_contract_is_a_top_driver(data):
    """Domain sanity check: short tenure and loose contracts should matter."""
    X_train, X_test, y_train, y_test = data
    model = train_random_forest(X_train, y_train)
    importances = feature_importance(model, X_train.columns)
    top_5 = set(importances.head(5).index)
    assert {"tenure", "contract_level"} & top_5


def test_evaluate_without_predict_proba_omits_roc_auc(data):
    # LinearSVC has no predict_proba, so evaluate() must take the branch that
    # skips roc_auc instead of raising on the missing attribute.
    X_train, X_test, y_train, y_test = data
    model = LinearSVC(max_iter=2000, random_state=0)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert "roc_auc" not in metrics
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


@pytest.mark.parametrize("test_size", [0.1, 0.3, 0.5])
def test_split_data_respects_requested_test_size(features_and_target, test_size):
    X, y = features_and_target
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    assert len(X_test) == pytest.approx(len(X) * test_size, abs=1)
    assert len(X_train) + len(X_test) == len(X)


def test_split_data_is_reproducible(features_and_target):
    X, y = features_and_target
    first = split_data(X, y)
    second = split_data(X, y)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[2], second[2])

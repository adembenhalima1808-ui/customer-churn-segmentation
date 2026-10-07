"""Churn classification: train/test split, baseline, and two classifiers."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def split_data(features: pd.DataFrame, target: pd.Series, test_size: float = 0.2):
    """Stratified train/test split so both sets keep the same churn rate."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def train_baseline(X_train, y_train) -> DummyClassifier:
    """Predicts the majority class; the bar every real model must clear."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train):
    """Scaled logistic regression, balanced for the ~27% churn rate."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    )
    # See churn.segmentation._fit_quietly: this numpy/BLAS build raises spurious
    # floating-point flags on finite, bounded input during the matmul-heavy
    # solver step. Metrics are checked for finiteness in the tests.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Return accuracy, precision, recall, f1, and ROC-AUC for the positive class."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    return {k: round(float(v), 4) for k, v in metrics.items()}


def feature_importance(model: RandomForestClassifier, feature_names) -> pd.Series:
    """Random forest feature importances, sorted descending."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

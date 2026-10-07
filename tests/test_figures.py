"""Tests for report figures: each function must write a readable, non-empty
PNG and leave no figures open behind it."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from churn.segmentation import KSweep
from churn.figures import (
    confusion_matrix_figure,
    elbow_silhouette_figure,
    feature_importance_figure,
    segment_profile_figure,
)


def _is_nonempty_png(path):
    return path.exists() and path.stat().st_size > 0 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def sweep():
    return KSweep(k_values=[2, 3, 4, 5], inertia=[400.0, 300.0, 250.0, 220.0], silhouette=[0.2, 0.3, 0.36, 0.28])


@pytest.fixture
def profile():
    return pd.DataFrame(
        {"tenure": [45.8, 25.7, 8.2, 61.7], "MonthlyCharges": [25.6, 81.7, 46.1, 90.9]},
        index=[0, 1, 2, 3],
    )


@pytest.fixture
def importances():
    values = [0.24, 0.18, 0.15, 0.13, 0.12, 0.09, 0.06, 0.03]
    names = [f"feature_{i}" for i in range(len(values))]
    return pd.Series(values, index=names).sort_values(ascending=False)


def test_elbow_silhouette_figure_writes_readable_png(sweep, tmp_path):
    out = elbow_silhouette_figure(sweep, chosen_k=4, out_path=tmp_path / "elbow.png")
    assert _is_nonempty_png(out)


def test_elbow_silhouette_figure_closes_its_figure(sweep, tmp_path):
    elbow_silhouette_figure(sweep, chosen_k=4, out_path=tmp_path / "elbow.png")
    assert plt.get_fignums() == []


def test_segment_profile_figure_writes_readable_png(profile, tmp_path):
    labels = {0: "Legacy low-spend", 1: "Growing high-value", 2: "New / at-risk", 3: "Loyal premium"}
    out = segment_profile_figure(profile, labels, tmp_path / "segments.png")
    assert _is_nonempty_png(out)


def test_segment_profile_figure_handles_unlabeled_segments(profile, tmp_path):
    # labels dict missing an entry should fall back to the raw segment id, not raise
    out = segment_profile_figure(profile, labels={0: "Legacy low-spend"}, out_path=tmp_path / "segments.png")
    assert _is_nonempty_png(out)
    assert plt.get_fignums() == []


def test_feature_importance_figure_writes_readable_png(importances, tmp_path):
    out = feature_importance_figure(importances, tmp_path / "importance.png", top_n=5)
    assert _is_nonempty_png(out)


def test_feature_importance_figure_top_n_larger_than_series_does_not_raise(importances, tmp_path):
    out = feature_importance_figure(importances, tmp_path / "importance.png", top_n=50)
    assert _is_nonempty_png(out)
    assert plt.get_fignums() == []


def test_confusion_matrix_figure_writes_readable_png(tmp_path):
    y_test = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    out = confusion_matrix_figure(y_test, y_pred, tmp_path / "confusion.png")
    assert _is_nonempty_png(out)
    assert plt.get_fignums() == []


def test_confusion_matrix_figure_handles_single_class_predictions(tmp_path):
    # A degenerate case (model predicts only one class) must still render, not raise.
    y_test = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    out = confusion_matrix_figure(y_test, y_pred, tmp_path / "confusion.png")
    assert _is_nonempty_png(out)

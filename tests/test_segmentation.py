"""Tests for k-means segmentation."""

import numpy as np
import pandas as pd
import pytest

from churn.data import clean, load_raw
from churn.features import build_features, segment_matrix
from churn.segmentation import fit_segments, profile_segments, scale, sweep_k


@pytest.fixture(scope="module")
def matrix():
    return segment_matrix(build_features(clean(load_raw())))


@pytest.fixture(scope="module")
def scaled(matrix):
    values, _ = scale(matrix)
    return values


def test_scaling_gives_zero_mean_unit_variance(scaled):
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-6)


def test_sweep_returns_one_score_per_k(scaled):
    sweep = sweep_k(scaled, k_values=range(2, 5))
    assert sweep.k_values == [2, 3, 4]
    assert len(sweep.inertia) == 3
    assert len(sweep.silhouette) == 3


def test_inertia_decreases_as_k_grows(scaled):
    sweep = sweep_k(scaled, k_values=range(2, 6))
    assert all(a > b for a, b in zip(sweep.inertia, sweep.inertia[1:]))


def test_silhouette_is_within_bounds(scaled):
    sweep = sweep_k(scaled, k_values=range(2, 5))
    assert all(-1 <= s <= 1 for s in sweep.silhouette)


def test_best_k_is_argmax_silhouette(scaled):
    sweep = sweep_k(scaled, k_values=range(2, 6))
    assert sweep.best_k() == sweep.k_values[int(np.argmax(sweep.silhouette))]


def test_sweep_values_are_finite(scaled):
    sweep = sweep_k(scaled, k_values=range(2, 5))
    assert np.all(np.isfinite(sweep.inertia))
    assert np.all(np.isfinite(sweep.silhouette))


def test_fit_centers_are_finite(scaled):
    model, _ = fit_segments(scaled, k=4)
    assert np.all(np.isfinite(model.cluster_centers_))


def test_fit_is_reproducible(scaled):
    _, first = fit_segments(scaled, k=4)
    _, second = fit_segments(scaled, k=4)
    assert np.array_equal(first, second)


def test_labels_cover_requested_clusters(scaled):
    _, labels = fit_segments(scaled, k=4)
    assert set(np.unique(labels)) == {0, 1, 2, 3}


def test_profile_sizes_sum_to_population(matrix, scaled):
    _, labels = fit_segments(scaled, k=4)
    profile = profile_segments(matrix, labels)
    assert profile["size"].sum() == len(matrix)
    assert round(profile["share"].sum(), 2) == 1.0


def test_profile_contains_feature_means(matrix, scaled):
    _, labels = fit_segments(scaled, k=3)
    profile = profile_segments(matrix, labels)
    assert "tenure" in profile.columns
    assert "MonthlyCharges" in profile.columns
    assert isinstance(profile.index[0], (int, np.integer))


def test_profile_does_not_modify_input(matrix, scaled):
    before = matrix.copy()
    _, labels = fit_segments(scaled, k=3)
    profile_segments(matrix, labels)
    pd.testing.assert_frame_equal(matrix, before)

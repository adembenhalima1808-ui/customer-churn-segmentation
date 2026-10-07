"""End-to-end test for the report pipeline: segmentation + churn models,
figures, and the metrics file it writes to disk."""

import json

import pytest

from churn.report import FIGURES_DIR, REPORTS_DIR, SEGMENTATION_K, run


@pytest.fixture(scope="module")
def results():
    return run()


def test_segmentation_section_has_one_entry_per_segment(results):
    segmentation = results["segmentation"]
    assert segmentation["k"] == SEGMENTATION_K
    assert len(segmentation["segments"]) == SEGMENTATION_K


def test_segmentation_silhouette_is_within_bounds(results):
    assert -1 <= results["segmentation"]["silhouette"] <= 1


def test_segment_shares_sum_to_one(results):
    shares = [segment["share"] for segment in results["segmentation"]["segments"].values()]
    assert round(sum(shares), 2) == 1.0


def test_churn_models_section_has_baseline_and_two_classifiers(results):
    models = results["churn_models"]
    assert set(models) == {"baseline", "logistic_regression", "random_forest"}
    for metrics in models.values():
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0


def test_random_forest_beats_baseline_on_f1(results):
    models = results["churn_models"]
    assert models["random_forest"]["f1"] > models["baseline"]["f1"]


def test_top_churn_drivers_are_sorted_descending(results):
    values = list(results["top_churn_drivers"].values())
    assert values == sorted(values, reverse=True)
    assert len(values) == 5


def test_metrics_file_matches_returned_results(results):
    on_disk = json.loads((REPORTS_DIR / "metrics.json").read_text())
    assert on_disk == results


@pytest.mark.parametrize(
    "filename",
    ["elbow_silhouette.png", "segment_profiles.png", "feature_importance.png", "confusion_matrix.png"],
)
def test_each_figure_is_written_and_nonempty(results, filename):
    path = FIGURES_DIR / filename
    assert path.exists()
    assert path.stat().st_size > 0

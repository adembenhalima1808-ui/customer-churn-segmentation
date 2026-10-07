"""End-to-end report: segmentation + churn model, figures, and a metrics file."""

import json
from pathlib import Path

from churn.data import clean, load_raw
from churn.features import build_features, segment_matrix
from churn.figures import (
    confusion_matrix_figure,
    elbow_silhouette_figure,
    feature_importance_figure,
    segment_profile_figure,
)
from churn.models import evaluate, feature_importance, split_data, train_baseline, train_logistic, train_random_forest
from churn.segmentation import fit_segments, profile_segments, scale, sweep_k

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Business-readable names for the k=4 segmentation, assigned after inspecting
# the profile (short tenure here means under a year on average).
SEGMENT_LABELS = {
    0: "Legacy low-spend",
    1: "Growing high-value",
    2: "New / at-risk",
    3: "Loyal premium",
}

SEGMENTATION_K = 4


def run() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw())
    features = build_features(df)

    # --- Segmentation ---
    seg_matrix = segment_matrix(features)
    scaled, _ = scale(seg_matrix)
    sweep = sweep_k(scaled)
    _, labels = fit_segments(scaled, SEGMENTATION_K)
    profile = profile_segments(seg_matrix, labels)

    elbow_silhouette_figure(sweep, SEGMENTATION_K, FIGURES_DIR / "elbow_silhouette.png")
    segment_profile_figure(profile, SEGMENT_LABELS, FIGURES_DIR / "segment_profiles.png")

    # --- Churn prediction ---
    X = features
    y = df["Churn"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_metrics = evaluate(train_baseline(X_train, y_train), X_test, y_test)
    logistic_metrics = evaluate(train_logistic(X_train, y_train), X_test, y_test)
    forest = train_random_forest(X_train, y_train)
    forest_metrics = evaluate(forest, X_test, y_test)

    importances = feature_importance(forest, X_train.columns)
    feature_importance_figure(importances, FIGURES_DIR / "feature_importance.png")
    confusion_matrix_figure(y_test, forest.predict(X_test), FIGURES_DIR / "confusion_matrix.png")

    results = {
        "segmentation": {
            "k": SEGMENTATION_K,
            "silhouette": sweep.silhouette[sweep.k_values.index(SEGMENTATION_K)],
            "segments": {
                SEGMENT_LABELS[int(seg)]: {
                    "size": int(row["size"]),
                    "share": float(row["share"]),
                    "avg_tenure_months": float(row["tenure"]),
                    "avg_monthly_charges": float(row["MonthlyCharges"]),
                }
                for seg, row in profile.iterrows()
            },
        },
        "churn_models": {
            "baseline": baseline_metrics,
            "logistic_regression": logistic_metrics,
            "random_forest": forest_metrics,
        },
        "top_churn_drivers": importances.head(5).round(4).to_dict(),
    }

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()

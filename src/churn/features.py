"""Customer-level feature engineering shared by clustering and classification."""

import pandas as pd

CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}

ADD_ON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Numeric columns used for k-means. Kept small on purpose: segments should
# describe value and loyalty, not every binary flag in the file.
SEGMENT_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "contract_level",
    "num_add_ons",
]

BINARY_FLAGS = {
    "gender_male": ("gender", "Male"),
    "partner": ("Partner", "Yes"),
    "dependents": ("Dependents", "Yes"),
    "senior": ("SeniorCitizen", 1),
    "paperless_billing": ("PaperlessBilling", "Yes"),
    "phone_service": ("PhoneService", "Yes"),
    "has_internet": ("InternetService", lambda v: v != "No"),
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature frame indexed like the input.

    Raises ValueError if an expected categorical value is unknown, so that
    silent mis-encoding cannot reach the models.
    """
    if not df["Contract"].isin(CONTRACT_ORDER).all():
        unknown = sorted(set(df["Contract"]) - set(CONTRACT_ORDER))
        raise ValueError(f"Unknown contract types: {unknown}")

    out = pd.DataFrame(index=df.index)
    out["tenure"] = df["tenure"].astype(float)
    out["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    out["TotalCharges"] = df["TotalCharges"].astype(float)
    out["contract_level"] = df["Contract"].map(CONTRACT_ORDER).astype(float)
    out["num_add_ons"] = (df[ADD_ON_SERVICES] == "Yes").sum(axis=1).astype(float)

    for name, (column, rule) in BINARY_FLAGS.items():
        if callable(rule):
            out[name] = df[column].map(rule).astype(float)
        else:
            out[name] = (df[column] == rule).astype(float)

    out["avg_monthly_to_total"] = _safe_ratio(out["TotalCharges"], out["tenure"] + 1)
    return out


def segment_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Subset of features used for clustering."""
    return features[SEGMENT_FEATURES].copy()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.where(denominator != 0, 1.0)).astype(float)

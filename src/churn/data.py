"""Loading and cleaning of the raw Telco churn file."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_churn.csv"

REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "Churn",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the CSV exactly as delivered, failing early if columns are missing."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: numeric totals, binary target, no duplicate IDs.

    TotalCharges is stored as text and holds a blank for customers with zero
    tenure. Those rows are kept and set to 0.0, since they have not been billed yet.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].str.strip(), errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})
    if out["Churn"].isna().any():
        raise ValueError("Churn column contains values other than Yes/No")

    out = out.drop_duplicates(subset="customerID", keep="first")
    return out.reset_index(drop=True)

"""Tests for loading and cleaning the raw dataset."""

import pandas as pd
import pytest

from churn.data import RAW_PATH, clean, load_raw


def test_raw_file_loads_with_expected_row_count():
    df = load_raw()
    assert len(df) == 7043


def test_load_raw_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"customerID": ["1"], "tenure": [3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw(bad)


def test_clean_converts_total_charges_to_numeric():
    df = clean(load_raw())
    assert pd.api.types.is_float_dtype(df["TotalCharges"])


def test_blank_total_charges_become_zero():
    df = clean(load_raw())
    zero_tenure = df[df["tenure"] == 0]
    assert len(zero_tenure) == 11
    assert (zero_tenure["TotalCharges"] == 0.0).all()


def test_churn_is_binary():
    df = clean(load_raw())
    assert set(df["Churn"].unique()) == {0, 1}


def test_churn_rate_matches_known_value():
    df = clean(load_raw())
    assert round(df["Churn"].mean(), 4) == 0.2654


def test_clean_drops_duplicate_customer_ids():
    raw = load_raw()
    doubled = pd.concat([raw, raw.head(5)], ignore_index=True)
    assert len(clean(doubled)) == len(raw)


def test_clean_rejects_unknown_churn_labels():
    raw = load_raw().head(3).copy()
    raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Yes/No"):
        clean(raw)


def test_raw_path_points_at_dataset():
    assert RAW_PATH.exists()


def test_clean_handles_whitespace_only_total_charges():
    raw = load_raw().head(3).copy()
    raw.loc[0, "TotalCharges"] = "   "
    df = clean(raw)
    assert df.loc[0, "TotalCharges"] == 0.0


def test_clean_does_not_mutate_input():
    raw = load_raw()
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)

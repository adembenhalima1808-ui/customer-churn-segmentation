"""Tests for customer feature engineering."""

import pandas as pd
import pytest

from churn.data import clean, load_raw
from churn.features import SEGMENT_FEATURES, _safe_ratio, build_features, segment_matrix


@pytest.fixture(scope="module")
def cleaned():
    return clean(load_raw())


@pytest.fixture(scope="module")
def features(cleaned):
    return build_features(cleaned)


def test_feature_frame_keeps_row_count(cleaned, features):
    assert len(features) == len(cleaned)
    assert features.index.equals(cleaned.index)


def test_no_missing_values(features):
    assert not features.isna().any().any()


def test_contract_level_is_ordinal(features, cleaned):
    month = features.loc[cleaned["Contract"] == "Month-to-month", "contract_level"]
    two_year = features.loc[cleaned["Contract"] == "Two year", "contract_level"]
    assert (month == 0).all()
    assert (two_year == 2).all()


def test_num_add_ons_range(features):
    assert features["num_add_ons"].between(0, 6).all()


def test_has_internet_matches_service_column(features, cleaned):
    expected = (cleaned["InternetService"] != "No").astype(float)
    assert (features["has_internet"] == expected).all()


def test_binary_flags_are_zero_or_one(features):
    flags = ["gender_male", "partner", "dependents", "senior", "paperless_billing", "phone_service", "has_internet"]
    for col in flags:
        assert set(features[col].unique()) <= {0.0, 1.0}


def test_ratio_is_finite_for_zero_tenure(features):
    assert features["avg_monthly_to_total"].notna().all()


def test_unknown_contract_raises(cleaned):
    broken = cleaned.head(5).copy()
    broken.loc[0, "Contract"] = "Weekly"
    with pytest.raises(ValueError, match="Unknown contract"):
        build_features(broken)


def test_segment_matrix_selects_expected_columns(features):
    seg = segment_matrix(features)
    assert list(seg.columns) == SEGMENT_FEATURES


def test_segment_matrix_is_a_copy(features):
    seg = segment_matrix(features)
    seg.iloc[0, 0] = -999
    assert features.iloc[0, 0] != -999


def test_safe_ratio_guards_against_zero_denominator():
    # build_features never hits this directly (its denominator is tenure + 1,
    # always >= 1), so the zero-guard needs its own direct test.
    numerator = pd.Series([10.0, 20.0])
    denominator = pd.Series([0.0, 5.0])
    result = _safe_ratio(numerator, denominator)
    assert result.tolist() == [10.0, 4.0]


def test_safe_ratio_normal_division():
    numerator = pd.Series([9.0])
    denominator = pd.Series([3.0])
    assert _safe_ratio(numerator, denominator).tolist() == [3.0]

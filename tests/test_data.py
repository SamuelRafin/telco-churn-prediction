"""Tests for data loading and cleaning."""

import pandas as pd
import pytest

from churn.data import (
    clean_data,
    get_feature_types,
    split_features_target,
    summarize,
)


def test_total_charges_becomes_numeric(raw_sample, config):
    """The text TotalCharges column is coerced to a float dtype."""
    cleaned = clean_data(raw_sample, config)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_blank_total_charges_filled_with_zero(raw_sample, config):
    """The tenure-0 customer's blank TotalCharges becomes 0.0, not NaN."""
    cleaned = clean_data(raw_sample, config)
    zero_tenure = cleaned.loc[cleaned["tenure"] == 0]

    assert len(zero_tenure) == 1
    assert zero_tenure["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_target_mapped_to_binary(raw_sample, config):
    """Churn Yes/No becomes 1/0 integers."""
    cleaned = clean_data(raw_sample, config)
    target = config.data.target_column

    assert set(cleaned[target].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(cleaned[target])
    # One 'Yes' in the fixture.
    assert cleaned[target].sum() == 1


def test_id_column_dropped(raw_sample, config):
    """The identifier column is removed."""
    cleaned = clean_data(raw_sample, config)
    assert config.data.id_column not in cleaned.columns


def test_row_count_preserved(raw_sample, config):
    """Cleaning drops no rows."""
    cleaned = clean_data(raw_sample, config)
    assert len(cleaned) == len(raw_sample)


def test_input_not_mutated(raw_sample, config):
    """Cleaning returns a new frame and leaves the input untouched."""
    before = raw_sample.copy()
    clean_data(raw_sample, config)
    pd.testing.assert_frame_equal(raw_sample, before)


def test_no_missing_values_after_cleaning(raw_sample, config):
    """The cleaned frame has no NaNs."""
    cleaned = clean_data(raw_sample, config)
    assert int(cleaned.isna().sum().sum()) == 0


def test_split_features_target(raw_sample, config):
    """Splitting separates the target from the features."""
    cleaned = clean_data(raw_sample, config)
    features, labels = split_features_target(cleaned, config)

    assert config.data.target_column not in features.columns
    assert len(features) == len(labels)
    assert features.shape[1] == cleaned.shape[1] - 1


def test_split_features_target_missing_target(raw_sample, config):
    """A frame without the target raises KeyError."""
    cleaned = clean_data(raw_sample, config)
    without_target = cleaned.drop(columns=[config.data.target_column])

    with pytest.raises(KeyError):
        split_features_target(without_target, config)


def test_feature_types_partition_all_columns(raw_sample, config):
    """Every feature is classified as either numeric or categorical."""
    cleaned = clean_data(raw_sample, config)
    features, _ = split_features_target(cleaned, config)
    numeric, categorical = get_feature_types(features)

    assert set(numeric) | set(categorical) == set(features.columns)
    assert not set(numeric) & set(categorical)
    # Known numeric columns from the Telco schema.
    assert "tenure" in numeric
    assert "MonthlyCharges" in numeric
    assert "TotalCharges" in numeric
    assert "SeniorCitizen" in numeric
    assert "Contract" in categorical


def test_summarize_mentions_churn_rate(raw_sample, config):
    """The summary string reports the churn rate."""
    cleaned = clean_data(raw_sample, config)
    text = summarize(cleaned, config)

    assert "Churn rate" in text
    assert "Rows" in text

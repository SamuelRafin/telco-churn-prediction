"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.data import clean_data, get_feature_types, split_features_target
from churn.features import build_pipeline, build_preprocessor, get_feature_names


def _features(raw_sample, config) -> pd.DataFrame:
    cleaned = clean_data(raw_sample, config)
    features, _ = split_features_target(cleaned, config)
    return features


def test_preprocessor_expands_categoricals(raw_sample, config):
    """One-hot encoding produces more columns than the raw input."""
    features = _features(raw_sample, config)
    preprocessor = build_preprocessor(features)

    transformed = preprocessor.fit_transform(features)
    numeric, categorical = get_feature_types(features)

    # Numeric columns pass through 1:1; each categorical expands to its levels.
    expected = len(numeric) + sum(features[c].nunique() for c in categorical)
    assert transformed.shape == (len(features), expected)
    assert transformed.shape[1] > features.shape[1]


def test_transformed_output_has_no_nans(raw_sample, config):
    """The transformed matrix contains no missing values."""
    features = _features(raw_sample, config)
    transformed = build_preprocessor(features).fit_transform(features)
    assert not np.isnan(transformed).any()


def test_numeric_columns_are_standardised(raw_sample, config):
    """Numeric columns are scaled to roughly zero mean."""
    features = _features(raw_sample, config)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)

    names = get_feature_names(preprocessor)
    numeric, _ = get_feature_types(features)

    for column in numeric:
        index = names.index(column)
        assert abs(transformed[:, index].mean()) < 1e-9


def test_unseen_category_does_not_raise(raw_sample, config):
    """A category unseen during fit encodes as zeros rather than erroring."""
    features = _features(raw_sample, config)
    preprocessor = build_preprocessor(features).fit(features)

    unseen = features.iloc[[0]].copy()
    unseen.loc[:, "Contract"] = "Five year"  # never seen in training

    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_build_pipeline_structure(raw_sample, config):
    """The pipeline exposes a preprocessor step and a model step."""
    features = _features(raw_sample, config)
    pipeline = build_pipeline(features, LogisticRegression())

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_pipeline_fits_and_predicts(raw_sample, config):
    """An end-to-end pipeline trains and produces valid probabilities."""
    cleaned = clean_data(raw_sample, config)
    features, labels = split_features_target(cleaned, config)

    pipeline = build_pipeline(features, LogisticRegression())
    pipeline.fit(features, labels)

    probabilities = pipeline.predict_proba(features)[:, 1]
    assert probabilities.shape == (len(features),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_feature_names_match_transformed_width(raw_sample, config):
    """get_feature_names returns one name per transformed column."""
    cleaned = clean_data(raw_sample, config)
    features, labels = split_features_target(cleaned, config)

    pipeline = build_pipeline(features, LogisticRegression())
    pipeline.fit(features, labels)

    transformed = pipeline.named_steps["preprocessor"].transform(features)
    names = get_feature_names(pipeline)

    assert len(names) == transformed.shape[1]
    assert len(names) == len(set(names))  # names are unique

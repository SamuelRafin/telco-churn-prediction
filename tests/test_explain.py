"""Tests for SHAP explainability.

Uses a small synthetic dataset and a shallow model so SHAP stays fast.
"""

import numpy as np
import pandas as pd
import pytest

from churn.explain import (
    compute_shap_values,
    explain_customer,
    global_importance,
    transform_features,
)
from churn.features import build_pipeline, get_feature_names


@pytest.fixture
def fitted_setup(config):
    """A fitted tree pipeline plus the data it was trained on."""
    from xgboost import XGBClassifier

    rng = np.random.default_rng(3)
    n = 200
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n)

    features = pd.DataFrame(
        {
            "SeniorCitizen": rng.integers(0, 2, n),
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        }
    )
    # Churn driven mainly by low tenure.
    labels = pd.Series((tenure < 24).astype(int))

    pipeline = build_pipeline(
        features,
        XGBClassifier(
            n_estimators=25, max_depth=3, eval_metric="logloss", random_state=0
        ),
    )
    pipeline.fit(features, labels)
    return pipeline, features


def test_transform_features_names_match_width(fitted_setup):
    """The transformed frame is labelled with the pipeline's feature names."""
    pipeline, features = fitted_setup
    transformed = transform_features(pipeline, features)

    assert list(transformed.columns) == get_feature_names(pipeline)
    assert len(transformed) == len(features)


def test_shap_values_shape_matches_transformed(fitted_setup):
    """SHAP values align with the transformed feature space."""
    pipeline, features = fitted_setup
    sample = features.head(40)

    shap_values, transformed = compute_shap_values(pipeline, sample, features)

    assert shap_values.ndim == 2
    assert shap_values.shape == transformed.shape


def test_shap_values_are_finite(fitted_setup):
    """No NaN or infinite SHAP values."""
    pipeline, features = fitted_setup
    shap_values, _ = compute_shap_values(pipeline, features.head(30), features)
    assert np.isfinite(shap_values).all()


def test_global_importance_is_ranked(fitted_setup):
    """Importance is sorted descending by mean absolute SHAP."""
    pipeline, features = fitted_setup
    shap_values, transformed = compute_shap_values(pipeline, features, features)

    ranking = global_importance(shap_values, transformed, top_n=10)

    assert len(ranking) <= 10
    assert list(ranking["mean_abs_shap"]) == sorted(
        ranking["mean_abs_shap"], reverse=True
    )
    assert (ranking["mean_abs_shap"] >= 0).all()


def test_global_importance_has_direction_flags(fitted_setup):
    """Each ranked feature carries a churn-direction flag."""
    pipeline, features = fitted_setup
    shap_values, transformed = compute_shap_values(pipeline, features, features)
    ranking = global_importance(shap_values, transformed)

    assert "pushes_toward_churn" in ranking.columns
    assert ranking["pushes_toward_churn"].dtype == bool
    assert ranking["value_shap_corr"].between(-1, 1).all()


def test_tenure_direction_is_retention(fitted_setup):
    """Since churn was generated from low tenure, tenure must push to retention."""
    pipeline, features = fitted_setup
    shap_values, transformed = compute_shap_values(pipeline, features, features)
    ranking = global_importance(shap_values, transformed, top_n=50)

    tenure_row = ranking.loc[ranking["feature"] == "tenure"].iloc[0]
    assert not tenure_row["pushes_toward_churn"]


def test_explain_customer_returns_probability_and_factors(fitted_setup):
    """A single-customer explanation reports a probability and ranked factors."""
    pipeline, features = fitted_setup
    customer = features.iloc[[0]]

    explanation = explain_customer(pipeline, customer, features, top_n=5)

    assert 0.0 <= explanation["churn_probability"] <= 1.0
    assert len(explanation["top_factors"]) == 5

    for factor in explanation["top_factors"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert factor["direction"] in {
            "increases churn risk",
            "decreases churn risk",
        }


def test_explain_customer_factors_sorted_by_magnitude(fitted_setup):
    """Contributing factors come back strongest-first."""
    pipeline, features = fitted_setup
    explanation = explain_customer(pipeline, features.iloc[[1]], features, top_n=6)

    magnitudes = [abs(f["shap_value"]) for f in explanation["top_factors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_customer_rejects_multiple_rows(fitted_setup):
    """Passing more than one row is an error."""
    pipeline, features = fitted_setup
    with pytest.raises(ValueError):
        explain_customer(pipeline, features.head(3), features)

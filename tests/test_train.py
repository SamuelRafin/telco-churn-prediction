"""Tests for the training harness.

These use a small synthetic dataset so the suite stays fast. They verify
wiring and contracts, not model quality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.train import (
    BASELINE_MODEL,
    SEARCH_SPACES,
    DataSplit,
    build_models,
    compare_models,
    select_best_model,
    split_data,
    train_final_model,
)


@pytest.fixture
def synthetic_frame(config) -> pd.DataFrame:
    """A larger synthetic frame with learnable signal, for training tests."""
    rng = np.random.default_rng(0)
    n = 300

    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)

    # Churn probability falls with tenure and rises with monthly charges.
    logit = -0.05 * tenure + 0.02 * monthly - 1.0
    probability = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < probability).astype(int)

    return pd.DataFrame(
        {
            "SeniorCitizen": rng.integers(0, 2, n),
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "gender": rng.choice(["Male", "Female"], n),
            "Contract": contract,
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            config.data.target_column: churn,
        }
    )


def test_split_is_stratified(synthetic_frame, config):
    """Train and test splits preserve the churn rate."""
    split = split_data(synthetic_frame, config)

    assert isinstance(split, DataSplit)
    overall = synthetic_frame[config.data.target_column].mean()
    assert split.y_train.mean() == pytest.approx(overall, abs=0.05)
    assert split.y_test.mean() == pytest.approx(overall, abs=0.05)


def test_split_sizes_match_config(synthetic_frame, config):
    """The test fraction follows config.split.test_size."""
    split = split_data(synthetic_frame, config)
    total = len(synthetic_frame)
    expected_test = int(round(total * config.split.test_size))

    assert len(split.x_test) == pytest.approx(expected_test, abs=1)
    assert len(split.x_train) + len(split.x_test) == total


def test_split_has_no_overlap(synthetic_frame, config):
    """No row appears in both train and test."""
    split = split_data(synthetic_frame, config)
    assert not set(split.x_train.index) & set(split.x_test.index)


def test_split_is_reproducible(synthetic_frame, config):
    """The same seed yields the same split."""
    first = split_data(synthetic_frame, config)
    second = split_data(synthetic_frame, config)
    assert list(first.x_train.index) == list(second.x_train.index)


def test_build_models_includes_baseline(synthetic_frame, config):
    """The candidate set contains the baseline and the tree models."""
    split = split_data(synthetic_frame, config)
    models = build_models(split.y_train, config)

    assert BASELINE_MODEL in models
    assert "random_forest" in models
    assert "xgboost" in models


def test_search_spaces_cover_all_models(synthetic_frame, config):
    """Every candidate model has a tuning search space defined."""
    split = split_data(synthetic_frame, config)
    models = build_models(split.y_train, config)
    assert set(models) == set(SEARCH_SPACES)


def test_search_space_params_are_pipeline_prefixed():
    """Search params target the pipeline's model step."""
    for space in SEARCH_SPACES.values():
        for key in space:
            assert key.startswith("model__")


def test_compare_models_returns_ranked_results(synthetic_frame, config):
    """Comparison produces one row per model, sorted by the selection metric."""
    split = split_data(synthetic_frame, config)
    results, best_params = compare_models(split.x_train, split.y_train, config)

    assert len(results) == 3
    assert best_params == {}  # no tuning requested

    column = f"{config.model.selection_metric}_mean"
    assert column in results.columns
    # Sorted descending.
    assert list(results[column]) == sorted(results[column], reverse=True)


def test_compare_models_metrics_in_valid_range(synthetic_frame, config):
    """All reported mean metrics sit in [0, 1]."""
    split = split_data(synthetic_frame, config)
    results, _ = compare_models(split.x_train, split.y_train, config)

    for column in results.columns:
        if column.endswith("_mean"):
            assert (results[column] >= 0).all()
            assert (results[column] <= 1).all()


def test_select_best_model_picks_top_row(synthetic_frame, config):
    """Selection returns the model with the highest selection metric."""
    split = split_data(synthetic_frame, config)
    results, _ = compare_models(split.x_train, split.y_train, config)

    best = select_best_model(results, config)
    column = f"{config.model.selection_metric}_mean"
    assert results.loc[best, column] == results[column].max()


def test_select_best_model_unknown_metric_raises(synthetic_frame, config):
    """An absent selection metric raises KeyError."""
    frame = pd.DataFrame({"other_mean": [0.5]}, index=["model"])
    with pytest.raises(KeyError):
        select_best_model(frame, config)


def test_train_final_model_produces_fitted_pipeline(synthetic_frame, config):
    """The final model is a fitted pipeline that yields valid probabilities."""
    split = split_data(synthetic_frame, config)
    pipeline = train_final_model(BASELINE_MODEL, split, config)

    assert isinstance(pipeline, Pipeline)
    probabilities = pipeline.predict_proba(split.x_test)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_train_final_model_applies_params(synthetic_frame, config):
    """Tuned parameters are applied before fitting."""
    split = split_data(synthetic_frame, config)
    pipeline = train_final_model(
        BASELINE_MODEL, split, config, params={"model__C": 0.05}
    )
    assert pipeline.named_steps["model"].C == 0.05


def test_save_and_load_pipeline_roundtrip(synthetic_frame, config, tmp_path, monkeypatch):
    """A saved pipeline reloads and predicts identically."""
    import churn.train as train_module

    split = split_data(synthetic_frame, config)
    pipeline = train_final_model(BASELINE_MODEL, split, config)

    target = tmp_path / "model.joblib"
    monkeypatch.setattr(type(config), "model_path", property(lambda self: target))

    saved_path = train_module.save_pipeline(pipeline, config)
    assert saved_path.is_file()

    reloaded = train_module.load_pipeline(config)
    np.testing.assert_allclose(
        pipeline.predict_proba(split.x_test)[:, 1],
        reloaded.predict_proba(split.x_test)[:, 1],
    )


def test_load_pipeline_missing_raises(config, tmp_path, monkeypatch):
    """Loading a nonexistent artifact raises FileNotFoundError."""
    import churn.train as train_module

    missing = tmp_path / "absent.joblib"
    monkeypatch.setattr(type(config), "model_path", property(lambda self: missing))

    with pytest.raises(FileNotFoundError):
        train_module.load_pipeline(config)

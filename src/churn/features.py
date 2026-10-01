"""Preprocessing pipeline.

Scaling for the numeric columns, one-hot for the categoricals, bundled into a
ColumnTransformer. Keeping this inside a Pipeline with the estimator is what
stops the scaler from seeing validation data during cross-validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import get_feature_types


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Unfitted transformer. `features` is only used to work out column types.

    handle_unknown="ignore" is for serving: an unseen category encodes as zeros
    rather than blowing up mid-request.
    """
    numeric, categorical = get_feature_types(features)

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(features: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    """Preprocessor + estimator as one object, so persisting it persists both."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", estimator),
        ]
    )


def get_feature_names(pipeline_or_transformer) -> list[str]:
    """Transformed column names, for labelling SHAP values and coefficients.

    Accepts either a fitted Pipeline or a fitted ColumnTransformer.
    """
    transformer = pipeline_or_transformer
    if isinstance(pipeline_or_transformer, Pipeline):
        transformer = pipeline_or_transformer.named_steps["preprocessor"]

    return list(transformer.get_feature_names_out())

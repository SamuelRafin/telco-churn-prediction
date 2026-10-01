"""Training, model comparison and selection.

Logistic regression as the baseline plus two tree models, compared with
stratified CV on the training split, then the winner refit on the whole
training set and saved.

Imbalance is handled per-estimator (class_weight, or scale_pos_weight for
XGBoost). Selection is on ROC-AUC / PR-AUC; accuracy would be misleading at
26.5% positives.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import Config, load_config
from churn.data import load_and_clean, split_features_target
from churn.features import build_pipeline

# average_precision is PR-AUC, more informative than ROC-AUC when positives
# are rare.
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}

N_FOLDS = 5

# The baseline every other model must beat to justify its extra complexity.
BASELINE_MODEL = "logistic_regression"


@dataclass
class DataSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def summary(self) -> str:
        return (
            f"Train: {len(self.x_train):,} rows "
            f"({self.y_train.mean():.2%} churn)\n"
            f"Test:  {len(self.x_test):,} rows "
            f"({self.y_test.mean():.2%} churn)"
        )


def split_data(df: pd.DataFrame, config: Config | None = None) -> DataSplit:
    """Stratified train/test split, so both sides keep the same churn rate."""
    config = config or load_config()
    features, labels = split_features_target(df, config)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.split.test_size,
        random_state=config.project.random_seed,
        stratify=labels if config.split.stratify else None,
    )
    return DataSplit(x_train, x_test, y_train, y_test)


def build_models(y_train: pd.Series, config: Config | None = None) -> dict:
    """The candidate estimators, unfitted, each set up for the imbalance."""
    config = config or load_config()
    seed = config.project.random_seed

    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    scale_pos_weight = n_negative / n_positive if n_positive else 1.0

    return {
        BASELINE_MODEL: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        ),
    }


# Search spaces for the optional tuning pass. Prefixed with ``model__``
# because the estimator sits inside a Pipeline step named "model".
SEARCH_SPACES: dict[str, dict] = {
    "random_forest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [6, 8, 12, 16, None],
        "model__min_samples_leaf": [1, 3, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [2, 3, 4, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 3.0, 10.0],
        "model__min_child_weight": [1, 3, 5],
    },
    # Only the regularisation strength is searched. 'penalty' is deprecated in
    # scikit-learn >= 1.8, and L2 is already the default.
    BASELINE_MODEL: {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
}

N_SEARCH_ITER = 25


def tune_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config | None = None,
    n_iter: int = N_SEARCH_ITER,
):
    """Randomised search over the whole pipeline, not just the estimator.

    Searching the pipeline means preprocessing refits inside each fold, so the
    search can't leak.
    """
    config = config or load_config()
    estimator = build_models(y_train, config)[name]
    pipeline = build_pipeline(x_train, estimator)

    folds = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=config.project.random_seed,
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=SEARCH_SPACES[name],
        n_iter=n_iter,
        scoring=config.model.selection_metric,
        cv=folds,
        random_state=config.project.random_seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search


def cross_validate_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config | None = None,
) -> dict[str, float]:
    """CV one pipeline, returning {metric}_mean / {metric}_std for each scorer."""
    config = config or load_config()
    folds = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=config.project.random_seed,
    )

    scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=folds,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
    )

    results: dict[str, float] = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        results[f"{metric}_mean"] = float(np.mean(values))
        results[f"{metric}_std"] = float(np.std(values))
    return results


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config | None = None,
    tune: bool = False,
    n_iter: int = N_SEARCH_ITER,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """CV every candidate and rank them.

    Returns (results, best_params). results is indexed by model name, sorted by
    the selection metric. best_params is empty unless tune=True.
    """
    config = config or load_config()
    models = build_models(y_train, config)

    rows: dict[str, dict[str, float]] = {}
    best_params: dict[str, dict] = {}

    for name, estimator in models.items():
        if tune:
            search = tune_model(name, x_train, y_train, config, n_iter)
            pipeline = search.best_estimator_
            best_params[name] = {
                key: value for key, value in search.best_params_.items()
            }
        else:
            pipeline = build_pipeline(x_train, estimator)

        rows[name] = cross_validate_model(pipeline, x_train, y_train, config)

    results = pd.DataFrame(rows).T
    sort_column = f"{config.model.selection_metric}_mean"
    return results.sort_values(sort_column, ascending=False), best_params


def select_best_model(results: pd.DataFrame, config: Config | None = None) -> str:
    """Name of the highest scorer on the configured selection metric."""
    config = config or load_config()
    column = f"{config.model.selection_metric}_mean"

    if column not in results.columns:
        raise KeyError(
            f"Selection metric '{column}' not found. "
            f"Available: {list(results.columns)}"
        )
    return str(results[column].idxmax())


def train_final_model(
    name: str,
    split: DataSplit,
    config: Config | None = None,
    params: dict | None = None,
) -> Pipeline:
    """Refit one model on the full training split.

    params takes pipeline-style keys from the tuning pass, e.g.
    {"model__max_depth": 8}.
    """
    config = config or load_config()
    estimator = build_models(split.y_train, config)[name]
    pipeline = build_pipeline(split.x_train, estimator)

    if params:
        pipeline.set_params(**params)

    pipeline.fit(split.x_train, split.y_train)
    return pipeline


def save_pipeline(pipeline: Pipeline, config: Config | None = None):
    config = config or load_config()
    path = config.model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(config: Config | None = None) -> Pipeline:
    config = config or load_config()
    path = config.model_path

    if not path.is_file():
        raise FileNotFoundError(
            f"No model artifact at {path}. Run 'python -m churn.train' first."
        )
    return joblib.load(path)


def format_results(results: pd.DataFrame, config: Config | None = None) -> str:
    config = config or load_config()
    display = pd.DataFrame(
        {
            "ROC-AUC": results["roc_auc_mean"].map("{:.4f}".format)
            + " +/- "
            + results["roc_auc_std"].map("{:.3f}".format),
            "PR-AUC": results["average_precision_mean"].map("{:.4f}".format)
            + " +/- "
            + results["average_precision_std"].map("{:.3f}".format),
            "F1": results["f1_mean"].map("{:.4f}".format),
            "Recall": results["recall_mean"].map("{:.4f}".format),
            "Precision": results["precision_mean"].map("{:.4f}".format),
        }
    )
    return display.to_string()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn models.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a randomised hyperparameter search for each model.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=N_SEARCH_ITER,
        help=f"Search iterations per model when tuning (default {N_SEARCH_ITER}).",
    )
    args = parser.parse_args(argv)

    config = load_config()

    print("Loading and cleaning data...")
    df = load_and_clean(config)
    split = split_data(df, config)
    print(split.summary())

    mode = f"tuned search ({args.n_iter} iters/model)" if args.tune else "default params"
    print(f"\nCross-validating {N_FOLDS} folds on the training split [{mode}]...")
    results, best_params = compare_models(
        split.x_train, split.y_train, config, tune=args.tune, n_iter=args.n_iter
    )

    print("\nModel comparison")
    print("=" * 78)
    print(format_results(results, config))

    metric = config.model.selection_metric
    baseline_score = results.loc[BASELINE_MODEL, f"{metric}_mean"]
    best_name = select_best_model(results, config)
    best_score = results.loc[best_name, f"{metric}_mean"]

    print(f"\nBaseline ({BASELINE_MODEL}) {metric}: {baseline_score:.4f}")
    print(f"Best model: {best_name} ({metric}: {best_score:.4f})")
    print(f"Improvement over baseline: {best_score - baseline_score:+.4f}")

    if best_params:
        print("\nBest hyperparameters found:")
        for name, params in best_params.items():
            print(f"  {name}:")
            for key, value in sorted(params.items()):
                print(f"    {key.replace('model__', '')} = {value}")

    print(f"\nRefitting '{best_name}' on the full training split...")
    pipeline = train_final_model(
        best_name, split, config, params=best_params.get(best_name)
    )
    model_path = save_pipeline(pipeline, config)
    print(f"Model saved to {model_path}")

    # Persist the comparison so the README and reports can quote real numbers.
    metrics_path = config.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_metric": metric,
        "n_folds": N_FOLDS,
        "tuned": bool(args.tune),
        "baseline_model": BASELINE_MODEL,
        "best_model": best_name,
        "cv_results": results.to_dict(orient="index"),
        "best_params": best_params,
        "split": {
            "n_train": int(len(split.x_train)),
            "n_test": int(len(split.x_test)),
            "train_churn_rate": float(split.y_train.mean()),
            "test_churn_rate": float(split.y_test.mean()),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"CV metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

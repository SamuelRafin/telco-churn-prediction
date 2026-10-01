"""SHAP explanations, global and per-customer.

Since the model lives inside a pipeline, SHAP runs against the model step on
already-transformed data, with feature names pulled off the fitted preprocessor
so the output is readable rather than feature_0..feature_44.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.config import Config, load_config  # noqa: E402
from churn.data import load_and_clean  # noqa: E402
from churn.features import get_feature_names  # noqa: E402
from churn.train import load_pipeline, split_data  # noqa: E402

# Cap the background sample so KernelExplainer-style fallbacks stay tractable.
MAX_BACKGROUND = 500
TOP_N = 15


def transform_features(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessor and keep the column names."""
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    names = get_feature_names(pipeline)
    return pd.DataFrame(transformed, columns=names, index=features.index)


def build_explainer(pipeline: Pipeline, background: pd.DataFrame):
    """Pick the right explainer: exact ones for trees and linear models,
    model-agnostic fallback for anything else."""
    model = pipeline.named_steps["model"]
    model_name = type(model).__name__

    if model_name in {
        "XGBClassifier",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
    }:
        return shap.TreeExplainer(model)

    if model_name == "LogisticRegression":
        return shap.LinearExplainer(model, background)

    return shap.Explainer(model.predict_proba, background)


def _positive_class_values(values: np.ndarray) -> np.ndarray:
    """Flatten SHAP output to 2-D for the positive class.

    Shapes vary by model family: (n, features) for binary tree and linear
    models, (n, features, classes) for ones that emit per-class values.
    """
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def compute_shap_values(
    pipeline: Pipeline,
    features: pd.DataFrame,
    background: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (shap_values, transformed_features), both n_samples rows."""
    transformed = transform_features(pipeline, features)

    background_source = features if background is None else background
    transformed_background = transform_features(pipeline, background_source)
    if len(transformed_background) > MAX_BACKGROUND:
        transformed_background = transformed_background.sample(
            MAX_BACKGROUND, random_state=0
        )

    explainer = build_explainer(pipeline, transformed_background)
    raw_values = explainer.shap_values(transformed)
    values = _positive_class_values(raw_values)

    return values, transformed


def global_importance(
    shap_values: np.ndarray, transformed: pd.DataFrame, top_n: int = TOP_N
) -> pd.DataFrame:
    """Rank by mean |SHAP|, with a direction for each feature.

    Direction is the correlation between a feature's value and its SHAP value,
    not the signed mean. The signed mean doesn't work: most customers don't
    churn, so nearly every feature ends up with a negative average
    contribution and everything looks protective.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    correlations = np.zeros(shap_values.shape[1])
    values = transformed.to_numpy()
    for index in range(shap_values.shape[1]):
        feature_column = values[:, index]
        shap_column = shap_values[:, index]
        # Constant column, no direction to infer.
        if feature_column.std() > 0 and shap_column.std() > 0:
            correlations[index] = float(
                np.corrcoef(feature_column, shap_column)[0, 1]
            )

    ranking = pd.DataFrame(
        {
            "feature": transformed.columns,
            "mean_abs_shap": mean_abs,
            "value_shap_corr": correlations,
            "pushes_toward_churn": correlations > 0,
        }
    )
    return ranking.sort_values("mean_abs_shap", ascending=False).head(top_n)


def explain_customer(
    pipeline: Pipeline,
    customer: pd.DataFrame,
    background: pd.DataFrame,
    top_n: int = 8,
) -> dict:
    """Probability plus the top contributing features for one customer."""
    if len(customer) != 1:
        raise ValueError(f"Expected exactly one customer row, got {len(customer)}.")

    probability = float(pipeline.predict_proba(customer)[0, 1])
    shap_values, transformed = compute_shap_values(pipeline, customer, background)

    contributions = pd.DataFrame(
        {
            "feature": transformed.columns,
            "value": transformed.iloc[0].to_numpy(),
            "shap_value": shap_values[0],
        }
    )
    contributions["abs_shap"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_shap", ascending=False).head(top_n)

    return {
        "churn_probability": probability,
        "top_factors": [
            {
                "feature": row.feature,
                "shap_value": float(row.shap_value),
                "direction": "increases churn risk"
                if row.shap_value > 0
                else "decreases churn risk",
            }
            for row in top.itertuples()
        ],
    }


def plot_global_importance(ranking: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of mean absolute SHAP values, coloured by direction."""
    ordered = ranking.sort_values("mean_abs_shap")
    colors = [
        "#E45756" if pushes else "#4C78A8"
        for pushes in ordered["pushes_toward_churn"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(ordered) + 1.5)))
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color=colors)

    ax.set_xlabel("Mean |SHAP value| (average impact on churn prediction)")
    ax.set_title(
        "Global feature importance\n"
        "red = higher values push toward churn, blue = toward retention"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray, transformed: pd.DataFrame, top_n: int = TOP_N
) -> plt.Figure:
    """SHAP beeswarm summary plot."""
    fig = plt.figure(figsize=(9, max(4, 0.4 * top_n + 2)))
    shap.summary_plot(
        shap_values,
        transformed,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP summary: feature value vs impact on churn")
    plt.tight_layout()
    return fig


def plot_customer_explanation(explanation: dict) -> plt.Figure:
    """Waterfall-style bar chart for one customer's top factors."""
    factors = explanation["top_factors"][::-1]
    names = [f["feature"] for f in factors]
    values = [f["shap_value"] for f in factors]
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(names) + 1.5)))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", lw=1)

    probability = explanation["churn_probability"]
    ax.set_xlabel("SHAP value (push toward churn ->)")
    ax.set_title(f"Why this customer? Predicted churn probability: {probability:.1%}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def generate_explanation_figures(
    pipeline: Pipeline,
    features: pd.DataFrame,
    config: Config | None = None,
    sample_size: int = 1000,
) -> tuple[dict, pd.DataFrame, dict]:
    """Returns (saved_paths, global_ranking, customer_explanation).

    sample_size caps the rows used for the global pass, purely for speed.
    """
    config = config or load_config()
    output_dir = config.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = (
        features.sample(sample_size, random_state=config.project.random_seed)
        if len(features) > sample_size
        else features
    )

    shap_values, transformed = compute_shap_values(pipeline, sample, sample)
    ranking = global_importance(shap_values, transformed)

    # Explain the riskiest customer in the sample, since that's the one a
    # retention team would actually be calling.
    probabilities = pipeline.predict_proba(sample)[:, 1]
    riskiest = sample.iloc[[int(np.argmax(probabilities))]]
    explanation = explain_customer(pipeline, riskiest, sample)

    figures = {
        "shap_global_importance": plot_global_importance(ranking),
        "shap_summary": plot_shap_summary(shap_values, transformed),
        "shap_customer_explanation": plot_customer_explanation(explanation),
    }

    saved = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        saved[name] = path

    return saved, ranking, explanation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explain the churn model with SHAP.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Rows used for the global explanation (default 1000).",
    )
    args = parser.parse_args(argv)

    config = load_config()
    pipeline = load_pipeline(config)

    df = load_and_clean(config)
    split = split_data(df, config)

    model_name = type(pipeline.named_steps["model"]).__name__
    print(f"Explaining {model_name} on {len(split.x_test):,} held-out customers...")

    saved, ranking, explanation = generate_explanation_figures(
        pipeline, split.x_test, config, args.sample_size
    )

    print("\nTop churn drivers (global)")
    print("=" * 68)
    print(f"{'Feature':<40}{'Mean |SHAP|':>14}{'Direction':>14}")
    print("-" * 68)
    for row in ranking.itertuples():
        direction = "-> churn" if row.pushes_toward_churn else "-> retain"
        print(f"{row.feature:<40}{row.mean_abs_shap:>14.4f}{direction:>14}")

    print(
        f"\nHighest-risk customer in sample: "
        f"{explanation['churn_probability']:.1%} churn probability"
    )
    print("Top contributing factors:")
    for factor in explanation["top_factors"]:
        print(
            f"  {factor['feature']:<40}{factor['shap_value']:+.4f}  "
            f"{factor['direction']}"
        )

    print("\nFigures written:")
    for name, path in saved.items():
        print(f"  {name:28s} -> {path.relative_to(path.parents[2])}")

    # Persist the global ranking alongside the other metrics.
    metrics_path = config.metrics_path
    payload = {}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    payload["shap_global_importance"] = ranking.to_dict(orient="records")
    payload["shap_example_customer"] = explanation

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSHAP importance saved to {metrics_path}")


if __name__ == "__main__":
    main()

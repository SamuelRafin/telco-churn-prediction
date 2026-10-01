"""EDA plots and the numbers quoted in the notebook and README.

Lives here rather than only in the notebook so the charts can be regenerated
without re-running cells by hand. Everything takes a cleaned frame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so figures can be generated headlessly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import pandas as pd  # noqa: E402

from churn.config import Config, load_config  # noqa: E402
from churn.data import (  # noqa: E402
    get_feature_types,
    load_and_clean,
    split_features_target,
)

RETAINED_COLOR = "#4C78A8"
CHURNED_COLOR = "#E45756"


def churn_rate_by(
    df: pd.DataFrame, column: str, target: str = "Churn"
) -> pd.Series:
    """Churn rate per level of a column, highest first."""
    return df.groupby(column)[target].mean().sort_values(ascending=False)


def tenure_bucket(df: pd.DataFrame) -> pd.Series:
    """Bucket tenure (in months) into readable ranges."""
    return pd.cut(
        df["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )


def plot_class_balance(df: pd.DataFrame, target: str = "Churn") -> plt.Figure:
    """Bar chart of retained vs churned customers."""
    counts = df[target].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(labels, counts.values, color=[RETAINED_COLOR, CHURNED_COLOR])
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Class balance: churn is the minority class")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_churn_rate_by_column(
    df: pd.DataFrame, column: str, target: str = "Churn"
) -> plt.Figure:
    """Horizontal bar chart of churn rate across one column's levels."""
    rates = churn_rate_by(df, column, target).sort_values()
    overall = df[target].mean()

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(rates) + 1.5)))
    ax.barh([str(i) for i in rates.index], rates.values, color=CHURNED_COLOR)
    ax.axvline(
        overall,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Overall ({overall:.1%})",
    )

    for y, value in enumerate(rates.values):
        ax.text(value, y, f" {value:.1%}", va="center", fontsize=9)

    ax.set_title(f"Churn rate by {column}")
    ax.set_xlabel("Churn rate")
    ax.set_xlim(0, min(1.0, rates.max() * 1.25))
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_churn_by_tenure(df: pd.DataFrame, target: str = "Churn") -> plt.Figure:
    """Churn rate across tenure buckets, with customer counts."""
    working = df.copy()
    working["tenure_bucket"] = tenure_bucket(working)

    grouped = working.groupby("tenure_bucket", observed=True)[target].agg(
        ["mean", "size"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        grouped.index.astype(str),
        grouped["mean"].values,
        color=CHURNED_COLOR,
    )

    for x, (rate, size) in enumerate(zip(grouped["mean"], grouped["size"])):
        ax.text(
            x,
            rate,
            f"{rate:.1%}\nn={size:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Churn rate falls sharply with tenure")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, grouped["mean"].max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(
    df: pd.DataFrame, target: str = "Churn"
) -> plt.Figure:
    """Overlaid histograms of numeric features, split by churn outcome."""
    columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, len(columns), figsize=(14, 4))

    for ax, column in zip(axes, columns):
        retained = df.loc[df[target] == 0, column]
        churned = df.loc[df[target] == 1, column]

        ax.hist(
            [retained, churned],
            bins=30,
            stacked=False,
            density=True,
            color=[RETAINED_COLOR, CHURNED_COLOR],
            label=["Retained", "Churned"],
        )
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=9)
    fig.suptitle("Numeric feature distributions by churn outcome", y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, target: str = "Churn") -> plt.Figure:
    """Correlation heatmap of numeric features plus the target."""
    features, _ = split_features_target(df, load_config())
    numeric, _ = get_feature_types(features)
    columns = numeric + [target]

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.5 else "black",
            )

    ax.set_title("Correlation: numeric features vs churn")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_categorical_churn_grid(
    df: pd.DataFrame, columns: list[str], target: str = "Churn"
) -> plt.Figure:
    """Small-multiples grid of churn rate across several categorical columns."""
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes = axes.flatten()
    overall = df[target].mean()

    for ax, column in zip(axes, columns):
        rates = churn_rate_by(df, column, target).sort_values()
        ax.barh([str(i) for i in rates.index], rates.values, color=CHURNED_COLOR)
        ax.axvline(overall, color="black", linestyle="--", linewidth=1)
        ax.set_title(column, fontsize=11)
        ax.set_xlim(0, 0.75)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Churn rate by category (dashed line = overall rate)", y=1.0, fontsize=13
    )
    fig.tight_layout()
    return fig


def compute_insights(df: pd.DataFrame, target: str = "Churn") -> dict:
    """The headline numbers, so the README and notebook can't drift apart."""
    working = df.copy()
    working["tenure_bucket"] = tenure_bucket(working)

    contract_rates = churn_rate_by(working, "Contract", target)
    tenure_rates = working.groupby("tenure_bucket", observed=True)[target].mean()
    internet_rates = churn_rate_by(working, "InternetService", target)
    payment_rates = churn_rate_by(working, "PaymentMethod", target)

    numeric_corr = working[["tenure", "MonthlyCharges", "TotalCharges", target]].corr()[
        target
    ]

    return {
        "n_customers": int(len(working)),
        "overall_churn_rate": float(working[target].mean()),
        "contract_churn_rates": {str(k): float(v) for k, v in contract_rates.items()},
        "tenure_churn_rates": {str(k): float(v) for k, v in tenure_rates.items()},
        "internet_churn_rates": {str(k): float(v) for k, v in internet_rates.items()},
        "payment_churn_rates": {str(k): float(v) for k, v in payment_rates.items()},
        "corr_with_churn": {
            "tenure": float(numeric_corr["tenure"]),
            "MonthlyCharges": float(numeric_corr["MonthlyCharges"]),
            "TotalCharges": float(numeric_corr["TotalCharges"]),
        },
        "median_tenure_churned": float(
            working.loc[working[target] == 1, "tenure"].median()
        ),
        "median_tenure_retained": float(
            working.loc[working[target] == 0, "tenure"].median()
        ),
        "mean_monthly_churned": float(
            working.loc[working[target] == 1, "MonthlyCharges"].mean()
        ),
        "mean_monthly_retained": float(
            working.loc[working[target] == 0, "MonthlyCharges"].mean()
        ),
    }


def generate_all_figures(
    df: pd.DataFrame | None = None, config: Config | None = None
) -> dict[str, Path]:
    """Write every EDA figure to the reports dir, returning {name: path}."""
    config = config or load_config()
    if df is None:
        df = load_and_clean(config)

    target = config.data.target_column
    output_dir = config.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
    ]

    figures = {
        "class_balance": plot_class_balance(df, target),
        "churn_by_contract": plot_churn_rate_by_column(df, "Contract", target),
        "churn_by_tenure": plot_churn_by_tenure(df, target),
        "numeric_distributions": plot_numeric_distributions(df, target),
        "correlation_heatmap": plot_correlation_heatmap(df, target),
        "categorical_grid": plot_categorical_churn_grid(df, grid_columns, target),
    }

    saved: dict[str, Path] = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        saved[name] = path

    return saved


def main() -> None:
    config = load_config()
    df = load_and_clean(config)

    saved = generate_all_figures(df, config)
    insights = compute_insights(df, config.data.target_column)

    print("EDA figures written:")
    for name, path in saved.items():
        print(f"  {name:24s} -> {path.relative_to(path.parents[2])}")

    print("\nKey insights")
    print("=" * 50)
    print(f"Customers: {insights['n_customers']:,}")
    print(f"Overall churn rate: {insights['overall_churn_rate']:.2%}")

    print("\nChurn rate by contract:")
    for level, rate in insights["contract_churn_rates"].items():
        print(f"  {level:16s} {rate:.2%}")

    print("\nChurn rate by tenure bucket (months):")
    for level, rate in insights["tenure_churn_rates"].items():
        print(f"  {level:16s} {rate:.2%}")

    print("\nChurn rate by internet service:")
    for level, rate in insights["internet_churn_rates"].items():
        print(f"  {level:16s} {rate:.2%}")

    print("\nCorrelation with churn:")
    for name, value in insights["corr_with_churn"].items():
        print(f"  {name:16s} {value:+.3f}")

    print(
        f"\nMedian tenure: churned {insights['median_tenure_churned']:.0f} mo "
        f"vs retained {insights['median_tenure_retained']:.0f} mo"
    )
    print(
        f"Mean monthly charges: churned ${insights['mean_monthly_churned']:.2f} "
        f"vs retained ${insights['mean_monthly_retained']:.2f}"
    )


if __name__ == "__main__":
    main()

"""Loading and cleaning the Telco churn CSV.

Three things need fixing in the raw file: TotalCharges is stored as text with
blanks for tenure-0 customers, Churn is a Yes/No string, and customerID is
dead weight. Column names come from config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import PROJECT_ROOT, Config, load_config

# SeniorCitizen is already 0/1 in the raw data so it goes in with the numerics.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_raw_data(config: Config | None = None) -> pd.DataFrame:
    """Read the CSV as-is, no cleaning."""
    config = config or load_config()
    raw_path = _resolve(config.data.raw_path)

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Download the IBM Telco "
            "Customer Churn CSV (e.g. from Kaggle) and save it there, or "
            "update 'data.raw_path' in config.yaml."
        )

    return pd.read_csv(raw_path)


def clean_data(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Return a modelling-ready copy of the raw frame.

    The TotalCharges blanks belong to customers with tenure 0, who haven't been
    billed yet, so 0.0 is the actual value rather than an imputation.
    """
    config = config or load_config()
    data = df.copy()

    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    target = config.data.target_column
    if target in data.columns and data[target].dtype == object:
        positive = config.data.positive_label.strip()
        data[target] = (data[target] == positive).astype(int)

    id_column = config.data.id_column
    if id_column in data.columns:
        data = data.drop(columns=[id_column])

    return data.reset_index(drop=True)


def load_and_clean(config: Config | None = None) -> pd.DataFrame:
    config = config or load_config()
    return clean_data(load_raw_data(config), config)


def split_features_target(
    df: pd.DataFrame, config: Config | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into (X, y)."""
    config = config or load_config()
    target = config.data.target_column

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    return df.drop(columns=[target]), df[target]


def get_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) column names in their original order."""
    numeric = [c for c in features.columns if c in NUMERIC_COLUMNS]
    categorical = [c for c in features.columns if c not in NUMERIC_COLUMNS]
    return numeric, categorical


def summarize(df: pd.DataFrame, config: Config | None = None) -> str:
    """Shape, missing values, churn rate and the numeric/categorical split."""
    config = config or load_config()
    target = config.data.target_column

    lines = [
        "Cleaned dataset summary",
        "=" * 40,
        f"Rows:    {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Missing values: {int(df.isna().sum().sum())}",
    ]

    if target in df.columns:
        churn_rate = df[target].mean()
        churned = int(df[target].sum())
        lines += [
            "",
            f"Churn rate: {churn_rate:.2%} ({churned:,} of {len(df):,} customers)",
            f"Class balance: {len(df) - churned:,} retained / {churned:,} churned",
        ]

        features, _ = split_features_target(df, config)
        numeric, categorical = get_feature_types(features)
        lines += [
            "",
            f"Numeric features ({len(numeric)}): {', '.join(numeric)}",
            f"Categorical features ({len(categorical)}): {', '.join(categorical)}",
        ]

    return "\n".join(lines)


def main() -> None:
    config = load_config()
    cleaned = load_and_clean(config)
    print(summarize(cleaned, config))

    output_path = _resolve(config.data.processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"\nProcessed data written to {output_path}")


if __name__ == "__main__":
    main()

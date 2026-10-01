"""Scoring customers with the saved pipeline.

The artifact holds preprocessing and model together, so inference applies the
same transforms that training fitted. Threshold comes from config.yaml so the
API's yes/no matches what was evaluated.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import Config, load_config
from churn.data import NUMERIC_COLUMNS
from churn.train import load_pipeline

# Order matters, matches the Telco schema.
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

RISK_BANDS = (
    (0.75, "critical", "Immediate outreach with a strong retention offer"),
    (0.50, "high", "Proactive contact and targeted incentive"),
    (0.25, "medium", "Monitor and include in nurture campaigns"),
    (0.00, "low", "No action needed"),
)


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Cached, otherwise joblib load dominates per-request latency."""
    return load_pipeline()


def risk_band(probability: float) -> tuple[str, str]:
    for floor, label, action in RISK_BANDS:
        if probability >= floor:
            return label, action
    return "low", "No action needed"


def to_dataframe(records: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """One or more records into a frame with columns in the expected order.

    Raises on missing columns rather than filling them in, since a quietly
    wrong score is worse than an error.
    """
    if isinstance(records, dict):
        records = [records]

    frame = pd.DataFrame(records)

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    frame = frame[FEATURE_COLUMNS].copy()

    # Coerce numerics so string inputs (e.g. from a web form) still work.
    for column in FEATURE_COLUMNS:
        if column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[[c for c in FEATURE_COLUMNS if c in NUMERIC_COLUMNS]].isna().any().any():
        raise ValueError("Numeric fields must be valid numbers.")

    return frame


def predict_proba(
    records: dict[str, Any] | list[dict[str, Any]],
    pipeline: Pipeline | None = None,
) -> list[float]:
    pipeline = pipeline or get_model()
    frame = to_dataframe(records)
    return [float(p) for p in pipeline.predict_proba(frame)[:, 1]]


def predict(
    records: dict[str, Any] | list[dict[str, Any]],
    pipeline: Pipeline | None = None,
    config: Config | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Probability, decision, risk band and suggested action per customer."""
    config = config or load_config()
    cutoff = config.model.decision_threshold if threshold is None else threshold

    probabilities = predict_proba(records, pipeline)

    results = []
    for probability in probabilities:
        band, action = risk_band(probability)
        results.append(
            {
                "churn_probability": round(probability, 4),
                "will_churn": bool(probability >= cutoff),
                "threshold": float(cutoff),
                "risk_band": band,
                "recommended_action": action,
            }
        )
    return results


def predict_one(
    record: dict[str, Any],
    pipeline: Pipeline | None = None,
    config: Config | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    return predict(record, pipeline, config, threshold)[0]


EXAMPLE_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.35,
    "TotalCharges": 188.70,
}


def main() -> None:
    config = load_config()
    result = predict_one(EXAMPLE_CUSTOMER, config=config)

    print("Example customer (new, month-to-month, fiber, no add-ons)")
    print("=" * 60)
    for key in ("tenure", "Contract", "InternetService", "MonthlyCharges"):
        print(f"  {key:<20}{EXAMPLE_CUSTOMER[key]}")

    print("\nPrediction")
    print("-" * 60)
    print(f"  Churn probability   {result['churn_probability']:.1%}")
    print(f"  Decision threshold  {result['threshold']:.4f}")
    print(f"  Will churn          {result['will_churn']}")
    print(f"  Risk band           {result['risk_band']}")
    print(f"  Recommended action  {result['recommended_action']}")


if __name__ == "__main__":
    main()

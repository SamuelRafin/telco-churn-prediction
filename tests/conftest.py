"""Shared pytest fixtures."""

import pandas as pd
import pytest

from churn.config import load_config


@pytest.fixture
def config():
    """The project configuration."""
    return load_config()


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A small synthetic frame mirroring the raw Telco schema and its quirks.

    Row 3 reproduces the real dataset's edge case: a tenure-0 customer whose
    ``TotalCharges`` is a blank string.
    """
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAA", "0002-BBBB", "0003-CCCC", "0004-DDDD"],
            "gender": ["Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 34, 12, 0],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "No"],
            "InternetService": ["DSL", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "Yes", "No"],
            "TechSupport": ["No", "No", "No", "Yes"],
            "StreamingTV": ["No", "No", "Yes", "No"],
            "StreamingMovies": ["No", "No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 89.10, 45.00],
            # Blank string for the tenure-0 customer, as in the real data.
            "TotalCharges": ["29.85", "1889.5", "1069.2", " "],
            "Churn": ["No", "No", "Yes", "No"],
        }
    )

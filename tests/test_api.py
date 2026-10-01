"""Tests for the prediction module and the FastAPI service.

API tests need a trained artifact. If none exists, they are skipped with a
clear message rather than failing, so a fresh clone can still run the suite.
"""

import pytest
from fastapi.testclient import TestClient

from churn.config import load_config
from churn.predict import (
    EXAMPLE_CUSTOMER,
    FEATURE_COLUMNS,
    predict_one,
    risk_band,
    to_dataframe,
)

MODEL_AVAILABLE = load_config().model_path.is_file()
requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="No trained model artifact. Run 'python -m churn.train' first.",
)


@pytest.fixture
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


# --- predict module -------------------------------------------------------


def test_to_dataframe_orders_columns():
    """Column order matches what the model expects."""
    frame = to_dataframe(EXAMPLE_CUSTOMER)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 1


def test_to_dataframe_accepts_list():
    """A list of records produces one row each."""
    frame = to_dataframe([EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER])
    assert len(frame) == 2


def test_to_dataframe_missing_column_raises():
    """A missing feature is reported rather than guessed."""
    incomplete = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "tenure"}
    with pytest.raises(ValueError, match="Missing required feature columns"):
        to_dataframe(incomplete)


def test_to_dataframe_coerces_numeric_strings():
    """Numeric fields arriving as strings are converted."""
    record = dict(EXAMPLE_CUSTOMER, tenure="12", MonthlyCharges="70.5")
    frame = to_dataframe(record)

    assert frame["tenure"].iloc[0] == 12
    assert frame["MonthlyCharges"].iloc[0] == 70.5


def test_to_dataframe_rejects_non_numeric():
    """Garbage in a numeric field raises."""
    record = dict(EXAMPLE_CUSTOMER, tenure="not-a-number")
    with pytest.raises(ValueError, match="valid numbers"):
        to_dataframe(record)


@pytest.mark.parametrize(
    "probability,expected",
    [(0.95, "critical"), (0.60, "high"), (0.30, "medium"), (0.05, "low")],
)
def test_risk_band_thresholds(probability, expected):
    """Probabilities map to the documented risk bands."""
    band, action = risk_band(probability)
    assert band == expected
    assert action


@requires_model
def test_predict_one_returns_expected_keys():
    """A scored customer carries all response fields."""
    result = predict_one(EXAMPLE_CUSTOMER)

    for key in (
        "churn_probability",
        "will_churn",
        "threshold",
        "risk_band",
        "recommended_action",
    ):
        assert key in result

    assert 0.0 <= result["churn_probability"] <= 1.0


@requires_model
def test_predict_respects_threshold_override():
    """An explicit threshold changes the decision."""
    always = predict_one(EXAMPLE_CUSTOMER, threshold=0.0)
    never = predict_one(EXAMPLE_CUSTOMER, threshold=1.01)

    assert always["will_churn"] is True
    assert never["will_churn"] is False


@requires_model
def test_high_risk_profile_scores_above_low_risk():
    """A new month-to-month fiber customer is riskier than a loyal one."""
    low_risk = dict(
        EXAMPLE_CUSTOMER,
        tenure=70,
        Contract="Two year",
        InternetService="DSL",
        OnlineSecurity="Yes",
        TechSupport="Yes",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=45.0,
        TotalCharges=3150.0,
    )

    high = predict_one(EXAMPLE_CUSTOMER)["churn_probability"]
    low = predict_one(low_risk)["churn_probability"]
    assert high > low


# --- API ------------------------------------------------------------------


def test_health_endpoint(client):
    """Health always answers, reporting whether the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


def test_root_endpoint(client):
    """Root points at the docs."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_openapi_schema_available(client):
    """The OpenAPI schema generates without error."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


@requires_model
def test_model_info_endpoint(client):
    """Model info reports the model type and feature count."""
    response = client.get("/model-info")
    assert response.status_code == 200

    body = response.json()
    assert body["n_features"] == len(body["feature_names"])
    assert body["n_features"] > len(FEATURE_COLUMNS)  # one-hot expansion
    assert 0.0 < body["decision_threshold"] < 1.0


@requires_model
def test_predict_endpoint_valid_payload(client):
    """A valid customer returns a probability and a decision."""
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["will_churn"], bool)
    assert body["risk_band"] in {"low", "medium", "high", "critical"}


def test_predict_endpoint_missing_field_returns_422(client):
    """An incomplete payload is rejected by validation."""
    incomplete = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_endpoint_invalid_category_returns_422(client):
    """An out-of-vocabulary category is rejected."""
    bad = dict(EXAMPLE_CUSTOMER, Contract="Five year")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_negative_tenure_returns_422(client):
    """Range constraints are enforced."""
    bad = dict(EXAMPLE_CUSTOMER, tenure=-5)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_wrong_type_returns_422(client):
    """A non-numeric charge is rejected."""
    bad = dict(EXAMPLE_CUSTOMER, MonthlyCharges="expensive")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


@requires_model
def test_batch_endpoint_scores_all(client):
    """Batch scoring returns one prediction per customer."""
    payload = {"customers": [EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_batch_endpoint_empty_returns_422(client):
    """An empty batch is invalid."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422

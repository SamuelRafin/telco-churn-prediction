"""Client helper tests. Network is mocked, no server needed."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.client import (
    ApiError,
    build_payload,
    check_health,
    format_probability,
    request_prediction,
    risk_label,
)


def test_build_payload_maps_senior_yes_no():
    assert build_payload({"SeniorCitizen": "Yes"})["SeniorCitizen"] == 1
    assert build_payload({"SeniorCitizen": "No"})["SeniorCitizen"] == 0


def test_build_payload_maps_senior_bool():
    assert build_payload({"SeniorCitizen": True})["SeniorCitizen"] == 1
    assert build_payload({"SeniorCitizen": False})["SeniorCitizen"] == 0


def test_build_payload_casts_tenure_to_int():
    payload = build_payload({"tenure": 12.0})
    assert payload["tenure"] == 12
    assert isinstance(payload["tenure"], int)


def test_build_payload_rounds_charges():
    payload = build_payload({"MonthlyCharges": 70.456, "TotalCharges": 1234.5678})
    assert payload["MonthlyCharges"] == 70.46
    assert payload["TotalCharges"] == 1234.57


def test_build_payload_preserves_other_fields():
    payload = build_payload({"Contract": "Two year", "gender": "Male"})
    assert payload["Contract"] == "Two year"
    assert payload["gender"] == "Male"


def test_build_payload_does_not_mutate_input():
    original = {"SeniorCitizen": "Yes", "tenure": 5.0}
    snapshot = dict(original)
    build_payload(original)
    assert original == snapshot


def test_format_probability():
    # Avoiding values like 0.8765 here, which formats as 87.6% not 87.7% because
    # of how it's stored in binary.
    assert format_probability(0.88) == "88.0%"
    assert format_probability(0.0687) == "6.9%"
    assert format_probability(0.5) == "50.0%"
    assert format_probability(0.0) == "0.0%"
    assert format_probability(1.0) == "100.0%"


def test_risk_label_known_bands():
    for band in ("critical", "high", "medium", "low"):
        assert risk_label(band)


def test_risk_label_unknown_band_has_fallback():
    assert risk_label("unheard-of")
    assert risk_label("")


def _mock_response(status: int = 200, json_body: dict | None = None, text: str = ""):
    response = MagicMock()
    response.status_code = status
    response.ok = 200 <= status < 300
    response.json.return_value = json_body or {}
    response.text = text
    if not response.ok:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    return response


@patch("app.client.requests.get")
def test_check_health_ok(mock_get):
    mock_get.return_value = _mock_response(
        200, {"status": "ok", "model_loaded": True}
    )
    status = check_health("http://test")

    assert status.reachable
    assert status.model_loaded


@patch("app.client.requests.get")
def test_check_health_no_model(mock_get):
    """Reachable but degraded, which the sidebar shows differently."""
    mock_get.return_value = _mock_response(
        200, {"status": "degraded", "model_loaded": False, "detail": "missing"}
    )
    status = check_health("http://test")

    assert status.reachable
    assert not status.model_loaded
    assert "missing" in status.message


@patch("app.client.requests.get", side_effect=requests.exceptions.ConnectionError())
def test_check_health_unreachable(mock_get):
    status = check_health("http://test")
    assert not status.reachable
    assert "Cannot reach" in status.message


@patch("app.client.requests.get", side_effect=requests.exceptions.Timeout())
def test_check_health_timeout(mock_get):
    status = check_health("http://test")
    assert not status.reachable
    assert "timed out" in status.message.lower()


@patch("app.client.requests.post")
def test_request_prediction_success(mock_post):
    expected = {
        "churn_probability": 0.88,
        "will_churn": True,
        "threshold": 0.4549,
        "risk_band": "critical",
        "recommended_action": "Immediate outreach",
    }
    mock_post.return_value = _mock_response(200, expected)

    assert request_prediction({"tenure": 2}, "http://test") == expected


@patch("app.client.requests.post")
def test_request_prediction_strips_trailing_slash(mock_post):
    mock_post.return_value = _mock_response(200, {"churn_probability": 0.1})
    request_prediction({}, "http://test/")

    assert mock_post.call_args[0][0] == "http://test/predict"


@patch("app.client.requests.post", side_effect=requests.exceptions.ConnectionError())
def test_request_prediction_connection_error(mock_post):
    """Error should tell the user how to start the API, not just that it failed."""
    with pytest.raises(ApiError, match="uvicorn"):
        request_prediction({}, "http://test")


@patch("app.client.requests.post", side_effect=requests.exceptions.Timeout())
def test_request_prediction_timeout(mock_post):
    with pytest.raises(ApiError, match="too long"):
        request_prediction({}, "http://test")


@patch("app.client.requests.post")
def test_request_prediction_validation_error(mock_post):
    """422 should name the offending field."""
    mock_post.return_value = _mock_response(
        422,
        {"detail": [{"loc": ["body", "Contract"], "msg": "unexpected value"}]},
    )

    with pytest.raises(ApiError, match="Contract"):
        request_prediction({}, "http://test")


@patch("app.client.requests.post")
def test_request_prediction_model_unavailable(mock_post):
    mock_post.return_value = _mock_response(503, {"detail": "no model"})

    with pytest.raises(ApiError, match="churn.train"):
        request_prediction({}, "http://test")


@patch("app.client.requests.post")
def test_request_prediction_server_error(mock_post):
    mock_post.return_value = _mock_response(500, text="boom")

    with pytest.raises(ApiError, match="500"):
        request_prediction({}, "http://test")

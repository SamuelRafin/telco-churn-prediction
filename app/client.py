"""HTTP bits for the Streamlit app.

Separate from the UI so the payload building and error handling can be tested
without standing up a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 10

RISK_LABELS: dict[str, str] = {
    "critical": "Critical risk",
    "high": "High risk",
    "medium": "Medium risk",
    "low": "Low risk",
}


class ApiError(Exception):
    pass


@dataclass
class HealthStatus:
    reachable: bool
    model_loaded: bool
    message: str


def build_payload(form_values: dict[str, Any]) -> dict[str, Any]:
    """Form values into something the API will accept."""
    payload = dict(form_values)

    # Schema wants 0/1 but Yes/No reads better in a dropdown.
    senior = payload.get("SeniorCitizen")
    if isinstance(senior, str):
        payload["SeniorCitizen"] = 1 if senior == "Yes" else 0
    elif isinstance(senior, bool):
        payload["SeniorCitizen"] = int(senior)

    for field in ("tenure",):
        if field in payload:
            payload[field] = int(payload[field])

    for field in ("MonthlyCharges", "TotalCharges"):
        if field in payload:
            payload[field] = round(float(payload[field]), 2)

    return payload


def check_health(
    api_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT
) -> HealthStatus:
    """Never raises. The sidebar has to render even with the API down."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        body = response.json()
        loaded = bool(body.get("model_loaded"))
        return HealthStatus(
            reachable=True,
            model_loaded=loaded,
            message=(
                "API online, model loaded"
                if loaded
                else f"API online but no model: {body.get('detail', 'unknown')}"
            ),
        )
    except requests.exceptions.ConnectionError:
        return HealthStatus(False, False, "Cannot reach the API. Is it running?")
    except requests.exceptions.Timeout:
        return HealthStatus(False, False, "API timed out.")
    except requests.exceptions.RequestException as error:
        return HealthStatus(False, False, f"API error: {error}")


def request_prediction(
    payload: dict[str, Any],
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST to /predict. Raises ApiError with something readable on failure."""
    url = f"{api_url.rstrip('/')}/predict"

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as error:
        raise ApiError(
            "Cannot reach the API. Start it with:\n"
            "    uvicorn api.main:app --reload"
        ) from error
    except requests.exceptions.Timeout as error:
        raise ApiError("The API took too long to respond.") from error
    except requests.exceptions.RequestException as error:
        raise ApiError(f"Request failed: {error}") from error

    if response.status_code == 422:
        raise ApiError(f"The API rejected the input: {_format_validation(response)}")

    if response.status_code == 503:
        raise ApiError(
            "The API is running but no trained model is available. "
            "Run 'python -m churn.train' first."
        )

    if not response.ok:
        raise ApiError(f"API returned {response.status_code}: {response.text[:200]}")

    return response.json()


def _format_validation(response: requests.Response) -> str:
    """Flatten a FastAPI 422 body into one line."""
    try:
        details = response.json().get("detail", [])
        parts = []
        for item in details[:5]:
            location = " -> ".join(str(p) for p in item.get("loc", []) if p != "body")
            parts.append(f"{location}: {item.get('msg', 'invalid')}")
        return "; ".join(parts) if parts else response.text[:200]
    except ValueError:
        return response.text[:200]


def format_probability(probability: float) -> str:
    return f"{probability:.1%}"


def risk_label(risk_band: str) -> str:
    """Display label for a risk band, falling back for unexpected values."""
    return RISK_LABELS.get(risk_band, risk_band.title() or "Unknown")

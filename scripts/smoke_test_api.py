"""Quick check against a running API.

    uvicorn api.main:app
    python scripts/smoke_test_api.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

HIGH_RISK = {
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

LOW_RISK = dict(
    HIGH_RISK,
    tenure=70,
    Contract="Two year",
    InternetService="DSL",
    OnlineSecurity="Yes",
    TechSupport="Yes",
    PaymentMethod="Credit card (automatic)",
    MonthlyCharges=45.0,
    TotalCharges=3150.0,
)


def post(payload, path="/predict"):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {}


def main() -> int:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        health = json.loads(response.read())
    print(f"health: {health}")

    if not health.get("model_loaded"):
        print("No model loaded. Run: python -m churn.train")
        return 1

    _, high = post(HIGH_RISK)
    _, low = post(LOW_RISK)
    print(f"high risk: {high['churn_probability']:.3f} ({high['risk_band']})")
    print(f"low risk:  {low['churn_probability']:.3f} ({low['risk_band']})")

    # Sanity check: a two-year contract customer of six years shouldn't outrank
    # a two-month fiber customer on churn risk.
    if high["churn_probability"] <= low["churn_probability"]:
        print("FAIL: risk ordering is backwards")
        return 1

    status, _ = post(dict(HIGH_RISK, Contract="Five year"))
    if status != 422:
        print(f"FAIL: bad category returned {status}, expected 422")
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as error:
        print(f"Can't reach {BASE_URL}: {error}")
        print("Start it with: uvicorn api.main:app")
        sys.exit(2)

"""Streamlit front end. Needs the API up first:

    uvicorn api.main:app --reload
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.client import (  # noqa: E402
    DEFAULT_API_URL,
    ApiError,
    build_payload,
    check_health,
    request_prediction,
    risk_label,
)

PRESETS: dict[str, dict] = {
    "High-risk (new, month-to-month, fiber)": {
        "gender": "Female",
        "SeniorCitizen": "No",
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
    },
    "Low-risk (loyal, two-year contract)": {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.15,
        "TotalCharges": 4362.20,
    },
}

INTERNET_OPTIONS = ["Yes", "No", "No internet service"]


def init_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = dict(PRESETS["High-risk (new, month-to-month, fiber)"])


def sidebar() -> str:
    """Draws the sidebar, returns the API URL to use."""
    st.sidebar.header("Configuration")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

    st.sidebar.subheader("API status")
    health = check_health(api_url)

    if health.reachable and health.model_loaded:
        st.sidebar.success(health.message)
    elif health.reachable:
        st.sidebar.warning(health.message)
    else:
        st.sidebar.error(health.message)
        st.sidebar.code("uvicorn api.main:app --reload", language="bash")

    st.sidebar.divider()
    st.sidebar.subheader("Load a preset")
    for name in PRESETS:
        if st.sidebar.button(name, use_container_width=True):
            st.session_state.profile = dict(PRESETS[name])
            st.rerun()

    return api_url


def customer_form() -> dict | None:
    """Returns the form values on submit, None otherwise."""
    profile = st.session_state.profile

    def index_of(options: list, key: str) -> int:
        value = profile.get(key)
        return options.index(value) if value in options else 0

    with st.form("customer"):
        st.subheader("Customer profile")

        demographics, account = st.columns(2)

        with demographics:
            st.markdown("**Demographics**")
            gender = st.selectbox(
                "Gender", ["Female", "Male"], index=index_of(["Female", "Male"], "gender")
            )
            senior = st.selectbox(
                "Senior citizen", ["No", "Yes"], index=index_of(["No", "Yes"], "SeniorCitizen")
            )
            partner = st.selectbox(
                "Has partner", ["No", "Yes"], index=index_of(["No", "Yes"], "Partner")
            )
            dependents = st.selectbox(
                "Has dependents", ["No", "Yes"], index=index_of(["No", "Yes"], "Dependents")
            )

        with account:
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, int(profile.get("tenure", 12)))
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                index=index_of(["Month-to-month", "One year", "Two year"], "Contract"),
            )
            paperless = st.selectbox(
                "Paperless billing",
                ["Yes", "No"],
                index=index_of(["Yes", "No"], "PaperlessBilling"),
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=index_of(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    "PaymentMethod",
                ),
            )

        st.markdown("**Services**")
        phone_col, internet_col, addon_col = st.columns(3)

        with phone_col:
            phone = st.selectbox(
                "Phone service", ["Yes", "No"], index=index_of(["Yes", "No"], "PhoneService")
            )
            multiple = st.selectbox(
                "Multiple lines",
                ["Yes", "No", "No phone service"],
                index=index_of(["Yes", "No", "No phone service"], "MultipleLines"),
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "StreamingTV"),
            )

        with internet_col:
            internet = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
                index=index_of(["DSL", "Fiber optic", "No"], "InternetService"),
            )
            security = st.selectbox(
                "Online security",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "OnlineSecurity"),
            )
            backup = st.selectbox(
                "Online backup",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "OnlineBackup"),
            )

        with addon_col:
            protection = st.selectbox(
                "Device protection",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "DeviceProtection"),
            )
            support = st.selectbox(
                "Tech support",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "TechSupport"),
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                INTERNET_OPTIONS,
                index=index_of(INTERNET_OPTIONS, "StreamingMovies"),
            )

        st.markdown("**Charges**")
        monthly_col, total_col = st.columns(2)
        with monthly_col:
            monthly = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=1000.0,
                value=float(profile.get("MonthlyCharges", 70.0)),
                step=1.0,
            )
        with total_col:
            total = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(profile.get("TotalCharges", 840.0)),
                step=10.0,
            )

        submitted = st.form_submit_button(
            "Predict churn risk", type="primary", use_container_width=True
        )

    if not submitted:
        return None

    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def render_result(result: dict) -> None:
    probability = result["churn_probability"]

    st.subheader("Prediction")

    left, middle, right = st.columns(3)
    left.metric("Churn probability", f"{probability:.1%}")
    middle.metric("Risk band", risk_label(result["risk_band"]))
    right.metric(
        "Verdict",
        "Likely to churn" if result["will_churn"] else "Likely to stay",
    )

    st.progress(min(max(probability, 0.0), 1.0))

    if result["will_churn"]:
        st.error(f"**Recommended action:** {result['recommended_action']}")
    else:
        st.success(f"**Recommended action:** {result['recommended_action']}")

    st.caption(
        f"Threshold {result['threshold']:.4f}, picked to maximise recall while "
        "keeping precision at 50% or better."
    )

    with st.expander("Raw API response"):
        st.json(result)


def main() -> None:
    st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
    init_state()

    st.title("Telco Customer Churn Prediction")
    st.caption(
        "Scores a customer's churn risk via the FastAPI service. "
        "Load a preset from the sidebar if you don't fancy filling in 19 fields."
    )

    api_url = sidebar()
    form_values = customer_form()

    if form_values is None:
        st.info("Fill in the form and hit **Predict churn risk**, or load a preset from the sidebar.")
        return

    payload = build_payload(form_values)

    try:
        with st.spinner("Scoring customer..."):
            result = request_prediction(payload, api_url)
    except ApiError as error:
        st.error(str(error))
        return

    render_result(result)


if __name__ == "__main__":
    main()

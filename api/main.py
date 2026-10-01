"""FastAPI service for the churn model.

No auth on any of this. It's a local demo; anything network-facing would need
an API key, rate limiting and TLS first.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from churn.config import load_config
from churn.predict import EXAMPLE_CUSTOMER, get_model, predict

MAX_BATCH_SIZE = 500

# Literals rather than plain str so a bad category gets a 422 instead of being
# silently encoded as unknown and scored anyway.
YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """One customer, matching the Telco schema."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if the customer is a senior.")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company.")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    model_config = {"json_schema_extra": {"example": EXAMPLE_CUSTOMER}}


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Probability of churn, 0-1.")
    will_churn: bool = Field(description="True if probability >= threshold.")
    threshold: float = Field(description="Decision threshold applied.")
    risk_band: str = Field(description="low | medium | high | critical.")
    recommended_action: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_type: str
    decision_threshold: float
    n_features: int
    feature_names: list[str]


_state: dict[str, object] = {"model_loaded": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model so the first request isn't the slow one."""
    try:
        get_model()
        _state["model_loaded"] = True
    except FileNotFoundError as error:
        # Stay up anyway so /health can say what's wrong.
        _state["model_loaded"] = False
        _state["error"] = str(error)
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Predicts the probability that a telecom customer will churn. "
        "Wraps a scikit-learn pipeline (preprocessing + gradient-boosted model) "
        "with a decision threshold tuned to favour recall."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    """Model, or a 503 saying why not."""
    try:
        return get_model()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {error}",
        ) from error


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    loaded = bool(_state["model_loaded"])
    if not loaded:
        # Re-check in case training finished after the server came up.
        try:
            get_model()
            _state["model_loaded"] = True
            loaded = True
        except FileNotFoundError as error:
            _state["error"] = str(error)

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        detail=None if loaded else str(_state["error"]),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    pipeline = _require_model()
    config = load_config()

    from churn.features import get_feature_names

    names = get_feature_names(pipeline)
    return ModelInfoResponse(
        model_type=type(pipeline.named_steps["model"]).__name__,
        decision_threshold=config.model.decision_threshold,
        n_features=len(names),
        feature_names=names,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_customer(customer: CustomerFeatures) -> PredictionResponse:
    """Score a single customer's churn risk."""
    pipeline = _require_model()
    result = predict(customer.model_dump(), pipeline=pipeline)[0]
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score several customers in one request."""
    pipeline = _require_model()
    records = [customer.model_dump() for customer in request.customers]
    results = predict(records, pipeline=pipeline)

    return BatchResponse(
        predictions=[PredictionResponse(**result) for result in results],
        count=len(results),
    )


@app.get("/", tags=["ops"])
def root() -> dict[str, str]:
    return {
        "service": "Telco Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }

"""FastAPI service that serves the trained XGBoost churn pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import Customer, PredictionResponse, HealthResponse

# Reuse the exact same feature engineering used in training
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import add_features  # noqa: E402
from src.preprocessing import clean  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_xgb_pipeline.joblib"
MODEL_VERSION = "1.0.0"

_bundle = None


def _load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python -m src.train` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        print("[api] Model loaded successfully.")
    except Exception as e:  # noqa
        print(f"[api] WARNING: model not loaded at startup: {e}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serves a tuned XGBoost pipeline predicting the probability that a telecom customer churns.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _risk_band(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


@app.get("/", tags=["meta"])
def root():
    return {"message": "Customer Churn Prediction API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = MODEL_PATH.exists()
    return HealthResponse(status="ok" if loaded else "degraded",
                          model_loaded=loaded, model_version=MODEL_VERSION)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: Customer):
    try:
        bundle = _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pipeline = bundle["pipeline"]
    row = pd.DataFrame([customer.model_dump()])
    # Apply the same clean + feature engineering as training
    row = clean(row) if "Churn" in row.columns else row
    row = add_features(row)
    # Align to training columns
    for col in bundle["feature_columns"]:
        if col not in row.columns:
            row[col] = 0
    row = row[bundle["feature_columns"]]

    proba = float(pipeline.predict_proba(row)[:, 1][0])
    return PredictionResponse(
        churn=bool(proba >= 0.5),
        churn_probability=round(proba, 4),
        risk_band=_risk_band(proba),
        model_version=MODEL_VERSION,
    )

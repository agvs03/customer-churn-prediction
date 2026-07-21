"""Smoke tests for the churn API. Requires a trained model in models/."""
import os
import pytest
from fastapi.testclient import TestClient

from api.main import app, MODEL_PATH

client = TestClient(app)

MODEL_MISSING = not MODEL_PATH.exists()
skip_if_no_model = pytest.mark.skipif(MODEL_MISSING, reason="trained model not present")

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 191.4,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


@skip_if_no_model
def test_predict_shape():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn", "churn_probability", "risk_band", "model_version"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}


@skip_if_no_model
def test_high_risk_customer():
    # New fiber, month-to-month, electronic check -> should skew high risk
    r = client.post("/predict", json=SAMPLE)
    assert r.json()["churn_probability"] > 0.3

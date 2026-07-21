"""Pydantic request/response schemas for the churn API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Customer(BaseModel):
    """Raw customer record - matches the Telco dataset schema (pre-feature-eng)."""
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: Literal["Yes", "No"] = "Yes"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(1, ge=0, le=100)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: str = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(70.0, ge=0)
    TotalCharges: float = Field(70.0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.7, "TotalCharges": 191.4,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn: bool
    churn_probability: float
    risk_band: Literal["Low", "Medium", "High"]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

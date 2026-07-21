"""Feature engineering: derive 25+ predictive features from the cleaned data.

All transforms are stateless (row-wise) so the same function is reused at
training time and inside the FastAPI service for single-record inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Service columns whose "Yes" counts as an active add-on
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
STREAMING_COLS = ["StreamingTV", "StreamingMovies"]
SUPPORT_COLS = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]


def _is_yes(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.lower() == "yes").astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with 25+ engineered features appended."""
    df = df.copy()

    tenure = df["tenure"].astype(float)
    monthly = df["MonthlyCharges"].astype(float)
    total = df["TotalCharges"].astype(float)

    # 1. Binary add-on flags (6 features)
    for col in ADDON_COLS:
        df[f"has_{col}"] = _is_yes(df[col])

    # 7. Count of active add-on services
    df["num_addon_services"] = df[[f"has_{c}" for c in ADDON_COLS]].sum(axis=1)

    # 8. Count of active streaming services
    df["num_streaming_services"] = df[[f"has_{c}" for c in STREAMING_COLS]].sum(axis=1)

    # 9. Count of protection/support services
    df["num_support_services"] = df[[f"has_{c}" for c in SUPPORT_COLS]].sum(axis=1)

    # 10. Has any internet service
    df["has_internet"] = (df["InternetService"].astype(str) != "No").astype(int)

    # 11. Uses fiber optic (higher-churn segment)
    df["is_fiber"] = (df["InternetService"].astype(str) == "Fiber optic").astype(int)

    # 12. Has phone service
    df["has_phone"] = _is_yes(df["PhoneService"])

    # 13. Multiple phone lines
    df["has_multiple_lines"] = (df["MultipleLines"].astype(str) == "Yes").astype(int)

    # 14. Tenure in years
    df["tenure_years"] = tenure / 12.0

    # 15. Tenure buckets (ordinal 0-4)
    df["tenure_bucket"] = pd.cut(
        tenure, bins=[-1, 6, 12, 24, 48, 1e9], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # 16. New customer flag (<= 3 months)
    df["is_new_customer"] = (tenure <= 3).astype(int)

    # 17. Long-tenure loyal flag (>= 48 months)
    df["is_loyal"] = (tenure >= 48).astype(int)

    # 18. Charges per tenure month (spend velocity)
    df["charges_per_tenure"] = total / (tenure + 1.0)

    # 19. Expected vs actual total (billing consistency)
    df["expected_total"] = monthly * tenure
    df["total_charge_ratio"] = total / (df["expected_total"] + 1.0)

    # 20. Monthly charge relative to population mean
    df["monthly_vs_avg"] = monthly - monthly.mean()

    # 21. High monthly charge flag (top quartile)
    df["is_high_spender"] = (monthly > monthly.quantile(0.75)).astype(int)

    # 22. Average revenue per active service
    df["revenue_per_service"] = monthly / (df["num_addon_services"] + 1.0)

    # 23. Contract commitment score (0 month, 1 one-year, 2 two-year)
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_score"] = df["Contract"].map(contract_map).fillna(0).astype(int)

    # 24. Month-to-month flag (highest churn contract)
    df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)

    # 25. Automatic payment flag (lower churn)
    df["auto_payment"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)

    # 26. Electronic check flag (higher churn segment)
    df["is_electronic_check"] = (df["PaymentMethod"].astype(str) == "Electronic check").astype(int)

    # 27. Paperless billing flag
    df["is_paperless"] = _is_yes(df["PaperlessBilling"])

    # 28. Family account (partner or dependents)
    df["has_family"] = ((_is_yes(df["Partner"]) + _is_yes(df["Dependents"])) > 0).astype(int)

    # 29. Senior living alone (risk segment)
    df["senior_alone"] = ((df["SeniorCitizen"] == 1) & (df["has_family"] == 0)).astype(int)

    # 30. Services-to-charge efficiency
    df["services_per_dollar"] = (df["num_addon_services"] + df["has_phone"] + df["has_internet"]) / (monthly + 1.0)

    # 31. Interaction: fiber + month-to-month (very high churn)
    df["fiber_month_to_month"] = df["is_fiber"] * df["is_month_to_month"]

    # 32. Interaction: no support + high charge
    df["no_support_high_charge"] = ((df["num_support_services"] == 0) & (df["is_high_spender"] == 1)).astype(int)

    # 33. Lifetime value proxy
    df["clv_proxy"] = monthly * df["contract_score"].replace({0: 1, 1: 12, 2: 24})

    return df

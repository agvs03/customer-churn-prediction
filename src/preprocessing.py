"""Data cleaning for the Telco churn dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

DROP_COLS = ["customerID"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data: fix dtypes, blanks, duplicates, target encoding."""
    df = df.copy()

    # Drop identifier columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # TotalCharges arrives as object with blank strings for tenure==0 customers
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # A blank TotalCharges means the customer just joined -> 0 spend so far
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # Normalise SeniorCitizen to Yes/No categorical for consistency, keep numeric too
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != before:
        print(f"[preprocessing] Dropped {before - len(df)} duplicate rows")

    # Encode target to 0/1
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    # Impute any residual numeric NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df

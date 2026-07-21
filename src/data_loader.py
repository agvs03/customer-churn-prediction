"""Load the churn dataset.

Priority:
  1. Real IBM/Kaggle Telco CSV if present (config.data.telco_csv)
  2. Previously generated synthetic CSV (config.data.raw_csv)
  3. Generate a fresh synthetic dataset on the fly
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import CFG, PROJECT_ROOT


def load_raw() -> pd.DataFrame:
    data_cfg = CFG["data"]
    telco = PROJECT_ROOT / data_cfg["telco_csv"]
    raw = PROJECT_ROOT / data_cfg["raw_csv"]

    if telco.exists():
        print(f"[data_loader] Using real Telco dataset: {telco.name}")
        return pd.read_csv(telco)

    if raw.exists():
        print(f"[data_loader] Using cached synthetic dataset: {raw.name}")
        return pd.read_csv(raw)

    print("[data_loader] No dataset found - generating synthetic data...")
    from data.generate_data import generate
    df = generate(rows=data_cfg.get("synthetic_rows", 7043), seed=CFG.get("seed", 42))
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    return df

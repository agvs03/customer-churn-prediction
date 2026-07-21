"""Load project configuration from config.yaml with sane fallbacks."""
from __future__ import annotations

import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


CFG = load_config()
SEED = CFG.get("seed", 42)
TARGET = CFG.get("target", "Churn")

"""Pytest root configuration.

Ensures the repository root is on sys.path so tests can import the
`api` and `src` packages regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Prediction utility for the persisted best default-risk pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_preparation import RAW_MODEL_FEATURES

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def load_best_pipeline(model_path: Path | str):
    """Load the saved preprocessing-and-classifier pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path.resolve()}")
    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved object is not a probabilistic classifier.")
    return model


def predict_default_risk(
    new_loans: pd.DataFrame,
    model_path: Path | str,
) -> pd.DataFrame:
    """Return default class, probability, and risk band for new loans."""
    if not isinstance(new_loans, pd.DataFrame):
        raise TypeError("new_loans must be a pandas DataFrame.")
    if new_loans.empty:
        raise ValueError("new_loans must contain at least one row.")

    missing = sorted(set(RAW_MODEL_FEATURES) - set(new_loans.columns))
    if missing:
        raise ValueError("Missing required origination columns: " + ", ".join(missing))

    model = load_best_pipeline(model_path)
    probabilities = model.predict_proba(new_loans)[:, 1]
    predictions = model.predict(new_loans).astype(int)
    categories = np.select(
        [
            probabilities < LOW_RISK_THRESHOLD,
            probabilities < HIGH_RISK_THRESHOLD,
        ],
        ["Low risk", "Medium risk"],
        default="High risk",
    )

    return pd.DataFrame(
        {
            "predicted_class": predictions,
            "probability_of_default": probabilities,
            "risk_category": categories,
        },
        index=new_loans.index,
    )

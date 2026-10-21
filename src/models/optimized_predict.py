"""Inference utility for the calibrated optimized default-risk model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimized_evaluation import OptimizedRiskModel


def predict_optimized_default_risk(
    loans: pd.DataFrame,
    model_path: Path | str,
) -> pd.DataFrame:
    """Return calibrated risk estimates using the saved locked threshold."""
    model = OptimizedRiskModel.load(model_path)
    return model.predict_frame(loans)

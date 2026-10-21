"""Train, evaluate, and persist the default-risk baseline models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_preparation import (  # noqa: E402
    ADDITIONAL_EXCLUDED_COLUMNS,
    FINAL_MODEL_FEATURES,
    LEAKAGE_AND_IDENTIFIER_COLUMNS,
    RANDOM_STATE,
    RAW_MODEL_FEATURES,
    TARGET_COLUMN,
    format_target_distribution,
    prepare_data,
)
from src.models.evaluate_models import (  # noqa: E402
    evaluate_models,
    save_best_model,
    save_training_metadata,
    write_supervisor_summary,
)
from src.models.train_models import train_models  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv"


def parse_args() -> argparse.Namespace:
    """Parse the optional cleaned-data path."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate default-risk models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned CSV dataset.",
    )
    return parser.parse_args()


def main(data_path: Path | str | None = None) -> None:
    """Run the complete leakage-safe baseline training workflow."""
    selected_data_path = Path(data_path) if data_path else parse_args().data_path
    models_dir = PROJECT_ROOT / "models"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    results_dir = PROJECT_ROOT / "reports" / "model_results"

    print(f"Loading: {selected_data_path}")
    prepared = prepare_data(selected_data_path)
    print("Target distribution:")
    print(format_target_distribution(prepared.target_distribution))
    print(f"Split: {len(prepared.X_train):,} train / " f"{len(prepared.X_test):,} test")
    print("Final model features: " + ", ".join(FINAL_MODEL_FEATURES))

    fitted_models = train_models(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        preprocessor=prepared.preprocessor,
        models_dir=models_dir,
    )
    comparison = evaluate_models(
        fitted_models=fitted_models,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        figures_dir=figures_dir,
        model_results_dir=results_dir,
    )
    best_model_name = save_best_model(
        fitted_models=fitted_models,
        comparison=comparison,
        output_path=models_dir / "best_default_model.joblib",
    )

    metadata = {
        "data_path": str(selected_data_path.resolve()),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "train_rows": len(prepared.X_train),
        "test_rows": len(prepared.X_test),
        "raw_model_features": RAW_MODEL_FEATURES,
        "final_model_features": FINAL_MODEL_FEATURES,
        "numerical_features": prepared.numerical_features,
        "categorical_features": prepared.categorical_features,
        "excluded_dataset_columns": prepared.excluded_columns,
        "required_leakage_exclusions": LEAKAGE_AND_IDENTIFIER_COLUMNS,
        "additional_conservative_exclusions": ADDITIONAL_EXCLUDED_COLUMNS,
        "best_model": best_model_name,
        "selection_rule": "ROC-AUC descending, then recall descending",
    }
    save_training_metadata(results_dir / "training_metadata.json", metadata)
    write_supervisor_summary(
        comparison=comparison,
        best_model_name=best_model_name,
        target_distribution=prepared.target_distribution,
        output_path=results_dir / "default_prediction_summary.md",
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected model: {best_model_name}")
    print(f"Reports: {results_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()

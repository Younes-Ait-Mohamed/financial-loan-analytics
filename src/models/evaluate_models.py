"""Evaluation and reporting for fitted default-risk classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

plt.switch_backend("Agg")


def _slugify(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def _clean_feature_names(pipeline: Pipeline) -> list[str]:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [name.split("__", maxsplit=1)[-1] for name in names]


def _comparison_to_markdown(comparison: pd.DataFrame) -> str:
    """Format the compact comparison table without optional dependencies."""
    columns = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in comparison.loc[:, columns].itertuples(index=False):
        values = [str(row.model)] + [f"{value:.4f}" for value in row[1:]]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _save_confusion_matrix(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    figures_dir: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No default", "Default"],
    )
    display.plot(cmap="Blues", values_format=",d", colorbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"confusion_matrix_{_slugify(model_name)}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def _save_feature_scores(
    model_name: str,
    pipeline: Pipeline,
    model_results_dir: Path,
    figures_dir: Path,
) -> None:
    classifier = pipeline.named_steps["classifier"]
    feature_names = _clean_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        score_name = "importance"
        scores = classifier.feature_importances_
        file_stem = f"feature_importance_{_slugify(model_name)}"
        title = f"{model_name} - Feature Importance"
    elif hasattr(classifier, "coef_"):
        score_name = "coefficient"
        scores = classifier.coef_[0]
        file_stem = "logistic_regression_coefficients"
        title = "Logistic Regression - Coefficients"
    else:
        return

    score_table = pd.DataFrame({"feature": feature_names, score_name: scores})
    score_table["absolute_score"] = score_table[score_name].abs()
    score_table = score_table.sort_values(
        "absolute_score", ascending=False
    ).reset_index(drop=True)
    score_table.to_csv(model_results_dir / f"{file_stem}.csv", index=False)

    top_scores = score_table.head(20).sort_values(score_name)
    colors = (
        ["#2C7FB8"] * len(top_scores)
        if score_name == "importance"
        else ["#D95F0E" if value < 0 else "#2C7FB8" for value in top_scores[score_name]]
    )
    plt.figure(figsize=(9, 6))
    plt.barh(top_scores["feature"], top_scores[score_name], color=colors)
    plt.title(title)
    plt.xlabel(score_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{file_stem}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def evaluate_models(
    fitted_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path | str,
    model_results_dir: Path | str,
) -> pd.DataFrame:
    """Evaluate all models and save tabular and visual diagnostics."""
    if not fitted_models:
        raise ValueError("No fitted models were supplied for evaluation.")

    figure_path = Path(figures_dir)
    results_path = Path(model_results_dir)
    figure_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    curve_data: dict[str, dict[str, np.ndarray]] = {}

    for model_name, pipeline in fitted_models.items():
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

        report = pd.DataFrame(
            classification_report(
                y_test,
                predictions,
                target_names=["No default", "Default"],
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        report.to_csv(
            results_path / f"classification_report_{_slugify(model_name)}.csv"
        )

        _save_confusion_matrix(model_name, y_test, predictions, figure_path)
        _save_feature_scores(model_name, pipeline, results_path, figure_path)

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision_values, recall_values, _ = precision_recall_curve(
            y_test, probabilities
        )
        curve_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision_values,
            "recall": recall_values,
        }

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "recall"], ascending=[False, False])
        .reset_index(drop=True)
    )
    comparison.to_csv(results_path / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 6))
    for row in comparison.itertuples(index=False):
        values = curve_data[row.model]
        plt.plot(
            values["fpr"],
            values["tpr"],
            label=f"{row.model} (AUC={row.roc_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Default Prediction - ROC Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_path / "roc_curve_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    baseline = float(y_test.mean())
    plt.figure(figsize=(8, 6))
    for row in comparison.itertuples(index=False):
        values = curve_data[row.model]
        plt.plot(
            values["recall"],
            values["precision"],
            label=row.model,
        )
    plt.axhline(
        baseline,
        linestyle="--",
        color="grey",
        label=f"Class prevalence ({baseline:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Default Prediction - Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_path / "precision_recall_curve_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    return comparison


def save_best_model(
    fitted_models: dict[str, Pipeline],
    comparison: pd.DataFrame,
    output_path: Path | str,
) -> str:
    """Persist the model ranked first by ROC-AUC, then recall."""
    if comparison.empty:
        raise ValueError("The model comparison table is empty.")
    best_model_name = str(comparison.iloc[0]["model"])
    if best_model_name not in fitted_models:
        raise KeyError(f"Fitted model not found: {best_model_name}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_model_name], destination)
    return best_model_name


def save_training_metadata(
    output_path: Path | str,
    metadata: dict[str, object],
) -> None:
    """Save reproducibility and leakage-audit metadata as JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def write_supervisor_summary(
    comparison: pd.DataFrame,
    best_model_name: str,
    target_distribution: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Write a concise, presentation-ready Markdown model summary."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    best = comparison.loc[comparison["model"] == best_model_name].iloc[0]
    default_share = float(
        target_distribution.loc[target_distribution["class"] == 1, "proportion"].iloc[0]
    )
    result_table = _comparison_to_markdown(comparison)

    summary = f"""# Default-Risk Modelling Summary

## 1. Business objective

Estimate each loan's probability of default within 12 months using information
available at or near origination. The score can support credit decisions,
portfolio monitoring, and risk-based review.

## 2. Binary target

`has_default_within_12_months` equals `1` for a default within 12 months and `0`
otherwise. Rows with an unknown target were excluded from supervised training.
The observed default share is {default_share:.2%}.

## 3. Leakage prevention

Identifiers, loan status, repayment activity, balances, arrears, default timing,
early repayment, and every other post-issuance outcome field were excluded.
`projected_npv_return` was also excluded because its calculation timestamp has
not been verified. The retained raw predictors are country, issuance date,
issued amount, initial interest rate, initial duration, combined income, and
customer risk rating.

## 4. Preprocessing pipeline

Issuance year and month are derived from `loan_issued_at`. Numerical values use
median imputation followed by standardisation. Categorical values use
most-frequent imputation and one-hot encoding with unknown-category handling.
Preprocessing is fitted only on the training set inside each saved pipeline.

## 5. Models selected

Logistic Regression provides an explainable linear benchmark. Random Forest
captures nonlinear interactions through many independent trees. Gradient
Boosting builds trees sequentially to correct previous errors and is a strong
tabular-data benchmark.

## 6. Evaluation

All models use the same stratified 80/20 split and untouched test set. Accuracy
is reported but is insufficient for an imbalanced credit-risk target. Recall
measures how many actual defaults are detected; a false negative is a future
defaulter predicted as safe. Precision limits false positives, where a
non-defaulter is flagged as risky. F1 balances precision and recall, while
ROC-AUC measures probability ranking across thresholds.

## 7. Results

{result_table}

## 8. Selected model

**{best_model_name}** was selected because it ranked first by ROC-AUC and then
recall. Its test ROC-AUC is {best.roc_auc:.4f}, recall is {best.recall:.4f}, and
F1-score is {best.f1_score:.4f}.

## 9. Limitations and next steps

This is an untuned baseline using a random stratified split and a default 0.50
decision threshold. Next steps are out-of-time validation, threshold selection
based on financial costs, probability calibration, controlled hyperparameter
tuning, fairness checks, explainability, and dashboard integration.
"""
    destination.write_text(summary, encoding="utf-8")

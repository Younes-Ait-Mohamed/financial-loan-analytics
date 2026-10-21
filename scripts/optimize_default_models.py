"""Run the professional time-aware default-risk optimization workflow."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.optimized_evaluation import OptimizedRiskModel  # noqa: E402
from src.models.optimized_evaluation import (  # noqa: E402
    compare_calibrators,
    extract_model_scores,
    operational_metrics,
    ranking_metrics,
    select_threshold_policies,
    threshold_analysis,
)
from src.models.optimized_features import (  # noqa: E402
    CONSERVATIVE_EXCLUSIONS,
    FEATURE_CONFIGURATIONS,
    POST_ORIGINATION_EXCLUSIONS,
    SAFE_RAW_FEATURES,
    TARGET_COLUMN,
    required_raw_features,
)
from src.models.optimized_training import MODEL_COMPLEXITY  # noqa: E402
from src.models.optimized_training import (  # noqa: E402
    CandidateConfiguration,
    build_optimized_pipeline,
    candidate_to_dict,
    evaluate_candidate_cv,
    feature_screening_candidates,
    load_temporal_partitions,
    positive_class_ratio,
    rolling_monthly_folds,
    screening_candidates,
    tuning_candidates,
)

plt.switch_backend("Agg")

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "optimized_model_results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "optimized_figures"
MODELS_DIR = PROJECT_ROOT / "models" / "optimized"
BEST_MODEL_PATH = MODELS_DIR / "best_optimized_default_model.joblib"


def parse_args() -> argparse.Namespace:
    """Parse reproducible experiment options."""
    parser = argparse.ArgumentParser(
        description="Optimize leakage-safe loan default models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Cleaned loan dataset CSV.",
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=6,
        help="Random configurations per finalist model family.",
    )
    parser.add_argument(
        "--top-models",
        type=int,
        default=4,
        help="Number of screened model families to tune.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
        help="Bootstrap samples for selected-model confidence intervals.",
    )
    parser.add_argument(
        "--resume-screening",
        action="store_true",
        help="Reuse completed feature/model screening CSVs.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_cv_candidates(
    candidates: list[CandidateConfiguration],
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    stage_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a controlled list of candidates and retain every fold result."""
    summaries = []
    fold_tables = []
    total = len(candidates)
    for number, candidate in enumerate(candidates, start=1):
        print(
            f"[{stage_name} {number}/{total}] "
            f"{candidate.model_name} | {candidate.feature_configuration}"
        )
        try:
            summary, fold_table = evaluate_candidate_cv(
                candidate=candidate,
                train=train,
                folds=folds,
            )
            summary["status"] = "completed"
            summary["candidate_number"] = number
            summaries.append(summary)
            fold_table.insert(0, "candidate_number", number)
            fold_table.insert(1, "model", candidate.model_name)
            fold_table.insert(
                2,
                "feature_configuration",
                candidate.feature_configuration,
            )
            fold_tables.append(fold_table)
            print(
                "  "
                f"PR-AUC={summary['cv_pr_auc']:.4f}, "
                f"ROC-AUC={summary['cv_roc_auc']:.4f}, "
                f"score={summary['stability_adjusted_score']:.4f}"
            )
        except Exception as error:  # noqa: BLE001
            summaries.append(
                {
                    "model": candidate.model_name,
                    "feature_configuration": candidate.feature_configuration,
                    "scaling": candidate.scaling,
                    "parameters": json.dumps(
                        candidate.model_parameters,
                        sort_keys=True,
                        default=str,
                    ),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "candidate_number": number,
                }
            )
            print(f"  FAILED: {type(error).__name__}: {error}")

    summary_table = pd.DataFrame(summaries)
    successful = summary_table[summary_table["status"] == "completed"].copy()
    if successful.empty:
        raise RuntimeError(f"No successful candidates in stage: {stage_name}")
    successful = successful.sort_values(
        "stability_adjusted_score",
        ascending=False,
    )
    failed = summary_table[summary_table["status"] != "completed"]
    summary_table = pd.concat([successful, failed], ignore_index=True)
    fold_table = (
        pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    )
    return summary_table, fold_table


def parse_parameters(value: str) -> dict[str, Any]:
    """Parse parameters persisted in a CV summary table."""
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def candidate_from_row(row: pd.Series) -> CandidateConfiguration:
    """Reconstruct a candidate from a tuning result row."""
    return CandidateConfiguration(
        model_name=str(row["model"]),
        feature_configuration=str(row["feature_configuration"]),
        model_parameters=parse_parameters(str(row["parameters"])),
        scaling=str(row["scaling"]),
    )


def pareto_frontier(
    leaderboard: pd.DataFrame,
    metrics: list[str],
) -> pd.Series:
    """Identify candidates not dominated across all requested metrics."""
    values = leaderboard[metrics].to_numpy(dtype="float64")
    is_pareto = np.ones(len(leaderboard), dtype=bool)
    for index, row in enumerate(values):
        if not is_pareto[index]:
            continue
        dominates = np.all(values >= row, axis=1) & np.any(
            values > row,
            axis=1,
        )
        if dominates.any():
            is_pareto[index] = False
    return pd.Series(is_pareto, index=leaderboard.index)


def choose_locked_model(leaderboard: pd.DataFrame) -> str:
    """Select on validation only, preferring simplicity for near ties."""
    candidates = leaderboard[leaderboard["pareto_optimal"]].copy()
    top_score = float(candidates["deployment_score"].max())
    near_best = candidates[candidates["deployment_score"] >= top_score - 0.005].copy()
    near_best["complexity_rank"] = near_best["model"].map(MODEL_COMPLEXITY)
    selected = near_best.sort_values(
        ["complexity_rank", "deployment_score"],
        ascending=[True, False],
    ).iloc[0]
    return str(selected["model"])


def bootstrap_confidence_intervals(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int,
) -> pd.DataFrame:
    """Calculate stratified bootstrap intervals at the locked threshold."""
    y_array = np.asarray(y_true, dtype="int8")
    positive_indices = np.flatnonzero(y_array == 1)
    negative_indices = np.flatnonzero(y_array == 0)
    rng = np.random.default_rng(42)
    metrics = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
        "balanced_accuracy",
        "mcc",
    ]
    samples = {metric: [] for metric in metrics}

    for _ in range(iterations):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )
        indices = np.concatenate([sampled_positive, sampled_negative])
        result = operational_metrics(
            y_array[indices],
            probabilities[indices],
            threshold,
        )
        for metric in metrics:
            samples[metric].append(float(result[metric]))

    rows = []
    for metric, values in samples.items():
        rows.append(
            {
                "metric": metric,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.50)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate selected-model stability across meaningful test subgroups."""
    frame = test.copy()
    frame["probability"] = probabilities
    frame["issue_month"] = frame["loan_issued_at"].dt.to_period("M").astype(str)
    frame["income_availability"] = np.where(
        frame["combined_income"].isna(),
        "Missing income",
        "Income available",
    )
    frame["loan_size_band"] = pd.cut(
        frame["issued_amount"],
        bins=[-np.inf, 500, 1500, 3000, 6000, np.inf],
        labels=["≤€500", "€501–1,500", "€1,501–3,000", "€3,001–6,000", ">€6,000"],
    ).astype(str)

    rows = []
    grouping_columns = [
        "issue_month",
        "country",
        "customer_risk_rating",
        "income_availability",
        "loan_size_band",
    ]
    for grouping in grouping_columns:
        for group, subset in frame.groupby(grouping, dropna=False):
            if len(subset) < 100 or subset[TARGET_COLUMN].nunique() < 2:
                continue
            indices = frame.index.get_indexer(subset.index)
            result = operational_metrics(
                subset[TARGET_COLUMN],
                probabilities[indices],
                threshold,
            )
            rows.append(
                {
                    "grouping": grouping,
                    "group": str(group),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def save_screening_plot(
    table: pd.DataFrame,
    path: Path,
    title: str,
    label_column: str,
) -> None:
    """Save paired CV ROC-AUC and PR-AUC bars."""
    completed = table[table["status"] == "completed"].copy()
    completed = completed.sort_values("stability_adjusted_score")
    labels = completed[label_column].astype(str)
    positions = np.arange(len(completed))
    height = max(4.5, len(completed) * 0.55)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(
        positions - 0.17,
        completed["cv_roc_auc"],
        height=0.32,
        label="ROC-AUC",
        color="#2E86AB",
    )
    ax.barh(
        positions + 0.17,
        completed["cv_pr_auc"],
        height=0.32,
        label="PR-AUC",
        color="#D1495B",
    )
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Cross-validation score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_threshold_plot(
    table: pd.DataFrame,
    selected_threshold: float,
    path: Path,
) -> None:
    """Save validation threshold trade-offs for the locked model."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "MCC"),
    ]:
        ax.plot(table["threshold"], table[metric], label=label, linewidth=1.8)
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Locked threshold: {selected_threshold:.1%}",
    )
    ax.set(
        title="Validation threshold trade-offs",
        xlabel="Decision threshold",
        ylabel="Metric value",
        xlim=(0, min(1.0, float(table["threshold"].quantile(0.99)))),
        ylim=(0, 1),
    )
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    selected_name: str,
    validation_row: pd.Series,
    test_row: pd.Series,
    feature_name: str,
    model_path: Path,
) -> None:
    """Write a concise supervisor-facing summary."""
    summary = f"""# Optimized Default-Risk Model Summary

## Scientific design

- Source-verified origination fields only; `nr_of_payments` and every lifecycle
  field are excluded as leakage.
- Expanding monthly cross-validation within January–September 2024.
- October–November 2024 fits probability-calibration candidates.
- December 2024 selects the calibration method and operating threshold.
- January–27 April 2025 is evaluated once as the untouched out-of-time test.

## Selected configuration

- Model: **{selected_name}**
- Feature configuration: **{feature_name}**
- Calibration: **{validation_row['calibration_method']}**
- Locked threshold: **{validation_row['selected_threshold']:.2%}**

Selection was made on validation data using the Pareto frontier, a pre-registered
multi-metric deployment score, temporal stability, and a simplicity preference
for near ties.

## Untouched 2025 test result

| Metric | Value |
| --- | ---: |
| Accuracy | {test_row['accuracy']:.4f} |
| Precision | {test_row['precision']:.4f} |
| Recall | {test_row['recall']:.4f} |
| F1-score | {test_row['f1_score']:.4f} |
| Balanced accuracy | {test_row['balanced_accuracy']:.4f} |
| MCC | {test_row['mcc']:.4f} |
| ROC-AUC | {test_row['roc_auc']:.4f} |
| PR-AUC | {test_row['pr_auc']:.4f} |
| PR-AUC lift | {test_row['pr_auc_lift']:.2f}x |

Confusion matrix: TN={int(test_row['true_negatives']):,},
FP={int(test_row['false_positives']):,},
FN={int(test_row['false_negatives']):,},
TP={int(test_row['true_positives']):,}.

## Deployment interpretation

This model ranks risk among loans that were already issued by Bondora. It is not
validated on rejected applications and should not be presented as a universal
loan-approval model. Interest rate and Bondora's existing risk rating are suitable
only for an at-origination/post-offer score. Country requires governance review.

Saved bundle: `{model_path}`
"""
    path.write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for directory in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("Loading chronological partitions...")
    partitions = load_temporal_partitions(args.data_path)
    partitions.split_summary.to_csv(
        RESULTS_DIR / "temporal_split_summary.csv",
        index=False,
    )
    print(partitions.split_summary.to_string(index=False))

    train = partitions.train.reset_index(drop=True)
    folds = rolling_monthly_folds(train)
    class_ratio = positive_class_ratio(train[TARGET_COLUMN])

    feature_summary_path = RESULTS_DIR / "feature_configuration_screening.csv"
    feature_folds_path = RESULTS_DIR / "feature_configuration_folds.csv"
    model_summary_path = RESULTS_DIR / "model_family_screening.csv"
    model_folds_path = RESULTS_DIR / "model_family_folds.csv"
    can_resume = args.resume_screening and all(
        path.exists()
        for path in [
            feature_summary_path,
            feature_folds_path,
            model_summary_path,
            model_folds_path,
        ]
    )

    if can_resume:
        print("\nStages 1–2: reusing completed screening results")
        feature_summary = pd.read_csv(feature_summary_path)
        feature_folds = pd.read_csv(feature_folds_path)
        model_summary = pd.read_csv(model_summary_path)
        model_folds = pd.read_csv(model_folds_path)
    else:
        print("\nStage 1: controlled feature-set screening")
        feature_summary, feature_folds = run_cv_candidates(
            candidates=feature_screening_candidates(),
            train=train,
            folds=folds,
            stage_name="feature",
        )
        feature_summary.to_csv(feature_summary_path, index=False)
        feature_folds.to_csv(feature_folds_path, index=False)

    successful_features = feature_summary[feature_summary["status"] == "completed"]
    top_feature_names = (
        successful_features.head(2)["feature_configuration"].astype(str).tolist()
    )
    selected_screen_feature = top_feature_names[0]
    print(f"Selected screening feature set: {selected_screen_feature}")

    if not can_resume:
        print("\nStage 2: broad algorithm screening")
        model_summary, model_folds = run_cv_candidates(
            candidates=screening_candidates(selected_screen_feature),
            train=train,
            folds=folds,
            stage_name="model",
        )
        model_summary.to_csv(model_summary_path, index=False)
        model_folds.to_csv(model_folds_path, index=False)

    successful_models = model_summary[model_summary["status"] == "completed"]
    top_model_names = (
        successful_models.head(args.top_models)["model"].astype(str).tolist()
    )
    if "Logistic Regression" not in top_model_names:
        top_model_names.append("Logistic Regression")
    top_model_names = list(dict.fromkeys(top_model_names))
    print("Finalist families: " + ", ".join(top_model_names))

    print("\nStage 3: randomized hyperparameter and feature tuning")
    tuning_summaries = []
    tuning_folds = []
    best_candidates: dict[str, CandidateConfiguration] = {}
    for model_name in top_model_names:
        print(f"\nTuning {model_name}")
        candidates = tuning_candidates(
            model_name=model_name,
            feature_configurations=top_feature_names,
            class_ratio=class_ratio,
            iterations=args.tuning_iterations,
        )
        summary, fold_table = run_cv_candidates(
            candidates=candidates,
            train=train,
            folds=folds,
            stage_name=f"tune {_slugify(model_name)}",
        )
        summary.insert(0, "tuned_family", model_name)
        fold_table.insert(0, "tuned_family", model_name)
        tuning_summaries.append(summary)
        tuning_folds.append(fold_table)
        best_row = summary[summary["status"] == "completed"].iloc[0]
        best_candidates[model_name] = candidate_from_row(best_row)

    tuning_summary = pd.concat(tuning_summaries, ignore_index=True)
    tuning_fold_table = pd.concat(tuning_folds, ignore_index=True)
    tuning_summary.to_csv(
        RESULTS_DIR / "hyperparameter_tuning_results.csv",
        index=False,
    )
    tuning_fold_table.to_csv(
        RESULTS_DIR / "hyperparameter_tuning_folds.csv",
        index=False,
    )

    print("\nStage 4: validation, calibration, and threshold optimization")
    X_train = train[SAFE_RAW_FEATURES]
    y_train = train[TARGET_COLUMN]
    validation = partitions.validation
    calibration = partitions.calibration
    selection = partitions.threshold_selection

    validation_rows = []
    finalist_objects: dict[str, dict[str, Any]] = {}
    for model_name, candidate in best_candidates.items():
        print(f"Fitting finalist: {model_name}")
        pipeline = build_optimized_pipeline(candidate, class_ratio)
        fit_start = perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        validation_scores, score_type = extract_model_scores(
            pipeline,
            validation[SAFE_RAW_FEATURES],
        )
        validation_ranking = ranking_metrics(
            validation[TARGET_COLUMN],
            validation_scores,
        )
        calibration_scores, _ = extract_model_scores(
            pipeline,
            calibration[SAFE_RAW_FEATURES],
        )
        selection_scores, _ = extract_model_scores(
            pipeline,
            selection[SAFE_RAW_FEATURES],
        )
        calibrator, calibration_table = compare_calibrators(
            score_type=score_type,
            calibration_scores=calibration_scores,
            calibration_target=calibration[TARGET_COLUMN],
            selection_scores=selection_scores,
            selection_target=selection[TARGET_COLUMN],
        )
        calibrated_selection = calibrator.transform(selection_scores)
        threshold_table = threshold_analysis(
            selection[TARGET_COLUMN],
            calibrated_selection,
        )
        threshold_policies = select_threshold_policies(threshold_table)
        selected_threshold = float(
            threshold_policies.loc["balanced_composite", "threshold"]
        )
        selection_metrics = operational_metrics(
            selection[TARGET_COLUMN],
            calibrated_selection,
            selected_threshold,
        )
        mcc_scaled = (float(selection_metrics["mcc"]) + 1.0) / 2.0
        deployment_score = (
            0.35 * validation_ranking["normalized_pr_auc"]
            + 0.20 * validation_ranking["roc_auc"]
            + 0.20 * mcc_scaled
            + 0.15 * float(selection_metrics["balanced_accuracy"])
            + 0.10 * float(selection_metrics["f1_score"])
        )

        inference_start = perf_counter()
        extract_model_scores(
            pipeline,
            validation[SAFE_RAW_FEATURES].head(10_000),
        )
        inference_seconds_10k = perf_counter() - inference_start

        row = {
            "model": model_name,
            "feature_configuration": candidate.feature_configuration,
            "parameters": json.dumps(
                candidate.model_parameters,
                sort_keys=True,
                default=str,
            ),
            "scaling": candidate.scaling,
            "calibration_method": calibrator.method,
            "selected_threshold": selected_threshold,
            "fit_seconds": fit_seconds,
            "inference_seconds_10k": inference_seconds_10k,
            "deployment_score": deployment_score,
            **{f"validation_{key}": value for key, value in validation_ranking.items()},
            **{f"selection_{key}": value for key, value in selection_metrics.items()},
        }
        validation_rows.append(row)
        slug = _slugify(model_name)
        calibration_table.to_csv(
            RESULTS_DIR / f"calibration_comparison_{slug}.csv",
            index=False,
        )
        threshold_table.to_csv(
            RESULTS_DIR / f"threshold_analysis_{slug}.csv",
            index=False,
        )
        threshold_policies.to_csv(RESULTS_DIR / f"threshold_policies_{slug}.csv")
        finalist_objects[model_name] = {
            "candidate": candidate,
            "pipeline": pipeline,
            "calibrator": calibrator,
            "threshold": selected_threshold,
            "threshold_table": threshold_table,
            "validation_scores": validation_scores,
        }

    validation_leaderboard = pd.DataFrame(validation_rows)
    validation_leaderboard["pareto_optimal"] = pareto_frontier(
        validation_leaderboard,
        [
            "validation_roc_auc",
            "validation_pr_auc",
            "selection_f1_score",
            "selection_balanced_accuracy",
            "selection_mcc",
        ],
    )
    validation_leaderboard = validation_leaderboard.sort_values(
        "deployment_score",
        ascending=False,
    ).reset_index(drop=True)
    selected_model_name = choose_locked_model(validation_leaderboard)
    validation_leaderboard["locked_selection"] = (
        validation_leaderboard["model"] == selected_model_name
    )
    validation_leaderboard.to_csv(
        RESULTS_DIR / "validation_model_leaderboard.csv",
        index=False,
    )

    selected_object = finalist_objects[selected_model_name]
    selection_manifest = {
        "locked_before_test": True,
        "selected_model": selected_model_name,
        "candidate": candidate_to_dict(selected_object["candidate"]),
        "calibration_method": selected_object["calibrator"].method,
        "threshold": selected_object["threshold"],
        "selection_rule": (
            "Pareto frontier plus pre-registered deployment score; "
            "prefer lower complexity within 0.005"
        ),
        "selection_metrics": validation_leaderboard[
            validation_leaderboard["model"] == selected_model_name
        ]
        .iloc[0]
        .to_dict(),
    }
    save_json(RESULTS_DIR / "locked_selection_before_test.json", selection_manifest)
    print(
        f"Locked before test: {selected_model_name} at "
        f"{selected_object['threshold']:.2%}"
    )

    print("\nStage 5: one-time untouched 2025 test evaluation")
    test = partitions.test
    test_rows = []
    test_probabilities: dict[str, np.ndarray] = {}
    for model_name, artifact in finalist_objects.items():
        raw_scores, _ = extract_model_scores(
            artifact["pipeline"],
            test[SAFE_RAW_FEATURES],
        )
        probabilities = artifact["calibrator"].transform(raw_scores)
        test_probabilities[model_name] = probabilities
        metrics = operational_metrics(
            test[TARGET_COLUMN],
            probabilities,
            artifact["threshold"],
        )
        test_rows.append(
            {
                "model": model_name,
                "locked_selection": model_name == selected_model_name,
                "calibration_method": artifact["calibrator"].method,
                **metrics,
            }
        )

    test_leaderboard = pd.DataFrame(test_rows).sort_values(
        "locked_selection",
        ascending=False,
    )
    test_leaderboard.to_csv(
        RESULTS_DIR / "untouched_test_leaderboard.csv",
        index=False,
    )
    selected_test_row = test_leaderboard[
        test_leaderboard["model"] == selected_model_name
    ].iloc[0]
    selected_validation_row = validation_leaderboard[
        validation_leaderboard["model"] == selected_model_name
    ].iloc[0]

    selected_candidate = selected_object["candidate"]
    selected_feature_config = FEATURE_CONFIGURATIONS[
        selected_candidate.feature_configuration
    ]
    bundle = OptimizedRiskModel(
        pipeline=selected_object["pipeline"],
        calibrator=selected_object["calibrator"],
        threshold=float(selected_object["threshold"]),
        model_name=selected_model_name,
        raw_features=required_raw_features(selected_feature_config),
        feature_configuration=asdict(selected_feature_config),
        metadata={
            "target": TARGET_COLUMN,
            "test_period": ["2025-01-01", "2025-04-27"],
            "training_period": ["2024-01-01", "2024-09-30"],
            "calibration_period": ["2024-10-01", "2024-11-30"],
            "threshold_period": ["2024-12-01", "2024-12-31"],
            "selected_before_test": True,
        },
    )
    bundle.save(BEST_MODEL_PATH)

    selected_probabilities = test_probabilities[selected_model_name]
    confidence_intervals = bootstrap_confidence_intervals(
        y_true=test[TARGET_COLUMN],
        probabilities=selected_probabilities,
        threshold=bundle.threshold,
        iterations=args.bootstrap_iterations,
    )
    confidence_intervals.to_csv(
        RESULTS_DIR / "selected_model_bootstrap_intervals.csv",
        index=False,
    )
    subgroup_table = subgroup_metrics(
        test=test,
        probabilities=selected_probabilities,
        threshold=bundle.threshold,
    )
    subgroup_table.to_csv(
        RESULTS_DIR / "selected_model_subgroup_metrics.csv",
        index=False,
    )

    print("\nStage 6: interpretation and figures")
    importance_sample = partitions.validation.sample(
        n=min(20_000, len(partitions.validation)),
        random_state=42,
    )
    importance = permutation_importance(
        selected_object["pipeline"],
        importance_sample[SAFE_RAW_FEATURES],
        importance_sample[TARGET_COLUMN],
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "raw_feature": SAFE_RAW_FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(
        RESULTS_DIR / "selected_model_permutation_importance.csv",
        index=False,
    )

    save_screening_plot(
        feature_summary,
        FIGURES_DIR / "feature_configuration_screening.png",
        "Feature Configuration - Rolling Temporal CV",
        "feature_configuration",
    )
    save_screening_plot(
        model_summary,
        FIGURES_DIR / "model_family_screening.png",
        "Model Family - Rolling Temporal CV",
        "model",
    )
    save_threshold_plot(
        selected_object["threshold_table"],
        bundle.threshold,
        FIGURES_DIR / "selected_model_threshold_tradeoff.png",
    )

    predictions = (selected_probabilities >= bundle.threshold).astype("int8")
    ConfusionMatrixDisplay.from_predictions(
        test[TARGET_COLUMN],
        predictions,
        display_labels=["No default", "Default"],
        cmap="Blues",
        values_format=",d",
        colorbar=False,
    )
    plt.title(f"{selected_model_name} - Untouched 2025 Test")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "selected_model_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probabilities in test_probabilities.items():
        RocCurveDisplay.from_predictions(
            test[TARGET_COLUMN],
            probabilities,
            name=model_name,
            ax=ax,
        )
    ax.set_title("Untouched 2025 Test - ROC Curves")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "untouched_test_roc_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probabilities in test_probabilities.items():
        PrecisionRecallDisplay.from_predictions(
            test[TARGET_COLUMN],
            probabilities,
            name=model_name,
            ax=ax,
        )
    ax.axhline(
        float(test[TARGET_COLUMN].mean()),
        linestyle="--",
        color="grey",
        label="Default prevalence",
    )
    ax.set_title("Untouched 2025 Test - Precision-Recall Curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "untouched_test_precision_recall_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_importance = importance_table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        plot_importance["raw_feature"],
        plot_importance["importance_mean"],
        xerr=plot_importance["importance_std"],
        color="#2E86AB",
        alpha=0.9,
    )
    ax.set(
        title="Selected Model - Validation Permutation Importance",
        xlabel="Decrease in PR-AUC after permutation",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "selected_model_permutation_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    artifact_size_mb = BEST_MODEL_PATH.stat().st_size / (1024**2)
    metadata = {
        "data_path": str(args.data_path.resolve()),
        "target": TARGET_COLUMN,
        "safe_raw_features": SAFE_RAW_FEATURES,
        "post_origination_exclusions": POST_ORIGINATION_EXCLUSIONS,
        "conservative_exclusions": CONSERVATIVE_EXCLUSIONS,
        "temporal_split": partitions.split_summary.to_dict(orient="records"),
        "rolling_cv_folds": len(folds),
        "tuning_iterations_per_finalist": args.tuning_iterations,
        "screened_models": model_summary["model"].tolist(),
        "catboost_status": "Not installed or evaluated; retained as an optional challenger.",
        "svm_policy": "LinearSVC used; nonlinear kernel SVM is unsuitable at this scale.",
        "class_imbalance_policy": (
            "Class weights / scale_pos_weight tuned; no synthetic samples cross time."
        ),
        "selected_model": selected_model_name,
        "selected_candidate": candidate_to_dict(selected_candidate),
        "calibration_method": bundle.calibrator.method,
        "decision_threshold": bundle.threshold,
        "artifact_path": str(BEST_MODEL_PATH),
        "artifact_size_mb": artifact_size_mb,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Issued-loan population only; rejected applications are absent.",
            "No borrower identifier is available for repeated-customer grouping.",
            "Country needs governance and fairness review.",
            "Risk rating is Bondora's existing at-issue score, not an independent input.",
            "Financial costs were unavailable; threshold uses a pre-registered balance.",
        ],
    }
    save_json(RESULTS_DIR / "optimized_training_metadata.json", metadata)
    write_summary(
        path=RESULTS_DIR / "optimized_default_model_summary.md",
        selected_name=selected_model_name,
        validation_row=selected_validation_row,
        test_row=selected_test_row,
        feature_name=selected_candidate.feature_configuration,
        model_path=BEST_MODEL_PATH,
    )

    print("\nLocked selected model:")
    print(selected_model_name)
    print("\nUntouched test metrics:")
    print(
        test_leaderboard.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved model: {BEST_MODEL_PATH} ({artifact_size_mb:.2f} MB)")
    print(f"Reports: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

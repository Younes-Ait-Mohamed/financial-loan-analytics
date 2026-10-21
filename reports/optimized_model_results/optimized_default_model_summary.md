# Optimized Default-Risk Model Summary

## Scientific design

- Source-verified origination fields only; `nr_of_payments` and every lifecycle
  field are excluded as leakage.
- Expanding monthly cross-validation within January–September 2024.
- October–November 2024 fits probability-calibration candidates.
- December 2024 selects the calibration method and operating threshold.
- January–27 April 2025 is evaluated once as the untouched out-of-time test.

## Locked configuration

- Model: **Logistic Regression**
- Feature configuration: **stacked_engineered**
- Parameters: **C = 0.1, L2 regularization, no class weighting**
- Preprocessing: **robust scaling**
- Calibration: **isotonic**
- Threshold: **4.00%**

The model was locked before opening the test using a Pareto frontier,
multi-metric deployment score, temporal stability, and a lower-complexity
preference within a pre-defined 0.005 near-tie margin.

## Untouched 2025 test result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5364 |
| Precision | 0.1176 |
| Recall | 0.7986 |
| F1-score | 0.2051 |
| Balanced accuracy | 0.6569 |
| MCC | 0.1652 |
| ROC-AUC | 0.7247 |
| PR-AUC | 0.1747 |
| PR-AUC lift | 2.33x |

Confusion matrix: TN=25,157, FP=23,677, FN=796, TP=3,157.

The 4% threshold is provisional. It favors recall and flags about half of the
test portfolio, so a financial cost matrix and review-capacity constraint are
required before deployment.

## Champion–challenger interpretation

Linear SVM achieved stronger test ROC-AUC (0.7280), PR-AUC (0.1793), F1
(0.2300), balanced accuracy (0.6717), and MCC (0.1863), but lower recall
(0.6959). It cannot replace the locked Logistic Regression after this test has
been observed. It is recorded as the challenger and requires a new untouched
future cohort.

## Scope and limitations

This model ranks risk among loans already issued by Bondora. It is not validated
on rejected applications and should not be presented as a universal approval
model. Interest rate and Bondora's existing risk rating make it an
at-origination/post-offer overlay. Country requires governance review.

The fast experiment used three rolling folds, three tuning candidates per
finalist, and 100 bootstrap resamples. It is a professional progress result,
not an exhaustive production search.

Saved bundle:
`C:\Users\aitmo\financial-loan-analytics\models\optimized\best_optimized_default_model.joblib`

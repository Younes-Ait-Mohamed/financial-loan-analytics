# Default-Risk Modelling Summary

## 1. Business objective

Estimate each loan's probability of default within 12 months using information
available at or near origination. The score can support credit decisions,
portfolio monitoring, and risk-based review.

## 2. Binary target

`has_default_within_12_months` equals `1` for a default within 12 months and `0`
otherwise. Rows with an unknown target were excluded from supervised training.
The observed default share is 14.74%.

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

| model | accuracy | precision | recall | f1_score | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.8595 | 0.5647 | 0.2039 | 0.2996 | 0.7580 |
| Logistic Regression | 0.7454 | 0.3039 | 0.5633 | 0.3948 | 0.7380 |
| Random Forest | 0.7847 | 0.3385 | 0.4822 | 0.3978 | 0.7379 |

## 8. Selected model

**Gradient Boosting** was selected because it ranked first by ROC-AUC and then
recall. Its test ROC-AUC is 0.7580, recall is 0.2039, and
F1-score is 0.2996.

## 9. Limitations and next steps

This is an untuned baseline using a random stratified split and a default 0.50
decision threshold. Next steps are out-of-time validation, threshold selection
based on financial costs, probability calibration, controlled hyperparameter
tuning, fairness checks, explainability, and dashboard integration.

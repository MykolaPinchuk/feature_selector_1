# Feature Selection Report - rwm5yr

## Run Summary
- Must keep features (11): age, docvis, edlevel, edlevel2, educ, female, hhninc, id, kids, married, outwork
- Useful features (0): None
- Dropped features (4): edlevel1, edlevel3, edlevel4, hospvis
- Overfit suspects: None
- Final chosen feature count: 11

## Candidate Set Metrics (VAL)

| Candidate | ROC-AUC | PR-AUC | PR-AUC@10 |
| --- | --- | --- | --- |
| baseline_all | 0.8411 | 0.4018 | 0.4858 |
| filtered | 0.8382 | 0.3936 | 0.4766 |
| aggressive | 0.8391 | 0.4005 | 0.4836 |

## Permutation / SHAP / Gain Importance (Top 15 by mean delta)

| Feature | Mean Δ Metric | Normalized Δ | Std Δ | SHAP | Gain |
| --- | --- | --- | --- | --- | --- |
| outwork | 0.0994 | 1.000 | 0.0062 | 0.791560 | 13.087845 |
| age | 0.0791 | 0.796 | 0.0067 | 0.435296 | 2.398767 |
| hhninc | 0.0767 | 0.772 | 0.0111 | 0.567396 | 2.655891 |
| id | 0.0725 | 0.729 | 0.0073 | 0.389638 | 2.166674 |
| educ | 0.0410 | 0.413 | 0.0051 | 0.327688 | 2.216707 |
| female | 0.0187 | 0.189 | 0.0026 | 0.225575 | 3.130532 |
| kids | 0.0165 | 0.166 | 0.0008 | 0.244125 | 2.056881 |
| docvis | 0.0112 | 0.113 | 0.0025 | 0.203571 | 1.889017 |
| edlevel2 | 0.0067 | 0.068 | 0.0028 | 0.064458 | 3.194046 |
| married | 0.0067 | 0.067 | 0.0031 | 0.054661 | 1.994385 |
| edlevel | 0.0037 | 0.037 | 0.0020 | 0.071943 | 1.941945 |
| hospvis | 0.0000 | 0.000 | 0.0005 | 0.016469 | 1.567215 |
| edlevel4 | -0.0000 | -0.000 | 0.0000 | 0.000725 | 0.000000 |
| edlevel1 | -0.0001 | -0.001 | 0.0001 | 0.004283 | 2.324305 |
| edlevel3 | -0.0008 | -0.008 | 0.0006 | 0.017214 | 1.780820 |

## Static Filter Report

- Leakage: None
- Constant: None
- Quasi_Constant: None
- Missingness: None
- Duplicates: None

## Outputs in results/rwm5yr_20251130T043827Z

- candidate_metrics.csv
- chosen_features.txt
- dropped_features.txt
- gain_importance.csv
- must_keep.txt
- permutation_importance.csv
- shap_importance.csv
- static_filters.json
- summary.json
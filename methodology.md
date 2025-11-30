## Methodology Overview

This document summarizes the feature-selection workflow implemented in `feature_selector_1`, linking the algorithmic steps from `prd.md` to the concrete modules in `feature_selector/`.

### Problem Setting
- Time-ordered binary classification (`0/1` target column) for XGBoost models.
- Global splits: `TRAIN` (earliest), `VAL` (next), `TEST` (final OOT, untouched until the end).
- Goals: reduce feature set while preserving OOT performance, keep compute budgets low, avoid unreliable in-sample feature importance.

---

## Pipeline Stages

1. **Data ingestion** (`feature_selector.data`):
   - Accepts CSV, Parquet, or pickle inputs (or `pydataset` name).
   - Validates target is binary integer and ensures all feature columns are already numeric/encoded.
   - Optional `target_binarize_threshold` converts numeric targets to indicators (e.g., `hospvis > 0`).
   - Parses the time column into `datetime64`.

2. **Static pre-filtering** (`feature_selector.preprocessing`):
   - Optional leakage drop list.
   - Removes constant or quasi-constant columns (default ≥99.5% single value).
   - Removes high-missingness columns (default ≥99% missing).
   - Drops duplicate columns via hash comparison.
   - Reports each removal reason.

3. **Time-aware splits** (`feature_selector.splits`):
   - Sorts by time and slices into TRAIN/VAL/TEST using configurable fractions.
   - Within TRAIN, creates `TRAIN_FS` (early) and `HOLDOUT_FS` (later) with `fs_holdout_frac`.
   - Optionally subsamples `TRAIN_FS` rows (contiguous window) to control FS training cost.
   - Samples an `FS_EVAL` slice from `HOLDOUT_FS` for SHAP/permutation evaluation.

4. **Feature selection models** (`feature_selector.fs_models`):
   - Trains a small ensemble (default 3) of moderately regularized XGBoost classifiers on `TRAIN_FS_SUB`, validating on `HOLDOUT_FS`.
   - Seeds differ per model to diversify trees.
   - Hyperparameters expose `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `eta`, `n_estimators`, `reg_lambda`, `reg_alpha`, `gamma`, and `scale_pos_weight` (auto-computed on TRAIN when enabled).

5. **SHAP triage**:
   - TreeSHAP computed on `FS_EVAL` for each FS model.
   - Mean absolute SHAP across the ensemble ranks features.
   - Two blocks defined from ranks:
     - **Top block (T)**: default top 60 features → candidates for detailed permutation.
     - **Bottom block (B)**: default bottom 60 → group permutation test.

6. **Permutation importance on OOT slice**:
   - Uses selected metric (ROC-AUC by default) computed on `FS_EVAL`.
   - For each model and each feature in (T):
     - Shuffle column values within `FS_EVAL`.
     - Compute metric delta (baseline − permuted).
   - Group permutation for (B) by shuffling all bottom features simultaneously.
   - Aggregate across ensemble (mean, std) and normalize relative to the maximum delta.

7. **Feature classification & overfit diagnostics**:
   - **Must keep**: top-N permutation ranks or normalized delta ≥ threshold.
   - **Useful**: positive but smaller deltas (or un-permuted features outside bottom block).
   - **Dropped**: deltas ≤ minimum positive threshold; bottom block dropped if grouped delta is negligible.
   - **Overfit suspects**: features with high train gain but poor permutation ranks.
   - Build candidate feature sets:
     - `baseline_all` = post-filter all features.
     - `filtered` = remove dropped + overfit suspects.
     - `aggressive` = must-keep subset.

8. **Ablation models & final evaluation**:
   - Retrain XGBoost on full TRAIN, evaluate on VAL for each candidate set.
   - Metrics recorded:
     - ROC-AUC (`sklearn.metrics.roc_auc_score`).
     - PR-AUC (average precision).
     - PR-AUC@10% minority rate (sample-weighted average precision).
   - Choose final feature set as the smallest candidate whose main metric is within 0.5% relative of the best VAL score.
   - Train a final model on TRAIN using the chosen features and score it on TRAIN, VAL, and TEST; results are saved in `final_metrics.csv`.

9. **Persistence & reporting**:
   - CLI writes JSON summary, candidate metrics, permutation SHAP/gain tables, and chosen feature lists under `results/<dataset>_<timestamp>/`.
   - `report.md` in each run folder offers a human-readable recap.

---

## Metrics

| Metric/Column         | Where it appears                         | Description                                                                                                                           | Location in code                                        |
| --------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `roc_auc`             | `candidate_metrics.csv` / report table   | ROC curve area on VAL for a candidate feature set.                                                                                    | `feature_selector.metrics.roc_auc`                     |
| `prauc`               | `candidate_metrics.csv` / report table   | Average precision (PR-AUC) on VAL without reweighting.                                                                                | `feature_selector.metrics.pr_auc`                      |
| `prauc_at_10`         | `candidate_metrics.csv` / report table   | Average precision after reweighting negatives to mimic a 10% positive rate (cost-sensitive view).                                    | `feature_selector.metrics.pr_auc_at_10`                |
| `mean_delta`          | `permutation_importance.csv` / report    | Average drop in the chosen FS metric (baseline − permuted) when the feature is shuffled on `FS_EVAL`, aggregated across FS models.   | `feature_selector.fs_models.compute_permutation_importance` |
| `std_delta`           | `permutation_importance.csv` / report    | Standard deviation of the per-model permutation deltas, indicating variability/noise.                                                | Same as above                                          |
| `normalized_delta`    | `permutation_importance.csv` / report    | `mean_delta` divided by the maximum delta (plus epsilon) to give a [0,1] relative score within the permuted block.                   | `feature_selector.pipeline._aggregate_permutation`     |
| `shap_importance`     | `permutation_importance.csv`, `shap_*.csv` | Mean absolute TreeSHAP value on `FS_EVAL`, averaged across the FS ensemble.                                                           | `feature_selector.fs_models.compute_shap_importance`   |
| `gain_importance`     | `permutation_importance.csv`, `gain_*.csv` | Booster gain importance on `TRAIN_FS_SUB`; useful for overfit diagnostics.                                                           | `feature_selector.fs_models.gain_importance`           |
| `overfit_score`       | `permutation_importance.csv`             | Rank percentile difference (`perm_rank - gain_rank`); positive values flag features strong in train gain but weak out-of-time.       | `feature_selector.pipeline._detect_overfit`            |

These metrics collectively guide triage (SHAP), usefulness (permutation), and diagnostics (gain vs permutation, PR-AUC@10 for imbalanced performance sensitivity).

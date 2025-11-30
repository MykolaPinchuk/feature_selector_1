Here is a concrete end-to-end algorithm that matches everything we discussed: time-ordered data, OOT val/test, XGB, distrust of gain FI, and avoiding full RFE.

I’ll write it as something you can turn almost directly into code.

---

## 0. Notation and setup

Assume you have:

* Time-ordered data with features (X) and labels (y).
* Global splits by time:

  * `TRAIN` (earliest)
  * `VAL` (OOT, used for model/FS tuning)
  * `TEST` (final OOT, used once at the end)

Goal: select a subset of features that generalizes well on future data, using as few heavy model runs as possible.

---

## 1. Static pre-filtering (no modeling yet)

Input: `TRAIN` slice only.

1. **Leakage filter (must-have)**

   * Drop any feature that uses information from at or after the label time:

     * Aggregates that include future events.
     * Post-hoc outcomes (e.g., “chargeback occurred later”).
   * This is domain work; do it by feature lineage, not statistics.

2. **Constant / quasi-constant**

   * For each feature (j):

     * If `var(X_train[:, j]) == 0`, drop.
     * Optionally drop if the most frequent category/value covers > 99–99.5% of rows and you have no reason to keep the tail.

3. **Extreme missingness**

   * For each feature (j):

     * Compute missing rate on `TRAIN`.
     * If missing rate > 98–99% and there is no strong reason to keep it, drop.

4. **Exact duplicates**

   * Detect features with identical values across all rows (e.g., via hashes column-wise) and keep only one representative.

Denote the remaining feature set as (F_0).

---

## 2. Define inner FS splits to respect time

You already have `TRAIN`, `VAL`, `TEST`.

Inside `TRAIN`, define a time-respecting split:

* `TRAIN_FS` = early part of `TRAIN`
* `HOLDOUT_FS` = later part of `TRAIN` (but still earlier than `VAL`)

Heuristic: `HOLDOUT_FS` ~20–30% of `TRAIN` by time, contiguous.

This gives:

* `TRAIN_FS`: for fitting FS models.
* `HOLDOUT_FS`: for computing FI on “future” data, without touching global `VAL`.

---

## 3. Subsample and define FS model config

We do FS on a subsample to save compute.

From `TRAIN_FS`:

1. **Row subsample (optional but recommended)**

   * If `TRAIN_FS` is very large, sample N rows (e.g., 100k–300k) in a time-aware way:

     * Either take a contiguous window.
     * Or stratify by time buckets and sample proportionally.
   * Call this `TRAIN_FS_SUB`.

2. **Model hyperparameters for FS models**

   * Use a moderately regularized, not over-tuned XGB config, e.g.:

     * `max_depth`: 4–6
     * `min_child_weight`: 5–20
     * `subsample`: 0.7–0.9
     * `colsample_bytree`: 0.7–1.0
     * `lambda` (L2): 1–5
     * `eta`: 0.05–0.2
     * `n_estimators`: 200–400 (or early stopping on `HOLDOUT_FS`)

   * Metrics: ROC-AUC / PR-AUC / logloss, whatever you care about.

We will train several FS models with different random seeds / row subsamples.

---

## 4. Train a small ensemble of FS models

Choose (M) (e.g., 3–5) FS models:

For each (m \in {1,\dots,M}):

1. Optionally draw a slightly different row subsample from `TRAIN_FS_SUB`.
2. Set `random_state = m` (or any varying seed).
3. Train XGB on `TRAIN_FS_SUB` with the FS hyperparams and early stopping on `HOLDOUT_FS` if you want.

Store each trained model (f^{(m)}).

---

## 5. Compute SHAP on OOT slice for triage

We now compute SHAP on an OOT slice to see **where the model actually puts its weight**, but we will *not* equate that with usefulness.

1. **Define evaluation slice for FI**

   * Use `HOLDOUT_FS` (since it is OOT relative to `TRAIN_FS`).
   * Optionally subsample rows if large (e.g., 10k–50k rows).
   * Call this `FS_EVAL`.

2. **Compute SHAP for each FS model**

   * For each model (f^{(m)}):

     * Compute SHAP values on `FS_EVAL` (tree SHAP).
     * For each feature (j \in F_0), compute:
       [
       \text{shap_imp}^{(m)}*j = \mathbb{E}*{x \in FS_EVAL} \left[| \text{SHAP}_j(x) |\right]
       ]
   * Optionally also compute **gain FI** on `TRAIN_FS_SUB` for diagnostics only.

3. **Aggregate SHAP across models**

   * For each feature (j):

     * `mean_shap_j = mean_m shap_imp^{(m)}_j`
   * Rank features by `mean_shap_j` (descending).

4. **SHAP-based triage**

   * Let (N = |F_0|) (after static pre-filters).
   * Define:

     * Top block (T): top ~40–80 features by `mean_shap_j` (you choose size; 60 is a good default for ~150 total features).
     * Bottom block (B): the lowest ~40–80 features by `mean_shap_j`.
     * Middle block (M): the rest.

Interpretation:

* (T): model is clearly relying on these somewhere.
* (B): model barely uses them (on OOT slice).
* (M): ambiguous.

We will do detailed permutation only on (T) and group tests on (B).

---

## 6. Permutation FI on OOT (core usefulness signal)

We now do permutation FI on the **same OOT slice** `FS_EVAL`.

Define your main metric (L) (e.g., ROC-AUC, PRAUC, logloss).

1. **Baseline metric**

   * For each FS model (f^{(m)}), compute `metric_base^{(m)}` on `FS_EVAL`.

2. **Permutation for top block T**

   * For each FS model (f^{(m)}):

     * For each feature (j \in T):

       * Shuffle (X_j) within `FS_EVAL` (keep labels and other features fixed).
       * Compute `metric_perm^{(m)}(j)` on the permuted `FS_EVAL`.
       * Define **permutation importance**:
         [
         \Delta^{(m)}_j = \text{metric_base}^{(m)} - \text{metric_perm}^{(m)}(j)
         ]
       * If your metric is something you want to maximize (AUC), positive (\Delta) = good feature. For loss metrics, reverse the sign appropriately (e.g., (\Delta = \text{metric_perm} - \text{metric_base})).

3. **Group permutation for bottom block B (optional but efficient)**

   * For each model (f^{(m)}), permute **all features in (B) together**:

     * Shuffle each feature (j \in B) independently across rows.
     * Compute `metric_perm_B^{(m)}`.
     * Group importance:
       [
       \Delta_B^{(m)} = \text{metric_base}^{(m)} - \text{metric_perm_B}^{(m)}
       ]
   * If (\Delta_B) is ~0, you know the entire bottom block is safe to drop.
   * If (\Delta_B) is non-negligible, you may choose to:

     * Either leave (B) as “maybe keep”.
     * Or sample a few features in (B) for individual permutation.

---

## 7. Aggregate permutation FI and classify features

For each feature (j \in T):

1. Aggregate across models:

   * `mean_perm_j = mean_m Δ^{(m)}_j`
   * `std_perm_j = std_m Δ^{(m)}_j`

2. Normalize **within T**:

   * Let `max_perm = max_j mean_perm_j` over (j \in T).
   * Define normalized permutation FI:
     [
     p_j = \frac{\text{mean_perm}_j}{\text{max_perm} + \epsilon}
     ]
     where (\epsilon) is small.

Now classify features using simple thresholds:

Define:

* **Must-keep** ((K)):

  * Features (j) with:

    * (p_j \ge p_{\text{high}}) (e.g., (p_{\text{high}} = 0.2)–0.3), or
    * In top K_perm (e.g., top 20 features by `mean_perm_j`) and non-negative `mean_perm_j`.
* **Useful but not critical** ((U)):

  * (0 < p_j < p_{\text{high}}) and `mean_perm_j` statistically > 0 (e.g., > 2× std of metric noise).
* **Useless-on-OOT** ((Z_T)):

  * `mean_perm_j` ≈ 0 or negative (within noise band around 0).

For the bottom block (B):

* If group (\Delta_B) ≈ 0:

  * Label all (j \in B) as **useless ((Z_B))**.
* Else:

  * Either keep them in middle group (U) for now, or inspect a subset individually.

Let:

* (Z = Z_T \cup Z_B): candidate-to-drop.
* (K): must-keep.
* (U): optional (if you want to keep some extra features).

---

## 8. Overfitting diagnostics using gain/SHAP vs permutation

Now optionally use gain/SHAP vs permutation to flag overfit features, mostly for documentation or more aggressive pruning.

For each feature (j):

1. Get:

   * `gain_train_j` from at least one FS model (importance_type="gain" on `TRAIN_FS_SUB`).
   * `mean_shap_j` computed earlier on `FS_EVAL`.
   * `mean_perm_j` from permutation.

2. Convert each to rank percentiles:

   * `rank_gain_j` ∈ [0,1] (0 = most important).
   * `rank_shap_j` ∈ [0,1].
   * `rank_perm_j` ∈ [0,1].

3. Define a simple overfit score, e.g.:
   [
   \text{overfit_score}_j = \text{rank_perm}_j - \text{rank_gain}_j
   ]
   (positive: high gain, low permutation).

Treat as **highly suspicious**:

* Features where:

  * `rank_gain_j` < 0.2 (top 20% by gain), AND
  * `rank_perm_j` > 0.8 (bottom 20% by permutation), AND
  * `mean_perm_j` ≈ 0.

You can then:

* Keep them in a separate set (O) (“overfit-suspect”) even if SHAP is high.
* Either:

  * Drop them together with (Z), or
  * Test dropping them as a block in one ablation.

---

## 9. Final ablation models (small number of full runs)

Now move from FS models to near-final models using the **original** `TRAIN` and `VAL` splits.

Define candidate feature sets:

1. **All features**:

   * (F_{\text{all}} = F_0) (after static filters).

2. **Filtered set (main candidate)**:

   * (F_{\text{filtered}} = (K \cup U) \setminus O_{dropped})
   * Where (O_{dropped}) is either:

     * (Z) (all useless + overfit-suspect), or
     * (Z) plus subset of (O) you choose to drop.

3. **Aggressive minimal set (optional)**:

   * (F_{\text{aggr}} = K) (must-keep only), or (K \cup U_{\text{top few}}).

Train 2–3 full models:

* **Model A (baseline)**:

  * Train tuned XGB (your normal hyperparams, early stopping on `VAL`) on `TRAIN` with `F_all`.
  * Get metric on `VAL`: `metric_A`.

* **Model B (filtered)**:

  * Same hyperparams, `TRAIN`, features `F_filtered`.
  * Metric on `VAL`: `metric_B`.

* **Model C (aggressive; optional)**:

  * Same hyperparams, `TRAIN`, features `F_aggr`.
  * Metric on `VAL`: `metric_C`.

Decision rule:

* Choose the smallest feature set whose metric is within a tolerance of the best model:

  * Let `metric_best_val = max(metric_A, metric_B, metric_C)` (for maximization metrics).
  * Define tolerance, e.g. `tol = 0.5–1%` relative:
    [
    \text{metric}(S) \ge (1 - \text{tol}) \times \text{metric_best_val}
    ]
  * Among feature sets satisfying this, choose the **smallest** (fewest features).

Denote the chosen feature set as (F^*).

---

## 10. Final training and TEST evaluation

Finally:

1. Retrain your production-style model using:

   * Either `TRAIN` or `TRAIN ∪ VAL` (depending on your policy),
   * Features (F^*),
   * Final tuned hyperparams.

2. Evaluate once on `TEST`.

   * Do not use `TEST` for any further FS or hyperparameter decisions.

If you are satisfied with `TEST` performance and calibration, lock in (F^*) and the model.

---

## 11. Practical notes / knobs you can tune

* Number of FS models (M):
  3–5 is usually enough; more improves stability but increases compute.

* Size of top SHAP block (T):

  * For 150 features, 40–80 is a reasonable range.
  * If you suspect high redundancy, err on slightly larger T.

* Permutation sample size:

  * 10k–50k rows usually enough to estimate FI reliably.
  * You can adjust based on noise in the metric.

* Thresholds:

  * (p_{\text{high}}) ≈ 0.2–0.3 (keep any feature whose permutation FI is ≥ 20–30% of the maximum).
  * You can also enforce a minimum absolute Δ in metric (e.g., AUC drop ≥ 0.002).

* Metrics:

  * For imbalanced classification, PR-AUC or cost-sensitive metrics may be more diagnostic than ROC-AUC.

---

This procedure gives you:

* FS primarily driven by **permutation FI on out-of-time data**.
* SHAP used only for:

  * triaging which features to permute,
  * seeing how the model uses features,
  * overfit diagnostics.
* Very few expensive model runs:

  * FS phase: (M) models with small trees and subsampled data.
  * Final phase: 2–3 full models on `TRAIN`/`VAL`.

You avoid RFE-style 150-iteration loops but still get a defensible, robust feature subset tailored to OOT generalization.

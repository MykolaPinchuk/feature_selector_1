from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureSelectorConfig
from .data import ensure_binary_target, ensure_numeric_features, parse_time_column
from .fs_models import (
    FSModel,
    PermutationOutputs,
    SHAPResult,
    compute_permutation_importance,
    compute_shap_importance,
    gain_importance,
    train_fs_ensemble,
)
from .metrics import MetricFunc, compute_metrics, get_metric
from .preprocessing import StaticFilterReport, apply_static_filters
from .splits import FSSplits, GlobalSplits, contiguous_subsample, create_fs_splits, sample_eval_slice, time_ordered_split

VAL_TOLERANCE = 0.005  # 0.5%


@dataclass
class FeatureSelectionResult:
    must_keep: List[str]
    useful: List[str]
    dropped: List[str]
    overfit_suspect: List[str]
    filtered_features: List[str]
    candidate_metrics: Dict[str, Dict[str, float]]
    chosen_features: List[str]
    permutation_importance: pd.DataFrame
    shap_importance: pd.Series
    gain_importance: pd.Series
    static_filter_report: Dict[str, List[str]]


class FeatureSelectorPipeline:
    def __init__(self, config: FeatureSelectorConfig):
        self.config = config
        self.metric_func: MetricFunc = get_metric(config.main_metric)
        self.rng = np.random.default_rng(config.random_seed)

    def run(self, df: pd.DataFrame) -> FeatureSelectionResult:
        df = df.copy()
        df[self.config.time_col] = parse_time_column(df, self.config.time_col)
        ensure_binary_target(df, self.config.target_col)

        feature_cols = [c for c in df.columns if c not in {self.config.target_col, self.config.time_col}]
        ensure_numeric_features(df, feature_cols)

        filtered_features, static_report = apply_static_filters(
            df,
            feature_cols,
            self.config.leakage_features,
            self.config.quasi_constant_threshold,
            self.config.missingness_threshold,
        )
        if not filtered_features:
            raise ValueError("All features were removed during static filtering")

        df = df[[self.config.time_col, self.config.target_col] + filtered_features].copy()
        splits = time_ordered_split(df, self.config.time_col, self.config.split_config)
        fs_splits = create_fs_splits(splits.train, self.config.time_col, self.config.split_config)
        train_fs_sub = contiguous_subsample(
            fs_splits.train_fs,
            self.config.split_config.train_fs_subsample_size,
            self.rng,
        )
        eval_size = self.config.resolve_fs_eval_size()
        fs_eval = sample_eval_slice(fs_splits.holdout_fs, eval_size, self.rng)

        ensemble = self._train_fs_models(train_fs_sub, fs_splits, filtered_features)
        shap_result = compute_shap_importance(ensemble, fs_eval[filtered_features])
        top_features, bottom_features = self._triage_features(shap_result, filtered_features)

        perm_outputs = compute_permutation_importance(
            ensemble,
            fs_eval[filtered_features],
            fs_eval[self.config.target_col],
            top_features,
            bottom_features,
            self.metric_func,
            self.config.permutation_config,
            self.rng,
        )
        aggregated = self._aggregate_permutation(perm_outputs, shap_result)

        classification = self._classify_features(
            aggregated,
            bottom_features,
            perm_outputs.bottom_block_deltas,
        )

        gain_scores = gain_importance(ensemble[0].model, filtered_features)
        permutation_table = self._build_permutation_table(aggregated, shap_result, gain_scores)

        overfit_suspect = self._detect_overfit(permutation_table)
        candidate_sets = self._build_candidate_sets(filtered_features, classification, overfit_suspect)
        candidate_metrics, chosen_features = self._run_ablation_models(splits, candidate_sets)

        return FeatureSelectionResult(
            must_keep=classification["must_keep"],
            useful=classification["useful"],
            dropped=classification["dropped"],
            overfit_suspect=overfit_suspect,
            filtered_features=filtered_features,
            candidate_metrics=candidate_metrics,
            chosen_features=chosen_features,
            permutation_importance=permutation_table,
            shap_importance=shap_result.mean_importance,
            gain_importance=gain_scores,
            static_filter_report=static_report.summary(),
        )

    def _train_fs_models(
        self,
        train_fs_sub: pd.DataFrame,
        fs_splits: FSSplits,
        feature_cols: List[str],
    ) -> List[FSModel]:
        X_train = train_fs_sub[feature_cols]
        y_train = train_fs_sub[self.config.target_col]
        X_valid = fs_splits.holdout_fs[feature_cols]
        y_valid = fs_splits.holdout_fs[self.config.target_col]
        return train_fs_ensemble(
            X_train,
            y_train,
            X_valid,
            y_valid,
            self.config.model_config,
        )

    def _triage_features(
        self,
        shap_result: SHAPResult,
        feature_cols: List[str],
    ) -> tuple[List[str], List[str]]:
        shap_sorted = shap_result.mean_importance.sort_values(ascending=False)
        top_k = min(self.config.permutation_config.shap_top_k, len(feature_cols))
        bottom_k = min(self.config.permutation_config.shap_bottom_k, len(feature_cols))

        top_features = shap_sorted.head(top_k).index.tolist()
        bottom_features = [col for col in shap_sorted.tail(bottom_k).index if col not in top_features]
        return top_features, bottom_features

    def _aggregate_permutation(
        self,
        outputs: PermutationOutputs,
        shap_result: SHAPResult,
    ) -> pd.DataFrame:
        rows = []
        for feature, deltas in outputs.per_feature_deltas.items():
            if not deltas:
                continue
            rows.append(
                {
                    "feature": feature,
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": float(np.std(deltas)),
                }
            )
        permuted_features = set(outputs.per_feature_deltas.keys())
        if rows:
            perm_df = pd.DataFrame(rows).set_index("feature")
            max_delta = perm_df["mean_delta"].max()
            denom = max_delta + self.config.permutation_config.importance_norm_epsilon
            perm_df["normalized_delta"] = perm_df["mean_delta"] / denom
        else:
            perm_df = pd.DataFrame(columns=["mean_delta", "std_delta", "normalized_delta"])
        shap_series = shap_result.mean_importance
        perm_df = perm_df.reindex(shap_series.index, fill_value=0.0)
        perm_df["has_permutation"] = perm_df.index.map(lambda c: c in permuted_features)
        perm_df["shap_importance"] = shap_series
        return perm_df

    def _classify_features(
        self,
        perm_df: pd.DataFrame,
        bottom_features: List[str],
        bottom_block_deltas: List[float],
    ) -> Dict[str, List[str]]:
        conf = self.config.permutation_config
        must_keep: List[str] = []
        useful: List[str] = []
        dropped: List[str] = []

        perm_sorted = perm_df.loc[perm_df["has_permutation"], "mean_delta"].sort_values(ascending=False)
        top_n_features = set(perm_sorted.head(conf.must_keep_top_n).index)
        bottom_set = set(bottom_features)

        for feature, row in perm_df.iterrows():
            if feature in bottom_set:
                continue
            if not row.get("has_permutation", False):
                useful.append(feature)
                continue
            mean_delta = row["mean_delta"]
            norm = row["normalized_delta"]
            if mean_delta <= conf.min_positive_delta:
                dropped.append(feature)
            elif feature in top_n_features or norm >= conf.must_keep_threshold:
                must_keep.append(feature)
            else:
                useful.append(feature)

        if bottom_features:
            group_delta = float(np.mean(bottom_block_deltas)) if bottom_block_deltas else 0.0
            if abs(group_delta) <= conf.group_importance_tol:
                dropped.extend(bottom_features)
            else:
                useful.extend(bottom_features)
        dropped = sorted(set(dropped))
        useful = [f for f in useful if f not in dropped]
        return {
            "must_keep": sorted(set(must_keep)),
            "useful": sorted(set(useful)),
            "dropped": dropped,
        }

    def _build_permutation_table(
        self,
        perm_df: pd.DataFrame,
        shap_result: SHAPResult,
        gain_scores: pd.Series,
    ) -> pd.DataFrame:
        table = perm_df.copy()
        table["gain_importance"] = gain_scores.reindex(table.index).fillna(0.0)
        return table

    def _detect_overfit(self, table: pd.DataFrame) -> List[str]:
        if len(table) == 0:
            return []

        def rank_percentile(series: pd.Series, descending: bool) -> pd.Series:
            if len(series) <= 1:
                return pd.Series(np.zeros(len(series)), index=series.index)
            ranked = series.rank(ascending=not descending, method="average")
            denom = len(series) - 1
            return (ranked - 1) / denom

        gain_rank = rank_percentile(table["gain_importance"], descending=True)
        perm_rank = rank_percentile(table["mean_delta"], descending=True)
        table["overfit_score"] = perm_rank - gain_rank

        suspects = table[
            (gain_rank < 0.2)
            & (perm_rank > 0.8)
            & (table["mean_delta"].abs() <= self.config.permutation_config.min_positive_delta)
        ].index.tolist()
        return sorted(suspects)

    def _build_candidate_sets(
        self,
        filtered_features: List[str],
        classification: Dict[str, List[str]],
        overfit_suspect: List[str],
    ) -> Dict[str, List[str]]:
        must_keep = classification["must_keep"]
        useful = [f for f in classification["useful"] if f not in overfit_suspect]
        dropped = classification["dropped"] + overfit_suspect

        filtered = [f for f in filtered_features if f not in dropped]
        aggressive = must_keep if must_keep else filtered

        return {
            "baseline_all": filtered_features,
            "filtered": filtered,
            "aggressive": aggressive,
        }

    def _train_eval_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_set: List[str],
    ) -> Dict[str, float]:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            max_depth=self.config.model_config.max_depth,
            min_child_weight=self.config.model_config.min_child_weight,
            subsample=self.config.model_config.subsample,
            colsample_bytree=self.config.model_config.colsample_bytree,
            reg_lambda=self.config.model_config.reg_lambda,
            learning_rate=self.config.model_config.eta,
            n_estimators=self.config.model_config.n_estimators,
            random_state=self.config.random_seed,
            n_jobs=self.config.model_config.n_jobs,
            tree_method=self.config.model_config.tree_method,
            objective="binary:logistic",
            eval_metric= "auc",
            verbosity=0,
        )
        X_train = train_df[feature_set]
        y_train = train_df[self.config.target_col]
        X_val = val_df[feature_set]
        y_val = val_df[self.config.target_col]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.to_numpy(), preds)
        return metrics

    def _run_ablation_models(
        self,
        splits: GlobalSplits,
        candidate_sets: Dict[str, List[str]],
    ) -> tuple[Dict[str, Dict[str, float]], List[str]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, features in candidate_sets.items():
            if len(features) == 0:
                continue
            metrics[name] = self._train_eval_model(splits.train, splits.val, features)

        best_metric = -np.inf
        best_set: Optional[str] = None
        for name, metric_values in metrics.items():
            val = metric_values.get(self.config.main_metric)
            if val is not None and val > best_metric:
                best_metric = val
                best_set = name

        chosen = candidate_sets.get("baseline_all", [])
        if best_set is not None:
            tolerance_cutoff = (1 - VAL_TOLERANCE) * best_metric
            eligible = [
                (name, metrics[name][self.config.main_metric])
                for name in metrics
                if metrics[name][self.config.main_metric] >= tolerance_cutoff
            ]
            if eligible:
                eligible.sort(key=lambda item: (len(candidate_sets[item[0]]), -item[1]))
                chosen = candidate_sets[eligible[0][0]]
        return metrics, chosen

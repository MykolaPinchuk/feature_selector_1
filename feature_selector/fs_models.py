from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from .config import ModelConfig, PermutationConfig
from .metrics import MetricFunc


@dataclass
class FSModel:
    model: XGBClassifier
    seed: int


@dataclass
class SHAPResult:
    shap_values: pd.DataFrame
    mean_importance: pd.Series


@dataclass
class PermutationOutputs:
    per_feature_deltas: Dict[str, List[float]]
    baseline_metrics: List[float]
    bottom_block_deltas: List[float]


DEFAULT_EVAL_METRIC = "auc"


def train_fs_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_config: ModelConfig,
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        max_depth=model_config.max_depth,
        min_child_weight=model_config.min_child_weight,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        reg_lambda=model_config.reg_lambda,
        learning_rate=model_config.eta,
        n_estimators=model_config.n_estimators,
        random_state=seed,
        n_jobs=model_config.n_jobs,
        tree_method=model_config.tree_method,
        objective="binary:logistic",
        eval_metric=DEFAULT_EVAL_METRIC,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def train_fs_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_config: ModelConfig,
) -> List[FSModel]:
    ensemble: List[FSModel] = []
    for idx in range(model_config.n_models):
        seed = model_config.random_state_base + idx
        model = train_fs_model(X_train, y_train, X_valid, y_valid, model_config, seed)
        ensemble.append(FSModel(model=model, seed=seed))
    return ensemble


def compute_shap_importance(models: Iterable[FSModel], X_eval: pd.DataFrame) -> SHAPResult:
    shap_values: List[pd.Series] = []
    for item in models:
        explainer = shap.TreeExplainer(item.model.get_booster())
        values = explainer.shap_values(X_eval)
        if isinstance(values, list):
            # Class-wise list; take positive class
            values = values[1]
        abs_values = np.abs(values)
        mean_abs = abs_values.mean(axis=0)
        shap_values.append(pd.Series(mean_abs, index=X_eval.columns, name=f"model_{item.seed}"))
    shap_df = pd.DataFrame(shap_values)
    mean_importance = shap_df.mean(axis=0)
    return SHAPResult(shap_values=shap_df, mean_importance=mean_importance)


def _predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1]


def compute_permutation_importance(
    models: Iterable[FSModel],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    top_features: List[str],
    bottom_features: List[str],
    metric_func: MetricFunc,
    perm_config: PermutationConfig,
    rng: np.random.Generator,
) -> PermutationOutputs:
    feature_indices = {name: idx for idx, name in enumerate(X_eval.columns)}
    X_eval_np = X_eval.to_numpy()
    y_eval_np = y_eval.to_numpy()

    per_feature_deltas: Dict[str, List[float]] = {feat: [] for feat in top_features}
    baseline_metrics: List[float] = []
    bottom_block_deltas: List[float] = []

    for fs_model in models:
        baseline_pred = _predict_proba(fs_model.model, X_eval_np)
        baseline_metric = metric_func(y_eval_np, baseline_pred, None)
        baseline_metrics.append(baseline_metric)

        for feat in top_features:
            idx = feature_indices[feat]
            permuted = X_eval_np.copy()
            permuted_col = np.array(permuted[:, idx])
            rng.shuffle(permuted_col)
            permuted[:, idx] = permuted_col
            perm_pred = _predict_proba(fs_model.model, permuted)
            perm_metric = metric_func(y_eval_np, perm_pred, None)
            delta = baseline_metric - perm_metric
            per_feature_deltas[feat].append(delta)

        if bottom_features:
            permuted = X_eval_np.copy()
            for feat in bottom_features:
                idx = feature_indices[feat]
                permuted_col = np.array(permuted[:, idx])
                rng.shuffle(permuted_col)
                permuted[:, idx] = permuted_col
            perm_pred = _predict_proba(fs_model.model, permuted)
            perm_metric = metric_func(y_eval_np, perm_pred, None)
            bottom_block_deltas.append(baseline_metric - perm_metric)

    return PermutationOutputs(
        per_feature_deltas=per_feature_deltas,
        baseline_metrics=baseline_metrics,
        bottom_block_deltas=bottom_block_deltas,
    )


def gain_importance(model: XGBClassifier, feature_names: Iterable[str]) -> pd.Series:
    gain_scores = model.get_booster().get_score(importance_type="gain")
    data = {feat: gain_scores.get(feat, 0.0) for feat in feature_names}
    return pd.Series(data)

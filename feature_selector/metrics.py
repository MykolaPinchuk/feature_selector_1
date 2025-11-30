from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MetricFunc = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


def _weighted_prauc_at_rate(
    y_true: np.ndarray, y_score: np.ndarray, target_pos_rate: float = 0.1
) -> float:
    positives = y_true == 1
    n_pos = positives.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    w_pos = 1.0
    weight_ratio = (n_pos * (1 - target_pos_rate)) / (target_pos_rate * n_neg)
    w_neg = max(weight_ratio, 1e-6)
    sample_weight = np.where(positives, w_pos, w_neg)
    return average_precision_score(y_true, y_score, sample_weight=sample_weight)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    return roc_auc_score(y_true, y_score, sample_weight=sample_weight)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    return average_precision_score(y_true, y_score, sample_weight=sample_weight)


def pr_auc_at_10(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    return _weighted_prauc_at_rate(y_true, y_score, target_pos_rate=0.1)


METRIC_REGISTRY: Dict[str, MetricFunc] = {
    "roc_auc": roc_auc,
    "prauc": pr_auc,
    "prauc_at_10": pr_auc_at_10,
}


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {name: func(y_true, y_score, None) for name, func in METRIC_REGISTRY.items()}


def get_metric(metric_name: str) -> MetricFunc:
    normalized = metric_name.lower()
    if normalized not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{metric_name}'")
    return METRIC_REGISTRY[normalized]


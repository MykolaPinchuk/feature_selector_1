from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class StaticFilterReport:
    removed_for_leakage: List[str] = field(default_factory=list)
    removed_constant: List[str] = field(default_factory=list)
    removed_quasi_constant: List[str] = field(default_factory=list)
    removed_missingness: List[str] = field(default_factory=list)
    removed_duplicates: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "leakage": self.removed_for_leakage,
            "constant": self.removed_constant,
            "quasi_constant": self.removed_quasi_constant,
            "missingness": self.removed_missingness,
            "duplicates": self.removed_duplicates,
        }


def drop_leakage_features(
    df: pd.DataFrame, feature_cols: Iterable[str], leakage_features: Iterable[str] | None
) -> Tuple[List[str], List[str]]:
    if not leakage_features:
        return list(feature_cols), []
    leakage_set = set(leakage_features)
    kept = [c for c in feature_cols if c not in leakage_set]
    dropped = [c for c in feature_cols if c in leakage_set]
    return kept, dropped


def drop_constant_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for col in feature_cols:
        series = df[col]
        if series.nunique(dropna=False) == 1:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def drop_quasi_constant_features(
    df: pd.DataFrame, feature_cols: Iterable[str], threshold: float
) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for col in feature_cols:
        series = df[col]
        if series.isna().all():
            dropped.append(col)
            continue
        top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= threshold:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def drop_high_missingness_features(
    df: pd.DataFrame, feature_cols: Iterable[str], threshold: float
) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for col in feature_cols:
        missing_rate = df[col].isna().mean()
        if missing_rate >= threshold:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def drop_duplicate_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[List[str], List[str]]:
    if not feature_cols:
        return [], []
    hashes = {}
    dropped: List[str] = []
    kept: List[str] = []
    for col in feature_cols:
        col_hash = pd.util.hash_pandas_object(df[col], index=False).sum()
        if col_hash in hashes:
            dropped.append(col)
        else:
            hashes[col_hash] = col
            kept.append(col)
    return kept, dropped


def apply_static_filters(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    leakage_features: Iterable[str] | None,
    quasi_constant_threshold: float,
    missingness_threshold: float,
) -> tuple[list[str], StaticFilterReport]:
    report = StaticFilterReport()
    features, dropped = drop_leakage_features(df, feature_cols, leakage_features)
    report.removed_for_leakage = dropped

    features, dropped = drop_constant_features(df, features)
    report.removed_constant = dropped

    features, dropped = drop_quasi_constant_features(df, features, quasi_constant_threshold)
    report.removed_quasi_constant = dropped

    features, dropped = drop_high_missingness_features(df, features, missingness_threshold)
    report.removed_missingness = dropped

    features, dropped = drop_duplicate_features(df, features)
    report.removed_duplicates = dropped

    return features, report


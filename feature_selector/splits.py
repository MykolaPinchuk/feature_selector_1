from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass
class GlobalSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FSSplits:
    train_fs: pd.DataFrame
    holdout_fs: pd.DataFrame


def time_ordered_split(df: pd.DataFrame, time_col: str, config: SplitConfig) -> GlobalSplits:
    config.validate()
    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * config.train_frac)
    val_end = train_end + int(n * config.val_frac)
    if train_end <= 0 or val_end <= train_end:
        raise ValueError("Not enough rows to satisfy split configuration")

    train = ordered.iloc[:train_end]
    val = ordered.iloc[train_end:val_end]
    test = ordered.iloc[val_end:]
    if len(test) == 0:
        raise ValueError("Test split ended up empty; adjust SplitConfig")
    return GlobalSplits(train=train, val=val, test=test)


def create_fs_splits(train_df: pd.DataFrame, time_col: str, config: SplitConfig) -> FSSplits:
    ordered = train_df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    holdout_size = max(1, int(n * config.fs_holdout_frac))
    holdout_fs = ordered.iloc[-holdout_size:]
    train_fs = ordered.iloc[: n - holdout_size]
    if len(train_fs) == 0:
        raise ValueError("train_fs is empty; reduce fs_holdout_frac")
    return FSSplits(train_fs=train_fs, holdout_fs=holdout_fs)


def contiguous_subsample(df: pd.DataFrame, max_rows: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    start_max = len(df) - max_rows
    start_idx = rng.integers(0, max(1, start_max + 1))
    end_idx = start_idx + max_rows
    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def sample_eval_slice(df: pd.DataFrame, sample_size: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    idx = rng.choice(len(df), size=sample_size, replace=False)
    return df.iloc[idx].reset_index(drop=True)


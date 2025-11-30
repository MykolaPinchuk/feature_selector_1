from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class DataLoaderError(RuntimeError):
    pass


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pqt", ".pkl", ".pickle"}


def load_dataset(source: str) -> pd.DataFrame:
    """Load a dataset from disk or from pydataset if a file isn't found."""
    path = Path(source)
    if path.exists():
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DataLoaderError(f"Unsupported extension: {ext}")
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".parquet", ".pqt"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_pickle(path)
        return df

    try:
        from pydataset import data  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise DataLoaderError("pydataset is required to load built-in datasets") from exc

    try:
        df = data(source)
    except ValueError as exc:
        raise DataLoaderError(f"Dataset '{source}' is not available via pydataset") from exc
    return df


def ensure_binary_target(df: pd.DataFrame, target_col: str) -> None:
    values = sorted(df[target_col].dropna().unique())
    if len(values) > 2 or any(v not in {0, 1} for v in values):
        raise DataLoaderError(
            f"Target column '{target_col}' must be binary ints \"0/1\". Found unique values: {values[:5]}"
        )


def ensure_numeric_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> None:
    non_numeric: list[str] = []
    for col in feature_cols:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            continue
        non_numeric.append(col)
    if non_numeric:
        raise DataLoaderError(
            "Non-numeric feature columns detected. Please provide pre-encoded inputs: "
            + ", ".join(non_numeric)
        )


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    series = df[time_col]
    try:
        # Convert to string first to handle integer years cleanly.
        parsed = pd.to_datetime(series.astype(str))
    except Exception as exc:  # pragma: no cover
        raise DataLoaderError(f"Unable to parse time column '{time_col}'") from exc
    return parsed



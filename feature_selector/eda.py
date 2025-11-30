from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hist_plot(series: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(series.dropna(), bins=40, kde=False, color="steelblue")
    else:
        sns.countplot(x=series)
        plt.xticks(rotation=45)
    plt.title(f"Distribution: {series.name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_eda_report(df: pd.DataFrame, output_dir: Path, dataset_name: str | None = None) -> Path:
    output_dir = _ensure_dir(Path(output_dir))
    figures_dir = _ensure_dir(output_dir / "figures")
    name = dataset_name or "dataset"

    report_lines: list[str] = []
    report_lines.append(f"# EDA Report – {name}")
    report_lines.append("")
    report_lines.append(f"Rows: {len(df):,}, Columns: {df.shape[1]}")
    report_lines.append("")

    report_lines.append("## Histograms / Count Plots")
    report_lines.append("")
    for col in df.columns:
        output_path = figures_dir / f"hist_{col}.png"
        _hist_plot(df[col], output_path)
        report_lines.append(f"### {col}")
        report_lines.append(f"![{col}](figures/hist_{col}.png)")
        report_lines.append("")

    report_lines.append("## Summary Statistics")
    report_lines.append("")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T if len(numeric_cols) else pd.DataFrame()
    report_lines.append("| Column | dtype | Mean | Std | Min | 25% | 50% | 75% | Max | Skew | Kurtosis |")
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for col in numeric_cols:
        series = df[col]
        desc = stats.loc[col]
        skew = series.skew()
        kurt = series.kurtosis()
        report_lines.append(
            f"| {col} | {series.dtype} | {desc['mean']:.3f} | {desc['std']:.3f} | {desc['min']:.3f} | "
            f"{desc['25%']:.3f} | {desc['50%']:.3f} | {desc['75%']:.3f} | {desc['max']:.3f} | "
            f"{skew:.3f} | {kurt:.3f} |"
        )
    report_lines.append("")

    report_lines.append("## Correlation Matrix (numeric vs target)")
    report_lines.append("")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        report_lines.append(corr.to_string())
    else:
        report_lines.append("Not enough numeric columns for correlation matrix.")
    report_lines.append("")

    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines))
    return report_path

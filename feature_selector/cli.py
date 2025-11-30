from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import FeatureSelectorConfig, ModelConfig, PermutationConfig, SplitConfig
from .data import DataLoaderError, load_dataset
from .pipeline import FeatureSelectorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selector pipeline for XGB classifiers")
    parser.add_argument(
        "--data-source",
        default="rwm5yr",
        help="CSV/Parquet/Pickle path or pydataset name (default: rwm5yr)",
    )
    parser.add_argument("--target-col", required=True, help="Binary target column name")
    parser.add_argument("--time-col", required=True, help="Time column for ordering")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--fs-holdout-frac", type=float, default=0.25)
    parser.add_argument("--fs-eval-size", type=int, default=5000)
    parser.add_argument("--train-fs-subsample", type=int, default=15000)
    parser.add_argument("--main-metric", default="roc_auc", choices=["roc_auc", "prauc", "prauc_at_10"])
    parser.add_argument("--shap-top-k", type=int, default=60)
    parser.add_argument("--shap-bottom-k", type=int, default=60)
    parser.add_argument("--output-json", help="Optional path to dump JSON summary")
    parser.add_argument("--leakage-features", nargs="*", help="Known leakage features to drop")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for storing detailed artifacts (default: results)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FeatureSelectorConfig:
    split_config = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        fs_holdout_frac=args.fs_holdout_frac,
        fs_eval_sample_size=args.fs_eval_size,
        train_fs_subsample_size=args.train_fs_subsample,
    )
    perm_config = PermutationConfig(
        shap_top_k=args.shap_top_k,
        shap_bottom_k=args.shap_bottom_k,
    )
    model_config = ModelConfig()

    return FeatureSelectorConfig(
        target_col=args.target_col,
        time_col=args.time_col,
        main_metric=args.main_metric,
        split_config=split_config,
        permutation_config=perm_config,
        model_config=model_config,
        leakage_features=args.leakage_features,
        random_seed=args.random_seed,
    )


def result_to_dict(result) -> Dict[str, Any]:
    return {
        "must_keep": result.must_keep,
        "useful": result.useful,
        "dropped": result.dropped,
        "overfit_suspect": result.overfit_suspect,
        "chosen_features": result.chosen_features,
        "candidate_metrics": result.candidate_metrics,
        "static_filter_report": result.static_filter_report,
    }


def _slugify(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return safe or "dataset"


def _build_run_directory(base: Path, data_source: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = Path(data_source)
    dataset_label = path.stem if path.exists() else data_source
    slug = _slugify(dataset_label)
    return base / f"{slug}_{timestamp}"


def persist_detailed_outputs(result, run_dir: Path, summary: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    perm = result.permutation_importance.copy()
    perm.index.name = "feature"
    perm.to_csv(run_dir / "permutation_importance.csv")

    shap_df = result.shap_importance.rename("shap_importance").to_frame()
    shap_df.index.name = "feature"
    shap_df.to_csv(run_dir / "shap_importance.csv")

    gain_df = result.gain_importance.rename("gain_importance").to_frame()
    gain_df.index.name = "feature"
    gain_df.to_csv(run_dir / "gain_importance.csv")

    cand_df = pd.DataFrame(result.candidate_metrics).T
    cand_df.index.name = "candidate_set"
    cand_df.to_csv(run_dir / "candidate_metrics.csv")

    (run_dir / "chosen_features.txt").write_text("\n".join(result.chosen_features))
    (run_dir / "must_keep.txt").write_text("\n".join(result.must_keep))
    (run_dir / "dropped_features.txt").write_text("\n".join(result.dropped))

    (run_dir / "static_filters.json").write_text(json.dumps(result.static_filter_report, indent=2))

    print(f"Detailed artifacts saved to {run_dir}")


def main() -> None:
    args = parse_args()
    try:
        df = load_dataset(args.data_source)
    except DataLoaderError as exc:
        raise SystemExit(str(exc))

    config = build_config(args)
    pipeline = FeatureSelectorPipeline(config)
    result = pipeline.run(df)

    summary = result_to_dict(result)
    print(json.dumps(summary, indent=2))

    results_base = Path(args.results_dir)
    run_dir = _build_run_directory(results_base, args.data_source)
    persist_detailed_outputs(result, run_dir, summary)

    if args.output_json:
        path = Path(args.output_json)
        path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()

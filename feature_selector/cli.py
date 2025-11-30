from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from .config import FeatureSelectorConfig, ModelConfig, PermutationConfig, SplitConfig
from .data import DataLoaderError, load_dataset
from .eda import generate_eda_report
from .pipeline import FeatureSelectorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selector pipeline for XGB classifiers")
    parser.add_argument("--config", help="Path to experiment YAML config")
    parser.add_argument("--dataset-name", help="Friendly dataset identifier")
    parser.add_argument("--data-source", help="CSV/Parquet/Pickle path or pydataset name")
    parser.add_argument("--target-col", help="Binary target column name")
    parser.add_argument("--time-col", help="Time column for ordering")
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--val-frac", type=float)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--fs-holdout-frac", type=float)
    parser.add_argument("--fs-eval-size", type=int)
    parser.add_argument("--train-fs-subsample", type=int)
    parser.add_argument("--main-metric", choices=["roc_auc", "prauc", "prauc_at_10"])
    parser.add_argument("--shap-top-k", type=int)
    parser.add_argument("--shap-bottom-k", type=int)
    parser.add_argument("--output-json", help="Optional path to dump JSON summary")
    parser.add_argument("--leakage-features", nargs="*", help="Known leakage features to drop")
    parser.add_argument("--drop-features", nargs="*", help="Columns to exclude before FS")
    parser.add_argument(
        "--target-binarize-threshold",
        type=float,
        help="If set, convert target to 1 when value > threshold (useful for count targets)",
    )
    parser.add_argument("--model-max-depth", type=int, help="Override FS model max_depth")
    parser.add_argument("--model-min-child-weight", type=float, help="Override FS model min_child_weight")
    parser.add_argument("--model-subsample", type=float, help="Override FS model subsample")
    parser.add_argument("--model-colsample-bytree", type=float, help="Override FS model colsample_bytree")
    parser.add_argument("--model-reg-lambda", type=float, help="Override FS model L2 regularization")
    parser.add_argument("--model-reg-alpha", type=float, help="Override FS model L1 regularization")
    parser.add_argument("--model-gamma", type=float, help="Override FS model gamma")
    parser.add_argument("--model-eta", type=float, help="Override FS model learning rate")
    parser.add_argument("--model-n-estimators", type=int, help="Override FS model n_estimators")
    parser.add_argument("--model-scale-pos-weight", type=float, help="Override scale_pos_weight")
    parser.add_argument(
        "--auto-scale-pos-weight",
        action="store_true",
        help="Automatically set scale_pos_weight to negative/positive ratio on TRAIN",
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--results-dir", help="Directory for storing detailed artifacts")
    parser.add_argument("--eda-output-dir", help="Directory to store EDA artifacts")
    parser.add_argument("--run-eda", action="store_true", help="Enable EDA generation")
    return parser.parse_args()


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {cfg_path} must contain a mapping")
    return data


def _update_dataclass(instance, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(instance, key):
            setattr(instance, key, value)


def _merge_lists(config_values: Optional[Iterable[str]], cli_values: Optional[Iterable[str]]) -> Optional[list[str]]:
    merged: list[str] = []
    for source in (config_values, cli_values):
        if not source:
            continue
        merged.extend(source)
    if not merged:
        return None
    seen = dict.fromkeys(merged)
    return list(seen.keys())


def build_config(args: argparse.Namespace, config_data: Dict[str, Any]) -> tuple[FeatureSelectorConfig, Dict[str, Any]]:
    def cfg_value(key: str, default=None):
        return config_data.get(key, default)

    data_source = args.data_source or cfg_value("data_source") or "rwm5yr"
    dataset_name = args.dataset_name or cfg_value("dataset_name")
    dataset_slug = dataset_name or _slugify(Path(str(data_source)).stem if data_source else "dataset")
    if not dataset_name:
        dataset_name = dataset_slug

    target_col = args.target_col or cfg_value("target_col")
    time_col = args.time_col or cfg_value("time_col")
    if not target_col or not time_col:
        raise SystemExit("target_col and time_col must be specified via CLI or config file")

    main_metric = args.main_metric or cfg_value("main_metric", "roc_auc")

    split_config = SplitConfig()
    _update_dataclass(split_config, cfg_value("split_config", {}))
    overrides = {
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "fs_holdout_frac": args.fs_holdout_frac,
        "fs_eval_sample_size": args.fs_eval_size,
        "train_fs_subsample_size": args.train_fs_subsample,
    }
    _update_dataclass(split_config, overrides)

    perm_config = PermutationConfig()
    _update_dataclass(perm_config, cfg_value("permutation_config", {}))
    perm_overrides = {"shap_top_k": args.shap_top_k, "shap_bottom_k": args.shap_bottom_k}
    _update_dataclass(perm_config, perm_overrides)

    model_config = ModelConfig()
    _update_dataclass(model_config, cfg_value("model_config", {}))
    model_overrides = {
        "max_depth": args.model_max_depth,
        "min_child_weight": args.model_min_child_weight,
        "subsample": args.model_subsample,
        "colsample_bytree": args.model_colsample_bytree,
        "reg_lambda": args.model_reg_lambda,
        "reg_alpha": args.model_reg_alpha,
        "gamma": args.model_gamma,
        "eta": args.model_eta,
        "n_estimators": args.model_n_estimators,
        "scale_pos_weight": args.model_scale_pos_weight,
    }
    _update_dataclass(model_config, {k: v for k, v in model_overrides.items() if v is not None})
    if args.auto_scale_pos_weight or cfg_value("auto_scale_pos_weight"):
        model_config.auto_scale_pos_weight = True

    leakage_features = _merge_lists(cfg_value("leakage_features"), args.leakage_features)
    drop_features = _merge_lists(cfg_value("drop_features"), args.drop_features)

    target_threshold = args.target_binarize_threshold
    if target_threshold is None:
        target_threshold = cfg_value("target_binarize_threshold")

    random_seed = args.random_seed if args.random_seed is not None else cfg_value("random_seed", 13)

    results_dir = args.results_dir or cfg_value("results_dir")
    if results_dir is None:
        results_dir = Path("experiments") / dataset_slug / "runs"
    results_dir = Path(results_dir)

    eda_output_dir = args.eda_output_dir or cfg_value("eda_output_dir")
    if eda_output_dir is None:
        eda_output_dir = results_dir.parent / "eda"
    eda_output_dir = Path(eda_output_dir)

    run_eda = cfg_value("run_eda", False) or args.run_eda

    fs_config = FeatureSelectorConfig(
        target_col=target_col,
        time_col=time_col,
        main_metric=main_metric,
        split_config=split_config,
        permutation_config=perm_config,
        model_config=model_config,
        leakage_features=leakage_features,
        drop_features=drop_features,
        target_binarize_threshold=target_threshold,
        random_seed=random_seed,
        dataset_name=dataset_name,
    )

    experiment_settings = {
        "data_source": data_source,
        "dataset_name": dataset_name,
        "results_dir": results_dir,
        "eda_output_dir": eda_output_dir,
        "run_eda": run_eda,
        "output_json": args.output_json or cfg_value("output_json"),
    }
    return fs_config, experiment_settings


def result_to_dict(result) -> Dict[str, Any]:
    return {
        "must_keep": result.must_keep,
        "useful": result.useful,
        "dropped": result.dropped,
        "overfit_suspect": result.overfit_suspect,
        "chosen_features": result.chosen_features,
        "candidate_metrics": result.candidate_metrics,
        "static_filter_report": result.static_filter_report,
        "excluded_features": result.excluded_features,
        "final_split_metrics": result.final_split_metrics,
    }


def _slugify(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return safe or "dataset"


def _build_run_directory(base: Path, dataset_label: Optional[str], data_source: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label = dataset_label
    if not label:
        path = Path(data_source)
        label = path.stem if path.exists() else data_source
    slug = _slugify(label)
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

    if result.final_split_metrics:
        final_df = pd.DataFrame(result.final_split_metrics).T
        final_df.index.name = "split"
        final_df.to_csv(run_dir / "final_metrics.csv")

    (run_dir / "chosen_features.txt").write_text("\n".join(result.chosen_features))
    (run_dir / "must_keep.txt").write_text("\n".join(result.must_keep))
    (run_dir / "dropped_features.txt").write_text("\n".join(result.dropped))

    (run_dir / "static_filters.json").write_text(json.dumps(result.static_filter_report, indent=2))

    print(f"Detailed artifacts saved to {run_dir}")


def main() -> None:
    args = parse_args()
    config_data = _load_config_file(args.config)
    fs_config, experiment_settings = build_config(args, config_data)

    data_source = experiment_settings["data_source"]
    try:
        df = load_dataset(data_source)
    except DataLoaderError as exc:
        raise SystemExit(str(exc))

    if experiment_settings["run_eda"]:
        generate_eda_report(df, experiment_settings["eda_output_dir"], experiment_settings["dataset_name"])

    pipeline = FeatureSelectorPipeline(fs_config)
    result = pipeline.run(df)

    summary = result_to_dict(result)
    print(json.dumps(summary, indent=2))

    results_base = experiment_settings["results_dir"]
    run_dir = _build_run_directory(results_base, experiment_settings["dataset_name"], data_source)
    persist_detailed_outputs(result, run_dir, summary)

    output_json = experiment_settings.get("output_json")
    if output_json:
        path = Path(output_json)
        path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()

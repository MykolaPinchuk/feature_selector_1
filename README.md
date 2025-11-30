## feature_selector_1

POC for a computationally efficient feature selector targeted at time-ordered binary classification problems built with XGBoost. The pipeline follows the spec in `prd.md`: static pre-filtering, time-aware splits, an ensemble of light FS models with SHAP triage, targeted permutation FI, overfit diagnostics, and a small number of ablation models on VAL.

### Structure

- `feature_selector/` – Python package with the pipeline, configs, metrics, CLI, and helpers.
- `experiments/` – dataset-specific folders (configs + generated runs/EDA).
- `requirements.txt` – dependencies (Python 3.11).
- `agents.md`, `prd.md` – design docs/instructions.

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Usage

Provide a YAML config under `experiments/<dataset>/config.yml` (see `experiments/rwm5yr/config.yml` for a template). Then run:

```bash
python -m feature_selector.cli --config experiments/rwm5yr/config.yml
```

The config declares the data source (CSV/Parquet path or `pydataset` name), target/time columns, preprocessing knobs (drop lists, binarization threshold), split ratios, and any model overrides. CLI flags can still override config values if needed (e.g., `--model-max-depth 4`).

Each run writes artifacts to `experiments/<dataset>/runs/<timestamp>/`, and (if enabled) EDA bundles go to `experiments/<dataset>/eda/`.

High-level pipeline steps:

1. Static filters for leakage, constant/quasi-constant, high-missingness, and duplicate features.
2. Time-ordered TRAIN/VAL/TEST splits plus TRAIN_FS/HOLDOUT_FS inside TRAIN.
3. Row subsampling + an ensemble of lightweight FS XGB models.
4. SHAP on HOLDOUT_FS (`FS_EVAL`), triage into top/middle/bottom blocks.
5. Permutation FI on the SHAP top block plus optional grouped bottom block permutation.
6. Aggregation/classification into must-keep/useful/drop sets, overfit diagnostics via gain vs permutation.
7. Ablation models on VAL (all features vs filtered vs aggressive) with ROC-AUC, PR-AUC, and PR-AUC@10%.

Outputs are printed as JSON and stored inside the run directory (summary, permutation/SHAP/gain CSVs, metrics, feature lists, static-filter report). Programmatic use is still available:

```python
from feature_selector import FeatureSelectorConfig, FeatureSelectorPipeline
from feature_selector.data import load_dataset

df = load_dataset(\"rwm5yr\")
config = FeatureSelectorConfig(target_col=\"self\", time_col=\"year\")
result = FeatureSelectorPipeline(config).run(df)
print(result.must_keep)
```

### Notes

- Target column must be binary `0/1`. If the data contains raw categoricals, the loader raises an error to avoid implicit encoding.
- PRAUC@10 is implemented via sample-weight adjustment to simulate a 10% minority rate.
- Adjust FS parameters via CLI flags or directly on the config dataclasses (e.g., number of FS models, SHAP block sizes, permutation thresholds). Use `drop_features` to exclude columns upfront, `target_binarize_threshold` to convert count targets (e.g., `hospvis > 0`) into binary labels, and `model_config` overrides (depth, min-child weight, subsample/colsample, reg_lambda/alpha, gamma, eta, estimators, `scale_pos_weight` or `auto_scale_pos_weight`) to tighten regularization when needed.
- Exploratory data analysis artifacts (schema tables plus per-column histograms) live under `experiments/<dataset>/eda/` when `run_eda: true`.

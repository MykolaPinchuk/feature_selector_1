## feature_selector_1

POC for a computationally efficient feature selector targeted at time-ordered binary classification problems built with XGBoost. The pipeline follows the spec in `prd.md`: static pre-filtering, time-aware splits, an ensemble of light FS models with SHAP triage, targeted permutation FI, overfit diagnostics, and a small number of ablation models on VAL.

### Structure

- `feature_selector/` – Python package with the pipeline, configs, metrics, CLI, and helpers.
- `requirements.txt` – dependencies (Python 3.11).
- `agents.md`, `prd.md` – design docs/instructions.

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

Run the CLI (defaults to the `rwm5yr` dataset from `pydataset`, `self` as the target, `year` as the time column):

```bash
python -m feature_selector.cli \
  --target-col self \
  --time-col year \
  --main-metric roc_auc \
  --results-dir results \
  --output-json artifacts/rwm5yr_fs.json
```

To point at your own data, provide `--data-source path/to/data.csv` (Parquet and pickle supported). Columns must already be numeric (categorical encoding handled upstream). The CLI performs:

1. Static filters for leakage, constant/quasi-constant, high-missingness, and duplicate features.
2. Time-ordered TRAIN/VAL/TEST splits plus TRAIN_FS/HOLDOUT_FS inside TRAIN.
3. Row subsampling + an ensemble of lightweight FS XGB models.
4. SHAP on HOLDOUT_FS (`FS_EVAL`), triage into top/middle/bottom blocks.
5. Permutation FI on the SHAP top block plus optional grouped bottom block permutation.
6. Aggregation/classification into must-keep/useful/drop sets, overfit diagnostics via gain vs permutation.
7. Ablation models on VAL (all features vs filtered vs aggressive) with ROC-AUC, PR-AUC, and PR-AUC@10%.

Outputs are printed as JSON (and optionally written to `--output-json`). Programmatic use is available via:

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
- Adjust FS parameters via CLI flags or directly on the config dataclasses (e.g., number of FS models, SHAP block sizes, permutation thresholds).
- Each run also writes detailed artifacts (summary JSON, permutation/shap/gain CSVs, candidate metric table, and feature lists) to `results/<dataset>_<timestamp>/` for manual inspection.

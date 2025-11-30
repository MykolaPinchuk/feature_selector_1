# rwm5yr Experiment

- **Source**: `pydataset` (`rwm5yr` German health registry).
- **Target**: `hospvis > 0` (binarized).
- **Time column**: `year` (1984–1988).
- **Feature drops**: `id` (avoids memorization).
- **Model overrides**: Moderate depth (3), high `min_child_weight` (30), subsample/cosample 0.6, gamma 0.5, L1/L2 regularization, auto `scale_pos_weight`.

Run:

```bash
python -m feature_selector.cli --config experiments/rwm5yr/config.yml
```

Artifacts land in `experiments/rwm5yr/runs/<timestamp>/` and the EDA bundle is saved under `experiments/rwm5yr/eda/`.

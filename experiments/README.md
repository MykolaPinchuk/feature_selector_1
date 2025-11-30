# Experiments

Each dataset lives in its own subdirectory (e.g., `experiments/rwm5yr`). A dataset folder typically contains:

- `config.yml`: declarative settings for data source, target/time columns, preprocessing knobs, and model overrides.
- `runs/`: auto-generated artifacts for each pipeline execution (timestamped subfolders).
- `eda/`: generated exploratory analysis (report + histogram PNGs) when `run_eda` is enabled in the config.
- Optional dataset-specific notes/README files.

Run an experiment via the CLI:

```bash
python -m feature_selector.cli --config experiments/<dataset>/config.yml
```

You can still override individual settings on the CLI (e.g., `--run-eda` or `--model-max-depth 4`) without editing the config.

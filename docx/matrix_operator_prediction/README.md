# Matrix-operator prediction figures

Place prediction CSV files (with columns `duration_us`, `pred_total_us`, `m`, `n`, `k`, `type`) under **`predictions_input/`** next to `plot_prediction_effect.py`. Files should be named `*_predicted.csv`.

Figures and regenerated metric CSVs are written to **`figures/`** in this directory (`prediction_effect_*.{png,pdf}`, `prediction_metrics_*.csv`).

Bundled **`data/`** contains example `prediction_metrics_*.csv` summaries (no separate `derived_metrics/` subdirectory).

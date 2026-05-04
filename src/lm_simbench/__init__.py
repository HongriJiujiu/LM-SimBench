# -*- coding: utf-8 -*-
"""
LM-SimBench benchmark metrics and task helpers.

Import after adding the ``benchmark`` directory to ``PYTHONPATH``, for example::

    PYTHONPATH=/path/to/repo/benchmark/src python your_script.py

    from lm_simbench import mae, iteration_time_metrics_ms, strategy_ranking_summary
"""

from .component import (
    comm_time_augmented_frame,
    component_regression_metrics,
    merge_pred_true_arrays,
    message_size_bucket,
    regression_metrics_by_group,
)
from .metrics import mae, r2_score, regression_metrics, smape, spearman_correlation
from .ranking import (
    evaluations_from_config_dicts,
    macro_averaged_top_k_hit_rate,
    spearman_iteration_time_ranking,
    strategy_ranking_summary,
    top_k_hit_rate,
    top_k_regret_ms,
    top_k_relative_regret,
    true_best_config,
)
from .system import (
    aggregate_config_peak_memory_mb,
    align_configs_to_series,
    feasibility_from_peak_mb,
    iteration_time_metrics_ms,
    memory_feasibility_classification_metrics,
    memory_peak_mae_mb,
    memory_peak_metrics_mb,
)

__all__ = [
    "aggregate_config_peak_memory_mb",
    "align_configs_to_series",
    "comm_time_augmented_frame",
    "component_regression_metrics",
    "evaluations_from_config_dicts",
    "feasibility_from_peak_mb",
    "iteration_time_metrics_ms",
    "macro_averaged_top_k_hit_rate",
    "mae",
    "memory_feasibility_classification_metrics",
    "memory_peak_mae_mb",
    "memory_peak_metrics_mb",
    "merge_pred_true_arrays",
    "message_size_bucket",
    "r2_score",
    "regression_metrics",
    "regression_metrics_by_group",
    "smape",
    "spearman_correlation",
    "spearman_iteration_time_ranking",
    "strategy_ranking_summary",
    "top_k_hit_rate",
    "top_k_regret_ms",
    "top_k_relative_regret",
    "true_best_config",
]

__version__ = "0.1.0"

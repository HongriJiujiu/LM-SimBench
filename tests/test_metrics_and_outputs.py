"""Regression checks for ranking, target alignment and figure output preservation."""
from pathlib import Path
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from lm_simbench import (
    merge_pred_true_arrays, regression_metrics_by_group,
    strategy_ranking_summary, true_best_config, top_k_hit_rate,
    top_k_regret_ms, top_k_relative_regret, spearman_iteration_time_ranking,
)


class RankingTests(unittest.TestCase):
    def test_missing_true_time_cannot_report_zero_regret(self):
        with self.assertRaises(ValueError):
            top_k_regret_ms(["a", "b"], [1, np.nan], [2, 1], 1)

    def test_missing_predictions_cannot_report_a_hit(self):
        with self.assertRaises(ValueError):
            top_k_hit_rate(["a", "b"], [1, 2], [np.nan, np.nan], 1)

    def test_best_config_requires_valid_times(self):
        for values in ([np.nan, np.nan], [1, np.inf], [1, -1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                true_best_config(["a", "b"], values)

    def test_summary_rejects_invalid_candidates_even_without_topk(self):
        for configs, true, predicted in (
            ([], [], []),
            (["a", "a"], [1, 2], [1, 2]),
            (["a", "b"], [[1, 2]], [[1, 2]]),
            (["a"], [1, 2], [1, 2]),
            (["a", "b"], [1, 2], [np.inf, 2]),
        ):
            with self.subTest(configs=configs, true=true), self.assertRaises(ValueError):
                strategy_ranking_summary(configs, true, predicted, ks=[])

    def test_known_shortlist_penalties(self):
        result = strategy_ranking_summary(["slow", "best", "middle"], [30, 10, 20], [5, 15, 10], ks=[1, 2, 3])
        self.assertFalse(result["top1_hit"])
        self.assertEqual(result["top1_regret_ms"], 20)
        self.assertEqual(result["top1_relative_regret"], 2)
        self.assertEqual(result["top2_regret_ms"], 10)
        self.assertTrue(result["top3_hit"])
        self.assertEqual(result["top3_regret_ms"], 0)
        self.assertEqual(result["spearman_rho"], -1)

    def test_any_tied_measured_optimum_counts_as_a_hit(self):
        self.assertTrue(top_k_hit_rate(["a", "b", "c"], [10, 10, 20], [3, 1, 2], 1))

    def test_predicted_ties_preserve_candidate_order(self):
        self.assertFalse(top_k_hit_rate(["a", "b"], [20, 10], [1, 1], 1))
        self.assertEqual(top_k_regret_ms(["a", "b"], [20, 10], [1, 1], 1), 10)

    def test_candidate_generator_is_reused_for_all_metrics(self):
        result = strategy_ranking_summary(iter(["a", "b"]), [10, 20], [2, 1], ks=[1, 2])
        self.assertEqual(result["top1_regret_ms"], 10)
        self.assertEqual(result["top2_regret_ms"], 0)

    def test_k_must_be_a_positive_integer(self):
        for k in (0, -1, 1.5, "1"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                top_k_hit_rate(["a"], [1], [1], k)

    def test_relative_regret_denominator_must_be_positive(self):
        with self.assertRaises(ValueError):
            top_k_relative_regret(["a"], [0], [0], 1, epsilon_ms=0)

    def test_spearman_uses_the_same_finite_candidate_policy(self):
        with self.assertRaises(ValueError):
            spearman_iteration_time_ranking([1, 2], [1, np.nan])


class AlignmentTests(unittest.TestCase):
    def test_same_target_name_aligns_by_key(self):
        true = pd.DataFrame({"config": ["a", "b"], "time": [1, 2]})
        predicted = pd.DataFrame({"config": ["b", "a"], "time": [20, 10]})
        yt, yp = merge_pred_true_arrays(true, predicted, ["config"], "time", "time")
        np.testing.assert_array_equal(yt, [1, 2])
        np.testing.assert_array_equal(yp, [10, 20])

    def test_duplicate_keys_cannot_multiply_samples(self):
        unique = pd.DataFrame({"config": ["a"], "time": [1]})
        duplicate = pd.DataFrame({"config": ["a", "a"], "time": [1, 2]})
        for true, predicted in ((duplicate, unique), (unique, duplicate), (duplicate, duplicate)):
            with self.subTest(true_rows=len(true), predicted_rows=len(predicted)):
                with self.assertRaises(pd.errors.MergeError):
                    merge_pred_true_arrays(true, predicted, ["config"], "time", "time")

    def test_null_keys_cannot_match_each_other(self):
        frame = pd.DataFrame({"config": [None], "time": [1]})
        with self.assertRaises(ValueError):
            merge_pred_true_arrays(frame, frame, ["config"], "time", "time")

    def test_internal_column_names_do_not_replace_keys(self):
        true = pd.DataFrame({"__true_target": ["a"], "__pred_target": ["b"], "time": [1]})
        predicted = true.assign(time=2)
        yt, yp = merge_pred_true_arrays(true, predicted, ["__true_target", "__pred_target"], "time", "time")
        np.testing.assert_array_equal(yt, [1])
        np.testing.assert_array_equal(yp, [2])

    def test_outer_join_retains_unmatched_targets(self):
        true = pd.DataFrame({"config": ["a"], "time": [1]})
        predicted = pd.DataFrame({"config": ["b"], "time": [2]})
        yt, yp = merge_pred_true_arrays(true, predicted, ["config"], "time", "time", how="outer")
        self.assertEqual(len(yt), 2)
        self.assertEqual(np.count_nonzero(np.isnan(yt)), 1)
        self.assertEqual(np.count_nonzero(np.isnan(yp)), 1)

    def test_group_sample_count_matches_valid_pairs(self):
        frame = pd.DataFrame({"group": ["a"] * 4, "true": [1, 2, np.nan, 4], "pred": [1, 4, 3, np.inf]})
        row = regression_metrics_by_group(frame, group_cols=["group"], y_col_true="true", y_col_pred="pred").iloc[0]
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["n_total"], 4)
        self.assertEqual(row["mae"], 1)


class PredictionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "docx/case_2/matrix_operator_prediction/plot_prediction_effect.py"
        spec = importlib.util.spec_from_file_location("prediction_plot", path)
        cls.plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.plot)

    def run_bad_input(self, content, exception):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inputs, outputs = base / "data", base / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            previous = outputs / "prediction_effect_previous.pdf"
            previous.write_bytes(b"existing figure")
            summary = outputs / "prediction_metrics_current.csv"
            summary.write_text("existing metrics", encoding="utf-8")
            if content is not None:
                (inputs / "current_predicted.csv").write_text(content, encoding="utf-8")
            with patch.object(self.plot, "INPUT_DIR", inputs), patch.object(self.plot, "OUTPUT_DIR", outputs):
                with self.assertRaises(exception):
                    self.plot.main()
            self.assertEqual(previous.read_bytes(), b"existing figure")
            self.assertEqual(summary.read_text(encoding="utf-8"), "existing metrics")

    def test_missing_input_preserves_outputs(self):
        self.run_bad_input(None, FileNotFoundError)

    def test_bad_schema_preserves_outputs(self):
        self.run_bad_input("type\nMatMul\n", ValueError)

    def test_no_usable_predictions_preserves_outputs(self):
        self.run_bad_input("type,duration_us,pred_total_us,m,n,k\nMatMul,0,0,2,2,2\n", ValueError)


if __name__ == "__main__":
    unittest.main()

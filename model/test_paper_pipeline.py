"""Focused regression tests for paper metrics and result aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from model import model
from model import reporting


class SecurityThresholdTests(unittest.TestCase):
    def test_far_target_is_applied_without_test_labels(self):
        validation_labels = np.asarray([0, 0, 0, 0, 1, 1])
        validation_scores = np.asarray([0.1, 0.2, 0.3, 0.9, 0.7, 0.8])
        calibration = model.calibrate_far_threshold(
            validation_labels,
            validation_scores,
            target_far=0.05,
        )
        validation_metrics = model.biometric_metrics(
            validation_labels,
            validation_scores,
            calibration["threshold"],
        )
        self.assertLessEqual(calibration["far"], 0.05)
        self.assertLessEqual(validation_metrics["far"], 0.05)

    def test_infinite_roc_threshold_rejects_all_scores(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.9, 0.7, 0.8])
        calibration = model.calibrate_far_threshold(
            labels,
            scores,
            target_far=0.0,
        )
        self.assertGreater(calibration["threshold"], float(np.max(scores)))


class ReportingAggregationTests(unittest.TestCase):
    def test_macro_metrics_and_pooled_confusion_counts_are_distinct(self):
        configuration = {
            "users": {
                "user_001": {
                    "interaction_count_metrics": {
                        "5": {
                            "accuracy": 0.5,
                            "precision": 0.5,
                            "recall": 1.0,
                            "f1_score": 2.0 / 3.0,
                            "roc_auc": 0.75,
                            "far": 1.0,
                            "frr": 0.0,
                            "eer": 0.25,
                            "tp": 1,
                            "tn": 0,
                            "fp": 1,
                            "fn": 0,
                            "attempt_count": 2,
                            "average_duration_seconds": 4.0,
                        }
                    }
                },
                "user_002": {
                    "interaction_count_metrics": {
                        "5": {
                            "accuracy": 1.0,
                            "precision": 1.0,
                            "recall": 1.0,
                            "f1_score": 1.0,
                            "roc_auc": 1.0,
                            "far": 0.0,
                            "frr": 0.0,
                            "eer": 0.0,
                            "tp": 1,
                            "tn": 3,
                            "fp": 0,
                            "fn": 0,
                            "attempt_count": 4,
                            "average_duration_seconds": 6.0,
                        }
                    }
                },
            }
        }
        aggregate = reporting.aggregate_k(configuration, 5)
        self.assertAlmostEqual(aggregate["far"], 0.5)
        self.assertEqual(aggregate["fp"], 1)
        self.assertEqual(aggregate["tn"], 3)
        self.assertEqual(aggregate["attempts"], 6)
        self.assertAlmostEqual(aggregate["average_duration_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()

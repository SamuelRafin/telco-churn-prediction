"""Tests for evaluation metrics and threshold tuning."""

import numpy as np
import pytest

from churn.evaluate import (
    ThresholdChoice,
    choose_threshold,
    format_evaluation,
    metrics_at_threshold,
)


@pytest.fixture
def separable_scores():
    """Scores that separate the classes well, with some overlap."""
    rng = np.random.default_rng(7)
    n = 400
    y_true = rng.integers(0, 2, n)
    # Positives score higher on average, but the distributions overlap.
    scores = np.clip(rng.normal(0.35, 0.18, n) + y_true * 0.3, 0.01, 0.99)
    return y_true, scores


def test_choose_threshold_returns_valid_range(separable_scores):
    """The chosen threshold is a probability."""
    y_true, scores = separable_scores
    choice = choose_threshold(y_true, scores, precision_floor=0.5)

    assert isinstance(choice, ThresholdChoice)
    assert 0.0 < choice.threshold < 1.0


def test_choose_threshold_respects_precision_floor(separable_scores):
    """The selected point meets the precision floor."""
    y_true, scores = separable_scores
    floor = 0.55
    choice = choose_threshold(y_true, scores, precision_floor=floor)
    assert choice.precision >= floor - 1e-9


def test_choose_threshold_improves_recall_over_default(separable_scores):
    """Tuning for recall beats the default 0.5 threshold on recall."""
    y_true, scores = separable_scores
    choice = choose_threshold(y_true, scores, precision_floor=0.5)

    default_recall = metrics_at_threshold(y_true, scores, 0.5)["recall"]
    assert choice.recall >= default_recall


def test_lower_floor_allows_higher_recall(separable_scores):
    """Relaxing the precision floor permits at least as much recall."""
    y_true, scores = separable_scores
    strict = choose_threshold(y_true, scores, precision_floor=0.7)
    loose = choose_threshold(y_true, scores, precision_floor=0.4)
    assert loose.recall >= strict.recall


def test_unreachable_floor_falls_back_to_f1(separable_scores):
    """An impossible floor still returns a usable threshold."""
    y_true, scores = separable_scores
    choice = choose_threshold(y_true, scores, precision_floor=0.999)

    assert 0.0 < choice.threshold < 1.0
    assert choice.recall > 0


def test_metrics_at_threshold_confusion_counts_sum(separable_scores):
    """Confusion matrix cells sum to the sample size."""
    y_true, scores = separable_scores
    stats = metrics_at_threshold(y_true, scores, 0.5)

    total = (
        stats["true_negatives"]
        + stats["false_positives"]
        + stats["false_negatives"]
        + stats["true_positives"]
    )
    assert total == len(y_true)


def test_metrics_at_threshold_ranges(separable_scores):
    """All rate metrics fall in [0, 1]."""
    y_true, scores = separable_scores
    stats = metrics_at_threshold(y_true, scores, 0.45)

    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= stats[key] <= 1.0


def test_lower_threshold_increases_recall(separable_scores):
    """Recall is monotonically non-decreasing as the threshold falls."""
    y_true, scores = separable_scores
    high = metrics_at_threshold(y_true, scores, 0.7)["recall"]
    low = metrics_at_threshold(y_true, scores, 0.3)["recall"]
    assert low >= high


def test_extreme_thresholds_behave(separable_scores):
    """At threshold 0 everything is positive; near 1 almost nothing is."""
    y_true, scores = separable_scores

    all_positive = metrics_at_threshold(y_true, scores, 0.0)
    assert all_positive["recall"] == pytest.approx(1.0)
    assert all_positive["true_negatives"] == 0

    none_positive = metrics_at_threshold(y_true, scores, 1.0)
    assert none_positive["true_positives"] == 0


def test_format_evaluation_includes_key_sections(separable_scores):
    """The rendered report mentions the headline metrics."""
    y_true, scores = separable_scores
    choice = choose_threshold(y_true, scores, precision_floor=0.5)

    results = {
        "roc_auc": 0.85,
        "pr_auc": 0.66,
        "threshold_choice": choice.__dict__,
        "at_default_threshold": metrics_at_threshold(y_true, scores, 0.5),
        "at_tuned_threshold": metrics_at_threshold(y_true, scores, choice.threshold),
        "n_test": len(y_true),
    }

    text = format_evaluation(results)
    assert "ROC-AUC" in text
    assert "PR-AUC" in text
    assert "missed churners" in text
    assert "Tuned threshold" in text


def test_threshold_choice_summary_is_readable(separable_scores):
    """ThresholdChoice renders a one-line summary."""
    y_true, scores = separable_scores
    choice = choose_threshold(y_true, scores, precision_floor=0.5)
    summary = choice.summary()

    assert "threshold=" in summary
    assert "recall=" in summary

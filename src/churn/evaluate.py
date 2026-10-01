"""Held-out evaluation and threshold selection.

Accuracy is a trap here (predicting "nobody churns" scores 73%), so the report
leads with ROC-AUC, PR-AUC, recall and precision.

0.5 is also just a default, not a decision. A missed churner costs far more
than an unnecessary retention offer, so the threshold maximises recall subject
to a precision floor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.config import Config, load_config  # noqa: E402
from churn.data import load_and_clean  # noqa: E402
from churn.train import DataSplit, load_pipeline, split_data  # noqa: E402

# At 0.50 at least half of the flagged customers are genuine, which keeps a
# retention campaign from being mostly wasted budget.
DEFAULT_PRECISION_FLOOR = 0.50


@dataclass
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f1: float
    precision_floor: float

    def summary(self) -> str:
        return (
            f"threshold={self.threshold:.3f} "
            f"precision={self.precision:.3f} "
            f"recall={self.recall:.3f} "
            f"f1={self.f1:.3f} "
            f"(floor={self.precision_floor:.2f})"
        )


def choose_threshold(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> ThresholdChoice:
    """Most recall we can get while holding precision at or above the floor.

    Falls back to best F1 if the floor can't be met at all.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # thresholds is one shorter than precision/recall; the dropped point is the
    # degenerate "predict nothing positive" end of the curve.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = precision >= precision_floor

    if eligible.any():
        candidate_recalls = np.where(eligible, recall, -np.inf)
        index = int(np.argmax(candidate_recalls))
    else:
        denominator = precision + recall
        f1_scores = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        index = int(np.argmax(f1_scores))

    chosen_precision = float(precision[index])
    chosen_recall = float(recall[index])
    f1 = (
        2 * chosen_precision * chosen_recall / (chosen_precision + chosen_recall)
        if (chosen_precision + chosen_recall) > 0
        else 0.0
    )

    return ThresholdChoice(
        threshold=float(thresholds[index]),
        precision=chosen_precision,
        recall=chosen_recall,
        f1=float(f1),
        precision_floor=float(precision_floor),
    )


def metrics_at_threshold(
    y_true: np.ndarray | pd.Series, y_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Confusion counts and the usual rates at one cutoff."""
    y_pred = (y_scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = matrix.ravel()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (true_pos + true_neg) / matrix.sum()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negatives": int(true_neg),
        "false_positives": int(false_pos),
        "false_negatives": int(false_neg),
        "true_positives": int(true_pos),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    split: DataSplit,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> dict:
    """Score the test split, and report at both 0.5 and the tuned threshold."""
    y_true = split.y_test
    y_scores = pipeline.predict_proba(split.x_test)[:, 1]

    choice = choose_threshold(y_true, y_scores, precision_floor)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "threshold_choice": asdict(choice),
        "at_default_threshold": metrics_at_threshold(y_true, y_scores, 0.5),
        "at_tuned_threshold": metrics_at_threshold(y_true, y_scores, choice.threshold),
        "classification_report": classification_report(
            y_true,
            (y_scores >= choice.threshold).astype(int),
            target_names=["Retained", "Churned"],
            output_dict=True,
            zero_division=0,
        ),
        "n_test": int(len(y_true)),
    }


def plot_roc_curve(y_true, y_scores) -> plt.Figure:
    """ROC curve with the AUC annotated."""
    false_pos, true_pos, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(false_pos, true_pos, color="#4C78A8", lw=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_scores, choice: ThresholdChoice) -> plt.Figure:
    """Precision-recall curve with the chosen operating point marked."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#E45756", lw=2, label=f"Model (AP = {pr_auc:.3f})")
    ax.axhline(
        baseline,
        color="black",
        linestyle="--",
        lw=1,
        label=f"Random (AP = {baseline:.3f})",
    )
    ax.scatter(
        [choice.recall],
        [choice.precision],
        s=90,
        color="black",
        zorder=5,
        label=f"Chosen (t={choice.threshold:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve with operating point")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_scores, threshold: float) -> plt.Figure:
    """Annotated confusion matrix at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")

    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")

    names = [["True neg", "False pos"], ["False neg", "True pos"]]
    threshold_color = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{names[i][j]}\n{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matrix[i, j] > threshold_color else "black",
            )

    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true, y_scores, choice: ThresholdChoice) -> plt.Figure:
    """Precision, recall, and F1 as the decision threshold moves."""
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []

    for threshold in thresholds:
        stats = metrics_at_threshold(y_true, y_scores, threshold)
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        f1s.append(stats["f1"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precisions, label="Precision", color="#4C78A8", lw=2)
    ax.plot(thresholds, recalls, label="Recall", color="#E45756", lw=2)
    ax.plot(thresholds, f1s, label="F1", color="#72B7B2", lw=2)
    ax.axvline(
        choice.threshold,
        color="black",
        linestyle="--",
        lw=1.2,
        label=f"Chosen ({choice.threshold:.2f})",
    )
    ax.axhline(
        choice.precision_floor,
        color="grey",
        linestyle=":",
        lw=1.2,
        label=f"Precision floor ({choice.precision_floor:.2f})",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-off: recall vs precision")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def generate_evaluation_figures(
    pipeline: Pipeline,
    split: DataSplit,
    choice: ThresholdChoice,
    config: Config | None = None,
) -> dict[str, object]:
    config = config or load_config()
    y_true = split.y_test
    y_scores = pipeline.predict_proba(split.x_test)[:, 1]

    output_dir = config.figures_path
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "roc_curve": plot_roc_curve(y_true, y_scores),
        "precision_recall_curve": plot_precision_recall_curve(
            y_true, y_scores, choice
        ),
        "confusion_matrix": plot_confusion_matrix(y_true, y_scores, choice.threshold),
        "threshold_sweep": plot_threshold_sweep(y_true, y_scores, choice),
    }

    saved = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        saved[name] = path
    return saved


def update_config_threshold(threshold: float, config: Config | None = None) -> None:
    """Write the threshold back to config.yaml so the API serves at it too.

    Line-level replacement rather than a YAML round-trip, which would eat the
    comments.
    """
    config = config or load_config()
    from churn.config import DEFAULT_CONFIG_PATH

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if line.strip().startswith("decision_threshold:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}decision_threshold: {threshold:.4f}"
            break

    DEFAULT_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_evaluation(results: dict) -> str:
    default = results["at_default_threshold"]
    tuned = results["at_tuned_threshold"]

    lines = [
        "Held-out test set evaluation",
        "=" * 60,
        f"Test rows: {results['n_test']:,}",
        "",
        "Ranking metrics (threshold-independent):",
        f"  ROC-AUC: {results['roc_auc']:.4f}",
        f"  PR-AUC:  {results['pr_auc']:.4f}",
        "",
        f"{'Metric':<12}{'@0.50 (default)':>18}{'@tuned':>14}{'change':>12}",
        "-" * 60,
    ]

    for key in ("accuracy", "precision", "recall", "f1"):
        change = tuned[key] - default[key]
        lines.append(
            f"{key:<12}{default[key]:>18.4f}{tuned[key]:>14.4f}{change:>+12.4f}"
        )

    lines += [
        "",
        f"Tuned threshold: {tuned['threshold']:.4f}",
        "",
        "Confusion matrix at tuned threshold:",
        f"  True negatives:  {tuned['true_negatives']:,}",
        f"  False positives: {tuned['false_positives']:,}  (needless offers)",
        f"  False negatives: {tuned['false_negatives']:,}  (missed churners)",
        f"  True positives:  {tuned['true_positives']:,}  (caught churners)",
        "",
        (
            f"Business read: at the tuned threshold the model catches "
            f"{tuned['recall']:.1%} of churners "
            f"(vs {default['recall']:.1%} at 0.50), "
            f"trading {tuned['false_positives'] - default['false_positives']:+,} "
            "false positives for "
            f"{default['false_negatives'] - tuned['false_negatives']:+,} "
            "fewer missed churners."
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained churn model.")
    parser.add_argument(
        "--precision-floor",
        type=float,
        default=DEFAULT_PRECISION_FLOOR,
        help=(
            "Minimum precision when tuning the threshold "
            f"(default {DEFAULT_PRECISION_FLOOR})."
        ),
    )
    parser.add_argument(
        "--write-threshold",
        action="store_true",
        help="Write the tuned threshold back into config.yaml.",
    )
    args = parser.parse_args(argv)

    config = load_config()

    pipeline = load_pipeline(config)
    df = load_and_clean(config)
    split = split_data(df, config)

    results = evaluate_pipeline(pipeline, split, args.precision_floor)
    print(format_evaluation(results))

    choice = ThresholdChoice(**results["threshold_choice"])
    saved = generate_evaluation_figures(pipeline, split, choice, config)

    print("\nFigures written:")
    for name, path in saved.items():
        print(f"  {name:24s} -> {path.relative_to(path.parents[2])}")

    # Merge evaluation results into the metrics file alongside CV results.
    metrics_path = config.metrics_path
    payload = {}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    payload["test_evaluation"] = results
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nMetrics updated in {metrics_path}")

    if args.write_threshold:
        update_config_threshold(choice.threshold, config)
        print(f"Wrote decision_threshold={choice.threshold:.4f} to config.yaml")


if __name__ == "__main__":
    main()

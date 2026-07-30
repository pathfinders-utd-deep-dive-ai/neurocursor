"""Generate paper-ready NeuroCursor tables, figures, and a results report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "far",
    "frr",
    "eer",
)
DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1",
    "roc_auc": "ROC-AUC",
    "far": "FAR",
    "frr": "FRR",
    "eer": "EER",
}
MODEL_LABELS = {
    "cnn-gru__full": "CNN–GRU",
    "cnn-only__full": "CNN-only",
    "gru-only__full": "GRU-only",
    "logistic-regression__full": "Logistic regression",
    "svm__full": "SVM",
    "random-forest__full": "Random forest",
    "knn__full": "KNN",
    "gradient-boosting__full": "Gradient boosting",
    "cnn-gru__raw": "Raw",
    "cnn-gru__raw-kinematic": "Raw + kinematic",
    "cnn-gru__full__sample-split": "Random sample-level",
}
COLORS = ("#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _available_user_metrics(configuration: dict[str, Any], k: int):
    for user, result in configuration["users"].items():
        metrics = result["interaction_count_metrics"].get(str(k), {})
        if metrics.get("status") != "unavailable":
            yield user, metrics


def aggregate_k(
    configuration: dict[str, Any],
    k: int = 5,
    operating_point: str = "balanced",
) -> dict[str, Any]:
    users = list(_available_user_metrics(configuration, k))
    selected = [
        metrics if operating_point == "balanced" else metrics["security_metrics"]
        for _, metrics in users
    ]
    if not selected:
        raise ValueError(f"No K={k} metrics are available.")
    aggregate = {
        name: float(np.mean([metrics[name] for metrics in selected]))
        for name in METRICS
    }
    for name in ("tp", "tn", "fp", "fn"):
        aggregate[name] = int(sum(metrics[name] for metrics in selected))
    aggregate["users"] = len(users)
    aggregate["attempts"] = int(
        sum(metrics["attempt_count"] for _, metrics in users)
    )
    aggregate["average_duration_seconds"] = float(
        np.mean([metrics["average_duration_seconds"] for _, metrics in users])
    )
    return aggregate


def aggregate_movement(configuration: dict[str, Any]) -> dict[str, Any]:
    selected = [
        result["movement_metrics"]
        for result in configuration["users"].values()
    ]
    aggregate = {
        name: float(np.mean([metrics[name] for metrics in selected]))
        for name in METRICS
    }
    for name in ("tp", "tn", "fp", "fn"):
        aggregate[name] = int(sum(metrics[name] for metrics in selected))
    aggregate["users"] = len(selected)
    return aggregate


def macro_roc(configuration: dict[str, Any], k: int = 5):
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for _, metrics in _available_user_metrics(configuration, k):
        curve = metrics["roc_curve"]
        false_positive = np.asarray(curve["false_positive_rate"], dtype=float)
        true_positive = np.asarray(curve["true_positive_rate"], dtype=float)
        curves.append(np.interp(grid, false_positive, true_positive))
    if not curves:
        raise ValueError(f"No K={k} ROC curves are available.")
    mean = np.mean(curves, axis=0)
    mean[0] = 0.0
    mean[-1] = 1.0
    return grid, mean


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def precise_percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def far_target_label(target: float) -> str:
    denominator = int(round(1.0 / target))
    return f"1 in {denominator:,} ({precise_percent(target)})"


def binomial_upper_bound(
    false_accepts: int,
    impostor_trials: int,
    confidence: float = 0.95,
) -> float:
    """One-sided exact Clopper-Pearson upper bound for pooled FAR."""
    if impostor_trials <= 0:
        raise ValueError("At least one impostor trial is required.")
    if false_accepts >= impostor_trials:
        return 1.0
    return float(
        beta.ppf(
            confidence,
            false_accepts + 1,
            impostor_trials - false_accepts,
        )
    )


def zero_event_trials_required(
    target_far: float,
    confidence: float = 0.95,
) -> int:
    """Trials needed for zero events to bound FAR below target."""
    return int(
        math.ceil(
            math.log(1.0 - confidence)
            / math.log(1.0 - target_far)
        )
    )


def number(value: float) -> str:
    return f"{value:.3f}"


def paper_metric_row(
    label: str,
    metrics: dict[str, Any],
    include_duration: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {"Configuration": label}
    for metric in METRICS:
        row[DISPLAY[metric]] = percent(metrics[metric])
    if include_duration:
        row["Duration (s)"] = f"{metrics['average_duration_seconds']:.2f}"
    return row


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data available._"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def plot_confusion_matrices(
    balanced: dict[str, Any],
    security: dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for axis, title, metrics in zip(
        axes,
        ("EER-balanced threshold", "FAR-target security threshold"),
        (balanced, security),
    ):
        matrix = np.asarray(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(2):
            denominator = max(1, int(matrix[row].sum()))
            for column in range(2):
                value = int(matrix[row, column])
                axis.text(
                    column,
                    row,
                    f"{value}\n({100 * value / denominator:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "#111827",
                    fontsize=11,
                    fontweight="bold",
                )
        axis.set(
            xticks=(0, 1),
            yticks=(0, 1),
            xticklabels=("Impostor", "Genuine"),
            yticklabels=("Impostor", "Genuine"),
            xlabel="Predicted identity",
            ylabel="Actual identity",
            title=title,
        )
    target = far_target_label(float(security["target_far"]))
    fig.suptitle(
        f"CNN–GRU pooled K=5 confusion matrices\n"
        f"Security policy target: {target}"
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc(
    configurations: dict[str, Any],
    names: Iterable[str],
    target_far: float,
    path: Path,
) -> None:
    primary = configurations["cnn-gru__full"]
    target_label = far_target_label(target_far)
    security = aggregate_k(primary, 5, operating_point="security")
    security_far = security["fp"] / max(1, security["fp"] + security["tn"])
    security_tpr = security["tp"] / max(1, security["tp"] + security["fn"])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.9),
        constrained_layout=True,
    )
    for color, name in zip(COLORS, names):
        x, y = macro_roc(configurations[name], 5)
        auc = aggregate_k(configurations[name], 5)["roc_auc"]
        label = f"{MODEL_LABELS[name]} ({auc:.3f})"
        axes[0].plot(x, y, color=color, linewidth=2.2, label=label)
        axes[1].step(
            np.maximum(x, target_far / 2.0),
            y,
            where="post",
            color=color,
            linewidth=2.2,
            label=label,
        )
    axes[0].plot(
        (0, 1),
        (0, 1),
        linestyle="--",
        color="#9ca3af",
        label="Chance",
    )
    axes[0].scatter(
        security_far,
        security_tpr,
        marker="*",
        s=130,
        color="#111827",
        zorder=5,
        label="Security operating point",
    )
    axes[0].set(
        xlabel="False acceptance rate",
        ylabel="True acceptance rate",
        title="K=5 ROC curves\nFull operating range",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(title="Model (macro AUC)", loc="lower right")

    plotted_far = max(security_far, target_far / 2.0)
    observed_label = (
        "Observed test FAR = 0"
        if security_far == 0.0
        else f"Observed test FAR = {precise_percent(security_far)}"
    )
    axes[1].axvline(
        target_far,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Policy target: {target_label}",
    )
    axes[1].scatter(
        plotted_far,
        security_tpr,
        marker="*",
        s=150,
        color="#111827",
        zorder=5,
        label=observed_label,
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="False acceptance rate (log scale)",
        ylabel="True acceptance rate",
        title=(
            "Low-FAR operating region\n"
            "Target shown; study is underpowered to validate it"
        ),
        xlim=(target_far / 2.0, 0.10),
        ylim=(0, 1.01),
    )
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_tradeoff(
    configuration: dict[str, Any],
    target_far: float,
    path: Path,
) -> None:
    counts = (1, 3, 5)
    balanced = [aggregate_k(configuration, k) for k in counts]
    security = [
        aggregate_k(configuration, k, operating_point="security")
        for k in counts
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for color, metric in zip(COLORS, ("far", "frr", "eer")):
        axis.plot(
            counts,
            [100 * item[metric] for item in balanced],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"Balanced {DISPLAY[metric]}",
        )
    axis.plot(
        counts,
        [100 * item["far"] for item in security],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="#111827",
        label="Security FAR",
    )
    axis.plot(
        counts,
        [100 * item["frr"] for item in security],
        marker="s",
        linestyle=":",
        linewidth=2.2,
        color="#6b7280",
        label="Security FRR",
    )
    target_label = far_target_label(target_far)
    axis.axhline(
        100.0 * target_far,
        color="#dc2626",
        linewidth=1.2,
        linestyle="--",
        label=f"Policy target: {target_label}",
    )
    axis.set(
        xlabel="Movements per authentication attempt (K)",
        ylabel="Rate (%)",
        title=(
            "Authentication error tradeoff by interaction count\n"
            f"Validation policy target: {target_label}"
        ),
        xticks=counts,
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    configurations: dict[str, Any],
    names: list[str],
    title: str,
    path: Path,
    grain: str = "k5",
    display_labels: list[str] | None = None,
) -> None:
    labels = display_labels or [MODEL_LABELS.get(name, name) for name in names]
    values = [
        (
            aggregate_k(configurations[name], 5)
            if grain == "k5"
            else aggregate_movement(configurations[name])
        )
        for name in names
    ]
    metrics = ("roc_auc", "far", "frr", "eer")
    x = np.arange(len(names))
    width = 0.19
    fig, axis = plt.subplots(figsize=(max(7.2, len(names) * 1.35), 4.9), constrained_layout=True)
    for offset, color, metric in zip(
        (-1.5, -0.5, 0.5, 1.5), COLORS, metrics
    ):
        axis.bar(
            x + offset * width,
            [100 * item[metric] for item in values],
            width,
            color=color,
            label=DISPLAY[metric],
        )
    axis.set(
        ylabel="Rate (%)",
        title=title,
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=4)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(configuration: dict[str, Any], path: Path) -> None:
    histories = [
        result["training_history"]
        for result in configuration["users"].values()
        if "training_history" in result
    ]
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    for axis, metric, label in (
        (axes[0], "loss", "Binary cross-entropy"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ):
        for history in histories:
            axis.plot(
                np.arange(1, len(history[metric]) + 1),
                history[metric],
                color="#2563eb",
                alpha=0.18,
            )
            validation_name = f"val_{metric}"
            axis.plot(
                np.arange(1, len(history[validation_name]) + 1),
                history[validation_name],
                color="#f59e0b",
                alpha=0.18,
            )
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.grid(alpha=0.2)
    axes[0].plot([], [], color="#2563eb", label="Training, one line/user")
    axes[0].plot([], [], color="#f59e0b", label="Validation, one line/user")
    axes[0].legend()
    fig.suptitle("CNN–GRU training histories")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def statistical_rows(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for user, metrics in _available_user_metrics(configuration, 5):
        statistics = metrics["statistics"]
        intervals = statistics["confidence_intervals"]
        effect_size = statistics["cohens_d"]
        effect_display = (
            number(effect_size)
            if np.isfinite(effect_size) and effect_size != 0.0
            else "not estimable"
        )
        rows.append(
            {
                "User": user,
                "Permutation p": f"{statistics['one_sided_permutation_p_value']:.5f}",
                "Cohen d": effect_display,
                "AUC 95% CI": (
                    f"{percent(intervals['roc_auc']['lower_95'])}–"
                    f"{percent(intervals['roc_auc']['upper_95'])}"
                ),
                "FAR 95% CI": (
                    f"{percent(intervals['far']['lower_95'])}–"
                    f"{percent(intervals['far']['upper_95'])}"
                ),
                "FRR 95% CI": (
                    f"{percent(intervals['frr']['lower_95'])}–"
                    f"{percent(intervals['frr']['upper_95'])}"
                ),
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_report(experiment_directory: Path, output_directory: Path) -> None:
    summary_path = experiment_directory / "experiment_summary.json"
    summary = load_json(summary_path)
    configurations = summary["configurations"]
    target = float(summary.get("target_far", 1.0 / 50_000.0))
    target_label = far_target_label(target)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    tables = output_directory / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    primary = configurations["cnn-gru__full"]
    balanced = aggregate_k(primary, 5)
    security = aggregate_k(primary, 5, operating_point="security")
    security["target_far"] = target
    impostor_trials = int(security["tn"] + security["fp"])
    pooled_security_far = float(security["fp"] / impostor_trials)
    upper_far_95 = binomial_upper_bound(
        int(security["fp"]),
        impostor_trials,
    )
    required_zero_event_trials = zero_event_trials_required(target)
    statistically_validated = upper_far_95 <= target

    neural_names = ["cnn-gru__full", "cnn-only__full", "gru-only__full"]
    classical_names = [
        "logistic-regression__full",
        "svm__full",
        "random-forest__full",
        "knn__full",
        "gradient-boosting__full",
    ]
    ablation_names = [
        "cnn-gru__raw",
        "cnn-gru__raw-kinematic",
        "cnn-gru__full",
    ]
    split_names = ["cnn-gru__full", "cnn-gru__full__sample-split"]

    neural_rows = [
        paper_metric_row(MODEL_LABELS[name], aggregate_k(configurations[name]))
        for name in neural_names
    ]
    classical_rows = [
        paper_metric_row(MODEL_LABELS[name], aggregate_k(configurations[name]))
        for name in classical_names
    ]
    k_rows = []
    for k in (1, 3, 5):
        metrics = aggregate_k(primary, k)
        k_rows.append(
            {
                "K": k,
                "FAR": percent(metrics["far"]),
                "FRR": percent(metrics["frr"]),
                "EER": percent(metrics["eer"]),
                "ROC-AUC": percent(metrics["roc_auc"]),
                "Duration (s)": f"{metrics['average_duration_seconds']:.2f}",
            }
        )
    ablation_rows = [
        {
            "Feature set": {
                "cnn-gru__raw": "Raw",
                "cnn-gru__raw-kinematic": "Raw + kinematic",
                "cnn-gru__full": "Full",
            }[name],
            "ROC-AUC": percent(aggregate_k(configurations[name])["roc_auc"]),
            "FAR": percent(aggregate_k(configurations[name])["far"]),
            "EER": percent(aggregate_k(configurations[name])["eer"]),
        }
        for name in ablation_names
    ]
    split_rows = [
        {
            "Split": (
                "Session-separated"
                if name == "cnn-gru__full"
                else MODEL_LABELS[name]
            ),
            "ROC-AUC": percent(
                aggregate_movement(configurations[name])["roc_auc"]
            ),
            "FAR": percent(
                aggregate_movement(configurations[name])["far"]
            ),
            "EER": percent(
                aggregate_movement(configurations[name])["eer"]
            ),
        }
        for name in split_names
    ]
    security_rows = [
        {
            "Operating point": "EER-balanced",
            "Policy target": "Not applicable",
            **{
                DISPLAY[name]: percent(balanced[name])
                for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "far", "frr", "eer")
            },
            "TN": balanced["tn"],
            "FP": balanced["fp"],
            "FN": balanced["fn"],
            "TP": balanced["tp"],
        },
        {
            "Operating point": "Validation FAR target",
            "Policy target": target_label,
            **{
                DISPLAY[name]: percent(security[name])
                for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "far", "frr", "eer")
            },
            "TN": security["tn"],
            "FP": security["fp"],
            "FN": security["fn"],
            "TP": security["tp"],
        },
    ]
    evidence_rows = [
        {
            "Policy FAR target": target_label,
            "Held-out impostor trials": impostor_trials,
            "False accepts": security["fp"],
            "Observed pooled FAR": precise_percent(pooled_security_far),
            "Macro FAR": precise_percent(security["far"]),
            "One-sided 95% FAR upper bound": precise_percent(upper_far_95),
            "Zero-event trials required for 95% validation": (
                required_zero_event_trials
            ),
            "Statistically validated": (
                "Yes" if statistically_validated else "No"
            ),
        }
    ]
    statistics = statistical_rows(primary)

    for filename, rows in (
        ("table_iv_neural_models.csv", neural_rows),
        ("classical_baselines.csv", classical_rows),
        ("table_v_interaction_counts.csv", k_rows),
        ("table_vi_feature_ablation.csv", ablation_rows),
        ("table_vii_split_comparison.csv", split_rows),
        ("security_operating_points.csv", security_rows),
        ("security_target_evidence.csv", evidence_rows),
        ("statistical_analysis_k5.csv", statistics),
    ):
        write_csv(tables / filename, rows)

    plot_confusion_matrices(
        balanced,
        security,
        figures / "confusion_matrix_k5.png",
    )
    plot_roc(
        configurations,
        neural_names,
        target,
        figures / "roc_curves_k5.png",
    )
    plot_k_tradeoff(
        primary,
        target,
        figures / "interaction_count_tradeoff.png",
    )
    plot_comparison(
        configurations,
        neural_names,
        "Neural model comparison at K=5",
        figures / "neural_model_comparison.png",
    )
    plot_comparison(
        configurations,
        classical_names,
        "Classical baseline comparison at K=5",
        figures / "classical_model_comparison.png",
    )
    plot_comparison(
        configurations,
        ablation_names,
        "Feature ablation at K=5",
        figures / "feature_ablation.png",
        display_labels=["Raw", "Raw + kinematic", "Full"],
    )
    plot_comparison(
        configurations,
        split_names,
        "Movement-level split comparison",
        figures / "split_comparison.png",
        grain="movement",
        display_labels=["Session-separated", "Random sample-level"],
    )
    plot_training_history(primary, figures / "training_history.png")

    dataset = summary["dataset"]
    observed_at_or_below_target = pooled_security_far <= target
    observed_statement = (
        "No false acceptances were observed in this held-out sample."
        if security["fp"] == 0
        else (
            f"{security['fp']} false acceptances were observed in "
            f"{impostor_trials} pooled impostor decisions."
        )
    )
    report = f"""# NeuroCursor complete real-data evaluation

## Result

The proposed CNN–GRU achieved **{percent(balanced['roc_auc'])} macro ROC-AUC**, **{percent(balanced['accuracy'])} macro accuracy**, **{percent(balanced['far'])} FAR**, **{percent(balanced['frr'])} FRR**, and **{percent(balanced['eer'])} EER** at the paper's K=5 EER-balanced operating point.

For the security operating point, thresholds were selected independently for each claimed identity using validation data only with a policy target of **{target_label}**. {observed_statement} Held-out K=5 macro FAR was **{precise_percent(security['far'])}**, pooled FAR was **{precise_percent(pooled_security_far)}**, and macro FRR was **{percent(security['frr'])}**.

This experiment **does not validate a true 1-in-50,000 FAR**. The one-sided exact 95% upper bound is **{precise_percent(upper_far_95)}**, compared with the {precise_percent(target)} target. Even with zero false acceptances, approximately **{required_zero_event_trials:,} independent impostor trials** are required to place a 95% upper bound below the target. The target therefore defines threshold policy; it is not a measured population-level guarantee.

## Data and protocol

- Anonymized users: **{dataset['users']}**
- Sessions / attempts: **{dataset['sessions']} / {dataset['attempts']}**
- Point-and-click movements: **{dataset['movements']}**
- Primary split: **{summary['split_type']}**
- Threshold selection: validation data only; test labels never select thresholds
- Reported table values: macro-average across one-vs-rest user verifiers
- Tables IV–VI use K=5 attempt-level decisions; Table V separately varies K
- Table VII uses movement-level metrics because random sample splitting does not preserve complete attempts
- Random seed: **{summary['runtime']['seed']}**
- Permutation tests / bootstrap intervals: **10,000 / 1,000 per user and K**

## Paper completion checklist

| Paper requirement | Completed output |
| --- | --- |
| CNN–GRU, CNN-only, GRU-only | Table IV, ROC figure, comparison figure |
| Logistic regression, SVM, random forest, KNN, gradient boosting | Classical table and figure |
| K=1, 3, 5 | Table V and tradeoff figure |
| Raw, raw + kinematic, full features | Table VI and ablation figure |
| Session-separated vs random sample split | Table VII and split figure |
| Accuracy, precision, recall, F1, ROC-AUC, FAR, FRR, EER | Tables IV and classical baseline table |
| One-sided permutation test, Cohen's d, bootstrap 95% CIs | Per-user statistical table |
| Confusion matrix | Pooled K=5 balanced and security matrices |
| Training behavior | Per-user training and validation history figure |

## Security operating point and confusion matrix

{markdown_table(security_rows)}

### Evidence for the 1-in-50,000 target

{markdown_table(evidence_rows)}

![CNN–GRU K=5 confusion matrices](figures/confusion_matrix_k5.png)

The confusion matrices pool decisions across the ten per-user verifiers. Percentages inside cells are normalized by actual class. Macro FAR/FRR in the table average user-specific rates, so they can differ slightly from a rate recomputed from pooled counts.

## Table IV — neural model comparison

{markdown_table(neural_rows)}

![Neural model comparison](figures/neural_model_comparison.png)

![Macro ROC curves](figures/roc_curves_k5.png)

An ROC curve is threshold-independent. The low-FAR panel marks the validation-calibrated security operating point and the 1-in-50,000 policy target. The target lies below this study's empirical resolution and is shown as a policy reference, not as a validated point on the population ROC.

## Classical baselines

{markdown_table(classical_rows)}

![Classical model comparison](figures/classical_model_comparison.png)

## Table V — number of interactions

{markdown_table(k_rows)}

![Interaction-count tradeoff](figures/interaction_count_tradeoff.png)

## Table VI — feature ablation

{markdown_table(ablation_rows)}

![Feature ablation](figures/feature_ablation.png)

## Table VII — split comparison

{markdown_table(split_rows)}

![Split comparison](figures/split_comparison.png)

The random sample-level split is included only as the paper's secondary sensitivity analysis. It can mix movements from the same session across partitions and must not replace the session-separated primary estimate.

## Statistical analysis at K=5

{markdown_table(statistics)}

The one-sided alternative is that genuine scores exceed impostor scores. Bootstrap intervals are stratified by class. These tests are run per claimed identity because the system trains one verifier per identity. Cohen's d is marked “not estimable” when a class has fewer than two attempts or the pooled within-class variance is effectively zero.

## Training diagnostics

![CNN–GRU training history](figures/training_history.png)

## Limitations

- This is a single deterministic split of a modest ten-user dataset, not an independent external validation cohort.
- One participant has only three eligible sessions, which makes that verifier's held-out estimates especially discrete and uncertain.
- Macro metrics weight each claimed identity equally; pooled confusion counts weight individual decisions.
- A lower FAR is not free: a stricter threshold generally raises FRR. Both are shown so the security gain is not presented without its usability cost.
- A study of {impostor_trials} pooled impostor decisions cannot validate a 1-in-50,000 FAR. Approximately {required_zero_event_trials:,} zero-false-accept trials are needed for a one-sided 95% upper bound below that rate.
- The paper's blank result tables are now filled from this dataset; these measurements should replace placeholders only if this is the intended study cohort.
- Classical baselines use documented temporal summary features. The paper names the baselines but does not specify every hyperparameter, so the exact reproducible choices remain in `model/model.py`.

## Reproduce

From the repository root, using an environment with `tensorflow==2.16.2`, `scikit-learn==1.6.1`, `scipy==1.15.1`, `matplotlib==3.10.0`, and `joblib==1.4.2`:

```bash
python -m pip install -r model/requirements-results.txt

python model/model.py experiment \\
  --data /path/to/paper-dataset.json \\
  --output-dir artifacts/paper-experiment \\
  --target-far {target:.8f} \\
  --seed {summary['runtime']['seed']} \\
  --permutations 10000 \\
  --bootstrap-samples 1000

python model/reporting.py \\
  --experiment-dir artifacts/paper-experiment \\
  --output-dir results/paper-real-data
```

Raw participant data, trained weights, and per-attempt scores are intentionally excluded from Git. The committed report contains anonymized aggregate results, paper tables, figures, and checksums.
"""
    (output_directory / "README.md").write_text(report, encoding="utf-8")

    compact_results = {
        "dataset": dataset,
        "split_type": summary["split_type"],
        "target_far": target,
        "target_far_label": target_label,
        "cnn_gru_k5_balanced": balanced,
        "cnn_gru_k5_security": security,
        "security_target_evidence": {
            "held_out_impostor_trials": impostor_trials,
            "false_accepts": int(security["fp"]),
            "observed_pooled_far": pooled_security_far,
            "observed_macro_far": float(security["far"]),
            "one_sided_95_percent_far_upper_bound": upper_far_95,
            "zero_event_trials_required_for_95_percent_validation": (
                required_zero_event_trials
            ),
            "observed_far_at_or_below_policy_target": (
                observed_at_or_below_target
            ),
            "statistically_validated_at_95_percent": statistically_validated,
        },
        "neural_models": {
            MODEL_LABELS[name]: aggregate_k(configurations[name])
            for name in neural_names
        },
        "classical_models": {
            MODEL_LABELS[name]: aggregate_k(configurations[name])
            for name in classical_names
        },
        "interaction_counts": {
            str(k): aggregate_k(primary, k) for k in (1, 3, 5)
        },
        "feature_ablations": {
            MODEL_LABELS[name]: aggregate_k(configurations[name])
            for name in ablation_names
        },
        "split_comparison_movement_level": {
            (
                "Session-separated"
                if name == "cnn-gru__full"
                else MODEL_LABELS[name]
            ): aggregate_movement(configurations[name])
            for name in split_names
        },
        "runtime": summary["runtime"],
    }
    write_json(output_directory / "RESULTS.json", compact_results)

    files = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    manifest = "\n".join(
        f"{sha256_file(path)}  {path.relative_to(output_directory)}"
        for path in files
    )
    (output_directory / "MANIFEST.sha256").write_text(
        manifest + "\n",
        encoding="utf-8",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NeuroCursor paper tables, figures, and report."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    generate_report(Path(args.experiment_dir), Path(args.output_dir))
    print(f"Saved paper-ready report to {args.output_dir}")


if __name__ == "__main__":
    main()

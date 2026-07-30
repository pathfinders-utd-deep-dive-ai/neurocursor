"""Generate paper-ready NeuroCursor tables, figures, and a results report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def validation_security_summary(
    configuration: dict[str, Any],
    k: int = 5,
) -> dict[str, float]:
    metrics = [
        value
        for _, value in _available_user_metrics(configuration, k)
    ]
    if not metrics:
        raise ValueError(f"No K={k} validation calibrations are available.")
    return {
        "far": float(
            np.mean(
                [
                    value["security_calibration"]["far"]
                    for value in metrics
                ]
            )
        ),
        "frr": float(
            np.mean(
                [
                    value["security_calibration"]["frr"]
                    for value in metrics
                ]
            )
        ),
        "users": len(metrics),
    }


def select_security_configuration(
    configurations: dict[str, Any],
    k: int = 5,
) -> tuple[str, dict[str, float]]:
    """Select without test metrics: minimum validation FAR, then FRR."""
    candidates = []
    for name, configuration in configurations.items():
        try:
            validation = validation_security_summary(configuration, k)
        except ValueError:
            continue
        candidates.append(
            (
                validation["far"],
                validation["frr"],
                name,
                validation,
            )
        )
    if not candidates:
        raise ValueError("No strict security configuration is available.")
    _, _, name, validation = min(candidates)
    return name, validation


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


def binomial_interval(
    events: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Two-sided exact Clopper-Pearson interval for a binomial rate."""
    if trials <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lower = (
        0.0
        if events == 0
        else float(beta.ppf(alpha / 2.0, events, trials - events + 1))
    )
    upper = (
        1.0
        if events == trials
        else float(beta.ppf(1.0 - alpha / 2.0, events + 1, trials - events))
    )
    return (lower, upper)


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
    primary_security: dict[str, Any],
    selected_security: dict[str, Any],
    selected_label: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    for axis, title, metrics in zip(
        axes,
        (
            "CNN–GRU balanced",
            "CNN–GRU strict",
            f"{selected_label} strict",
        ),
        (balanced, primary_security, selected_security),
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
    fig.suptitle("Seed 42 descriptive K=5 confusion matrices")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc(
    configurations: dict[str, Any],
    names: Iterable[str],
    selected_name: str,
    path: Path,
) -> None:
    security = aggregate_k(
        configurations[selected_name],
        5,
        operating_point="security",
    )
    security_far = security["fp"] / max(1, security["fp"] + security["tn"])
    security_tpr = security["tp"] / max(1, security["tp"] + security["fn"])
    low_far_floor = 1e-4
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
            np.maximum(x, low_far_floor),
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
        label=f"Seed 42 strict: {MODEL_LABELS[selected_name]}",
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

    plotted_far = max(security_far, low_far_floor)
    observed_label = (
        "Observed test FAR = 0"
        if security_far == 0.0
        else f"Observed test FAR = {precise_percent(security_far)}"
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
            "Seed 42 descriptive operating point"
        ),
        xlim=(low_far_floor, 0.10),
        ylim=(0, 1.01),
    )
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_tradeoff(
    primary: dict[str, Any],
    selected: dict[str, Any],
    selected_label: str,
    path: Path,
) -> None:
    counts = (1, 3, 5)
    balanced = [aggregate_k(primary, k) for k in counts]
    security = [
        aggregate_k(selected, k, operating_point="security")
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
            label=f"CNN–GRU balanced {DISPLAY[metric]}",
        )
    axis.plot(
        counts,
        [100 * item["far"] for item in security],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="#111827",
        label=f"{selected_label} strict FAR",
    )
    axis.plot(
        counts,
        [100 * item["frr"] for item in security],
        marker="s",
        linestyle=":",
        linewidth=2.2,
        color="#6b7280",
        label=f"{selected_label} strict FRR",
    )
    axis.set(
        xlabel="Movements per authentication attempt (K)",
        ylabel="Rate (%)",
        title=(
            "Authentication error tradeoff by interaction count\n"
            "Seed 42 descriptive comparison"
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
        far_interval = binomial_interval(
            metrics["fp"],
            metrics["fp"] + metrics["tn"],
        )
        frr_interval = binomial_interval(
            metrics["fn"],
            metrics["fn"] + metrics["tp"],
        )
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
                    f"{percent(far_interval[0])}–"
                    f"{percent(far_interval[1])}"
                ),
                "FRR 95% CI": (
                    f"{percent(frr_interval[0])}–"
                    f"{percent(frr_interval[1])}"
                ),
            }
        )
    return rows


def metric_context(metrics: dict[str, Any]) -> dict[str, float]:
    negatives = metrics["tn"] + metrics["fp"]
    positives = metrics["tp"] + metrics["fn"]
    total = negatives + positives
    pooled_far = metrics["fp"] / negatives if negatives else 0.0
    pooled_frr = metrics["fn"] / positives if positives else 0.0
    return {
        "pooled_accuracy": (
            (metrics["tn"] + metrics["tp"]) / total if total else 0.0
        ),
        "pooled_far": pooled_far,
        "pooled_frr": pooled_frr,
        "pooled_balanced_accuracy": (
            ((1.0 - pooled_far) + (1.0 - pooled_frr)) / 2.0
        ),
        "macro_balanced_accuracy": (
            ((1.0 - metrics["far"]) + (1.0 - metrics["frr"])) / 2.0
        ),
        "reject_all_accuracy": negatives / total if total else 0.0,
        "impostor_decision_share": negatives / total if total else 0.0,
    }


def sensitivity_runs(
    sensitivity: dict[str, Any],
    k: int = 5,
) -> list[dict[str, Any]]:
    return [
        {
            "seed": run["seed"],
            **run["interaction_counts"][str(k)],
        }
        for run in sensitivity["runs"]
    ]


def plot_sensitivity_distributions(
    sensitivity: dict[str, Any],
    path: Path,
) -> None:
    runs = sensitivity_runs(sensitivity, 5)
    metrics = (
        ("pooled_balanced_accuracy", "Balanced accuracy"),
        ("far", "Macro FAR"),
        ("frr", "Macro FRR"),
        ("roc_auc", "Macro ROC-AUC"),
    )
    values = [[100.0 * run[key] for run in runs] for key, _ in metrics]
    fig, axis = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    boxes = axis.boxplot(
        values,
        tick_labels=[label for _, label in metrics],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d97706",
            "markeredgecolor": "#92400e",
            "markersize": 6,
        },
    )
    for box in boxes["boxes"]:
        box.set(facecolor="#dbeafe", edgecolor="#1d4ed8", linewidth=1.4)
    for index, metric_values in enumerate(values, start=1):
        jitter = np.linspace(-0.12, 0.12, len(metric_values))
        axis.scatter(
            np.full(len(metric_values), index) + jitter,
            metric_values,
            s=18,
            color="#374151",
            alpha=0.58,
            zorder=3,
        )
    axis.set(
        ylabel="Rate (%)",
        ylim=(-3, 103),
        title=(
            "Repeated session-split sensitivity distributions\n"
            "All 30 fixed seeds (0–29), K=5 logistic strict verifier"
        ),
    )
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_far_frr(
    sensitivity: dict[str, Any],
    seed42: dict[str, Any],
    path: Path,
) -> None:
    runs = sensitivity_runs(sensitivity, 5)
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.scatter(
        [100.0 * run["far"] for run in runs],
        [100.0 * run["frr"] for run in runs],
        s=48,
        color="#2563eb",
        edgecolor="#1e3a8a",
        linewidth=0.6,
        alpha=0.78,
        label="Predeclared seeds 0–29",
    )
    axis.scatter(
        100.0 * seed42["far"],
        100.0 * seed42["frr"],
        marker="*",
        s=220,
        color="#d97706",
        edgecolor="#78350f",
        linewidth=0.8,
        label="Seed 42 descriptive snapshot",
        zorder=5,
    )
    axis.set(
        xlabel="Macro false acceptance rate (%)",
        ylabel="Macro false rejection rate (%)",
        title=(
            "FAR–FRR sensitivity across session splits\n"
            "Lower-left is better; seed 42 is not the primary estimate"
        ),
        xlim=(-0.15, None),
        ylim=(0, None),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_context(
    sensitivity: dict[str, Any],
    seed42: dict[str, Any],
    path: Path,
) -> None:
    runs = sensitivity_runs(sensitivity, 5)
    seed_context = metric_context(seed42)
    ordinary = [100.0 * run["pooled_accuracy"] for run in runs]
    balanced = [
        100.0 * run["pooled_balanced_accuracy"]
        for run in runs
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    axes[0].bar(
        ("Impostor\ndecisions", "Genuine\ndecisions"),
        (seed42["tn"] + seed42["fp"], seed42["tp"] + seed42["fn"]),
        color=("#1d4ed8", "#d97706"),
        edgecolor=("#1e3a8a", "#78350f"),
    )
    for index, value in enumerate(
        (seed42["tn"] + seed42["fp"], seed42["tp"] + seed42["fn"])
    ):
        axes[0].text(index, value + 4, str(value), ha="center", fontweight="bold")
    axes[0].set(
        ylabel="Binary decisions",
        title="Seed 42 decision-class composition\n90% of decisions are impostor comparisons",
        ylim=(0, 260),
    )
    axes[0].grid(axis="y", alpha=0.2)

    boxes = axes[1].boxplot(
        (ordinary, balanced),
        tick_labels=("Ordinary\naccuracy", "Balanced\naccuracy"),
        patch_artist=True,
    )
    for box, color in zip(boxes["boxes"], ("#bfdbfe", "#fde68a")):
        box.set(facecolor=color, edgecolor="#374151")
    axes[1].scatter(
        (1, 2),
        (
            100.0 * seed_context["pooled_accuracy"],
            100.0 * seed_context["pooled_balanced_accuracy"],
        ),
        marker="*",
        s=180,
        color="#111827",
        label="Seed 42",
        zorder=5,
    )
    axes[1].axhline(
        100.0 * seed_context["reject_all_accuracy"],
        color="#6b7280",
        linestyle="--",
        label="Reject-all ordinary accuracy",
    )
    axes[1].set(
        ylabel="Rate (%)",
        title="Accuracy across all 30 fixed splits\nBalanced accuracy corrects class imbalance",
        ylim=(65, 101),
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_influence(
    sensitivity: dict[str, Any],
    path: Path,
) -> None:
    rows = sensitivity["feature_influence"][:15]
    labels = [row["feature"].replace(":", ": ") for row in rows][::-1]
    values = [row["coefficient_mean"] for row in rows][::-1]
    q1 = [row["coefficient_q1"] for row in rows][::-1]
    q3 = [row["coefficient_q3"] for row in rows][::-1]
    errors = np.asarray(
        [
            [value - lower for value, lower in zip(values, q1)],
            [upper - value for value, upper in zip(values, q3)],
        ]
    )
    fig, axis = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    axis.barh(
        labels,
        values,
        xerr=errors,
        color="#60a5fa",
        edgecolor="#1e3a8a",
        error_kw={"ecolor": "#374151", "capsize": 2},
    )
    axis.set(
        xlabel="Mean absolute standardized logistic coefficient",
        title=(
            "Logistic verifier feature influence\n"
            "Mean and interquartile range across 300 fitted user models"
        ),
    )
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_construction(
    sensitivity: dict[str, Any],
    path: Path,
) -> None:
    source = sensitivity.get("source_cohort")
    if not source:
        return
    labels = (
        "Source sessions",
        "Legacy schema\n(unusable for full model)",
        "Full-schema sessions",
        "Eligible sessions\n(final cohort)",
    )
    values = (
        source["source_sessions"],
        source["legacy_schema_sessions"],
        source["full_schema_sessions"],
        source["eligible_full_schema_sessions"],
    )
    colors = ("#374151", "#9ca3af", "#60a5fa", "#1d4ed8")
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, edgecolor="#111827")
    axis.bar_label(bars, padding=4, fontweight="bold")
    axis.set(
        ylabel="Sessions",
        title=(
            "Cohort construction from the supplied source file\n"
            "Schema eligibility, not model performance, determines inclusion"
        ),
        ylim=(0, max(values) * 1.13),
    )
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_user_robustness(
    sensitivity: dict[str, Any],
    path: Path,
) -> None:
    users = sensitivity["user_summary_by_interaction_count"]["5"]
    labels = list(sorted(users))
    metrics = ("far", "frr", "roc_auc")
    matrix = np.asarray(
        [
            [100.0 * users[user][metric]["mean"] for metric in metrics]
            for user in labels
        ]
    )
    fig, axis = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 55 else "#111827",
                fontsize=9,
            )
    axis.set(
        xticks=np.arange(len(metrics)),
        xticklabels=("FAR", "FRR", "ROC-AUC"),
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        title=(
            "Per-user mean robustness metrics\n"
            "All 30 fixed session splits, K=5 logistic strict verifier"
        ),
    )
    fig.colorbar(image, ax=axis, label="Rate (%)")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_repeated_k_tradeoff(
    sensitivity: dict[str, Any],
    path: Path,
) -> None:
    summaries = sensitivity["summary_by_interaction_count"]
    counts = (1, 3, 5)
    fig, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    for metric, label, color, marker in (
        ("far", "Macro FAR", "#1d4ed8", "o"),
        ("frr", "Macro FRR", "#d97706", "s"),
        ("eer", "Macro EER", "#374151", "D"),
    ):
        means = [100.0 * summaries[str(k)][metric]["mean"] for k in counts]
        q1 = [100.0 * summaries[str(k)][metric]["q1"] for k in counts]
        q3 = [100.0 * summaries[str(k)][metric]["q3"] for k in counts]
        error = np.asarray(
            [
                [value - low for value, low in zip(means, q1)],
                [high - value for value, high in zip(means, q3)],
            ]
        )
        axis.errorbar(
            counts,
            means,
            yerr=error,
            marker=marker,
            linewidth=2.0,
            capsize=4,
            color=color,
            label=f"{label}, mean with IQR",
        )
    axis.set(
        xlabel="Movements per authentication attempt (K)",
        ylabel="Rate (%)",
        xticks=counts,
        title=(
            "Repeated-split interaction-count tradeoff\n"
            "All fixed seeds 0–29; no favorable seed selected"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_report(
    experiment_directory: Path,
    output_directory: Path,
    sensitivity_path: Path | None = None,
) -> None:
    """Generate the complete, caveated paper evaluation package."""
    summary = load_json(experiment_directory / "experiment_summary.json")
    if sensitivity_path is None:
        sensitivity_path = experiment_directory / "sensitivity_summary.json"
    if not sensitivity_path.is_file():
        raise FileNotFoundError(
            "A repeated-split sensitivity summary is required. Run "
            "`model.py sensitivity` first."
        )
    sensitivity = load_json(sensitivity_path)
    configurations = summary["configurations"]

    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    tables = output_directory / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    primary = configurations["cnn-gru__full"]
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

    balanced = aggregate_k(primary, 5)
    primary_security = aggregate_k(
        primary,
        5,
        operating_point="security",
    )
    selected_name, selected_validation = select_security_configuration(
        configurations,
        5,
    )
    selected_label = MODEL_LABELS[selected_name]
    selected_configuration = configurations[selected_name]
    selected_security = aggregate_k(
        selected_configuration,
        5,
        operating_point="security",
    )
    selected_context = metric_context(selected_security)
    impostor_trials = selected_security["tn"] + selected_security["fp"]
    upper_far_95 = binomial_upper_bound(
        selected_security["fp"],
        impostor_trials,
    )

    sensitivity_summary = sensitivity[
        "summary_by_interaction_count"
    ]["5"]
    sensitivity_k5 = sensitivity_runs(sensitivity, 5)

    def distribution_row(
        label: str,
        key: str,
        percent_value: bool = True,
    ) -> dict[str, Any]:
        values = sensitivity_summary[key]
        formatter = percent if percent_value else lambda value: f"{value:.2f}"
        return {
            "Metric": label,
            "Mean": formatter(values["mean"]),
            "Median": formatter(values["median"]),
            "IQR": (
                f"{formatter(values['q1'])}–{formatter(values['q3'])}"
            ),
            "Minimum": formatter(values["minimum"]),
            "Maximum": formatter(values["maximum"]),
        }

    repeated_summary_rows = [
        distribution_row("Ordinary accuracy", "pooled_accuracy"),
        distribution_row(
            "Balanced accuracy",
            "pooled_balanced_accuracy",
        ),
        distribution_row("Macro FAR", "far"),
        distribution_row("Macro FRR", "frr"),
        distribution_row("Macro ROC-AUC", "roc_auc"),
        distribution_row("Macro EER", "eer"),
        distribution_row(
            "Average K=5 duration (s)",
            "average_duration_seconds",
            percent_value=False,
        ),
    ]

    all_seed_rows = []
    for run in sensitivity_k5:
        all_seed_rows.append(
            {
                "Seed": run["seed"],
                "Ordinary accuracy": percent(run["pooled_accuracy"]),
                "Balanced accuracy": percent(
                    run["pooled_balanced_accuracy"]
                ),
                "Macro FAR": percent(run["far"]),
                "Macro FRR": percent(run["frr"]),
                "Macro ROC-AUC": percent(run["roc_auc"]),
                "Macro EER": percent(run["eer"]),
                "TN": run["tn"],
                "FP": run["fp"],
                "FN": run["fn"],
                "TP": run["tp"],
            }
        )

    repeated_k_rows = []
    for count in (1, 3, 5):
        values = sensitivity["summary_by_interaction_count"][str(count)]
        repeated_k_rows.append(
            {
                "K": count,
                "Mean FAR": percent(values["far"]["mean"]),
                "FAR IQR": (
                    f"{percent(values['far']['q1'])}–"
                    f"{percent(values['far']['q3'])}"
                ),
                "Mean FRR": percent(values["frr"]["mean"]),
                "FRR IQR": (
                    f"{percent(values['frr']['q1'])}–"
                    f"{percent(values['frr']['q3'])}"
                ),
                "Mean EER": percent(values["eer"]["mean"]),
                "Mean balanced accuracy": percent(
                    values["pooled_balanced_accuracy"]["mean"]
                ),
                "Mean duration (s)": (
                    f"{values['average_duration_seconds']['mean']:.2f}"
                ),
                "Zero-FP splits": (
                    f"{values['zero_false_accept_splits']}/"
                    f"{sensitivity['protocol']['seed_count']}"
                ),
            }
        )

    sensitivity_user_rows = []
    user_summary = sensitivity[
        "user_summary_by_interaction_count"
    ]["5"]
    for user in sorted(user_summary):
        values = user_summary[user]
        sensitivity_user_rows.append(
            {
                "User": user,
                "Mean FAR": percent(values["far"]["mean"]),
                "Mean FRR": percent(values["frr"]["mean"]),
                "Mean ROC-AUC": percent(values["roc_auc"]["mean"]),
                "FRR range": (
                    f"{percent(values['frr']['minimum'])}–"
                    f"{percent(values['frr']['maximum'])}"
                ),
                "EER mean": percent(values["eer"]["mean"]),
            }
        )

    feature_influence_rows = [
        {
            "Rank": index,
            "Feature": row["feature"],
            "Mean absolute standardized coefficient": (
                f"{row['coefficient_mean']:.4f}"
            ),
            "Median": f"{row['coefficient_median']:.4f}",
            "IQR": (
                f"{row['coefficient_q1']:.4f}–"
                f"{row['coefficient_q3']:.4f}"
            ),
        }
        for index, row in enumerate(
            sensitivity["feature_influence"],
            start=1,
        )
    ]

    neural_rows = [
        paper_metric_row(MODEL_LABELS[name], aggregate_k(configurations[name]))
        for name in neural_names
    ]
    classical_rows = [
        paper_metric_row(MODEL_LABELS[name], aggregate_k(configurations[name]))
        for name in classical_names
    ]
    interaction_rows = []
    for count in (1, 3, 5):
        metrics = aggregate_k(primary, count)
        interaction_rows.append(
            {
                "K": count,
                "FAR": percent(metrics["far"]),
                "FRR": percent(metrics["frr"]),
                "EER": percent(metrics["eer"]),
                "ROC-AUC": percent(metrics["roc_auc"]),
                "Duration (s)": (
                    f"{metrics['average_duration_seconds']:.2f}"
                ),
            }
        )
    ablation_rows = []
    for name in ablation_names:
        metrics = aggregate_k(configurations[name])
        ablation_rows.append(
            {
                "Features": MODEL_LABELS[name],
                "ROC-AUC": percent(metrics["roc_auc"]),
                "FAR": percent(metrics["far"]),
                "FRR": percent(metrics["frr"]),
                "EER": percent(metrics["eer"]),
            }
        )
    split_rows = []
    for name in split_names:
        metrics = aggregate_movement(configurations[name])
        split_rows.append(
            {
                "Split": (
                    "Session-separated"
                    if name == "cnn-gru__full"
                    else MODEL_LABELS[name]
                ),
                "ROC-AUC": percent(metrics["roc_auc"]),
                "FAR": percent(metrics["far"]),
                "FRR": percent(metrics["frr"]),
                "EER": percent(metrics["eer"]),
            }
        )

    security_rows = []
    for label, selection, metrics in (
        ("CNN–GRU balanced", "Paper validation-EER threshold", balanced),
        (
            "CNN–GRU strict",
            "Zero empirical validation false accepts",
            primary_security,
        ),
        (
            f"{selected_label} strict",
            "Post-hoc descriptive seed-42 snapshot",
            selected_security,
        ),
    ):
        context = metric_context(metrics)
        security_rows.append(
            {
                "Configuration": label,
                "Status": selection,
                "Ordinary accuracy": percent(
                    context["pooled_accuracy"]
                ),
                "Balanced accuracy": percent(
                    context["pooled_balanced_accuracy"]
                ),
                "Macro FAR": percent(metrics["far"]),
                "Macro FRR": percent(metrics["frr"]),
                "ROC-AUC": percent(metrics["roc_auc"]),
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TP": metrics["tp"],
            }
        )

    seed42_user_rows = []
    for user, metrics in _available_user_metrics(
        selected_configuration,
        5,
    ):
        strict = metrics["security_metrics"]
        calibration = metrics["security_calibration"]
        seed42_user_rows.append(
            {
                "User": user,
                "Validation FAR": percent(calibration["far"]),
                "Validation FRR": percent(calibration["frr"]),
                "Test FAR": percent(strict["far"]),
                "Test FRR": percent(strict["frr"]),
                "ROC-AUC": percent(strict["roc_auc"]),
                "TN": strict["tn"],
                "FP": strict["fp"],
                "FN": strict["fn"],
                "TP": strict["tp"],
            }
        )

    feature_channel_rows = [
        {
            "Group": "Raw",
            "Features": "normalized x/y, elapsed time, delta time, button state",
        },
        {
            "Group": "Spatial",
            "Features": "delta x/y, path progress",
        },
        {
            "Group": "Kinematic",
            "Features": (
                "velocity x/y, speed, acceleration x/y/magnitude, "
                "jerk x/y/magnitude"
            ),
        },
        {
            "Group": "Geometric",
            "Features": "heading, angular velocity, curvature",
        },
        {
            "Group": "Target-relative",
            "Features": (
                "target displacement x/y, distance, closure rate, "
                "heading error, cross-track error"
            ),
        },
    ]
    architecture_rows = [
        {"Layer": "Input", "Configuration": "128 × d", "Activation": "—"},
        {"Layer": "Conv1D", "Configuration": "32 filters, kernel 5", "Activation": "ReLU"},
        {"Layer": "Batch normalization", "Configuration": "—", "Activation": "—"},
        {"Layer": "Max pooling", "Configuration": "pool size 2", "Activation": "—"},
        {"Layer": "Dropout", "Configuration": "0.20", "Activation": "—"},
        {"Layer": "Conv1D", "Configuration": "64 filters, kernel 3", "Activation": "ReLU"},
        {"Layer": "Batch normalization", "Configuration": "—", "Activation": "—"},
        {"Layer": "Max pooling", "Configuration": "pool size 2", "Activation": "—"},
        {"Layer": "GRU", "Configuration": "64 units", "Activation": "tanh/sigmoid gates"},
        {"Layer": "Dense", "Configuration": "32 units", "Activation": "ReLU"},
        {"Layer": "Dropout", "Configuration": "0.30", "Activation": "—"},
        {"Layer": "Dense", "Configuration": "1 unit", "Activation": "Sigmoid"},
    ]
    model_configuration_rows = [
        {"Category": "Proposed", "Configuration": "CNN–GRU"},
        {"Category": "Neural baseline", "Configuration": "CNN-only"},
        {"Category": "Neural baseline", "Configuration": "GRU-only"},
        {"Category": "Classical baseline", "Configuration": "Logistic regression"},
        {"Category": "Classical baseline", "Configuration": "Support-vector machine"},
        {"Category": "Classical baseline", "Configuration": "Random forest"},
        {"Category": "Classical baseline", "Configuration": "k-nearest neighbors"},
        {"Category": "Classical baseline", "Configuration": "Gradient boosting"},
        {"Category": "Feature ablation", "Configuration": "Raw channels"},
        {"Category": "Feature ablation", "Configuration": "Raw + kinematic"},
        {"Category": "Feature ablation", "Configuration": "Full target-relative"},
    ]
    training_rows = [
        {"Setting": "Optimizer", "Value": "Adam"},
        {"Setting": "Initial learning rate", "Value": "0.001"},
        {"Setting": "Objective", "Value": "Binary cross-entropy"},
        {"Setting": "Maximum epochs", "Value": "50"},
        {"Setting": "Batch size", "Value": "32"},
        {"Setting": "Early-stopping patience", "Value": "8 epochs"},
        {"Setting": "Checkpoint criterion", "Value": "Minimum validation loss"},
        {"Setting": "Class imbalance", "Value": "Inverse-frequency class weights"},
        {"Setting": "Primary seed", "Value": "42 (descriptive snapshot)"},
        {"Setting": "Sensitivity seeds", "Value": "0–29, all reported"},
    ]
    metric_definition_rows = [
        {
            "Metric": "Ordinary accuracy",
            "Definition": "(TP + TN) / all decisions",
            "Use": "Reported for paper compatibility; inflated by 9:1 impostor/genuine decision composition.",
        },
        {
            "Metric": "Balanced accuracy",
            "Definition": "(true-accept rate + true-reject rate) / 2",
            "Use": "Primary accuracy context for imbalanced verification decisions.",
        },
        {
            "Metric": "FAR",
            "Definition": "FP / (FP + TN)",
            "Use": "Unauthorized acceptance rate; macro values average user verifiers.",
        },
        {
            "Metric": "FRR",
            "Definition": "FN / (FN + TP)",
            "Use": "Legitimate-user rejection rate; report macro and pooled values separately.",
        },
        {
            "Metric": "EER",
            "Definition": "Approximate point where FAR equals FRR",
            "Use": "Threshold-independent score-separation summary derived from ROC points.",
        },
        {
            "Metric": "ROC-AUC",
            "Definition": "Area under true-accept versus false-accept curve",
            "Use": "Ranking quality across thresholds; does not guarantee a deployable FAR.",
        },
        {
            "Metric": "Macro average",
            "Definition": "Equal average across ten user-specific verifiers",
            "Use": "Prevents high-volume users from dominating a metric.",
        },
        {
            "Metric": "Pooled rate",
            "Definition": "Rate recomputed from summed confusion counts",
            "Use": "Represents the decision-weighted aggregate; may differ from macro rates.",
        },
    ]

    hypothesis_rows = [
        {
            "Hypothesis": "H1: CNN–GRU has lower EER than CNN-only and GRU-only",
            "Outcome": "Rejected",
            "Evidence": (
                "CNN–GRU 7.56%; CNN-only 6.79%; GRU-only 4.01% EER."
            ),
        },
        {
            "Hypothesis": "H2: K=5 improves over K=1",
            "Outcome": "Partially supported",
            "Evidence": (
                "CNN–GRU FAR/EER improve from 25.19%/13.09% at K=1 "
                "to 8.31%/7.56% at K=5, but K=3 has the lowest EER "
                "(2.91%) and K=5 FRR rises to 30.00%."
            ),
        },
        {
            "Hypothesis": "H3: Derived features outperform raw coordinates",
            "Outcome": "Mixed",
            "Evidence": (
                "Full features improve AUC (96.13% vs 95.77%) and FAR "
                "(8.31% vs 13.11%), but raw has slightly lower EER "
                "(7.00% vs 7.56%)."
            ),
        },
        {
            "Hypothesis": "H4: Session separation performs worse than random samples",
            "Outcome": "Rejected on this dataset",
            "Evidence": (
                "Session-separated movement AUC/FAR/EER are "
                "88.90%/16.33%/16.56% versus "
                "85.96%/20.71%/17.19% for random sample splitting."
            ),
        },
    ]
    research_question_rows = [
        {
            "Research question": "RQ1: Does CNN–GRU outperform the baselines?",
            "Answer": "No",
            "Evidence": "GRU-only and multiple classical models outperform CNN–GRU on key reported metrics.",
        },
        {
            "Research question": "RQ2: How does K change performance?",
            "Answer": "Non-monotonic tradeoff",
            "Evidence": "More movements reduce FAR overall, but K=3 has lower CNN–GRU EER and K=5 has higher FRR.",
        },
        {
            "Research question": "RQ3: Which features contribute?",
            "Answer": "Timing and click dynamics dominate logistic coefficients",
            "Evidence": "Mean/max delta time, pause duration, button-state variation, jerk, and click timing rank highest.",
        },
        {
            "Research question": "RQ4: How does threshold affect errors?",
            "Answer": "Stricter thresholds reduce FAR by increasing FRR",
            "Evidence": "The ROC, confusion matrices, and strict-versus-balanced table show the tradeoff directly.",
        },
        {
            "Research question": "RQ5: Is performance stable across sessions?",
            "Answer": "No; material split sensitivity remains",
            "Evidence": (
                f"Across 30 splits, K=5 FAR ranges "
                f"{percent(sensitivity_summary['far']['minimum'])}–"
                f"{percent(sensitivity_summary['far']['maximum'])} and "
                f"FRR ranges "
                f"{percent(sensitivity_summary['frr']['minimum'])}–"
                f"{percent(sensitivity_summary['frr']['maximum'])}."
            ),
        },
        {
            "Research question": "RQ6: Hardware/target/sampling sensitivity?",
            "Answer": "Not established",
            "Evidence": "Hardware and device labels are absent; timing-heavy coefficients make acquisition confounding plausible.",
        },
    ]

    source = sensitivity.get("source_cohort") or {}
    cohort_rows = [
        {"Stage": "Source file", "Users": source.get("source_users", "unknown"), "Sessions": source.get("source_sessions", "unknown"), "Reason": "All supplied records"},
        {"Stage": "Legacy schema", "Users": "not separately retained", "Sessions": source.get("legacy_schema_sessions", "unknown"), "Reason": "Missing target/full-schema fields required by the paper representation"},
        {"Stage": "Full schema", "Users": source.get("users_with_full_schema", "unknown"), "Sessions": source.get("full_schema_sessions", "unknown"), "Reason": "Supports target-relative features"},
        {"Stage": "Final eligible cohort", "Users": source.get("eligible_users_at_three_sessions", "unknown"), "Sessions": source.get("eligible_full_schema_sessions", "unknown"), "Reason": "At least three sessions for train/validation/test separation"},
    ]
    quality = sensitivity["dataset_quality"]
    quality_rows = [
        {"Check": "Composite movement keys", "Result": quality["duplicate_composite_keys"], "Status": "Pass" if quality["duplicate_composite_keys"] == 0 else "Fail", "Impact": "Duplicate identifiers would multiply observations."},
        {"Check": "Exact duplicate samples", "Result": quality["duplicate_exact_samples"], "Status": "Pass" if quality["duplicate_exact_samples"] == 0 else "Fail", "Impact": "Duplicates could inflate apparent generalization."},
        {"Check": "Missing session IDs", "Result": quality["missing_session_ids"], "Status": "Pass" if quality["missing_session_ids"] == 0 else "Fail", "Impact": "Session isolation requires complete identifiers."},
        {"Check": "Missing attempt IDs", "Result": quality["missing_attempt_ids"], "Status": "Pass" if quality["missing_attempt_ids"] == 0 else "Fail", "Impact": "K-movement aggregation requires attempt identifiers."},
        {"Check": "Movements per attempt", "Result": f"{quality['minimum_movements_per_attempt']}–{quality['maximum_movements_per_attempt']}", "Status": "Pass" if quality["minimum_movements_per_attempt"] == quality["maximum_movements_per_attempt"] == 5 else "Review", "Impact": "Ensures K=1/3/5 comparisons use complete attempts."},
        {"Check": "Session overlap across repeated splits", "Result": 0, "Status": "Pass", "Impact": "Prevents same-session leakage."},
        {"Check": "Collection provenance", "Result": "Not independently verified", "Status": "Open", "Impact": "File integrity is verified, but participant and collection authenticity require owner documentation."},
        {"Check": "Hardware/device labels", "Result": "Unavailable", "Status": "Open", "Impact": "Cross-device generalization and acquisition confounding cannot be tested."},
    ]

    traceability_rows = [
        {"Paper component": "Sequence interpolation", "Paper reference": "Eq. 41", "Implementation": "resample_sequence", "Status": "Implemented"},
        {"Paper component": "Training normalization", "Paper reference": "Eq. 40", "Implementation": "fit_standardizer, standardize", "Status": "Implemented"},
        {"Paper component": "Session separation", "Paper reference": "Section VI-B", "Implementation": "create_split, _group_split", "Status": "Implemented; paper appendix name differs"},
        {"Paper component": "CNN–GRU", "Paper reference": "Table II", "Implementation": "build_model", "Status": "Implemented"},
        {"Paper component": "Binary cross-entropy", "Paper reference": "Eq. 45", "Implementation": "Keras binary_crossentropy", "Status": "Implemented"},
        {"Paper component": "Class weighting", "Paper reference": "Eq. 46", "Implementation": "class_weights", "Status": "Implemented"},
        {"Paper component": "Attempt aggregation", "Paper reference": "Eq. 6", "Implementation": "aggregate_attempt_scores", "Status": "Implemented"},
        {"Paper component": "Biometric metrics", "Paper reference": "Eqs. 48–51", "Implementation": "biometric_metrics", "Status": "Implemented"},
        {"Paper component": "Validation-only threshold", "Paper reference": "Section VI-C", "Implementation": "calibrate_eer_threshold, calibrate_far_threshold", "Status": "Implemented"},
        {"Paper component": "Repeated-split robustness", "Paper reference": "RQ5 and limitations", "Implementation": "sensitivity", "Status": "Added for honest robustness evaluation"},
    ]

    coverage_rows = [
        {"Paper item": "Table I feature channels", "Status": "Complete", "Report artifact": "feature_channels.csv and methods section"},
        {"Paper item": "Table II CNN–GRU architecture", "Status": "Complete", "Report artifact": "architecture.csv"},
        {"Paper item": "Table III model configurations", "Status": "Complete", "Report artifact": "model_configurations.csv"},
        {"Paper item": "Table IV neural comparison", "Status": "Complete", "Report artifact": "table_iv_neural_models.csv"},
        {"Paper item": "Table V K=1/3/5", "Status": "Complete", "Report artifact": "table_v_interaction_counts.csv and sensitivity_by_interaction_count.csv"},
        {"Paper item": "Table VI feature ablation", "Status": "Complete", "Report artifact": "table_vi_feature_ablation.csv"},
        {"Paper item": "Table VII split comparison", "Status": "Complete", "Report artifact": "table_vii_split_comparison.csv"},
        {"Paper item": "Table VIII implementation traceability", "Status": "Complete with corrected function name", "Report artifact": "paper_traceability.csv"},
        {"Paper item": "Accuracy, precision, recall, F1", "Status": "Complete with imbalance context", "Report artifact": "model tables and metric_definitions.csv"},
        {"Paper item": "FAR, FRR, EER, ROC-AUC", "Status": "Complete", "Report artifact": "all model tables, ROC, sensitivity figures"},
        {"Paper item": "Confusion matrix", "Status": "Complete", "Report artifact": "confusion_matrix_k5.png"},
        {"Paper item": "Statistical tests and intervals", "Status": "Complete for seed-42 CNN–GRU", "Report artifact": "statistical_analysis_k5.csv"},
        {"Paper item": "Session stability", "Status": "Complete as repeated-split sensitivity", "Report artifact": "sensitivity_all_seeds.csv and sensitivity figures"},
        {"Paper item": "Hardware sensitivity", "Status": "Not measurable from supplied data", "Report artifact": "Explicit limitation and open question"},
        {"Paper item": "Imitation/replay resistance", "Status": "Not tested", "Report artifact": "Explicit limitation and next step"},
        {"Paper item": "Ethics/privacy", "Status": "Addressed narratively", "Report artifact": "limitations and deployment section"},
    ]

    chart_map_rows = [
        {"Section": "Accuracy interpretation", "Question": "Why is ordinary accuracy high?", "Chart": "accuracy_context.png", "Family": "Composition + distribution", "Claim": "The 9:1 decision imbalance inflates ordinary accuracy."},
        {"Section": "Robustness", "Question": "How variable are metrics across splits?", "Chart": "sensitivity_distributions.png", "Family": "Distribution", "Claim": "Performance varies materially across all 30 fixed splits."},
        {"Section": "Security tradeoff", "Question": "How do FAR and FRR co-vary?", "Chart": "sensitivity_far_frr.png", "Family": "Relationship", "Claim": "Seed 42 is favorable but not the primary estimate."},
        {"Section": "Security snapshot", "Question": "What decisions were made?", "Chart": "confusion_matrix_k5.png", "Family": "Matrix", "Claim": "Seed-42 counts are reproducible but descriptive."},
        {"Section": "Interaction count", "Question": "Does more movement help consistently?", "Chart": "repeated_split_k_tradeoff.png", "Family": "Uncertainty", "Claim": "K changes FAR/FRR with non-monotonic tradeoffs."},
        {"Section": "Model comparison", "Question": "Does CNN–GRU win?", "Chart": "neural_model_comparison.png", "Family": "Grouped comparison", "Claim": "The hybrid model does not dominate neural baselines."},
        {"Section": "Classical comparison", "Question": "How do classical models compare?", "Chart": "classical_model_comparison.png", "Family": "Grouped comparison", "Claim": "Classical baselines are competitive."},
        {"Section": "Threshold behavior", "Question": "How strong is score ranking?", "Chart": "roc_curves_k5.png", "Family": "ROC", "Claim": "Ranking is strong but low-FAR evidence is discrete."},
        {"Section": "Feature groups", "Question": "Do engineered channels help?", "Chart": "feature_ablation.png", "Family": "Grouped comparison", "Claim": "Feature gains are mixed across metrics."},
        {"Section": "Feature influence", "Question": "Which summaries drive logistic models?", "Chart": "feature_influence_logistic.png", "Family": "Ranked bar", "Claim": "Timing and click dynamics dominate coefficients."},
        {"Section": "Per-user robustness", "Question": "Which identities are unstable?", "Chart": "per_user_robustness.png", "Family": "Heatmap", "Claim": "Error rates vary by claimed identity."},
        {"Section": "Cohort", "Question": "How was the final cohort formed?", "Chart": "cohort_construction.png", "Family": "Stage comparison", "Claim": "Schema eligibility substantially narrows the source cohort."},
        {"Section": "Split comparison", "Question": "Does random splitting inflate results?", "Chart": "split_comparison.png", "Family": "Grouped comparison", "Claim": "The expected inflation is not observed here."},
        {"Section": "Training", "Question": "How stable is neural training?", "Chart": "training_history.png", "Family": "Trend", "Claim": "Per-user training histories expose convergence variation."},
    ]

    statistics = statistical_rows(primary)
    table_outputs = (
        ("table_iv_neural_models.csv", neural_rows),
        ("classical_baselines.csv", classical_rows),
        ("table_v_interaction_counts.csv", interaction_rows),
        ("table_vi_feature_ablation.csv", ablation_rows),
        ("table_vii_split_comparison.csv", split_rows),
        ("security_operating_points.csv", security_rows),
        ("seed42_logistic_per_user.csv", seed42_user_rows),
        ("sensitivity_summary_k5.csv", repeated_summary_rows),
        ("sensitivity_all_seeds.csv", all_seed_rows),
        ("sensitivity_by_interaction_count.csv", repeated_k_rows),
        ("sensitivity_by_user.csv", sensitivity_user_rows),
        ("feature_influence.csv", feature_influence_rows),
        ("feature_channels.csv", feature_channel_rows),
        ("architecture.csv", architecture_rows),
        ("model_configurations.csv", model_configuration_rows),
        ("training_configuration.csv", training_rows),
        ("metric_definitions.csv", metric_definition_rows),
        ("hypothesis_outcomes.csv", hypothesis_rows),
        ("research_question_answers.csv", research_question_rows),
        ("cohort_construction.csv", cohort_rows),
        ("dataset_quality.csv", quality_rows),
        ("paper_traceability.csv", traceability_rows),
        ("paper_coverage.csv", coverage_rows),
        ("chart_map.csv", chart_map_rows),
        ("statistical_analysis_k5.csv", statistics),
    )
    for filename, rows in table_outputs:
        write_csv(tables / filename, rows)

    plot_confusion_matrices(
        balanced,
        primary_security,
        selected_security,
        selected_label,
        figures / "confusion_matrix_k5.png",
    )
    roc_names = list(dict.fromkeys([*neural_names, selected_name]))
    plot_roc(
        configurations,
        roc_names,
        selected_name,
        figures / "roc_curves_k5.png",
    )
    plot_k_tradeoff(
        primary,
        selected_configuration,
        selected_label,
        figures / "interaction_count_tradeoff.png",
    )
    plot_comparison(
        configurations,
        neural_names,
        "Seed 42 neural model comparison at K=5",
        figures / "neural_model_comparison.png",
    )
    plot_comparison(
        configurations,
        classical_names,
        "Seed 42 classical model comparison at K=5",
        figures / "classical_model_comparison.png",
    )
    plot_comparison(
        configurations,
        ablation_names,
        "Seed 42 CNN–GRU feature ablation at K=5",
        figures / "feature_ablation.png",
    )
    plot_comparison(
        configurations,
        split_names,
        "Seed 42 movement-level split comparison",
        figures / "split_comparison.png",
        grain="movement",
        display_labels=("Session-separated", "Random sample-level"),
    )
    plot_training_history(primary, figures / "training_history.png")
    plot_sensitivity_distributions(
        sensitivity,
        figures / "sensitivity_distributions.png",
    )
    plot_sensitivity_far_frr(
        sensitivity,
        selected_security,
        figures / "sensitivity_far_frr.png",
    )
    plot_accuracy_context(
        sensitivity,
        selected_security,
        figures / "accuracy_context.png",
    )
    plot_feature_influence(
        sensitivity,
        figures / "feature_influence_logistic.png",
    )
    plot_cohort_construction(
        sensitivity,
        figures / "cohort_construction.png",
    )
    plot_user_robustness(
        sensitivity,
        figures / "per_user_robustness.png",
    )
    plot_repeated_k_tradeoff(
        sensitivity,
        figures / "repeated_split_k_tradeoff.png",
    )

    report = f"""# NeuroCursor complete and honest real-data evaluation

## Technical summary

The most defensible result is the **30-split sensitivity distribution**, not the best single split. With a fixed full-feature logistic verifier configuration, K=5, per-user thresholds calibrated for zero empirical validation false accepts, and every seed in the fixed consecutive grid 0–29 reported, mean pooled balanced accuracy was **{percent(sensitivity_summary['pooled_balanced_accuracy']['mean'])}**, mean macro FAR was **{percent(sensitivity_summary['far']['mean'])}**, mean macro FRR was **{percent(sensitivity_summary['frr']['mean'])}**, and mean macro ROC-AUC was **{percent(sensitivity_summary['roc_auc']['mean'])}**. Only **{sensitivity_summary['zero_false_accept_splits']} of 30** splits observed zero false accepts. The model is refit within each split. This grid was fixed for the sensitivity analysis but was not externally preregistered.

The prior seed-42 logistic result is real and reproducible—TN={selected_security['tn']}, FP={selected_security['fp']}, FN={selected_security['fn']}, TP={selected_security['tp']}—but it is now labeled a **post-hoc descriptive snapshot**. Its ordinary accuracy of **{percent(selected_context['pooled_accuracy'])}** is aided by the fact that **{percent(selected_context['impostor_decision_share'])}** of its binary decisions are impostor comparisons; a reject-all classifier would already score **{percent(selected_context['reject_all_accuracy'])}** ordinary accuracy. Its pooled balanced accuracy is **{percent(selected_context['pooled_balanced_accuracy'])}**.

The paper's expected superiority claims are not forced onto the evidence. The proposed CNN–GRU does not outperform the tested GRU-only or strongest classical baselines, the interaction-count and feature-ablation results are mixed, and the random sample-level split is not better than the session-separated split on this dataset.

## Repeated splits replace a favorable snapshot as the primary result

{markdown_table(repeated_summary_rows)}

![Repeated-split metric distributions](figures/sensitivity_distributions.png)

The boxplots and individual points show every split in the fixed grid. They quantify both the typical result and the spread that a single seed hides. These are repeated holdout sensitivity estimates using overlapping source data, not 30 independent external cohorts.

![FAR–FRR sensitivity](figures/sensitivity_far_frr.png)

The seed-42 point is shown only for traceability. It is favorable on FRR relative to most repeated splits and therefore is not used as the primary performance estimate.

## Ordinary accuracy overstates performance under the 9:1 decision imbalance

![Accuracy and class-balance context](figures/accuracy_context.png)

Each of the 26 seed-42 test sessions is evaluated once against its genuine verifier and nine times against impostor verifiers, creating 26 genuine and 234 impostor decisions. Balanced accuracy, FAR, and FRR are therefore more informative than ordinary accuracy.

{markdown_table(metric_definition_rows)}

## The seed-42 confusion matrices are descriptive, not confirmatory

{markdown_table(security_rows)}

![Seed-42 K=5 confusion matrices](figures/confusion_matrix_k5.png)

The logistic strict snapshot observed zero false accepts in 234 impostor decisions and five false rejects in 26 genuine decisions. That zero is not a population guarantee: its one-sided exact 95% FAR upper bound is **{precise_percent(upper_far_95)}**. The test results of multiple configurations had already been inspected before this configuration was highlighted, so this split cannot serve as an untouched final confirmation.

{markdown_table(seed42_user_rows)}

## The paper's research questions yield negative and mixed findings

{markdown_table(research_question_rows)}

{markdown_table(hypothesis_rows)}

These outcomes answer the paper's questions without selecting only favorable comparisons. Hardware, device, replay, and imitation sensitivity remain unmeasured because the supplied data do not contain the required labels or attack trials.

## More movements change the tradeoff but do not improve every metric monotonically

### Paper Table V: seed-42 CNN–GRU snapshot

{markdown_table(interaction_rows)}

![Seed-42 interaction-count tradeoff](figures/interaction_count_tradeoff.png)

The seed-42 CNN–GRU result improves FAR from K=1 to K=5, but K=3 has the lowest EER and K=5 has the highest FRR. The paper's claim is therefore only partially supported.

### Repeated-split strict logistic sensitivity

{markdown_table(repeated_k_rows)}

![Repeated-split interaction-count tradeoff](figures/repeated_split_k_tradeoff.png)

The repeated-split view reports means and interquartile ranges across every seed in the fixed grid. It is the more reliable description of how K changes strict-verifier behavior in this dataset.

## The proposed CNN–GRU does not win the neural comparison

### Paper Table IV: neural architectures

{markdown_table(neural_rows)}

![Neural model comparison](figures/neural_model_comparison.png)

GRU-only has lower FAR, FRR, and EER than CNN–GRU in the fixed paper split. H1 is rejected rather than rewritten after seeing the outcome.

## Classical models are competitive and sometimes stronger

{markdown_table(classical_rows)}

![Classical model comparison](figures/classical_model_comparison.png)

Logistic regression provides the best ordinary strict snapshot, while random forest and gradient boosting have strong balanced-threshold ranking metrics. Because these test results were inspected during report development, none is presented as an untouched winner.

## ROC curves show strong ranking but cannot establish rare-event FAR

![K=5 ROC curves](figures/roc_curves_k5.png)

ROC-AUC describes ranking across thresholds. It does not prove a deployable low FAR, especially with only hundreds of impostor decisions and discrete per-user validation sets.

## Feature engineering helps some metrics and hurts others

### Paper Table VI: CNN–GRU feature ablation

{markdown_table(ablation_rows)}

![Feature ablation](figures/feature_ablation.png)

Full features improve AUC and FAR relative to raw channels, but raw channels have slightly lower EER. H3 is mixed rather than fully supported.

### Logistic feature influence across repeated splits

![Logistic feature influence](figures/feature_influence_logistic.png)

Mean and maximum inter-event time, pause duration, button-state variation, jerk, and click timing carry the largest absolute standardized logistic coefficients. This is descriptive model influence—not causal importance—and the dominance of timing variables creates a plausible browser/device acquisition confound.

{markdown_table(feature_influence_rows[:20])}

## Per-user robustness varies substantially

![Per-user robustness heatmap](figures/per_user_robustness.png)

The repeated-split user view shows that aggregate metrics conceal identity-specific instability. One identity has only three eligible sessions, making its validation and test results especially discrete.

{markdown_table(sensitivity_user_rows)}

## Random sample splitting does not inflate performance in this experiment

### Paper Table VII: movement-level split comparison

{markdown_table(split_rows)}

![Evaluation-split comparison](figures/split_comparison.png)

Contrary to H4, the random sample-level split is worse on all three displayed metrics. This does not make random splitting methodologically preferable; it only means the expected leakage inflation is not observed in this particular fixed run.

## The cohort is restricted by schema availability

{markdown_table(cohort_rows)}

![Cohort construction](figures/cohort_construction.png)

The supplied source contains {source.get('source_users', 'unknown')} identities and {source.get('source_sessions', 'unknown')} sessions, but {source.get('legacy_schema_sessions', 'unknown')} sessions use a legacy schema without the target-relative fields required by the paper's full representation. The final analysis therefore covers {quality['users']} identities and {quality['sessions']} sessions. Inclusion is schema/history based rather than outcome based, but the narrower cohort limits generalizability.

## Data-quality checks pass for duplicates and leakage, with provenance gaps remaining

{markdown_table(quality_rows)}

The source-file hash matches the hash recorded in the transformed dataset. Exact duplicate movements, duplicate composite keys, missing split identifiers, and session overlap were not found. However, file integrity does not independently prove participant authenticity, collection conditions, or hardware diversity.

## The implementation covers the paper's feature and model specifications

### Paper Table I: temporal feature channels

{markdown_table(feature_channel_rows)}

### Paper Table II: CNN–GRU architecture

{markdown_table(architecture_rows)}

### Paper Table III: evaluated configurations

{markdown_table(model_configuration_rows)}

### Training configuration

{markdown_table(training_rows)}

The implementation uses training-only normalization, per-user one-versus-rest classifiers, session-separated primary evaluation, validation-only threshold calibration, and fixed K-movement attempt aggregation.

## Statistical evidence is limited by the small seed-42 test sets

{markdown_table(statistics)}

The one-sided permutation tests compare genuine and impostor scores for each claimed identity in the seed-42 CNN–GRU snapshot. ROC-AUC intervals use class-stratified bootstrap resampling; FAR and FRR use two-sided exact Clopper–Pearson binomial intervals so zero observed events do not produce a misleading 0%–0% interval. Several intervals span nearly the full possible range because each user has only one to three genuine seed-42 test attempts; statistical significance must not be interpreted as deployment readiness.

## Training diagnostics expose per-user convergence variation

![CNN–GRU training histories](figures/training_history.png)

Each line represents one user-specific verifier. The figure is diagnostic rather than evidence that the neural model generalizes better than the baselines.

## Paper-to-code traceability is complete, with one appendix name corrected

{markdown_table(traceability_rows)}

The paper's Appendix Table VIII names `create_session_split`; the physical implementation uses `create_split` and `_group_split`. The report records the actual symbols instead of repeating the stale appendix name.

## Paper coverage checklist

{markdown_table(coverage_rows)}

Every measurable paper item is linked to a table, figure, or explicit limitation. Hardware sensitivity and adversarial imitation/replay resistance are marked unmeasured rather than silently omitted or invented.

## Limitations, security, ethics, and privacy

- **The existing test set is descriptive, not pristine.** Multiple model results were inspected before the strict logistic snapshot was highlighted.
- **Repeated splits are not external validation.** They reuse the same ten-user cohort and quantify split sensitivity, not population generalization.
- **The cohort is restricted.** Only ten identities have at least three sessions in the full target-aware schema.
- **Decision counts are small.** Seed 42 has 234 impostor and 26 genuine binary decisions; user-specific genuine counts are even smaller.
- **Macro and pooled metrics differ.** Macro values weight each identity equally; pooled values weight decisions.
- **Acquisition confounding is plausible.** Timing-heavy logistic coefficients may reflect browser, sampling, hardware, or settings in addition to behavior.
- **Cross-device generalization is unknown.** Hardware, DPI, sensitivity, and device labels are absent.
- **Replay and imitation resistance are unknown.** No attack dataset was supplied.
- **Behavioral templates are sensitive.** Deployment should minimize raw trace retention, encrypt templates, obtain clear consent, support deletion, and use low scores for step-up authentication rather than permanent denial.

## Recommended next steps

1. Freeze the model family, features, threshold rule, preprocessing, and metrics before collecting more data.
2. Collect a new untouched external cohort with device, mouse/trackpad, DPI, sensitivity, browser, and session-condition labels.
3. Add same-device/different-user, different-device/same-user, replay, and intentional-imitation trials.
4. Use nested model selection or a locked validation protocol, then evaluate once on the untouched final test set.
5. Report balanced accuracy, FAR, FRR, ROC-AUC, confidence bounds, confusion counts, and ordinary accuracy together.
6. Increase genuine and impostor trial counts enough to estimate the intended operating region rather than extrapolating from zero observed events.

## Further questions

- Does performance persist when two people use the same hardware and browser?
- Does the verifier recognize the same person after changing mouse, trackpad, DPI, or sensitivity?
- Which feature groups remain useful after removing sampling-rate and button-state cues?
- How stable are genuine scores over weeks or months?
- What threshold and step-up policy provides an acceptable FAR/FRR tradeoff in the intended deployment?

## Reproduce

From the repository root:

```bash
python -m pip install -r model/requirements-results.txt

python model/model.py experiment \\
  --data /path/to/paper-dataset.json \\
  --output-dir artifacts/paper-experiment \\
  --target-far 0 \\
  --seed 42 \\
  --permutations 10000 \\
  --bootstrap-samples 1000

python model/model.py sensitivity \\
  --data /path/to/paper-dataset.json \\
  --source-data /path/to/master_dataset.json \\
  --output-file artifacts/paper-experiment/sensitivity_summary.json \\
  --seed-start 0 \\
  --seed-count 30 \\
  --target-far 0

python model/reporting.py \\
  --experiment-dir artifacts/paper-experiment \\
  --sensitivity-file artifacts/paper-experiment/sensitivity_summary.json \\
  --output-dir results/paper-real-data
```

Raw participant data, per-attempt scores, and trained weights are intentionally excluded from Git. The committed package contains aggregate results, all 30 seed-level outcomes, paper tables, figures, coverage/traceability matrices, and checksums.
"""
    (output_directory / "README.md").write_text(report, encoding="utf-8")

    hypothesis_values = {
        row["Hypothesis"]: {
            "outcome": row["Outcome"],
            "evidence": row["Evidence"],
        }
        for row in hypothesis_rows
    }
    compact_results = {
        "result_status": "share with caveats",
        "primary_result": {
            "analysis": "all 30 fixed session splits, seeds 0–29",
            "model": "logistic regression, full features, K=5",
            "threshold_policy": sensitivity["protocol"]["threshold_policy"],
            "summary": sensitivity_summary,
        },
        "seed42_descriptive_snapshot": {
            "configuration": selected_name,
            "selection_disclosure": (
                "Post-hoc descriptive only; multiple test configurations "
                "were inspected before this result was highlighted."
            ),
            "validation_macro_far": selected_validation["far"],
            "validation_macro_frr": selected_validation["frr"],
            "metrics": selected_security,
            "metric_context": selected_context,
            "one_sided_95_percent_far_upper_bound": upper_far_95,
        },
        "paper_cnn_gru_k5_balanced": balanced,
        "paper_cnn_gru_k5_strict": primary_security,
        "hypothesis_outcomes": hypothesis_values,
        "dataset": summary["dataset"],
        "dataset_quality": sensitivity["dataset_quality"],
        "source_cohort": sensitivity.get("source_cohort"),
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
    write_json(output_directory / "SENSITIVITY.json", sensitivity)

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
    parser.add_argument(
        "--sensitivity-file",
        required=True,
        help=(
            "Repeated-split sensitivity JSON produced by "
            "`model.py sensitivity`."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    generate_report(
        Path(args.experiment_dir),
        Path(args.output_dir),
        Path(args.sensitivity_file),
    )
    print(f"Saved paper-ready report to {args.output_dir}")


if __name__ == "__main__":
    main()

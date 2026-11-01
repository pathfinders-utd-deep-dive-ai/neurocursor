"""Reproduce reviewer-requested input counts and grouped feature influence.

This script uses only the archived result bundle. It does not retrain models
or claim enrollment-size or sequence-order results that are absent from the
experiment.
"""

from __future__ import annotations

import ast
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY = ROOT / "results/paper-real-data-2026-07-29/SENSITIVITY.json"
MODEL_SOURCE = ROOT / "model/model.py"
PAPER_SOURCE = ROOT / "model/Paper/paper.tex"
RESPONSE_SOURCE = ROOT / "model/Paper/response_to_reviewers.tex"

FEATURE_GROUPS = (
    "RAW_FEATURES",
    "SPATIAL_FEATURES",
    "KINEMATIC_FEATURES",
    "GEOMETRIC_FEATURES",
    "TARGET_RELATIVE_FEATURES",
)
DERIVED_CLASSICAL_FEATURES = (
    "duration",
    "path_length",
    "straight_distance",
    "path_efficiency",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "max_acceleration",
    "pause_duration",
    "click_timing",
    "acceleration_energy",
    "jerk_energy",
    "dimensionless_jerk",
)


def model_feature_groups() -> dict[str, tuple[str, ...]]:
    """Read feature tuples from model.py without importing TensorFlow."""
    module = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    groups: dict[str, tuple[str, ...]] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in FEATURE_GROUPS:
                value = ast.literal_eval(node.value)
                groups[target.id] = tuple(str(item) for item in value)
    missing = set(FEATURE_GROUPS) - set(groups)
    if missing:
        raise AssertionError(f"Missing feature definitions: {sorted(missing)}")
    return groups


def expected_classical_names(features: tuple[str, ...]) -> set[str]:
    names = {
        f"{statistic}:{feature}"
        for statistic in ("mean", "std", "min", "max")
        for feature in features
    }
    names.update(f"derived:{name}" for name in DERIVED_CLASSICAL_FEATURES)
    return names


def verify_document_dimensions() -> None:
    paper = PAPER_SOURCE.read_text(encoding="utf-8")
    response = RESPONSE_SOURCE.read_text(encoding="utf-8")
    for source in (paper, response):
        assert "128\\times27" in source
        assert "classical 113" not in source
        assert "128\\times25" not in source


def feature_group(name: str) -> str:
    timing_click = (
        "delta_time",
        "elapsed_time",
        "button_state",
        "pause_duration",
        "click_timing",
        "derived:duration",
    )
    kinematic = ("velocity", "speed", "acceleration", "jerk")
    if any(token in name for token in timing_click):
        return "timing_and_click"
    if any(token in name for token in kinematic):
        return "kinematic"
    return "spatial_geometric_target"


def main() -> None:
    groups = model_feature_groups()
    raw = groups["RAW_FEATURES"]
    raw_spatial_kinematic = (
        raw + groups["SPATIAL_FEATURES"] + groups["KINEMATIC_FEATURES"]
    )
    full = tuple(
        feature for group in FEATURE_GROUPS for feature in groups[group]
    )
    feature_sets = {
        "raw": raw,
        "raw-spatial-kinematic": raw_spatial_kinematic,
        "full": full,
    }
    for name, features in feature_sets.items():
        count = len(features)
        classical = 4 * count + (13 if name == "full" else 0)
        print(f"{name}: neural=128x{count}; classical={classical}")

    document = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    rows = document["feature_influence"]
    actual_names = {str(row["feature"]) for row in rows}
    expected_names = expected_classical_names(full)
    assert actual_names == expected_names
    assert len(rows) == 4 * len(full) + len(DERIVED_CLASSICAL_FEATURES) == 121
    verify_document_dimensions()
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(feature_group(row["feature"]), []).append(
            float(row["coefficient_mean"])
        )

    for group, values in groups.items():
        print(
            f"{group}: n={len(values)}, mean={statistics.fmean(values):.6f}, "
            f"median={statistics.median(values):.6f}, max={max(values):.6f}"
        )


if __name__ == "__main__":
    main()

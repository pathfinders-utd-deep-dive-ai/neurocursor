"""Reproduce reviewer-requested input counts and grouped feature influence.

This script uses only the archived result bundle. It does not retrain models
or claim enrollment-size or sequence-order results that are absent from the
experiment.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY = ROOT / "results/paper-real-data-2026-07-29/SENSITIVITY.json"

FEATURE_COUNTS = {
    "raw": 5,
    "raw-kinematic": 17,
    "full": 25,
}


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
    for name, count in FEATURE_COUNTS.items():
        classical = 4 * count + (13 if name == "full" else 0)
        print(f"{name}: neural=128x{count}; classical={classical}")

    document = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    rows = document["feature_influence"]
    assert len(rows) == 121
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

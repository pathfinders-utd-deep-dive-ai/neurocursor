"""Paper-aligned NeuroCursor training, evaluation, and prediction pipeline.

The implementation follows the July 2026 NeuroCursor paper:

* raw point-and-click events are converted into the paper's temporal channels;
* trajectories are interpolated to 128 samples and normalized from training
  statistics only;
* one-versus-rest CNN-GRU verifiers use session/trial-separated splits;
* thresholds are calibrated only on validation data and frozen for testing;
* K=1, 3, and 5 movement attempts, baselines, ablations, and statistical
  analyses are available through the experiment command.

The paper intentionally contains blank results tables. This module computes
those results from supplied data; it never embeds or fabricates measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    MaxPooling1D,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


SEED = 42
EPSILON = 1e-8
SEQUENCE_LENGTH = 128
INTERACTION_COUNTS = (1, 3, 5)
DEFAULT_TARGET_FAR = 0.0

RAW_FEATURES = (
    "x_normalized",
    "y_normalized",
    "elapsed_time",
    "delta_time",
    "button_state",
)
SPATIAL_FEATURES = (
    "delta_x",
    "delta_y",
    "path_progress",
)
KINEMATIC_FEATURES = (
    "velocity_x",
    "velocity_y",
    "speed",
    "acceleration_x",
    "acceleration_y",
    "acceleration",
    "jerk_x",
    "jerk_y",
    "jerk",
)
GEOMETRIC_FEATURES = (
    "heading",
    "angular_velocity",
    "curvature",
)
TARGET_RELATIVE_FEATURES = (
    "running_path_efficiency",
    "target_delta_x",
    "target_delta_y",
    "target_distance",
    "target_closure_rate",
    "target_heading_error",
    "cross_track_error",
)
FULL_FEATURE_NAMES = (
    RAW_FEATURES
    + SPATIAL_FEATURES
    + KINEMATIC_FEATURES
    + GEOMETRIC_FEATURES
    + TARGET_RELATIVE_FEATURES
)
FEATURE_SETS = {
    "raw": RAW_FEATURES,
    "raw-kinematic": RAW_FEATURES + SPATIAL_FEATURES + KINEMATIC_FEATURES,
    "full": FULL_FEATURE_NAMES,
}
NEURAL_MODELS = ("cnn-gru", "cnn-only", "gru-only")
CLASSICAL_MODELS = (
    "logistic-regression",
    "svm",
    "random-forest",
    "knn",
    "gradient-boosting",
)


@dataclass(frozen=True)
class CursorDataset:
    sequences: list[np.ndarray]
    user_ids: np.ndarray
    session_ids: np.ndarray
    attempt_ids: np.ndarray
    movement_indexes: np.ndarray
    durations: np.ndarray
    feature_names: tuple[str, ...]
    source_format: str
    metadata: dict[str, Any]


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_json_default)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _first_present(mapping: Mapping[str, Any], names: Iterable[str]):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _time_scale(unit: str) -> float:
    normalized = unit.strip().lower()
    scales = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    if normalized not in scales:
        raise ValueError(
            f"Unsupported time_unit '{unit}'. Use seconds, milliseconds, "
            "microseconds, or nanoseconds."
        )
    return scales[normalized]


def _event_value(event: Mapping[str, Any], names: Sequence[str], label: str):
    value = _first_present(event, names)
    if value is None:
        raise ValueError(f"Raw event is missing {label}.")
    return float(value)


def _extract_screen(sample: Mapping[str, Any], defaults: Mapping[str, Any]):
    screen = sample.get("screen", defaults.get("screen", {}))
    if not isinstance(screen, Mapping):
        screen = {}

    events = sample.get("events") or sample.get("trajectory") or []
    first_event = events[0] if events and isinstance(events[0], Mapping) else {}
    width = _first_present(
        sample,
        ("screen_width", "canvas_width"),
    )
    height = _first_present(
        sample,
        ("screen_height", "canvas_height"),
    )
    width = width if width is not None else _first_present(
        screen, ("width", "screen_width", "canvas_width")
    )
    height = height if height is not None else _first_present(
        screen, ("height", "screen_height", "canvas_height")
    )
    width = width if width is not None else _first_present(
        first_event, ("screen_width", "canvas_width")
    )
    height = height if height is not None else _first_present(
        first_event, ("screen_height", "canvas_height")
    )
    if width is None or height is None:
        raise ValueError(
            "Raw samples require screen width and height for Eq. 39 "
            "normalization."
        )
    width = float(width)
    height = float(height)
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Screen width and height must be positive and finite.")
    return width, height


def _extract_target(sample: Mapping[str, Any]):
    target = sample.get("target", {})
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        if len(target) < 2:
            raise ValueError("Target arrays must contain x and y.")
        return float(target[0]), float(target[1])
    if not isinstance(target, Mapping):
        target = {}

    events = sample.get("events") or sample.get("trajectory") or []
    first_event = events[0] if events and isinstance(events[0], Mapping) else {}
    target_x = _first_present(sample, ("target_x",))
    target_y = _first_present(sample, ("target_y",))
    target_x = target_x if target_x is not None else _first_present(
        target, ("x", "target_x")
    )
    target_y = target_y if target_y is not None else _first_present(
        target, ("y", "target_y")
    )
    target_x = target_x if target_x is not None else _first_present(
        first_event, ("target_x",)
    )
    target_y = target_y if target_y is not None else _first_present(
        first_event, ("target_y",)
    )
    if target_x is None or target_y is None:
        raise ValueError(
            "Raw samples require the randomized target x and y coordinates."
        )
    return float(target_x), float(target_y)


def _unwrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def engineer_raw_features(
    sample: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float]:
    """Implement Eqs. 22-40 and the complete temporal channel table."""
    defaults = defaults or {}
    events = sample.get("events") or sample.get("trajectory")
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("Raw samples require at least two cursor events.")
    if not all(isinstance(event, Mapping) for event in events):
        raise ValueError("Every raw cursor event must be a JSON object.")

    width, height = _extract_screen(sample, defaults)
    target_x, target_y = _extract_target(sample)
    time_unit = str(sample.get("time_unit", defaults.get("time_unit", "seconds")))
    scale = _time_scale(time_unit)

    x = np.asarray(
        [_event_value(event, ("x", "cursor_x"), "cursor x") for event in events],
        dtype=np.float64,
    )
    y = np.asarray(
        [_event_value(event, ("y", "cursor_y"), "cursor y") for event in events],
        dtype=np.float64,
    )
    timestamps = np.asarray(
        [
            _event_value(event, ("t", "time", "timestamp"), "timestamp")
            for event in events
        ],
        dtype=np.float64,
    ) * scale
    button = np.asarray(
        [
            float(
                _first_present(
                    event,
                    ("button_state", "button", "pressed"),
                )
                or 0.0
            )
            for event in events
        ],
        dtype=np.float64,
    )

    if not np.isfinite(np.column_stack((x, y, timestamps, button))).all():
        raise ValueError("Raw cursor events contain NaN or infinity.")
    timestamp_differences = np.diff(timestamps)
    if np.any(timestamp_differences < 0):
        raise ValueError("Cursor timestamps must preserve recorded order.")

    x = x / width
    y = y / height
    target_x /= width
    target_y /= height
    elapsed = timestamps - timestamps[0]
    delta_time = np.zeros_like(elapsed)
    delta_time[1:] = np.maximum(timestamp_differences, EPSILON)

    delta_x = np.zeros_like(x)
    delta_y = np.zeros_like(y)
    delta_x[1:] = np.diff(x)
    delta_y[1:] = np.diff(y)

    velocity_x = np.zeros_like(x)
    velocity_y = np.zeros_like(y)
    velocity_x[1:] = delta_x[1:] / delta_time[1:]
    velocity_y[1:] = delta_y[1:] / delta_time[1:]
    speed = np.hypot(velocity_x, velocity_y)

    acceleration_x = np.zeros_like(x)
    acceleration_y = np.zeros_like(y)
    acceleration_x[1:] = np.diff(velocity_x) / delta_time[1:]
    acceleration_y[1:] = np.diff(velocity_y) / delta_time[1:]
    acceleration = np.hypot(acceleration_x, acceleration_y)

    jerk_x = np.zeros_like(x)
    jerk_y = np.zeros_like(y)
    jerk_x[1:] = np.diff(acceleration_x) / delta_time[1:]
    jerk_y[1:] = np.diff(acceleration_y) / delta_time[1:]
    jerk = np.hypot(jerk_x, jerk_y)

    step_distance = np.hypot(delta_x, delta_y)
    cumulative_path = np.cumsum(step_distance)
    total_path = float(cumulative_path[-1])
    path_progress = (
        cumulative_path / total_path
        if total_path > EPSILON
        else np.zeros_like(cumulative_path)
    )
    distance_from_start = np.hypot(x - x[0], y - y[0])
    running_path_efficiency = np.ones_like(cumulative_path)
    moving = cumulative_path > EPSILON
    running_path_efficiency[moving] = np.clip(
        distance_from_start[moving] / cumulative_path[moving],
        0.0,
        1.0,
    )

    heading = np.arctan2(velocity_y, velocity_x)
    angular_velocity = np.zeros_like(heading)
    angular_velocity[1:] = _unwrap_angle(np.diff(heading)) / delta_time[1:]
    curvature = np.zeros_like(speed)
    stable_speed = speed > EPSILON
    curvature[stable_speed] = (
        velocity_x[stable_speed] * acceleration_y[stable_speed]
        - velocity_y[stable_speed] * acceleration_x[stable_speed]
    ) / np.maximum(speed[stable_speed] ** 3, EPSILON)

    target_delta_x = target_x - x
    target_delta_y = target_y - y
    target_distance = np.hypot(target_delta_x, target_delta_y)
    target_closure_rate = np.zeros_like(target_distance)
    target_closure_rate[1:] = (
        target_distance[:-1] - target_distance[1:]
    ) / delta_time[1:]
    target_heading = np.arctan2(target_delta_y, target_delta_x)
    target_heading_error = _unwrap_angle(target_heading - heading)

    ideal_x = target_x - x[0]
    ideal_y = target_y - y[0]
    ideal_length = float(np.hypot(ideal_x, ideal_y))
    if ideal_length > EPSILON:
        cross_track_error = (
            ideal_x * (y - y[0]) - ideal_y * (x - x[0])
        ) / ideal_length
    else:
        cross_track_error = np.zeros_like(x)

    features = np.column_stack(
        (
            x,
            y,
            elapsed,
            delta_time,
            button,
            delta_x,
            delta_y,
            path_progress,
            velocity_x,
            velocity_y,
            speed,
            acceleration_x,
            acceleration_y,
            acceleration,
            jerk_x,
            jerk_y,
            jerk,
            heading,
            angular_velocity,
            curvature,
            running_path_efficiency,
            target_delta_x,
            target_delta_y,
            target_distance,
            target_closure_rate,
            target_heading_error,
            cross_track_error,
        )
    ).astype(np.float32)
    if features.shape[1] != len(FULL_FEATURE_NAMES):
        raise AssertionError("Feature-channel definition is inconsistent.")
    if not np.isfinite(features).all():
        raise ValueError("Feature engineering produced NaN or infinity.")
    return features, float(elapsed[-1])


def load_dataset(path: str) -> CursorDataset:
    """Load raw paper-aligned events or named precomputed feature sequences."""
    with open(path, "r", encoding="utf-8") as file:
        document = json.load(file)

    if isinstance(document, Mapping):
        samples = document.get("samples")
        defaults = document.get("defaults", {})
        metadata = dict(document.get("metadata", {}))
        declared_features = document.get("feature_names")
    else:
        samples = document
        defaults = {}
        metadata = {}
        declared_features = None
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must contain a non-empty 'samples' array.")
    if not isinstance(defaults, Mapping):
        raise ValueError("Dataset defaults must be a JSON object.")

    sequences: list[np.ndarray] = []
    user_ids: list[str] = []
    session_ids: list[str] = []
    attempt_ids: list[str] = []
    movement_indexes: list[int] = []
    durations: list[float] = []
    source_formats: set[str] = set()
    feature_names: tuple[str, ...] | None = None

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Sample {index} must be a JSON object.")
        user_id = str(sample.get("user_id", "")).strip()
        if not user_id:
            raise ValueError(f"Sample {index} is missing user_id.")

        if "events" in sample or "trajectory" in sample:
            sequence, duration = engineer_raw_features(sample, defaults)
            sample_feature_names = FULL_FEATURE_NAMES
            source_formats.add("raw-events")
        elif "sequence" in sample:
            sequence = np.asarray(sample["sequence"], dtype=np.float32)
            if sequence.ndim != 2 or sequence.shape[0] < 2:
                raise ValueError(
                    f"Sample {index} sequence must have shape "
                    "(time_steps, features) with at least two time steps."
                )
            if not np.isfinite(sequence).all():
                raise ValueError(f"Sample {index} contains NaN or infinity.")
            names = sample.get("feature_names", declared_features)
            if not isinstance(names, list) or len(names) != sequence.shape[1]:
                raise ValueError(
                    "Precomputed sequences require feature_names matching "
                    "their sequence width."
                )
            sample_feature_names = tuple(str(name) for name in names)
            duration = float(sample.get("duration", 0.0))
            source_formats.add("precomputed-features")
        else:
            raise ValueError(
                f"Sample {index} requires raw events/trajectory or sequence."
            )

        if feature_names is None:
            feature_names = tuple(sample_feature_names)
        elif tuple(sample_feature_names) != feature_names:
            raise ValueError("Every sample must use the same feature channels.")

        attempt_id = str(
            sample.get("attempt_id", sample.get("trial_id", ""))
        ).strip()
        movement_index = int(sample.get("movement_index", index))
        sequences.append(sequence)
        user_ids.append(user_id)
        session_ids.append(str(sample.get("session_id", "")).strip())
        attempt_ids.append(attempt_id)
        movement_indexes.append(movement_index)
        durations.append(duration)

    user_array = np.asarray(user_ids)
    if len(np.unique(user_array)) < 2:
        raise ValueError("The dataset must contain at least two users.")
    return CursorDataset(
        sequences=sequences,
        user_ids=user_array,
        session_ids=np.asarray(session_ids),
        attempt_ids=np.asarray(attempt_ids),
        movement_indexes=np.asarray(movement_indexes, dtype=int),
        durations=np.asarray(durations, dtype=np.float32),
        feature_names=feature_names or (),
        source_format="+".join(sorted(source_formats)),
        metadata=metadata,
    )


def feature_indexes(
    available_features: Sequence[str],
    feature_set: str,
) -> np.ndarray:
    required = FEATURE_SETS[feature_set]
    positions = {name: index for index, name in enumerate(available_features)}
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(
            f"Feature set '{feature_set}' is unavailable; missing: "
            + ", ".join(missing)
        )
    return np.asarray([positions[name] for name in required], dtype=int)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Implement Eq. 41 with per-channel linear interpolation."""
    old_length, feature_count = sequence.shape
    old_positions = np.linspace(0.0, 1.0, old_length)
    new_positions = np.linspace(0.0, 1.0, target_length)
    result = np.empty((target_length, feature_count), dtype=np.float32)
    for feature_index in range(feature_count):
        result[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )
    return result


def prepare_sequences(
    sequences: Sequence[np.ndarray],
    target_length: int,
    indexes: np.ndarray | None = None,
) -> np.ndarray:
    if indexes is None:
        selected = sequences
    else:
        selected = [sequence[:, indexes] for sequence in sequences]
    return np.stack(
        [resample_sequence(sequence, target_length) for sequence in selected]
    ).astype(np.float32)


def _group_split(
    user_ids: np.ndarray,
    group_ids: np.ndarray,
    test_size: float,
    validation_size: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for user_id in np.unique(user_ids):
        user_indexes = np.flatnonzero(user_ids == user_id)
        groups = np.unique(group_ids[user_indexes])
        if len(groups) < 3:
            raise ValueError(
                f"User '{user_id}' needs at least three distinct groups."
            )
        groups = groups.copy()
        rng.shuffle(groups)
        test_count = max(1, int(round(len(groups) * test_size)))
        validation_count = max(1, int(round(len(groups) * validation_size)))
        while test_count + validation_count >= len(groups):
            if validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"Could not split groups for '{user_id}'.")
        test_groups = set(groups[:test_count])
        validation_groups = set(
            groups[test_count : test_count + validation_count]
        )
        for index in user_indexes:
            group = group_ids[index]
            if group in test_groups:
                test.append(int(index))
            elif group in validation_groups:
                validation.append(int(index))
            else:
                train.append(int(index))
    return (
        np.asarray(sorted(train), dtype=int),
        np.asarray(sorted(validation), dtype=int),
        np.asarray(sorted(test), dtype=int),
    )


def create_sample_split(
    user_ids: np.ndarray,
    test_size: float,
    validation_size: float,
    seed: int,
):
    indexes = np.arange(len(user_ids))
    train_validation, test = train_test_split(
        indexes,
        test_size=test_size,
        random_state=seed,
        stratify=user_ids,
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed,
        stratify=user_ids[train_validation],
    )
    return (
        np.asarray(sorted(train), dtype=int),
        np.asarray(sorted(validation), dtype=int),
        np.asarray(sorted(test), dtype=int),
    )


def create_split(
    dataset: CursorDataset,
    test_size: float = 0.10,
    validation_size: float = 0.15,
    seed: int = SEED,
    mode: str = "auto",
):
    """Use session/trial isolation primarily and sample splitting secondarily."""
    if test_size <= 0 or validation_size <= 0:
        raise ValueError("Validation and test sizes must be positive.")
    if test_size + validation_size >= 1:
        raise ValueError("Training, validation, and test fractions are invalid.")

    session_ready = np.all(dataset.session_ids != "")
    attempt_ready = np.all(dataset.attempt_ids != "")
    if mode in ("auto", "session") and session_ready:
        try:
            return (
                *_group_split(
                    dataset.user_ids,
                    dataset.session_ids,
                    test_size,
                    validation_size,
                    seed,
                ),
                "session-separated",
            )
        except ValueError:
            if mode == "session":
                raise
    if mode in ("auto", "trial") and attempt_ready:
        try:
            trial_groups = np.char.add(
                np.char.add(dataset.session_ids.astype(str), "::"),
                dataset.attempt_ids.astype(str),
            )
            return (
                *_group_split(
                    dataset.user_ids,
                    trial_groups,
                    test_size,
                    validation_size,
                    seed,
                ),
                "trial-separated",
            )
        except ValueError:
            if mode == "trial":
                raise
    if mode in ("session", "trial"):
        raise ValueError(f"Requested {mode}-separated split is unavailable.")
    return (
        *create_sample_split(
            dataset.user_ids,
            test_size,
            validation_size,
            seed,
        ),
        "stratified-sample-secondary",
    )


def fit_standardizer(training_data: np.ndarray):
    """Implement Eq. 40 using training data only."""
    mean = training_data.mean(axis=(0, 1), keepdims=True)
    standard_deviation = training_data.std(axis=(0, 1), keepdims=True)
    standard_deviation[standard_deviation < 1e-7] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def standardize(data, mean, standard_deviation):
    return ((data - mean) / standard_deviation).astype(np.float32)


def _convolutional_block(inputs):
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.20)(x)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    return MaxPooling1D(pool_size=2)(x)


def build_model(
    sequence_length: int,
    feature_count: int,
    learning_rate: float,
    model_type: str = "cnn-gru",
):
    """Implement Table II plus the paper's CNN-only and GRU-only baselines."""
    inputs = Input(shape=(sequence_length, feature_count), name="trajectory")
    if model_type == "cnn-gru":
        x = _convolutional_block(inputs)
        x = GRU(64, dropout=0.20)(x)
    elif model_type == "cnn-only":
        x = _convolutional_block(inputs)
        x = GlobalAveragePooling1D()(x)
    elif model_type == "gru-only":
        x = GRU(64, dropout=0.20)(inputs)
    else:
        raise ValueError(f"Unknown neural model type: {model_type}")
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.30)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs, name=f"neurocursor_{model_type}")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


def class_weights(labels):
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    if negative_count == 0 or positive_count == 0:
        raise ValueError("Both genuine and impostor samples are required.")
    total = len(labels)
    return {
        0: float(total / (2 * negative_count)),
        1: float(total / (2 * positive_count)),
    }


def calibrate_eer_threshold(labels, scores) -> dict[str, float]:
    """Choose the EER operating threshold from validation data only."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold calibration requires both classes.")
    false_acceptance, true_positive, thresholds = roc_curve(labels, scores)
    false_rejection = 1.0 - true_positive
    index = int(np.argmin(np.abs(false_acceptance - false_rejection)))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite[0]) if finite.size else 0.5
    return {
        "threshold": threshold,
        "far": float(false_acceptance[index]),
        "frr": float(false_rejection[index]),
        "eer": float(
            (false_acceptance[index] + false_rejection[index]) / 2.0
        ),
    }


def calibrate_far_threshold(
    labels,
    scores,
    target_far: float = DEFAULT_TARGET_FAR,
) -> dict[str, float]:
    """Choose the lowest-FRR validation threshold whose FAR is within target."""
    if not 0.0 <= target_far < 1.0:
        raise ValueError("target_far must be in the interval [0, 1).")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold calibration requires both classes.")
    false_acceptance, true_positive, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(false_acceptance <= target_far + EPSILON)
    if eligible.size == 0:
        index = 0
    else:
        best_recall = np.max(true_positive[eligible])
        best = eligible[np.isclose(true_positive[eligible], best_recall)]
        index = int(best[-1])
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
    return {
        "target_far": float(target_far),
        "threshold": threshold,
        "far": float(false_acceptance[index]),
        "frr": float(1.0 - true_positive[index]),
    }


def roc_curve_points(labels, scores) -> dict[str, list[float]]:
    """Return serializable ROC points for paper figures."""
    false_acceptance, true_positive, _ = roc_curve(
        np.asarray(labels, dtype=int),
        np.asarray(scores, dtype=float).reshape(-1),
    )
    return {
        "false_positive_rate": false_acceptance.tolist(),
        "true_positive_rate": true_positive.tolist(),
    }


def biometric_metrics(labels, scores, threshold: float):
    """Implement Eqs. 48-51 using a previously frozen threshold."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    false_acceptance = fp / (fp + tn) if fp + tn else 0.0
    false_rejection = fn / (fn + tp) if fn + tp else 0.0

    roc_false_acceptance, roc_true_positive, roc_thresholds = roc_curve(
        labels, scores
    )
    roc_false_rejection = 1.0 - roc_true_positive
    eer_index = int(
        np.argmin(np.abs(roc_false_acceptance - roc_false_rejection))
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "far": float(false_acceptance),
        "frr": float(false_rejection),
        "eer": float(
            (
                roc_false_acceptance[eer_index]
                + roc_false_rejection[eer_index]
            )
            / 2.0
        ),
        "eer_threshold": float(roc_thresholds[eer_index]),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _cohens_d(genuine: np.ndarray, impostor: np.ndarray) -> float:
    if len(genuine) < 2 or len(impostor) < 2:
        return float("nan")
    pooled_variance = (
        (len(genuine) - 1) * np.var(genuine, ddof=1)
        + (len(impostor) - 1) * np.var(impostor, ddof=1)
    ) / (len(genuine) + len(impostor) - 2)
    if pooled_variance <= EPSILON:
        return float("nan")
    return float((np.mean(genuine) - np.mean(impostor)) / np.sqrt(pooled_variance))


def score_statistics(
    labels,
    scores,
    threshold: float,
    seed: int = SEED,
    permutations: int = 10000,
    bootstrap_samples: int = 1000,
):
    """One-sided permutation test, Cohen's d, and stratified bootstrap CIs."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    observed = float(np.mean(genuine) - np.mean(impostor))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(scores)
        difference = float(
            np.mean(permuted[labels == 1])
            - np.mean(permuted[labels == 0])
        )
        exceedances += difference >= observed
    p_value = (exceedances + 1) / (permutations + 1)

    tracked = ("accuracy", "precision", "recall", "f1_score", "roc_auc", "far", "frr", "eer")
    bootstrap: dict[str, list[float]] = {name: [] for name in tracked}
    for _ in range(bootstrap_samples):
        genuine_sample = rng.choice(genuine, size=len(genuine), replace=True)
        impostor_sample = rng.choice(impostor, size=len(impostor), replace=True)
        sampled_scores = np.concatenate((impostor_sample, genuine_sample))
        sampled_labels = np.concatenate(
            (
                np.zeros(len(impostor_sample), dtype=int),
                np.ones(len(genuine_sample), dtype=int),
            )
        )
        metrics = biometric_metrics(
            sampled_labels, sampled_scores, threshold
        )
        for name in tracked:
            bootstrap[name].append(metrics[name])
    confidence_intervals = {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for name, values in bootstrap.items()
        if values
    }
    return {
        "hypothesis": "mean_genuine_score > mean_impostor_score",
        "mean_genuine_score": float(np.mean(genuine)),
        "mean_impostor_score": float(np.mean(impostor)),
        "mean_difference": observed,
        "one_sided_permutation_p_value": float(p_value),
        "permutations": int(permutations),
        "cohens_d": _cohens_d(genuine, impostor),
        "bootstrap_samples": int(bootstrap_samples),
        "confidence_intervals": confidence_intervals,
    }


def aggregate_attempt_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    session_ids: np.ndarray,
    attempt_ids: np.ndarray,
    movement_indexes: np.ndarray,
    durations: np.ndarray,
    interaction_count: int,
):
    """Implement Eq. 6 using the first K chronological valid movements."""
    if np.any(attempt_ids == ""):
        return None
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, (user_id, session_id, attempt_id) in enumerate(
        zip(user_ids, session_ids, attempt_ids)
    ):
        groups.setdefault(
            (str(user_id), str(session_id), str(attempt_id)), []
        ).append(index)

    attempt_scores: list[float] = []
    attempt_labels: list[int] = []
    attempt_durations: list[float] = []
    for indexes in groups.values():
        ordered = sorted(indexes, key=lambda item: (movement_indexes[item], item))
        if len(ordered) < interaction_count:
            continue
        selected = np.asarray(ordered[:interaction_count], dtype=int)
        if len(np.unique(labels[selected])) != 1:
            raise ValueError("An attempt mixes genuine and impostor labels.")
        attempt_scores.append(float(np.mean(scores[selected])))
        attempt_labels.append(int(labels[selected[0]]))
        attempt_durations.append(float(np.sum(durations[selected])))
    if not attempt_scores or len(np.unique(attempt_labels)) < 2:
        return None
    return (
        np.asarray(attempt_scores, dtype=np.float32),
        np.asarray(attempt_labels, dtype=int),
        np.asarray(attempt_durations, dtype=np.float32),
    )


def interaction_count_metrics(
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    dataset: CursorDataset,
    validation_indexes: np.ndarray,
    test_indexes: np.ndarray,
    seed: int,
    permutations: int,
    bootstrap_samples: int,
    target_far: float,
    include_statistics: bool,
):
    results = {}
    for count in INTERACTION_COUNTS:
        validation_attempts = aggregate_attempt_scores(
            validation_scores,
            validation_labels,
            dataset.user_ids[validation_indexes],
            dataset.session_ids[validation_indexes],
            dataset.attempt_ids[validation_indexes],
            dataset.movement_indexes[validation_indexes],
            dataset.durations[validation_indexes],
            count,
        )
        test_attempts = aggregate_attempt_scores(
            test_scores,
            test_labels,
            dataset.user_ids[test_indexes],
            dataset.session_ids[test_indexes],
            dataset.attempt_ids[test_indexes],
            dataset.movement_indexes[test_indexes],
            dataset.durations[test_indexes],
            count,
        )
        if validation_attempts is None or test_attempts is None:
            results[str(count)] = {
                "status": "unavailable",
                "reason": "Attempts with both classes and at least K movements are required.",
            }
            continue
        validation_attempt_scores, validation_attempt_labels, _ = validation_attempts
        test_attempt_scores, test_attempt_labels, test_durations = test_attempts
        calibration = calibrate_eer_threshold(
            validation_attempt_labels, validation_attempt_scores
        )
        security_calibration = calibrate_far_threshold(
            validation_attempt_labels,
            validation_attempt_scores,
            target_far=target_far,
        )
        metrics = biometric_metrics(
            test_attempt_labels,
            test_attempt_scores,
            calibration["threshold"],
        )
        metrics["security_calibration"] = security_calibration
        metrics["security_metrics"] = biometric_metrics(
            test_attempt_labels,
            test_attempt_scores,
            security_calibration["threshold"],
        )
        metrics["roc_curve"] = roc_curve_points(
            test_attempt_labels,
            test_attempt_scores,
        )
        metrics["average_duration_seconds"] = float(np.mean(test_durations))
        metrics["attempt_count"] = int(len(test_attempt_scores))
        metrics["validation_calibration"] = calibration
        if include_statistics:
            metrics["statistics"] = score_statistics(
                test_attempt_labels,
                test_attempt_scores,
                calibration["threshold"],
                seed=seed + count,
                permutations=permutations,
                bootstrap_samples=bootstrap_samples,
            )
        results[str(count)] = metrics
    return results


def summarize_sequences(
    data: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Create fixed-dimensional features for the paper's classical baselines.

    The first block retains distributional summaries for every temporal
    channel. When the full representation is available, the second block
    implements Eqs. 26-35 and the click/pause summaries described in Sec. IV.
    """
    distributional = np.concatenate(
        (
            np.mean(data, axis=1),
            np.std(data, axis=1),
            np.min(data, axis=1),
            np.max(data, axis=1),
        ),
        axis=1,
    )
    if not feature_names:
        return distributional
    positions = {name: index for index, name in enumerate(feature_names)}
    required = {
        "elapsed_time",
        "delta_time",
        "button_state",
        "delta_x",
        "delta_y",
        "speed",
        "acceleration",
        "jerk",
        "target_distance",
    }
    if not required.issubset(positions):
        return distributional

    summaries = []
    for sequence in data:
        elapsed = sequence[:, positions["elapsed_time"]]
        delta_time = np.maximum(
            sequence[:, positions["delta_time"]], 0.0
        )
        button = sequence[:, positions["button_state"]]
        delta_x = sequence[:, positions["delta_x"]]
        delta_y = sequence[:, positions["delta_y"]]
        speed = sequence[:, positions["speed"]]
        acceleration = sequence[:, positions["acceleration"]]
        jerk = sequence[:, positions["jerk"]]
        duration = float(elapsed[-1] - elapsed[0])
        path_length = float(np.sum(np.hypot(delta_x, delta_y)))
        straight_distance = float(
            sequence[0, positions["target_distance"]]
        )
        path_efficiency = (
            straight_distance / path_length
            if path_length > EPSILON
            else 0.0
        )
        acceleration_energy = float(
            np.sum((acceleration ** 2) * delta_time)
        )
        jerk_energy = float(np.sum((jerk ** 2) * delta_time))
        dimensionless_jerk = (
            (duration ** 5 / max(path_length ** 2, EPSILON))
            * jerk_energy
        )
        pause_duration = float(
            np.sum(delta_time[speed <= np.percentile(speed, 10)])
        )
        click_indexes = np.flatnonzero(button > 0)
        click_timing = (
            float(elapsed[click_indexes[0]]) if click_indexes.size else duration
        )
        summaries.append(
            [
                duration,
                path_length,
                straight_distance,
                path_efficiency,
                float(np.mean(speed)),
                float(np.max(speed)),
                float(np.mean(acceleration)),
                float(np.max(acceleration)),
                pause_duration,
                click_timing,
                acceleration_energy,
                jerk_energy,
                dimensionless_jerk,
            ]
        )
    return np.concatenate(
        (distributional, np.asarray(summaries, dtype=np.float32)),
        axis=1,
    )


def build_classical_model(model_type: str, seed: int):
    if model_type == "logistic-regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if model_type == "svm":
        return SVC(
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    if model_type == "random-forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if model_type == "gradient-boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown classical model type: {model_type}")


def _ensure_both_classes(labels, split_name: str, claimed_user: str):
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"{split_name} data for {claimed_user} requires genuine and "
            "impostor samples."
        )


def _mean_user_metrics(user_results: Mapping[str, Mapping[str, Any]]):
    names = ("accuracy", "precision", "recall", "f1_score", "roc_auc", "far", "frr", "eer")
    return {
        name: float(
            np.mean(
                [
                    result["movement_metrics"][name]
                    for result in user_results.values()
                ]
            )
        )
        for name in names
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "keras": getattr(tf.keras, "__version__", "bundled-with-tensorflow"),
        "tensorflow_devices": [
            {"name": device.name, "type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "seed": SEED,
    }


def run_configuration(
    dataset: CursorDataset,
    split,
    output_directory: Path,
    args,
    model_type: str,
    feature_set: str,
    persist_models: bool,
    include_statistics: bool = True,
):
    train_indexes, validation_indexes, test_indexes, split_type = split
    selected_indexes = feature_indexes(dataset.feature_names, feature_set)
    unstandardized_data = prepare_sequences(
        dataset.sequences,
        args.sequence_length,
        selected_indexes,
    )
    mean, standard_deviation = fit_standardizer(
        unstandardized_data[train_indexes]
    )
    data = standardize(
        unstandardized_data, mean, standard_deviation
    )
    classical_data = None
    classical_mean = None
    classical_standard_deviation = None
    if model_type in CLASSICAL_MODELS:
        classical_data = summarize_sequences(
            unstandardized_data,
            FEATURE_SETS[feature_set],
        )
        classical_mean = classical_data[train_indexes].mean(
            axis=0, keepdims=True
        )
        classical_standard_deviation = classical_data[
            train_indexes
        ].std(axis=0, keepdims=True)
        classical_standard_deviation[
            classical_standard_deviation < 1e-7
        ] = 1.0
        classical_data = (
            classical_data - classical_mean
        ) / classical_standard_deviation
    models_directory = output_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)

    user_results: dict[str, Any] = {}
    manifest_users: dict[str, Any] = {}
    for claimed_user in sorted(np.unique(dataset.user_ids)):
        print(
            f"\n[{model_type}/{feature_set}] Training verifier for "
            f"{claimed_user}"
        )
        y_train = (
            dataset.user_ids[train_indexes] == claimed_user
        ).astype(np.float32)
        y_validation = (
            dataset.user_ids[validation_indexes] == claimed_user
        ).astype(np.float32)
        y_test = (
            dataset.user_ids[test_indexes] == claimed_user
        ).astype(np.float32)
        for name, labels in (
            ("training", y_train),
            ("validation", y_validation),
            ("test", y_test),
        ):
            _ensure_both_classes(labels, name, claimed_user)

        filename = safe_filename(claimed_user)
        training_history = None
        if model_type in NEURAL_MODELS:
            model = build_model(
                args.sequence_length,
                data.shape[2],
                args.learning_rate,
                model_type,
            )
            model_path = models_directory / f"{filename}.keras"
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                ),
                ModelCheckpoint(
                    filepath=str(model_path),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]
            history = model.fit(
                data[train_indexes],
                y_train,
                validation_data=(data[validation_indexes], y_validation),
                epochs=args.epochs,
                batch_size=args.batch_size,
                class_weight=class_weights(y_train),
                callbacks=callbacks,
                verbose=args.verbose,
            )
            training_history = {
                name: [float(value) for value in values]
                for name, values in history.history.items()
            }
            best_model = load_model(model_path)
            validation_scores = best_model.predict(
                data[validation_indexes],
                batch_size=args.batch_size,
                verbose=0,
            ).reshape(-1)
            test_scores = best_model.predict(
                data[test_indexes],
                batch_size=args.batch_size,
                verbose=0,
            ).reshape(-1)
            model_file = f"models/{model_path.name}"
        else:
            model = build_classical_model(model_type, args.seed)
            model.fit(classical_data[train_indexes], y_train.astype(int))
            validation_scores = model.predict_proba(
                classical_data[validation_indexes]
            )[:, 1]
            test_scores = model.predict_proba(
                classical_data[test_indexes]
            )[:, 1]
            model_path = models_directory / f"{filename}.joblib"
            joblib.dump(model, model_path)
            model_file = f"models/{model_path.name}"

        calibration = calibrate_eer_threshold(
            y_validation, validation_scores
        )
        security_calibration = calibrate_far_threshold(
            y_validation,
            validation_scores,
            target_far=args.target_far,
        )
        threshold = calibration["threshold"]
        movement_metrics = biometric_metrics(y_test, test_scores, threshold)
        movement_security_metrics = biometric_metrics(
            y_test,
            test_scores,
            security_calibration["threshold"],
        )
        movement_metrics["roc_curve"] = roc_curve_points(y_test, test_scores)
        k_metrics = interaction_count_metrics(
            validation_scores,
            test_scores,
            y_validation.astype(int),
            y_test.astype(int),
            dataset,
            validation_indexes,
            test_indexes,
            seed=args.seed,
            permutations=args.permutations,
            bootstrap_samples=args.bootstrap_samples,
            target_far=args.target_far,
            include_statistics=include_statistics,
        )
        user_results[str(claimed_user)] = {
            "validation_calibration": calibration,
            "security_calibration": security_calibration,
            "movement_metrics": movement_metrics,
            "movement_security_metrics": movement_security_metrics,
            "interaction_count_metrics": k_metrics,
        }
        if include_statistics:
            user_results[str(claimed_user)]["statistics"] = score_statistics(
                y_test,
                test_scores,
                threshold,
                seed=args.seed,
                permutations=args.permutations,
                bootstrap_samples=args.bootstrap_samples,
            )
        if training_history is not None:
            user_results[str(claimed_user)][
                "training_history"
            ] = training_history
        interaction_thresholds = {
            count: value["validation_calibration"]["threshold"]
            for count, value in k_metrics.items()
            if value.get("status") != "unavailable"
        }
        manifest_users[str(claimed_user)] = {
            "model_file": model_file,
            "thresholds": {
                "movement": threshold,
                **interaction_thresholds,
            },
            "security_thresholds": {
                "movement": security_calibration["threshold"],
                **{
                    count: value["security_calibration"]["threshold"]
                    for count, value in k_metrics.items()
                    if value.get("status") != "unavailable"
                },
            },
            "target_far": float(args.target_far),
            "default_interaction_count": 5,
            "default_operating_point": "security",
        }
        print(
            f"AUC={movement_metrics['roc_auc']:.4f} | "
            f"EER={movement_metrics['eer']:.4f} | "
            f"validation threshold={threshold:.4f}"
        )
        if model_type in NEURAL_MODELS:
            tf.keras.backend.clear_session()

    preprocessing = {
        "sequence_length": int(args.sequence_length),
        "feature_set": feature_set,
        "feature_count": int(len(selected_indexes)),
        "feature_names": list(FEATURE_SETS[feature_set]),
        "source_feature_names": list(dataset.feature_names),
        "source_format": dataset.source_format,
        "split_type": split_type,
        "target_far": float(args.target_far),
        "full_statistical_analysis": bool(include_statistics),
        "split_fractions": {
            "train": 1.0 - args.validation_size - args.test_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "mean": mean.reshape(-1).tolist(),
        "standard_deviation": standard_deviation.reshape(-1).tolist(),
    }
    if classical_mean is not None and classical_standard_deviation is not None:
        preprocessing["classical_summary_mean"] = (
            classical_mean.reshape(-1).tolist()
        )
        preprocessing["classical_summary_standard_deviation"] = (
            classical_standard_deviation.reshape(-1).tolist()
        )
    configuration = {
        "model_type": model_type,
        "feature_set": feature_set,
        "split_type": split_type,
        "users": user_results,
        "mean_user_metrics": _mean_user_metrics(user_results),
        "runtime": runtime_metadata(),
        "implementation_notes": {
            "neural_baselines": (
                "The paper names CNN-only and GRU-only baselines but does "
                "not specify their complete layer stacks. They reuse the "
                "corresponding proposed-model component and Dense head."
            ),
            "classical_features": (
                "Classical models use per-channel distribution summaries "
                "plus Eqs. 26-35 when the full channels are available."
            ),
            "feature_channels": (
                "The full set is the union of Table I and the prose list in "
                "Sec. V, retaining both speed and running path efficiency."
            ),
        },
    }
    if persist_models:
        save_json(output_directory / "preprocessing.json", preprocessing)
        save_json(
            output_directory / "manifest.json",
            {
                "model_type": model_type,
                "feature_set": feature_set,
                "users": manifest_users,
            },
        )
        save_json(output_directory / "metrics.json", configuration)
    return configuration


def train(args):
    tf.keras.utils.set_random_seed(args.seed)
    dataset = load_dataset(args.data)
    split = create_split(
        dataset,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        mode=args.split_mode,
    )
    output_directory = Path(args.output_dir)
    result = run_configuration(
        dataset,
        split,
        output_directory,
        args,
        model_type=args.model_type,
        feature_set=args.feature_set,
        persist_models=True,
    )
    print(f"\nSaved models and reproducibility metadata to {output_directory}")
    print(f"Evaluation split: {result['split_type']}")


def experiment(args):
    """Run Table III baselines and feature ablations on one fixed split."""
    tf.keras.utils.set_random_seed(args.seed)
    dataset = load_dataset(args.data)
    split = create_split(
        dataset,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        mode=args.split_mode,
    )
    output_directory = Path(args.output_dir)
    configurations = [
        ("cnn-gru", "full"),
        ("cnn-only", "full"),
        ("gru-only", "full"),
        *[(name, "full") for name in CLASSICAL_MODELS],
        ("cnn-gru", "raw"),
        ("cnn-gru", "raw-kinematic"),
    ]
    results = {}
    for model_type, feature_set in configurations:
        name = f"{model_type}__{feature_set}"
        results[name] = run_configuration(
            dataset,
            split,
            output_directory / name,
            args,
            model_type=model_type,
            feature_set=feature_set,
            persist_models=True,
            include_statistics=(name == "cnn-gru__full"),
        )
    split_comparison = {
        "primary": {
            "split_type": split[3],
            "mean_user_metrics": results["cnn-gru__full"][
                "mean_user_metrics"
            ],
        }
    }
    if split[3] != "stratified-sample-secondary":
        sample_split = create_split(
            dataset,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
            mode="sample",
        )
        sample_result = run_configuration(
            dataset,
            sample_split,
            output_directory / "cnn-gru__full__sample-split",
            args,
            model_type="cnn-gru",
            feature_set="full",
            persist_models=True,
            include_statistics=False,
        )
        results["cnn-gru__full__sample-split"] = sample_result
        split_comparison["secondary"] = {
            "split_type": sample_split[3],
            "mean_user_metrics": sample_result["mean_user_metrics"],
        }
    save_json(
        output_directory / "experiment_summary.json",
        {
            "split_type": split[3],
            "dataset": {
                "source_format": dataset.source_format,
                "users": int(len(np.unique(dataset.user_ids))),
                "movements": int(len(dataset.sequences)),
                "sessions": int(
                    len(
                        set(
                            zip(
                                dataset.user_ids.tolist(),
                                dataset.session_ids.tolist(),
                            )
                        )
                    )
                ),
                "attempts": int(
                    len(
                        set(
                            zip(
                                dataset.user_ids.tolist(),
                                dataset.session_ids.tolist(),
                                dataset.attempt_ids.tolist(),
                            )
                        )
                    )
                ),
                "metadata": dataset.metadata,
            },
            "target_far": float(args.target_far),
            "configurations": results,
            "paper_tables": {
                "table_iv_models": [
                    "cnn-gru__full",
                    "cnn-only__full",
                    "gru-only__full",
                ],
                "table_v_interaction_counts": list(INTERACTION_COUNTS),
                "table_vi_feature_ablations": [
                    "cnn-gru__raw",
                    "cnn-gru__raw-kinematic",
                    "cnn-gru__full",
                ],
                "table_iii_classical_baselines": list(CLASSICAL_MODELS),
                "table_vii_split_comparison": split_comparison,
            },
            "runtime": runtime_metadata(),
        },
    )
    print(f"\nSaved paper experiment suite to {output_directory}")


def _load_prediction_movements(path: str, preprocessing: Mapping[str, Any]):
    with open(path, "r", encoding="utf-8") as file:
        document = json.load(file)
    raw_movements = document.get("movements") if isinstance(document, Mapping) else None
    if raw_movements is None and isinstance(document, Mapping):
        raw_movements = [document]
    if not isinstance(raw_movements, list) or not raw_movements:
        raise ValueError("Prediction input requires one or more movements.")

    sequences = []
    available_names = tuple(preprocessing["source_feature_names"])
    for movement in raw_movements:
        if not isinstance(movement, Mapping):
            raise ValueError("Each prediction movement must be a JSON object.")
        if "events" in movement or "trajectory" in movement:
            sequence, _ = engineer_raw_features(
                movement,
                document.get("defaults", {}) if isinstance(document, Mapping) else {},
            )
            names = FULL_FEATURE_NAMES
        elif "sequence" in movement:
            sequence = np.asarray(movement["sequence"], dtype=np.float32)
            names = tuple(
                movement.get("feature_names", document.get("feature_names", []))
            )
            if sequence.ndim != 2 or len(names) != sequence.shape[1]:
                raise ValueError(
                    "Precomputed prediction sequences need matching feature_names."
                )
        else:
            raise ValueError("Movement requires events/trajectory or sequence.")
        if tuple(names) != available_names:
            raise ValueError(
                "Prediction channels differ from the trained source channels."
            )
        sequences.append(sequence)
    indexes = feature_indexes(
        available_names,
        str(preprocessing["feature_set"]),
    )
    return prepare_sequences(
        sequences,
        int(preprocessing["sequence_length"]),
        indexes,
    )


def predict(args):
    model_directory = Path(args.model_dir)
    with (model_directory / "manifest.json").open(
        "r", encoding="utf-8"
    ) as file:
        manifest = json.load(file)
    with (model_directory / "preprocessing.json").open(
        "r", encoding="utf-8"
    ) as file:
        preprocessing = json.load(file)

    user_config = manifest["users"].get(args.claimed_user)
    if user_config is None:
        raise ValueError(f"Unknown claimed user: {args.claimed_user}")
    unstandardized_data = _load_prediction_movements(
        args.input, preprocessing
    )
    if len(unstandardized_data) < args.required_movements:
        raise ValueError(
            f"Paper protocol requires {args.required_movements} movements; "
            f"received {len(unstandardized_data)}."
        )
    unstandardized_data = unstandardized_data[: args.required_movements]
    mean = np.asarray(preprocessing["mean"], dtype=np.float32).reshape(1, 1, -1)
    standard_deviation = np.asarray(
        preprocessing["standard_deviation"], dtype=np.float32
    ).reshape(1, 1, -1)
    model_type = manifest["model_type"]
    model_path = model_directory / user_config["model_file"]
    if model_type in NEURAL_MODELS:
        data = standardize(
            unstandardized_data, mean, standard_deviation
        )
        model = load_model(model_path)
        movement_scores = model.predict(data, verbose=0).reshape(-1)
    else:
        model = joblib.load(model_path)
        summaries = summarize_sequences(
            unstandardized_data, preprocessing["feature_names"]
        )
        summary_mean = np.asarray(
            preprocessing["classical_summary_mean"], dtype=np.float32
        ).reshape(1, -1)
        summary_standard_deviation = np.asarray(
            preprocessing["classical_summary_standard_deviation"],
            dtype=np.float32,
        ).reshape(1, -1)
        summaries = (
            summaries - summary_mean
        ) / summary_standard_deviation
        movement_scores = model.predict_proba(summaries)[:, 1]

    attempt_score = float(np.mean(movement_scores))
    thresholds = (
        user_config.get("security_thresholds", user_config["thresholds"])
        if args.operating_point == "security"
        else user_config["thresholds"]
    )
    threshold_key = str(args.required_movements)
    if threshold_key not in thresholds:
        raise ValueError(
            f"No validation-calibrated threshold is available for "
            f"K={args.required_movements}."
        )
    threshold = float(thresholds[threshold_key])
    accepted = attempt_score >= threshold
    print(
        json.dumps(
            {
                "claimed_user": args.claimed_user,
                "movement_count": int(len(movement_scores)),
                "movement_scores": movement_scores.tolist(),
                "attempt_score": attempt_score,
                "threshold": threshold,
                "operating_point": args.operating_point,
                "accepted": bool(accepted),
                "decision": "accept" if accepted else "reject",
            },
            indent=2,
        )
    )


def dataset_schema(_args):
    print(
        json.dumps(
            {
                "defaults": {
                    "time_unit": "milliseconds",
                    "screen": {"width": 1920, "height": 1080},
                },
                "samples": [
                    {
                        "user_id": "user_001",
                        "session_id": "session_01",
                        "attempt_id": "attempt_01",
                        "movement_index": 0,
                        "target": {"x": 1200, "y": 600},
                        "events": [
                            {
                                "timestamp": 0,
                                "x": 200,
                                "y": 300,
                                "button_state": 0,
                            },
                            {
                                "timestamp": 16,
                                "x": 220,
                                "y": 310,
                                "button_state": 0,
                            },
                        ],
                    }
                ],
            },
            indent=2,
        )
    )


def _add_training_arguments(parser):
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=("auto", "session", "trial", "sample"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--target-far",
        type=float,
        default=DEFAULT_TARGET_FAR,
        help=(
            "Validation FAR target for the security operating point "
            "(default: zero empirical validation false accepts)."
        ),
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)


def make_parser():
    parser = argparse.ArgumentParser(
        description="NeuroCursor paper-aligned biometric pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train one paper configuration."
    )
    _add_training_arguments(train_parser)
    train_parser.add_argument(
        "--model-type",
        choices=NEURAL_MODELS + CLASSICAL_MODELS,
        default="cnn-gru",
    )
    train_parser.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_SETS),
        default="full",
    )
    train_parser.set_defaults(function=train)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run the paper's model baselines and feature ablations.",
    )
    _add_training_arguments(experiment_parser)
    experiment_parser.set_defaults(function=experiment)

    predict_parser = subparsers.add_parser(
        "predict", help="Verify one claimed identity."
    )
    predict_parser.add_argument("--model-dir", default="artifacts")
    predict_parser.add_argument("--claimed-user", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument(
        "--required-movements",
        type=int,
        choices=INTERACTION_COUNTS,
        default=5,
    )
    predict_parser.add_argument(
        "--operating-point",
        choices=("security", "balanced"),
        default="security",
        help=(
            "Use the validation-calibrated FAR-target threshold by default, "
            "or the paper's EER-balanced threshold."
        ),
    )
    predict_parser.set_defaults(function=predict)

    schema_parser = subparsers.add_parser(
        "schema", help="Print the raw-event dataset schema."
    )
    schema_parser.set_defaults(function=dataset_schema)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.function(args)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

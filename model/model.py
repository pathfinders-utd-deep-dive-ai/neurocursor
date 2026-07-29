from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GRU,
    MaxPooling1D,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def load_dataset(path: str):
    """
    Expected JSON format:

    {
      "feature_names": ["x", "y", "dt", "vx", "vy", "ax", "ay"],
      "samples": [
        {
          "user_id": "user_001",
          "session_id": "session_01",
          "sequence": [[...], [...], ...]
        }
      ]
    }

    Each sample represents one cursor movement.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    samples = data["samples"] if isinstance(data, dict) else data
    feature_names = data.get("feature_names") if isinstance(data, dict) else None

    sequences = []
    user_ids = []
    session_ids = []
    feature_count = None

    for index, sample in enumerate(samples):
        user_id = str(sample["user_id"]).strip()
        sequence = np.asarray(sample["sequence"], dtype=np.float32)

        if sequence.ndim != 2 or sequence.shape[0] < 2:
            raise ValueError(
                f"Sample {index} must have shape (time_steps, features) "
                "and contain at least two time steps."
            )
        if not np.isfinite(sequence).all():
            raise ValueError(f"Sample {index} contains NaN or infinity.")

        if feature_count is None:
            feature_count = sequence.shape[1]
        elif sequence.shape[1] != feature_count:
            raise ValueError("Every sequence must use the same features.")

        sequences.append(sequence)
        user_ids.append(user_id)
        session_ids.append(str(sample.get("session_id", "")).strip())

    user_ids = np.asarray(user_ids)
    session_ids = np.asarray(session_ids)

    if len(np.unique(user_ids)) < 2:
        raise ValueError("The dataset must contain at least two users.")

    if feature_names is not None and len(feature_names) != feature_count:
        raise ValueError(
            "The number of feature names does not match the sequence width."
        )

    return sequences, user_ids, session_ids, feature_names


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Interpolate a variable-length movement to a fixed number of steps."""
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


def prepare_sequences(sequences, target_length: int) -> np.ndarray:
    return np.stack(
        [resample_sequence(sequence, target_length) for sequence in sequences]
    ).astype(np.float32)


def create_sample_split(
    user_ids,
    test_size: float,
    validation_size: float,
    seed: int,
):
    """Create a stratified sample-level split."""
    indexes = np.arange(len(user_ids))

    train_validation, test = train_test_split(
        indexes,
        test_size=test_size,
        random_state=seed,
        stratify=user_ids,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=user_ids[train_validation],
    )

    return train, validation, test


def create_session_split(
    user_ids,
    session_ids,
    test_size: float,
    validation_size: float,
    seed: int,
):
    """
    Split each user's sessions so no session appears in multiple sets.

    Every user needs at least three distinct sessions.
    """
    rng = np.random.default_rng(seed)
    train = []
    validation = []
    test = []

    for user_id in np.unique(user_ids):
        user_indexes = np.flatnonzero(user_ids == user_id)
        sessions = np.unique(session_ids[user_indexes])

        if len(sessions) < 3:
            raise ValueError(
                f"User '{user_id}' needs at least three distinct sessions "
                "for session-separated training, validation, and testing."
            )

        rng.shuffle(sessions)
        session_count = len(sessions)

        test_count = max(1, int(round(session_count * test_size)))
        validation_count = max(
            1,
            int(round(session_count * validation_size)),
        )

        while test_count + validation_count >= session_count:
            if test_count > validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(
                    f"Could not split sessions for user '{user_id}'."
                )

        test_sessions = set(sessions[:test_count])
        validation_sessions = set(
            sessions[test_count : test_count + validation_count]
        )

        for index in user_indexes:
            session_id = session_ids[index]
            if session_id in test_sessions:
                test.append(index)
            elif session_id in validation_sessions:
                validation.append(index)
            else:
                train.append(index)

    return (
        np.asarray(train, dtype=int),
        np.asarray(validation, dtype=int),
        np.asarray(test, dtype=int),
    )


def create_split(
    user_ids,
    session_ids,
    test_size: float,
    validation_size: float,
    seed: int,
):
    """
    Prefer a session-separated split. Fall back to a stratified sample split
    only when session IDs are missing.
    """
    if np.all(session_ids != ""):
        return (
            *create_session_split(
                user_ids,
                session_ids,
                test_size,
                validation_size,
                seed,
            ),
            "session-separated",
        )

    train, validation, test = create_sample_split(
        user_ids,
        test_size,
        validation_size,
        seed,
    )
    return train, validation, test, "stratified-sample"


def fit_standardizer(training_data: np.ndarray):
    """Fit preprocessing on training data only to avoid leakage."""
    mean = training_data.mean(axis=(0, 1), keepdims=True)
    standard_deviation = training_data.std(axis=(0, 1), keepdims=True)
    standard_deviation[standard_deviation < 1e-7] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def standardize(data, mean, standard_deviation):
    return ((data - mean) / standard_deviation).astype(np.float32)


def build_model(sequence_length: int, feature_count: int, learning_rate: float):
    """Create a compact CNN-GRU binary verification model."""
    inputs = Input(
        shape=(sequence_length, feature_count),
        name="trajectory",
    )

    x = Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.20)(x)

    x = Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = GRU(64, dropout=0.20)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.30)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="neurocursor_cnn_gru")
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
    negative_count = np.sum(labels == 0)
    positive_count = np.sum(labels == 1)

    if negative_count == 0 or positive_count == 0:
        raise ValueError("Both genuine and impostor samples are required.")

    total = len(labels)
    return {
        0: float(total / (2 * negative_count)),
        1: float(total / (2 * positive_count)),
    }


def biometric_metrics(labels, scores):
    """Calculate metrics at the threshold closest to the equal-error point."""
    labels = labels.astype(int)
    scores = scores.reshape(-1)

    false_acceptance, true_positive, thresholds = roc_curve(labels, scores)
    false_rejection = 1.0 - true_positive

    index = int(np.argmin(np.abs(false_acceptance - false_rejection)))
    threshold = float(thresholds[index])

    if not np.isfinite(threshold):
        threshold = 0.5

    predictions = (scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "far": float(false_acceptance[index]),
        "frr": float(false_rejection[index]),
        "eer": float(
            (false_acceptance[index] + false_rejection[index]) / 2
        ),
    }


def safe_filename(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def train(args):
    tf.keras.utils.set_random_seed(args.seed)

    sequences, user_ids, session_ids, feature_names = load_dataset(args.data)
    data = prepare_sequences(sequences, args.sequence_length)

    (
        train_indexes,
        validation_indexes,
        test_indexes,
        split_type,
    ) = create_split(
        user_ids,
        session_ids,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    mean, standard_deviation = fit_standardizer(data[train_indexes])
    data = standardize(data, mean, standard_deviation)

    output_directory = Path(args.output_dir)
    models_directory = output_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)

    preprocessing = {
        "sequence_length": args.sequence_length,
        "feature_count": int(data.shape[2]),
        "feature_names": feature_names,
        "split_type": split_type,
        "mean": mean.reshape(-1).tolist(),
        "standard_deviation": standard_deviation.reshape(-1).tolist(),
    }
    save_json(output_directory / "preprocessing.json", preprocessing)

    manifest = {
        "split_type": split_type,
        "users": {},
    }
    all_metrics = {}

    for claimed_user in sorted(np.unique(user_ids)):
        print(f"\nTraining model for {claimed_user}")

        y_train = (user_ids[train_indexes] == claimed_user).astype(np.float32)
        y_validation = (
            user_ids[validation_indexes] == claimed_user
        ).astype(np.float32)
        y_test = (user_ids[test_indexes] == claimed_user).astype(np.float32)

        for split_name, labels in (
            ("training", y_train),
            ("validation", y_validation),
            ("test", y_test),
        ):
            if len(np.unique(labels)) < 2:
                raise ValueError(
                    f"{split_name.title()} data for {claimed_user} does not "
                    "contain both genuine and impostor samples."
                )

        model = build_model(
            sequence_length=args.sequence_length,
            feature_count=data.shape[2],
            learning_rate=args.learning_rate,
        )

        model_path = models_directory / f"{safe_filename(claimed_user)}.keras"

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

        model.fit(
            data[train_indexes],
            y_train,
            validation_data=(data[validation_indexes], y_validation),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights(y_train),
            callbacks=callbacks,
            verbose=1,
        )

        best_model = load_model(model_path)
        test_scores = best_model.predict(
            data[test_indexes],
            batch_size=args.batch_size,
            verbose=0,
        ).reshape(-1)

        metrics = biometric_metrics(y_test, test_scores)
        all_metrics[claimed_user] = metrics

        manifest["users"][claimed_user] = {
            "model_file": f"models/{model_path.name}",
            "threshold": metrics["threshold"],
        }

        print(
            f"AUC={metrics['roc_auc']:.4f} | "
            f"EER={metrics['eer']:.4f} | "
            f"threshold={metrics['threshold']:.4f}"
        )

        tf.keras.backend.clear_session()

    save_json(output_directory / "manifest.json", manifest)
    save_json(output_directory / "metrics.json", all_metrics)

    print(f"\nSaved trained models to {output_directory}")
    print(f"Evaluation split: {split_type}")


def load_attempt(path: str):
    """
    Accepted prediction formats:

    {"sequence": [[...], ...]}

    or

    {"movements": [
        [[...], ...],
        [[...], ...]
    ]}
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "movements" in data:
        raw_movements = data["movements"]
    elif "sequence" in data:
        raw_movements = [data["sequence"]]
    else:
        raise ValueError("Input must contain 'sequence' or 'movements'.")

    movements = []
    for movement in raw_movements:
        movement = np.asarray(movement, dtype=np.float32)
        if movement.ndim != 2 or movement.shape[0] < 2:
            raise ValueError(
                "Each movement must have shape (time_steps, features)."
            )
        movements.append(movement)

    return movements


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

    movements = load_attempt(args.input)

    expected_features = preprocessing["feature_count"]
    if any(movement.shape[1] != expected_features for movement in movements):
        raise ValueError(
            f"Every movement must contain {expected_features} features."
        )

    data = prepare_sequences(
        movements,
        preprocessing["sequence_length"],
    )

    mean = np.asarray(
        preprocessing["mean"],
        dtype=np.float32,
    ).reshape(1, 1, -1)

    standard_deviation = np.asarray(
        preprocessing["standard_deviation"],
        dtype=np.float32,
    ).reshape(1, 1, -1)

    data = standardize(data, mean, standard_deviation)

    model = load_model(model_directory / user_config["model_file"])
    movement_scores = model.predict(data, verbose=0).reshape(-1)

    # This implements the arithmetic-mean score from the paper.
    attempt_score = float(np.mean(movement_scores))
    threshold = float(user_config["threshold"])
    accepted = attempt_score >= threshold

    print(
        json.dumps(
            {
                "claimed_user": args.claimed_user,
                "movement_scores": movement_scores.tolist(),
                "attempt_score": attempt_score,
                "threshold": threshold,
                "accepted": bool(accepted),
                "decision": "accept" if accepted else "reject",
            },
            indent=2,
        )
    )


def make_parser():
    parser = argparse.ArgumentParser(description="NeuroCursor CNN-GRU model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--test-size", type=float, default=0.20)
    train_parser.add_argument("--validation-size", type=float, default=0.20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(function=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-dir", default="artifacts")
    predict_parser.add_argument("--claimed-user", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.set_defaults(function=predict)

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

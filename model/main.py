import json
import joblib
import os
import random
import threading
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cnn_grumodel import MouseCNNGRU

_training_status = {"running": False, "progress": 0, "message": ""}
_training_lock = threading.Lock()

def get_training_status():
    with _training_lock:
        return dict(_training_status)

def _set_progress(pct, msg):
    with _training_lock:
        _training_status["progress"] = pct
        _training_status["message"] = msg

RANDOM_SEED = 42
DATA_DIR = Path("demo_data")
DATA_DIR.mkdir(exist_ok=True)
NORM_FILE = DATA_DIR / "norm_params.pt"
SCALER_FILE = DATA_DIR / "scaler.pkl"
LR_FILE = DATA_DIR / "lr_model.pkl"
THRESHOLD_FILE = DATA_DIR / "threshold.txt"


def _process_data_points(data_points):
    multi_dim = []
    for dp in data_points:
        if len(dp) < 13:
            multi_dim.append([dp["time"], dp["coords"][0], dp["coords"][1], dp["coords"][2]])
        else:
            multi_dim.append([dp["time"], dp["relative_x"], dp["relative_y"], dp["button_state"]])
    raw = np.array(multi_dim, dtype=np.float32)

    vel_x = np.zeros(128, dtype=np.float32); vel_x[1:] = np.diff(raw[:, 1])
    vel_y = np.zeros(128, dtype=np.float32); vel_y[1:] = np.diff(raw[:, 2])
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    acc_x = np.zeros(128, dtype=np.float32); acc_x[1:] = np.diff(vel_x)
    acc_y = np.zeros(128, dtype=np.float32); acc_y[1:] = np.diff(vel_y)
    jerk_x = np.zeros(128, dtype=np.float32); jerk_x[1:] = np.diff(acc_x)
    jerk_y = np.zeros(128, dtype=np.float32); jerk_y[1:] = np.diff(acc_y)
    dist = np.sqrt(raw[:, 1] ** 2 + raw[:, 2] ** 2)
    heading = np.arctan2(vel_y, vel_x)
    btn_diff = np.zeros(128, dtype=np.float32); btn_diff[1:] = np.diff(raw[:, 3])

    return np.column_stack([raw, vel_x, vel_y, speed, acc_x, acc_y, jerk_x, jerk_y, dist, heading, btn_diff]).astype(np.float32)


def _chunk_sessions(sessions):
    chunks = []
    for session in sessions:
        for i in range(0, len(session), 128):
            chunk = session[i:i + 128]
            if len(chunk) < 128:
                continue
            chunks.append(_process_data_points(chunk))
    if not chunks:
        return np.empty((0, 128, 14), dtype=np.float32)
    return np.array(chunks, dtype=np.float32)


def _extract_features(model, X_norm, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)), batch_size=32, shuffle=False)
    feats = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            feats.append(model.extract_features(batch_x).cpu().numpy())
    return np.vstack(feats)


def train(data, max_far=0.01):
    with _training_lock:
        _training_status["running"] = True
        _training_status["progress"] = 0
        _training_status["message"] = "Starting..."

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load master_dataset as background
    _set_progress(1, "Loading master dataset...")
    raw_data = json.load(open("master_dataset.json"))

    # Build list of (user_id, session) pairs and count sessions per user
    all_pairs = []
    counts = {}
    for uid, sessions in raw_data.items():
        counts[uid] = len(sessions)
        for session in sessions:
            all_pairs.append((uid, session))

    # Keep only users with >= 20 sessions (same as helpers.py)
    valid_users = sorted([uid for uid, c in counts.items() if c >= 20])
    valid_pairs = [(uid, s) for uid, s in all_pairs if uid in valid_users]
    num_users = len(valid_users)
    user_labels = {uid: i for i, uid in enumerate(valid_users)}

    # Split cycles 75/25 stratified by user (same as load_data)
    user_ids = [uid for uid, _ in valid_pairs]
    cycles = [s for _, s in valid_pairs]
    strat_labels = [user_labels[uid] for uid in user_ids]
    train_cycles, _ = train_test_split(
        cycles, train_size=0.75, random_state=RANDOM_SEED, stratify=strat_labels
    )

    # Compute mean/std from the 75% portion only
    _set_progress(5, "Chunking background cycles for normalization...")
    X_norm_ref_raw = _chunk_sessions(train_cycles)
    if len(X_norm_ref_raw) == 0:
        raise RuntimeError("No chunks from background training cycles")
    ref_tensor = torch.tensor(X_norm_ref_raw, dtype=torch.float32)
    mean = ref_tensor.mean(dim=(0, 1))
    std = ref_tensor.std(dim=(0, 1), unbiased=False)

    # Process all background cycles into chunks
    _set_progress(10, "Chunking all background cycles...")
    X_bg_raw = _chunk_sessions(cycles)

    # Split demo sessions: 80% for LR training, 20% held out for threshold
    n_train = max(1, int(len(data) * 0.8))
    if len(data) < 6:
        raise ValueError("Need at least 6 sessions")
    train_demo = data[:n_train]
    val_demo = data[n_train:]

    X_train_demo_raw = _chunk_sessions(train_demo)
    X_val_demo_raw = _chunk_sessions(val_demo)
    if len(X_train_demo_raw) == 0:
        raise ValueError("Not enough demo data to form any 128-length chunks")

    # Save normalization params
    torch.save({"mean": mean, "std": std, "num_users": num_users}, NORM_FILE)

    # Normalize
    eps = 1e-8
    mean_np = mean.numpy()
    std_np = std.numpy()
    X_bg_norm = (X_bg_raw - mean_np) / (std_np + eps)
    X_train_demo_norm = (X_train_demo_raw - mean_np) / (std_np + eps)
    X_val_demo_norm = (X_val_demo_raw - mean_np) / (std_np + eps)

    # Load CNN-GRU feature extractor
    _set_progress(20, "Loading CNN-GRU model...")
    model = MouseCNNGRU(num_classes=14, num_users=num_users)
    model.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device), strict=False)
    model.to(device)

    # Extract features
    _set_progress(30, "Extracting background features (CNN-GRU)...")
    X_bg_feats = _extract_features(model, X_bg_norm, device)
    _set_progress(60, "Extracting demo features (CNN-GRU)...")
    X_train_demo_feats = _extract_features(model, X_train_demo_norm, device)
    X_val_demo_feats = _extract_features(model, X_val_demo_norm, device)

    # Fit scaler on background features, transform all
    scaler = StandardScaler()
    X_bg_scaled = scaler.fit_transform(X_bg_feats)
    X_train_demo_scaled = scaler.transform(X_train_demo_feats)
    X_val_demo_scaled = scaler.transform(X_val_demo_feats)
    joblib.dump(scaler, SCALER_FILE)

    _set_progress(75, "Training logistic regression...")
    # Train LR on background + first n_train demo sessions
    X = np.vstack([X_bg_scaled, X_train_demo_scaled])
    y = np.hstack([np.zeros(len(X_bg_scaled)), np.ones(len(X_train_demo_scaled))])

    clf = LogisticRegression(
        l1_ratio=1, C=0.1, solver="liblinear", max_iter=2000,
        random_state=RANDOM_SEED, class_weight="balanced"
    )
    clf.fit(X, y)

    _set_progress(85, "Selecting threshold...")
    # Threshold: session-level on held-out sessions (like classify.py cycle-level)
    chunk_probs = clf.predict_proba(X_val_demo_scaled)[:, 1]
    session_probs = []
    offset = 0
    for session in val_demo:
        n = len(_chunk_sessions([session]))
        if n == 0:
            continue
        avg = float(np.mean(chunk_probs[offset:offset + n]))
        session_probs.append(avg)
        offset += n

    if len(session_probs) > 0:
        # ROC on held-out session-level probs vs background chunk probs
        n_bg = min(len(session_probs) * 5, len(X_bg_scaled))
        bg_idx = np.random.default_rng(RANDOM_SEED).choice(len(X_bg_scaled), n_bg, replace=False)
        bg_probs = clf.predict_proba(X_bg_scaled[bg_idx])[:, 1]
        all_probs = np.concatenate([session_probs, list(bg_probs)])
        all_y = np.concatenate([np.ones(len(session_probs)), np.zeros(n_bg)])
        fpr, tpr, thresholds = roc_curve(all_y, all_probs)
        valid = np.where(fpr <= max_far)[0]
        valid = valid[valid > 0]
        if len(valid) > 0:
            best_idx = valid[np.argmax(tpr[valid])]
        else:
            best_idx = np.argmax(tpr[1:]) + 1
        threshold = float(thresholds[best_idx])
    else:
        threshold = 0.5

    _set_progress(95, "Computing results...")
    # Compute train session-level pass counts
    train_chunk_probs = clf.predict_proba(X_train_demo_scaled)[:, 1]
    train_session_probs = []
    offset_t = 0
    for session in train_demo:
        n = len(_chunk_sessions([session]))
        if n == 0:
            continue
        avg = float(np.mean(train_chunk_probs[offset_t:offset_t + n]))
        train_session_probs.append(avg)
        offset_t += n
    train_ok = sum(1 for p in train_session_probs if p >= threshold)

    val_ok = sum(1 for p in session_probs if p >= threshold)

    stats = {
        "n_train": len(train_session_probs),
        "train_ok": train_ok,
        "n_val": len(session_probs),
        "val_ok": val_ok,
        "threshold": threshold,
        "far": float(fpr[best_idx]) if len(session_probs) > 0 else 0.0,
        "frr": float(1 - tpr[best_idx]) if len(session_probs) > 0 else 0.0,
    }

    _set_progress(98, "Saving model...")
    # Save LR and threshold
    joblib.dump(clf, LR_FILE)
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))

    with _training_lock:
        _training_status["running"] = False
        _training_status["progress"] = 100
        _training_status["message"] = "Done"
        _training_status["results"] = stats

    return clf, threshold, stats


def evaluate(data, model, threshold):
    if not NORM_FILE.exists() or not SCALER_FILE.exists():
        return {"ok": False, "prob": 0.0, "threshold": threshold}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = torch.load(NORM_FILE, map_location=device, weights_only=False)
    scaler = joblib.load(SCALER_FILE)

    X_raw = _chunk_sessions([data])
    if len(X_raw) == 0:
        return {"ok": False, "prob": 0.0, "threshold": threshold}

    eps = 1e-8
    X_norm = (X_raw - norm["mean"].numpy()) / (norm["std"].numpy() + eps)

    cnn_gru = MouseCNNGRU(num_classes=14, num_users=norm["num_users"])
    cnn_gru.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device), strict=False)
    cnn_gru.to(device)

    feats = _extract_features(cnn_gru, X_norm, device)
    feats_scaled = scaler.transform(feats)

    probs = model.predict_proba(feats_scaled)[:, 1]
    avg_prob = float(np.mean(probs))
    return {"ok": avg_prob >= threshold, "prob": avg_prob, "threshold": threshold}

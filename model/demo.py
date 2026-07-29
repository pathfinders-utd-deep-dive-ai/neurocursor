# AI coded with opencode
import eel
import glob
import json
import os
import pickle
import random
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cnn_grumodel import MouseCNNGRU
from main import (
    _chunk_sessions, _extract_features, _process_data_points,
    _set_progress, _training_lock, _training_status,
    evaluate, get_training_status, train,
    RANDOM_SEED, NORM_FILE, SCALER_FILE, LR_FILE, THRESHOLD_FILE,
)


def _playwright_version_key(path):
    """Natural-sort key for ms-playwright version dirs.

    Path layout is .../ms-playwright/chromium-1.10.0/chrome-linux64/chrome
    (or chromium_headless_shell-*). We pull out the version directory and
    fullmatch against the canonical pattern, so '(1, 10, 0)' > '(1, 9, 0)'
    and trailing cruft in the dir name is rejected.
    """
    parts = path.split("/")
    try:
        ms_idx = parts.index("ms-playwright")
        ver_dir = parts[ms_idx + 1]
    except (ValueError, IndexError):
        return ()
    m = re.fullmatch(r"chromium(?:_headless_shell)?-([\d.]+)", ver_dir)
    if not m:
        return ()
    return tuple(int(n) for n in m.group(1).split(".") if n.isdigit())


_APT_IDS = {"debian", "ubuntu", "pop", "linuxmint", "elementary", "kali"}
_DNF_IDS = {"fedora", "rhel", "centos", "rocky", "almalinux"}
_PACMAN_IDS = {"arch", "manjaro", "endeavouros", "artix"}
_ZYPPER_IDS = {"opensuse", "sles", "opensuse-tumbleweed", "opensuse-leap"}


def _distro_install_hint():
    """Return a distro-specific install command, or None if the distro is unknown.

    `ID_LIKE` is space-separated per the systemd spec (e.g. Manjaro is
    `ID=manjaro ID_LIKE="arch debian"`), so we split it into tokens and do
    set-intersection matching against the family sets.
    """
    try:
        with open("/etc/os-release") as f:
            info = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                info[k] = v.strip().strip('"').strip("'").lower()
    except (OSError, IOError):
        return None
    did = info.get("id", "")
    like = set(info.get("id_like", "").split())
    if did in _APT_IDS or like & _APT_IDS:
        return "sudo apt install chromium"
    if did in _DNF_IDS or like & _DNF_IDS:
        return "sudo dnf install chromium"
    if did in _PACMAN_IDS or like & _PACMAN_IDS:
        return "sudo pacman -S chromium"
    if did in _ZYPPER_IDS or like & _ZYPPER_IDS:
        return "sudo zypper install chromium"
    return None


def _find_chrome():
    """Locate a Chrome/Chromium/Edge/Brave executable.

    Eel's built-in detector only tries a handful of binary names on PATH; this
    extends the search to common distros, snap/flatpak installs, /opt, and
    Playwright-bundled chromium. Returns (path, source_label) or (None, None).

    A pre-existing CHROME_PATH env var is honored as-is.
    """
    # 1. Respect a user-provided CHROME_PATH first.
    existing = os.environ.get("CHROME_PATH")
    if existing and os.path.isfile(existing):
        return existing, "CHROME_PATH env"

    # 2. Common binary names on PATH (also catches /snap/bin/* which is on PATH on Ubuntu).
    names = [
        "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
        "chrome", "microsoft-edge", "microsoft-edge-stable",
        "brave", "brave-browser", "vivaldi", "vivaldi-stable",
    ]
    for name in names:
        p = shutil.which(name)
        if p:
            return p, f"PATH:{name}"

    # 3. Linux-only direct-path fallbacks. Skip on Windows since /snap/bin and
    #    /opt paths don't exist there, and X_OK semantics on Windows differ.
    if sys.platform != "win32":
        direct_candidates = [
            ("/snap/bin/chromium", "snap:chromium"),
            ("/snap/bin/google-chrome", "snap:google-chrome"),
            ("/opt/google/chrome/chrome", "opt:google-chrome"),
            ("/usr/bin/google-chrome", "apt:google-chrome"),
            ("/usr/bin/google-chrome-stable", "apt:google-chrome-stable"),
            ("/usr/local/bin/google-chrome", "manual:google-chrome"),
            ("/usr/local/bin/chromium", "manual:chromium"),
        ]
        for path, label in direct_candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path, label

        # 4. Flatpak exports (no single fixed binary — glob the app-id directory).
        flatpak_root = "/var/lib/flatpak/exports/bin"
        if os.path.isdir(flatpak_root):
            for entry in sorted(glob.glob(os.path.join(flatpak_root, "*"))):
                base = os.path.basename(entry).lower()
                if any(k in base for k in ("chrom", "chrome", "edge", "brave")):
                    if os.access(entry, os.X_OK):
                        return entry, f"flatpak:{base}"

        # 5. Playwright bundled chromium. Modern Playwright uses chrome-linux64/
        #    while older releases used chrome-linux/ — check both. Use a natural-
        #    sort key so '1.10.0' beats '1.9.0' (vs lexicographic where it wouldn't).
        for pattern in (
            "~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
            "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
            "~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux64/chrome",
            "~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/chrome",
        ):
            matches = sorted(glob.glob(os.path.expanduser(pattern)), key=_playwright_version_key)
            if matches:
                return matches[-1], "playwright"

    return None, None

DATA_DIR = Path("demo_data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
PROGRESS_FILE = DATA_DIR / "progress.json"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Persistent File Storage ---
users = {}
user_progress = {}
user_models = {}

def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _model_path(username):
    return MODELS_DIR / f"{username}.pkl"

def _load_all():
    global users, user_progress, user_models
    users = _load_json(USERS_FILE)
    user_progress = _load_json(PROGRESS_FILE)
    user_models = {}
    for p in MODELS_DIR.glob("*.pkl"):
        name = p.stem
        try:
            with open(p, "rb") as f:
                user_models[name] = pickle.load(f)
        except Exception:
            traceback.print_exc()

def _save_users():
    _save_json(USERS_FILE, users)

def _save_progress():
    _save_json(PROGRESS_FILE, user_progress)

def _save_model(username, model_data):
    with open(_model_path(username), "wb") as f:
        pickle.dump(model_data, f)
    user_models[username] = model_data

# --- Background Cache ---
BG_CHUNKS_FILE = CACHE_DIR / "bg_chunks.npy"
BG_NORM_FILE = CACHE_DIR / "bg_norm.pt"
BG_FEATS_FILE = CACHE_DIR / "bg_feats.npy"
BG_SCALER_FILE = CACHE_DIR / "bg_scaler.pkl"
BG_USERS_FILE = CACHE_DIR / "bg_num_users.txt"
_bg_cache_lock = threading.Lock()

def _ensure_bg_cache(device):
    """Load cached background data or build it on first run."""
    with _bg_cache_lock:
        bg_chunks_file = BG_CHUNKS_FILE
        bg_norm_file = BG_NORM_FILE
        bg_feats_file = BG_FEATS_FILE
        bg_scaler_file = BG_SCALER_FILE
        bg_users_file = BG_USERS_FILE

        if all(p.exists() for p in [bg_chunks_file, bg_norm_file, bg_feats_file, bg_scaler_file, bg_users_file]):
            bg_chunks = np.load(bg_chunks_file)
            bg_norm = torch.load(bg_norm_file, weights_only=False)
            bg_feats = np.load(bg_feats_file)
            bg_scaler = joblib.load(bg_scaler_file)
            with open(bg_users_file) as f:
                num_users = int(f.read())
            return bg_chunks, bg_norm, bg_feats, bg_scaler, num_users

        print("[cache] Building background cache (first run)...")
        raw_data = json.load(open("master_dataset.json"))

        all_pairs = []
        counts = {}
        for uid, sessions in raw_data.items():
            counts[uid] = len(sessions)
            for session in sessions:
                all_pairs.append((uid, session))
        valid_users = sorted([uid for uid, c in counts.items() if c >= 20])
        valid_pairs = [(uid, s) for uid, s in all_pairs if uid in valid_users]
        num_users = len(valid_users)
        user_labels = {uid: i for i, uid in enumerate(valid_users)}

        user_ids = [uid for uid, _ in valid_pairs]
        cycles = [s for _, s in valid_pairs]
        strat_labels = [user_labels[uid] for uid in user_ids]
        train_cycles, _ = train_test_split(cycles, train_size=0.75, random_state=RANDOM_SEED, stratify=strat_labels)

        X_norm_ref_raw = _chunk_sessions(train_cycles)
        ref_tensor = torch.tensor(X_norm_ref_raw, dtype=torch.float32)
        mean = ref_tensor.mean(dim=(0, 1))
        std = ref_tensor.std(dim=(0, 1), unbiased=False)

        X_bg_raw = _chunk_sessions(cycles)
        np.save(bg_chunks_file, X_bg_raw)

        eps = 1e-8
        mean_np = mean.numpy()
        std_np = std.numpy()
        X_bg_norm = (X_bg_raw - mean_np) / (std_np + eps)

        bg_norm = {"mean": mean, "std": std, "num_users": num_users}
        torch.save(bg_norm, bg_norm_file)

        model = MouseCNNGRU(num_classes=14, num_users=num_users)
        model.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device), strict=False)
        model.to(device)

        bg_feats = _extract_features(model, X_bg_norm, device)
        np.save(bg_feats_file, bg_feats)

        scaler = StandardScaler()
        scaler.fit(bg_feats)
        joblib.dump(scaler, bg_scaler_file)

        with open(bg_users_file, "w") as f:
            f.write(str(num_users))

        print(f"[cache] Cached {len(X_bg_raw)} bg chunks, {len(bg_feats)} feature vectors, {num_users} users")
        return X_bg_raw, bg_norm, bg_feats, scaler, num_users


def _fast_train(all_data, max_far=0.01):
    """Train using cached background data — avoids reprocessing master dataset."""
    with _training_lock:
        _training_status["running"] = True
        _training_status["progress"] = 0
        _training_status["message"] = "Starting..."
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load cached background
    bg_chunks, bg_norm, bg_feats, bg_scaler, num_users = _ensure_bg_cache(device)

    # Split demo sessions 80/20
    n_train = max(1, int(len(all_data) * 0.8))
    train_demo = all_data[:n_train]
    val_demo = all_data[n_train:]

    # Chunk demo
    X_train_demo_raw = _chunk_sessions(train_demo)
    X_val_demo_raw = _chunk_sessions(val_demo)

    # Normalize demo with cached bg stats
    eps = 1e-8
    mean_np = bg_norm["mean"].numpy()
    std_np = bg_norm["std"].numpy()
    X_train_demo_norm = (X_train_demo_raw - mean_np) / (std_np + eps)
    X_val_demo_norm = (X_val_demo_raw - mean_np) / (std_np + eps)

    # Load CNN-GRU and extract demo features
    model = MouseCNNGRU(num_classes=14, num_users=num_users)
    model.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device), strict=False)
    model.to(device)

    X_train_demo_feats = _extract_features(model, X_train_demo_norm, device)
    X_val_demo_feats = _extract_features(model, X_val_demo_norm, device)

    # Scale demo features with cached scaler
    X_train_demo_scaled = bg_scaler.transform(X_train_demo_feats)
    X_val_demo_scaled = bg_scaler.transform(X_val_demo_feats)

    # Train LR on cached bg features + demo train features
    X = np.vstack([bg_feats, X_train_demo_scaled])
    y = np.hstack([np.zeros(len(bg_feats)), np.ones(len(X_train_demo_scaled))])

    clf = LogisticRegression(
        l1_ratio=1, C=0.1, solver="liblinear", max_iter=2000,
        random_state=RANDOM_SEED, class_weight="balanced",
    )
    clf.fit(X, y)

    # Threshold selection (session-level on held-out demo)
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
        n_bg = min(len(session_probs) * 5, len(bg_feats))
        bg_idx = np.random.default_rng(RANDOM_SEED).choice(len(bg_feats), n_bg, replace=False)
        bg_probs = clf.predict_proba(bg_feats[bg_idx])[:, 1]
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
        far_val = float(fpr[best_idx])
        frr_val = float(1 - tpr[best_idx])
    else:
        threshold = 0.5
        far_val = 0.0
        frr_val = 0.0
        fpr = tpr = thresholds = None

    # Compute train session-level ok count
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

    # Precompute FAR→FRR curve for slider display
    if fpr is not None and len(session_probs) > 0:
        far_grid = np.logspace(-4, -1, 200)
        far_frr_curve = []
        for target_far in far_grid:
            valid = np.where(fpr <= target_far)[0]
            valid = valid[valid > 0]
            if len(valid) > 0:
                best = valid[np.argmax(tpr[valid])]
                th = float(thresholds[best])
                fa = float(fpr[best])
                fr = float(1 - tpr[best])
                tr_ok = int(sum(1 for p in train_session_probs if p >= th))
                vl_ok = int(sum(1 for p in session_probs if p >= th))
                far_frr_curve.append([float(target_far), th, fa, fr, tr_ok, vl_ok])
            else:
                far_frr_curve.append([float(target_far), 0.5, 1.0, 1.0, 0, 0])
        with open(CACHE_DIR / "far_frr_curve.json", "w") as f:
            json.dump(far_frr_curve, f)
    else:
        far_frr_curve = []

    stats = {
        "n_train": len(train_session_probs),
        "train_ok": train_ok,
        "n_val": len(session_probs),
        "val_ok": val_ok,
        "threshold": threshold,
        "far": far_val,
        "frr": frr_val,
        "far_frr_curve": far_frr_curve,
    }

    # Save files needed by evaluate()
    torch.save(bg_norm, NORM_FILE)
    joblib.dump(bg_scaler, SCALER_FILE)
    joblib.dump(clf, LR_FILE)
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))

    with _training_lock:
        _training_status["running"] = False
        _training_status["progress"] = 100
        _training_status["message"] = "Done"
        _training_status["results"] = stats

    return clf, threshold, stats


# Load persisted data on startup
_load_all()

# Expose Python functions directly to JavaScript

@eel.expose
def submit_auth(username, password, is_signup):
    """Handles login and signup directly."""
    if is_signup:
        if username in users:
            return False
        users[username] = password
        _save_users()
        user_progress[username] = {"sessions": 0, "mode": "train", "data": [], "test_sessions": 0, "test_data": []}
        _save_progress()
        return True
    else:
        return users.get(username) == password


@eel.expose
def get_user_progress(username):
    """Retrieves mode and session count for user."""
    if username in user_progress:
        p = user_progress[username]
        p["test_sessions"] = len(p.get("test_data", []))
        return p
    return {"mode": "train", "sessions": 0, "test_sessions": 0, "test_data": []}

@eel.expose
def get_training_progress():
    return get_training_status()

@eel.expose
def get_far_frr_curve():
    path = CACHE_DIR / "far_frr_curve.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@eel.expose
def retrain(username, max_far=0.01):
    """Retrains model on all collected train + test data.

    Blocks until done and returns the stats so the JS can update the UI
    without depending on the 1s poll cadence catching the running→done
    transition (which fails when retrain is faster than the poll interval).
    """
    if username not in user_progress:
        return {"ok": False, "reason": "no_user", "message": "No such user."}
    progress = user_progress[username]
    all_data = progress["data"] + progress.get("test_data", [])
    if len(all_data) < 6:
        return {
            "ok": False,
            "reason": "insufficient_data",
            "message": "Not enough sessions to retrain (need at least 6).",
        }

    try:
        t0 = time.perf_counter()
        model, threshold, stats = _fast_train(all_data, max_far=max_far)
        elapsed = time.perf_counter() - t0
        print(f"[benchmark] retrain took {elapsed:.3f}s")
        _save_model(username, {"model": model, "threshold": threshold})
        return {"ok": True, "stats": stats}
    except Exception as e:
        print("Retrain failed:", e)
        traceback.print_exc()
        with _training_lock:
            _training_status["running"] = False
            _training_status["progress"] = 0
            _training_status["message"] = f"Failed: {e}"
        return {"ok": False, "reason": str(e), "message": "Retrain failed: " + str(e)}


@eel.expose
def save_session(username, session_data, save_data=True):
    """Saves session data and triggers train() or evaluate()."""
    if username not in user_progress:
        return {"just_finished_training": False, "mode": None, "evaluate_result": None}

    progress = user_progress[username]
    just_finished = False
    evaluate_result = None

    if progress["mode"] == "train":
        progress["data"].append(session_data)
        progress["sessions"] += 1
        _save_progress()

        # Train model once 30 sessions are collected
        if progress["sessions"] >= 30:
            def _initial_train_thread():
                try:
                    t0 = time.perf_counter()
                    model, threshold, stats = _fast_train(progress["data"])
                    elapsed = time.perf_counter() - t0
                    print(f"[benchmark] initial train took {elapsed:.3f}s")
                    _save_model(username, {"model": model, "threshold": threshold})
                    progress["mode"] = "test"
                    progress["test_data"] = []
                    _save_progress()
                except Exception as e:
                    print("Initial training failed:", e)
                    traceback.print_exc()
                    progress["mode"] = "train"
                    _save_progress()
            thread = threading.Thread(target=_initial_train_thread, daemon=True)
            thread.start()
            progress["mode"] = "training"
            _save_progress()

    elif progress["mode"] == "test":
        if save_data:
            progress.setdefault("test_data", []).append(session_data)
        _save_progress()
        if username in user_models:
            evaluate_result = evaluate(session_data, user_models[username]["model"], user_models[username]["threshold"])

    return {
        "just_finished_training": just_finished,
        "mode": progress["mode"],
        "sessions": progress["sessions"],
        "test_sessions": len(progress.get("test_data", [])),
        "evaluate_result": evaluate_result,
    }


# Start the self-contained app.
# Point Eel to the folder containing your index.html.
_chrome_path, _chrome_source = _find_chrome()
if _chrome_path:
    os.environ["CHROME_PATH"] = _chrome_path
    print(f"[startup] Using browser at {_chrome_path} (via {_chrome_source})")
else:
    print("[startup] WARNING: No Chrome/Chromium/Edge/Brave found on this system.")
    hint = _distro_install_hint()
    if hint:
        print(f"[startup]   {hint}")
    else:
        print("[startup]   Install Chromium/Chrome via your distro's package manager.")
    print("[startup]   Or set CHROME_PATH=/absolute/path/to/chrome before running.")

eel.init('.')
eel.start('index.html', size=(1250, 700), shutdown_delay=30)
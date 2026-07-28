import eel
from main import train, evaluate

# --- Simple In-Memory Storage ---
users = {}
user_progress = {}
user_models = {}

# Dummy placeholders if your train/evaluate are in another file:
# from model_script import train, evaluate
def train(data):
    print(">>> Training model on collected 30 sessions...")
    fake_model = "trained_classifier"
    fake_threshold = 0.5
    return fake_model, fake_threshold

def evaluate(data, model, threshold):
    print(">>> Evaluating test session...")
    pass


# Expose Python functions directly to JavaScript

@eel.expose
def submit_auth(username, password, is_signup):
    """Handles login and signup directly."""
    if is_signup:
        if username in users:
            return False
        users[username] = password
        user_progress[username] = {"sessions": 0, "mode": "train", "data": []}
        return True
    else:
        return users.get(username) == password


@eel.expose
def get_user_progress(username):
    """Retrieves mode and session count for user."""
    if username in user_progress:
        return user_progress[username]
    return {"mode": "train", "sessions": 0}


@eel.expose
def save_session(username, session_data):
    """Saves session data and triggers train() or evaluate()."""
    if username not in user_progress:
        return {"just_finished_training": False}

    progress = user_progress[username]
    just_finished = False

    if progress["mode"] == "train":
        progress["data"].append(session_data)
        progress["sessions"] += 1

        # Train model once 30 sessions are collected
        if progress["sessions"] >= 30:
            model, threshold = train(progress["data"])
            user_models[username] = {"model": model, "threshold": threshold}
            progress["mode"] = "test"
            just_finished = True

    elif progress["mode"] == "test":
        if username in user_models:
            evaluate(session_data, user_models[username]["model"], user_models[username]["threshold"])

    return {"just_finished_training": just_finished}


# Start the self-contained app
# Point Eel to the folder containing your index.html
eel.init('.') 
eel.start('index.html', size=(1250, 700))
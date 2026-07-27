from flask import Flask, request, jsonify, render_template
import json, os

app = Flask(__name__)

def train_model(sessions):
    return {}

def run_verification(session_data, profile):
    return True

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/login/")
def login_page():
    return render_template("login.html")

@app.route("/signup/")
def signup_page():
    return render_template("signup.html")

@app.route("/api/signup/", methods=["POST"])
def api_signup():
    data = request.json
    username = data["username"]
    password = data["password"]

    db = {}
    if os.path.exists("data.json"):
        f = open("data.json", "r")
        db = json.load(f)
        f.close()

    if not username or not password or username in db:
        return "False"

    db[username] = {
        "password": password,
        "mode": "train",
        "sessions": [],
        "profile": None
    }

    f = open("data.json", "w")
    json.dump(db, f, indent=2)
    f.close()
    
    return "True"

@app.route("/api/login/", methods=["POST"])
def api_login():
    data = request.json
    username = data["username"]
    password = data["password"]

    db = {}
    if os.path.exists("data.json"):
        f = open("data.json", "r")
        db = json.load(f)
        f.close()

    if username in db and db[username]["password"] == password:
        return "True"
    return "False"

@app.route("/api/data/get/", methods=["POST"])
def api_data_get():
    data = request.json
    username = data["username"]
    
    db = {}
    if os.path.exists("data.json"):
        f = open("data.json", "r")
        db = json.load(f)
        f.close()

    user = db.get(username)
    if not user:
        return jsonify({"sessions": 0, "mode": "train"})
    return jsonify({"sessions": len(user["sessions"]), "mode": user["mode"]})

@app.route("/api/data/save/", methods=["POST"])
def api_data_save():
    data = request.json
    username = data["username"]
    session_data = data["data"]

    db = {}
    if os.path.exists("data.json"):
        f = open("data.json", "r")
        db = json.load(f)
        f.close()

    user = db[username]
    user["sessions"].append(session_data)

    just_finished = False
    if len(user["sessions"]) >= 30 and user["mode"] == "train":
        user["profile"] = train_model(user["sessions"])
        user["mode"] = "test"
        just_finished = True

    f = open("data.json", "w")
    json.dump(db, f, indent=2)
    f.close()

    return jsonify({
        "sessions": len(user["sessions"]),
        "mode": user["mode"],
        "just_finished_training": just_finished
    })

@app.route("/api/data/verify/", methods=["POST"])
def api_data_verify():
    data = request.json
    username = data["username"]
    session_data = data["data"]

    db = {}
    if os.path.exists("data.json"):
        f = open("data.json", "r")
        db = json.load(f)
        f.close()

    user = db[username]
    is_match = run_verification(session_data, user["profile"])
    return jsonify({"match": is_match})

if __name__ == "__main__":
    app.run(debug=True)
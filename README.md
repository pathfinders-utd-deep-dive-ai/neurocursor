NeuroCursor

Securing Identity Through Every Movement

NeuroCursor is a desktop behavioral-biometric prototype that uses cursor movement to verify an enrolled user. The system presents randomized point-and-click tasks, records mouse behavior, extracts motion features, and returns a probability-based verification result.

Prototype status: NeuroCursor is a research demo. It is not ready for production authentication. Do not use real passwords or sensitive production data.

Contributions

Hamid Abdul Mossa: Backend, model, and frontend development.

Shreehan Aalok Pathak: Frontend and model development.

Chetan Gangireddy: Early model development and research-paper contributions; unavailable for part of the project due to being out of town.

Shaurya Saxena: Majority of the research paper, majority of the presentation, and the model's mathematical features and architecture.

Demo Workflow

Create an account or log in.

Complete 30 enrollment sessions.

Each session records a short randomized cursor challenge.

The system converts the recorded data into 128-sample chunks.

A pretrained CNN-GRU extracts a behavioral feature vector.

A user-specific logistic regression model is trained against background-user data.

A validation threshold is selected under a configured false-acceptance limit.

Future attempts return a probability and a pass/fail result.

The background dataset uses users with at least 20 sessions. The interactive demo waits until the current user has completed 30 sessions before training that user’s verification model.

Architecture

Background Training

The CNN-GRU first learns to classify the users in master_dataset.json.

128-step trajectory
        ↓
1D CNN feature extraction
        ↓
GRU temporal modeling
        ↓
Dense classification layer

Implemented neural architecture:

Input: 128 × 14

Conv1D: 16 filters, kernel size 5

Batch normalization

Max pooling

Dropout: 0.20

Conv1D: 32 filters, kernel size 3

Conv1D: 64 filters, kernel size 3

Max pooling

Conv1D: 128 filters, kernel size 3

GRU: 256 hidden units

Dense: 256 units

Dropout: 0.30

Multiclass user output

The model saves:

best_neurocursor_model.pth
best_neurocursor_val_model.pth

Verification Stage

The demo uses the trained CNN-GRU as a feature extractor.

Recorded cursor session
        ↓
14-channel preprocessing
        ↓
CNN-GRU feature extraction
        ↓
StandardScaler
        ↓
Binary logistic regression
        ↓
Probability threshold
        ↓
Accepted or rejected

The logistic regression model treats:

the enrolled demo user as the positive class

the background users as the negative class

Chunk probabilities are averaged to create a session-level score.

Feature Engineering

Every complete 128-sample chunk contains 14 channels.

Raw channels

Time

Relative x-position

Relative y-position

Button state

Derived channels

Horizontal velocity

Vertical velocity

Speed

Horizontal acceleration

Vertical acceleration

Horizontal jerk

Vertical jerk

Distance from the target-relative origin

Heading

Button-state difference

The code supports the current fields:

{
  "time": 0,
  "relative_x": 0,
  "relative_y": 0,
  "button_state": 0
}

It also supports the older format:

{
  "time": 0,
  "coords": [0, 0, 0]
}

Incomplete chunks containing fewer than 128 samples are skipped.

Repository Structure

neurocursor/
├── model/
│   ├── demo.py              # Desktop demo and user workflow
│   ├── index.html           # Eel interface
│   ├── styles.css           # Additional interface styling
│   ├── main.py              # Preprocessing, training, and evaluation
│   ├── cnn_grumodel.py      # CNN-GRU architecture and training
│   ├── helpers.py           # Dataset loading and feature engineering
│   ├── classify.py          # Validation-stage classifier evaluation
│   ├── test.py              # Held-out test evaluation
│   ├── old_model.py         # Earlier model implementation
│   ├── master_dataset.json  # Background cursor dataset
│   └── best_neurocursor_model.pth
└── web/                     # Original web data-collection application

Requirements

Python 3

Chrome, Chromium, Edge, Brave, or another supported Chromium browser

master_dataset.json

best_neurocursor_model.pth

Python packages:

eel
joblib
numpy
pandas
scipy
scikit-learn
tensorboard
torch

Setup

Clone the repository and switch to the hamid branch:

git clone https://github.com/pathfinders-utd-deep-dive-ai/neurocursor.git
cd neurocursor
git switch hamid

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the dependencies:

python3 -m pip install --upgrade pip
python3 -m pip install eel joblib numpy pandas scipy scikit-learn tensorboard torch

Run the Demo

The Python files use relative paths. Run the program from the model directory:

cd model
python3 demo.py

The first launch may take longer because the program builds and stores a background-feature cache inside:

model/demo_data/cache/

The demo also stores local user progress and trained user models inside:

model/demo_data/

Train the CNN-GRU

Place master_dataset.json inside the model directory, then run:

cd model
python3 cnn_grumodel.py

Training uses:

random seed: 42

batch size: 32

AdamW optimizer

learning rate: 0.001

cross-entropy loss with label smoothing

early stopping

TensorBoard logging

View the training logs with:

tensorboard --logdir runs

Evaluate the Classifier

Validation evaluation:

cd model
python3 classify.py

Held-out test evaluation:

cd model
python3 test.py

Both scripts report:

Accuracy

Precision

Recall

F1 score

False acceptance rate

False rejection rate

Equal error estimate

They report results at both the individual 128-sample chunk level and the full-session or cycle level.

Current Results

The recorded final test results were:

Evaluation level

FAR

FRR

Chunk level

0%

100%

Cycle level

0%

66.7%

Cycle-level result: FAR 0% and FRR 66.7%.

These results show a highly conservative threshold. The tested configuration produced no false accepts, but it rejected many genuine attempts. The current model therefore requires further tuning and broader evaluation.

Browser Detection

demo.py attempts to locate a browser through:

CHROME_PATH

browser executables available on PATH

common Linux installation paths

Flatpak exports

Playwright Chromium installations

A custom browser path can be supplied before launch:

export CHROME_PATH="/absolute/path/to/chrome"
python3 demo.py

Important Limitations

The available dataset is limited.

Mouse behavior can change across devices and sessions.

The present threshold causes a high false rejection rate.

The code processes fixed 128-sample chunks.

Incomplete chunks are discarded.

The demo stores usernames and passwords locally in plain JSON.

The trained models and cache files are local prototype artifacts.

The system should be used as a supplemental signal, not as the only authentication factor.

Future Work

Collect data from more users.

Test more devices and environments.

Collect more sessions per user.

Improve feature extraction.

Test different chunk sizes.

Use failed attempts for later retraining.

Compare one, three, and five movements.

Compare with Random Forest, KNN, and other classifiers.

Reduce false rejections while keeping FAR low.

Improve credential and biometric-data protection.

Team

Shaurya Saxena

Chetan Gangireddy

Shreehan Aalok Pathak

Hamid Abdul Mossa

Developed through the 2026 Summer Deep Dive AI Program at The University of Texas at Dallas.
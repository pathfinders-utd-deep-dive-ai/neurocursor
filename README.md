NeuroCursor

Securing Identity Through Every Movement

NeuroCursor is a desktop behavioral-biometric research prototype that uses cursor movement dynamics to continuously verify an enrolled user's identity. By presenting randomized point-and-click tasks, recording fine-grained mouse kinematics, extracting spatial-temporal feature vectors, and scoring probability against background user baselines, NeuroCursor demonstrates real-time continuous behavioral authentication.

⚠️ Prototype Status & Safety Disclaimer > NeuroCursor is a research demonstration. It is not production-ready authentication software. Do not use real passwords, personal data, or sensitive production credentials.

👥 Project Team & Contributions

Shaurya Saxena: Primary research paper author, lead presentation developer, mathematical feature design, and model architecture.

Hamid Abdul Mossa: Full-stack pipeline development (Backend, ML model integration, and frontend interface).

Shreehan Aalok Pathak: Frontend design, interface integration, and model development.

Chetan Gangireddy: Early model architecture development, research contributions, and feature ideation.

📝 Note on Collaboration & Git Commit History > Much of the development was conducted interactively using VS Code Live Share. As a result, git commit histories across different branches and user profiles may not strictly reflect individual line-by-line contributions or real effort distribution.

🔄 Demo Workflow

[ Register / Log In ]
         │
         ▼
[ 30 Enrollment Sessions ] ──► (Randomized cursor trajectory challenges)
         │
         ▼
[ Chunk Processing ] ───────► (Extract 128-sample x 14-channel matrices)
         │
         ▼
[ CNN-GRU Feature Extractor ]
         │
         ▼
[ Logistic Regression Model ] ──► (Trained against background baseline user data)
         │
         ▼
[ Thresholding & Inference ] ─► (Generates confidence probability + Pass/Fail decision)


Authentication: Create a demo account or log into an existing local account.

Enrollment: Complete 30 cursor enrollment sessions consisting of short, randomized point-and-click movement tasks.

Data Chunking: The system standardizes recorded mouse trajectories into 128-sample sequence windows.

Feature Extraction: A pretrained CNN-GRU deep learning network extracts dense behavioral feature vectors.

Class-Model Training: A user-specific Logistic Regression classifier is dynamically trained using the user's vectors as positive samples and background dataset vectors as negative samples.

Threshold Calibration: A decision boundary threshold is tuned under a target false-acceptance constraint.

Verification Inference: Subsequent authentication attempts evaluate mouse trajectories, yielding an identity probability score and an explicit Pass / Fail output.

Note: Background user baselines are sourced from dataset participants with at least 20 recorded sessions. Interactive model training triggers once the active user completes 30 valid enrollment sessions.

🏗 System Architecture

1. Neural Architecture (CNN-GRU Feature Extractor)

The pretrained neural network leverages 1D Convolutional layers for spatially-local kinematics extraction, followed by GRU recurrent layers for sequential trajectory dynamics.

128-step x 14-channel trajectory ──► 1D CNN Layers ──► GRU Temporal Modeling ──► Dense Layer ──► User Embedding / Output


Detailed Network Layers

Layer Type

Configuration / Output Dimensions

Hyperparameters / Details

Input

128 × 14

128 spatial-temporal steps across 14 channels

Conv1D

16 Filters

Kernel Size: 5

Batch Normalization

-

Accelerates convergence

MaxPool1D

-

Downsampling

Dropout

Rate: 0.20

Regularization

Conv1D

32 Filters

Kernel Size: 3

Conv1D

64 Filters

Kernel Size: 3

MaxPool1D

-

Downsampling

Conv1D

128 Filters

Kernel Size: 3

GRU

256 Hidden Units

Recurrent temporal feature extraction

Dense

256 Output Units

Fully connected embedding layer

Dropout

Rate: 0.30

Regularization

Dense

Multiclass Classification

Background user identification head

Saved Checkpoints:

best_neurocursor_model.pth

best_neurocursor_val_model.pth

2. Verification Classifier Stage

Recorded Cursor Session
        │
        ▼
14-Channel Feature Preprocessing
        │
        ▼
Pretrained CNN-GRU Feature Extractor
        │
        ▼
StandardScaler Normalization
        │
        ▼
Binary Logistic Regression (User vs. Background)
        │
        ▼
Probability Averaging & Thresholding
        │
        ▼
[ ACCEPTED ] or [ REJECTED ]


📐 Feature Engineering

Each 128-sample chunk processes 14 channels derived from time, spatial position, and mouse state:

Feature Specification Matrix

Category

Channel Name

Description / Formula

Raw Signals

time

Timestamp delta / elapsed time



relative_x

$x$ coordinate relative to target origin



relative_y

$y$ coordinate relative to target origin



button_state

Mouse button state (0 = released, 1 = pressed)

Velocity

Horizontal Velocity

$v_x = \frac{\Delta x}{\Delta t}$



Vertical Velocity

$v_y = \frac{\Delta y}{\Delta t}$



Speed

$s = \sqrt{v_x^2 + v_y^2}$

Acceleration

Horizontal Acceleration

$a_x = \frac{\Delta v_x}{\Delta t}$



Vertical Acceleration

$a_y = \frac{\Delta v_y}{\Delta t}$

Jerk

Horizontal Jerk

$j_x = \frac{\Delta a_x}{\Delta t}$



Vertical Jerk

$j_y = \frac{\Delta a_y}{\Delta t}$

Spatial/Kinematic

Origin Distance

$d = \sqrt{x_{\text{rel}}^2 + y_{\text{rel}}^2}$



Heading Angle

$\theta = \arctan2(v_y, v_x)$



Button State Delta

$\Delta \text{button\_state}$

Accepted Input JSON Schemas

Primary Format:

{
  "time": 0,
  "relative_x": 0,
  "relative_y": 0,
  "button_state": 0
}


Legacy Format Support:

{
  "time": 0,
  "coords": [0, 0, 0]
}


Note: Any trajectories resulting in incomplete sequence windows containing fewer than 128 samples are safely skipped.

⚙️ Requirements & Installation

Requirements

Python: 3.9+

Supported Browsers: Google Chrome, Chromium, Microsoft Edge, Brave, or Playwright Chromium instances.

Setup Instructions

Clone Repository & Switch Branch:

git clone [https://github.com/pathfinders-utd-deep-dive-ai/neurocursor.git](https://github.com/pathfinders-utd-deep-dive-ai/neurocursor.git)
cd neurocursor
git switch hamid


Set Up Python Virtual Environment:

python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate


Install Dependencies:

python3 -m pip install --upgrade pip
python3 -m pip install eel joblib numpy pandas scipy scikit-learn tensorboard torch


🚀 Running the System

1. Launch Interactive Desktop Demo

Because paths are configured relative to the model/ directory, launch the script directly from model/:

cd model
python3 demo.py


First Run Behavior: The application will generate a cached feature repository in model/demo_data/cache/. User state and trained linear models are stored under model/demo_data/.

Custom Browser Path Setup (Optional): If automatic browser detection fails, manually define your Chromium binary path:

export CHROME_PATH="/path/to/chrome"
python3 demo.py


2. Train the Background Deep Network (CNN-GRU)

To retrain the base feature extractor on master_dataset.json:

cd model
python3 cnn_grumodel.py


Training Configuration

Optimizer: AdamW (learning rate: 0.001)

Batch Size: 32

Loss Function: Cross-Entropy Loss with Label Smoothing

Regularization: Dropout, Early Stopping, Seed = 42

Logging: TensorBoard (tensorboard --logdir runs)

3. Run Benchmark Evaluations

Evaluate system performance on validation and held-out test sets:

cd model
# Evaluate validation metrics
python3 classify.py

# Evaluate out-of-sample test performance
python3 test.py


📊 Experimental Results

The held-out evaluation yields the following error metrics:

Evaluation Granularity

False Acceptance Rate (FAR)

False Rejection Rate (FRR)

Chunk Level (128-sample windows)

0.0%

100.0%

Cycle Level (Full Session)

0.0%

66.7%

Metric Analysis

The evaluation metrics demonstrate an ultra-conservative decision threshold. While the system successfully eliminates False Acceptances ($0\%$ FAR), it exhibits a high False Rejection Rate ($66.7\%$ at session level). Fine-tuning the logistic decision probability hyper-parameters is necessary to achieve optimal Equal Error Rate (EER) trade-offs.

⚠️ Limitations & Security Considerations

Dataset Scale: Prototype baselines rely on a constrained pool of background users.

Hardware Variability: Kinematic features (acceleration, velocity, jerk) vary across physical hardware, mouse DPI settings, and display resolutions.

Conservative Decision Thresholds: High FRR requires additional threshold tuning.

Windowing Constraint: Fixed 128-sample chunks result in partial chunk discarding for short trajectories.

Credentials Storage: Credentials are currently stored in unencrypted local JSON files for prototype convenience.

Authentication Scope: Behavioral biometrics should serve as a continuous multi-factor authentication (MFA) signal rather than a standalone login credential.

🔮 Future Work

[ ] Expand dataset diversity with additional users, hardware setups, and input device types.

[ ] Implement adaptive chunking or variable-length sequence models (e.g., Transformers/LSTMs).

[ ] Benchmark against classical classifiers (Random Forest, SVM, k-NN) and anomaly detection baselines (Isolation Forests, One-Class SVMs).

[ ] Incorporate rejected authentications into continuous incremental user retraining loops.

[ ] Enhance storage security via standard password hashing algorithms (Argon2 / bcrypt) and encrypted biometrics storage.

🏫 Institutional Context

NeuroCursor was developed through the 2026 Summer Deep Dive AI Program at The University of Texas at Dallas (UTD).

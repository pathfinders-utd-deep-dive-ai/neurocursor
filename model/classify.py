print("[Classify - Imports (1/3)] Importing sklearn...")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from helpers import load_data
from cnn_grumodel import MouseCNNGRU
print("[Classify - Imports (2/3)] Importing torch...")
import torch
from torch.utils.data import TensorDataset, DataLoader
print("[Classify - Imports (3/3)] Importing numpy...")
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
# 3. Ensure deterministic cuDNN algorithms (GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_64_features(model, dataloader, device):
    model.eval()
    feature_list = []
    label_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device).permute(0, 2, 1) #Aids in reshaping the input to match the expected input shape of the model
            features = model.extract_features(batch_x)
            feature_list.append(features.cpu().numpy())
            label_list.append(batch_y.numpy())
            
    X_features = np.vstack(feature_list) 
    y_labels = np.concatenate(label_list)
    return X_features, y_labels

X_train, y_train, X_val, y_val, yes_user_id, user_labels = load_data()
yes_user_label = user_labels[yes_user_id]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_background_users = len(user_labels) - 1

model = MouseCNNGRU(num_classes=14, num_users=num_background_users)
model.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device))
model.to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

X_train_features, y_train_orig = extract_64_features(model, train_loader, device)
X_val_features, y_val_orig = extract_64_features(model, val_loader, device)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)

y_train = (y_train_orig == yes_user_label).astype(int)
y_val = (y_val_orig == yes_user_label).astype(int)

clf = LogisticRegression(l1_ratio=1, C=0.1, solver='liblinear', max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced')
clf.fit(X_train_features, y_train)

y_val_prob = clf.predict_proba(X_val_features)[:, 1]
print(y_val_prob)
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
frr = 1 - tpr

# Define your maximum allowed FAR target (e.g., 1% or 0.01)
MAX_FAR = 0.1
MAX_FRR = 1

# Find all indices where FAR is within our budget
valid_indices = np.where((fpr <= MAX_FAR) & (fpr > 0))[0]

if len(valid_indices) > 0:
    # Among valid thresholds, pick the one with the lowest FAR
    best_idx = valid_indices[np.argmin(fpr[valid_indices])]
    threshold = thresholds[best_idx]
else:
    # Fallback if no threshold meets the budget: pick the absolute minimum FAR
    print("fallbacked")
    best_idx = np.argmin(fpr)
    threshold = thresholds[best_idx]

y_val_pred = (y_val_prob >= threshold).astype(int)
print(f"Best threshold: {threshold:.4f}")
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel() #.ravel() flattens the confusion matrix into a 1D array, allowing us to unpack the values directly into tn, fp, fn, and tp.
far = fp / (fp + tn)
frr = fn / (fn + tp)
eer = (far + frr) / 2
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall: {rec:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print(f"Validation False Acceptance Rate (FAR): {far:.4f}")
print(f"Validation False Rejection Rate (FRR): {frr:.4f}")
print(f"Validation Equal Error Rate (EER): {eer:.4f}")
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

X_train, y_train, X_val, y_val, yes_user_id, user_labels, train_cycleids, val_cycleids, X_test, y_test, test_cycleids = load_data()
yes_user_label = user_labels[yes_user_id]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MouseCNNGRU(num_classes=14, num_users=len(user_labels))
model.load_state_dict(torch.load("best_neurocursor_model.pth", map_location=device))
model.to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

X_train_features, y_train_orig = extract_64_features(model, train_loader, device)
X_val_features, y_val_orig = extract_64_features(model, val_loader, device)
X_test_features, y_test_orig = extract_64_features(model, test_loader, device)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)
X_test_features = scaler.transform(X_test_features)

y_train = (y_train_orig == yes_user_label).astype(int)
y_val = (y_val_orig == yes_user_label).astype(int)
y_test = (y_test_orig == yes_user_label).astype(int)

clf = LogisticRegression(l1_ratio=1, C=0.1, solver='liblinear', max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced')
clf.fit(X_train_features, y_train)

y_val_prob_chunk = clf.predict_proba(X_val_features)[:, 1]
y_val_chunk = y_val.copy()

# chunk level stats
print("Chunk level stats:")

fpr, tpr, thresholds = roc_curve(y_val, y_val_prob_chunk)
frr = 1 - tpr

# Define your maximum allowed FAR target (e.g., 1% or 0.01)
MAX_FAR = 0.00002
MAX_FRR = 1

# Find all indices where FAR is within our budget
valid_indices = np.where((fpr <= MAX_FAR) )[0]
valid_indices = valid_indices[valid_indices > 0]  # Exclude the first index to avoid threshold=inf

if len(valid_indices) > 0:
    # Among valid thresholds, pick the one with the lowest FRR
    best_idx = valid_indices[np.argmin(frr[valid_indices])]
    threshold = thresholds[best_idx]
else:
    # Fallback if no threshold meets the budget: pick the absolute minimum FAR
    print("fallbacked")
    best_idx = np.argmin(fpr[1:]) + 1
    threshold = thresholds[best_idx]

y_test_prob_chunk = clf.predict_proba(X_test_features)[:, 1]
y_test_pred = (y_test_prob_chunk >= threshold).astype(int)

print(f"Best threshold: {threshold:.4f}")
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel() #.ravel() flattens the confusion matrix into a 1D array, allowing us to unpack the values directly into tn, fp, fn, and tp.
far = fp / (fp + tn)
frr = fn / (fn + tp)
eer = (far + frr) / 2
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test False Acceptance Rate (FAR): {far:.4f}")
print(f"Test False Rejection Rate (FRR): {frr:.4f}")
print(f"Test Equal Error Rate (EER): {eer:.4f}")

# overwrite y_test_prob with cycle accuracy version
y_test_prob = []
for cycle in set(test_cycleids):
    cycle_indices = [i for i, cycle_id in enumerate(test_cycleids) if cycle_id == cycle]
    cycle_predictions = clf.predict_proba(X_test_features[cycle_indices])[:, 1]
    prediction = np.mean(cycle_predictions)
    y_test_prob.append(prediction)

# overwrite y_test with cycle accuracy version
y_test_chunk = y_test
y_test = []
for cycle in set(test_cycleids):
    cycle_indices = [i for i, cycle_id in enumerate(test_cycleids) if cycle_id == cycle]
    cycle_labels = y_test_chunk[cycle_indices]
    y_test.append(cycle_labels[0])

print(y_test_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
frr = 1 - tpr

# Define your maximum allowed FAR target (e.g., 1% or 0.01)
MAX_FAR = 0.00002
MAX_FRR = 1

# Find all indices where FAR is within our budget
valid_indices = np.where((fpr <= MAX_FAR) )[0]
valid_indices = valid_indices[valid_indices > 0]  # Exclude the first index to avoid threshold=inf

if len(valid_indices) > 0:
    # Among valid thresholds, pick the one with the lowest FRR
    best_idx = valid_indices[np.argmin(frr[valid_indices])]
    threshold = thresholds[best_idx]
else:
    # Fallback if no threshold meets the budget: pick the absolute minimum FAR
    print("fallbacked")
    best_idx = np.argmin(fpr[1:]) + 1
    threshold = thresholds[best_idx]

y_test_pred = (y_test_prob >= threshold).astype(int)
print(f"Best threshold: {threshold:.4f}")
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel() #.ravel() flattens the confusion matrix into a 1D array, allowing us to unpack the values directly into tn, fp, fn, and tp.
far = fp / (fp + tn)
frr = fn / (fn + tp)
eer = (far + frr) / 2
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test False Acceptance Rate (FAR): {far:.4f}")
print(f"Test False Rejection Rate (FRR): {frr:.4f}")
print(f"Test Equal Error Rate (EER): {eer:.4f}")

print("[CNN-GRU - Imports (1/11)] Importing tensorboard...")
import tensorboard
print("[CNN-GRU - Imports (2/11)] Importing torch...")
import torch
print("[CNN-GRU - Imports (3/11)] Importing json...")
import json
print("[CNN-GRU - Imports (4/11)] Importing pandas...")
import pandas as pd
print("[CNN-GRU - Imports (5/11)] Importing random...")
import random
print("[CNN-GRU - Imports (6/11)] Importing sklearn...")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
print("[CNN-GRU - Imports (7/11)] Importing numpy...")
import numpy as np
print("[CNN-GRU - Imports (8/11)] Importing datetime...")
from datetime import datetime
print("[CNN-GRU - Imports (9/11)] Importing copy...")
import copy
print("[CNN-GRU - Imports (10/11)] Importing helpers...")
from helpers import load_data
print("[CNN-GRU - Imports (11/11)] Importing scipy...")
from scipy.stats import entropy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
# 3. Ensure deterministic cuDNN algorithms (GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

threshold_pct = 0.5

X_train, y_train, X_val, y_val, yes_user_id, user_labels = load_data()
yes_user_label = user_labels[yes_user_id]
# remove yes user from training set
train_mask = y_train != yes_user_label
X_train = X_train[train_mask]
y_train = y_train[train_mask]
val_mask = y_val != yes_user_label
X_val = X_val[val_mask]
y_val = y_val[val_mask]

# -- Taken from Gemini --
# RE-INDEX LABELS to 0..N-1
unique_users = torch.unique(y_train)
label_map = {old.item(): new for new, old in enumerate(unique_users)}

y_train = torch.tensor([label_map[y.item()] for y in y_train], dtype=torch.long)
y_val = torch.tensor([label_map[y.item()] for y in y_val], dtype=torch.long)

# Update total class count for the model
num_background_users = len(unique_users)

# 3. Create PyTorch DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=None, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

class MouseCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MouseCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_classes, 16, kernel_size=5)
        self.relu = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(16)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=3)
        self.norm2 = torch.nn.BatchNorm1d(32)
        self.conv3 = torch.nn.Conv1d(32, 64, kernel_size=3)
        self.norm3 = torch.nn.BatchNorm1d(64)
        self.conv4 = torch.nn.Conv1d(64, 128, kernel_size=3)
        self.norm4 = torch.nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.norm4(x)
        return x

# tensorflow units are pytorch hidden_size

class MouseCNNGRU(torch.nn.Module):
    def __init__(self, num_classes, num_users):
        super(MouseCNNGRU, self).__init__()
        self.cnn = MouseCNN(num_classes)
        self.gru = torch.nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        units = 128
        self.dense = torch.nn.Linear(128, units)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.dense2 = torch.nn.Linear(units, num_users)

    def extract_features(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        gru_out, hn = self.gru(x)
        x = hn[-1]
        return x

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        gru_out, hn = self.gru(x)
        x = hn[-1]
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

# Evaluate models
def CNNGRU_evaluate(model, X_val, y_val, criterion, device, batch_size=32):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    total_false_accepts = 0
    total_false_rejects = 0
    total_negatives = 0
    total_positives = 0
    # measure prediction frequency per user
    prediction_frequency = {}
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = batch_x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, sequence_length)
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y.long())
            running_loss += loss.item() * len(batch_y)
            output = torch.argmax(output.squeeze(-1), dim=1)
            for user in output.tolist():
                prediction_frequency[user] = prediction_frequency.get(user, 0) + 1
            correct += (output == batch_y).sum().item()
            total += len(batch_y)
            total_negatives += (batch_y == 0).sum().item()
            total_positives += (batch_y == 1).sum().item()
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    counts = np.zeros(num_background_users)
    for user_id, count in prediction_frequency.items():
        counts[user_id] = count
    p = counts / counts.sum()
    neff = 2 ** entropy(p[p > 0], base=2)
    return avg_loss, accuracy, neff


# Train the models

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_neurocursor_model.pth')
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def trainCNNGRU(X_train, y_train, X_val, y_val):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/neurocursor_cnngru_{timestamp}")
    model = MouseCNNGRU(num_classes=14, num_users=num_background_users)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = torch.tensor([(y_train == 0).sum().item() / (y_train == 1).sum().item()], dtype=torch.float32).to(device)
    print(f"Current negative:positive ratio: {pos_weight.item()}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    early_stopper = EarlyStopping(patience=50, min_delta=0.0001)
    num_epochs = 500
    batch_size = 32
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        total_false_accepts = 0
        total_false_rejects = 0
        total_negatives = 0
        total_positives = 0
        # measure prediction frequency per user
        prediction_frequency = {}
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = batch_x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, sequence_length)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
            output = torch.argmax(output.squeeze(-1), dim=1)
            for user in output.tolist():
                prediction_frequency[user] = prediction_frequency.get(user, 0) + 1
            correct += (output == batch_y).sum().item()
            total += len(batch_y)
            total_negatives += (batch_y == 0).sum().item()
            total_positives += (batch_y == 1).sum().item()
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        counts = np.zeros(num_background_users)

        for user_id, count in prediction_frequency.items():
            counts[user_id] = count
        p = counts / counts.sum()
        train_neff = 2 ** entropy(p[p > 0], base=2)
        val_loss, val_acc, val_neff = CNNGRU_evaluate(model, X_val, y_val, criterion, device, batch_size=batch_size)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar('Loss/train - Training:', train_loss, epoch)
        writer.add_scalar('Accuracy/train - Training:', train_acc, epoch)
        writer.add_scalar('Loss/train - Validation:', val_loss, epoch)
        writer.add_scalar('Accuracy/train - Validation:', val_acc, epoch)
        writer.add_scalar('Neff/train - Training:', train_neff, epoch)
        writer.add_scalar('Neff/train - Validation:', val_neff, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Train Neff: {train_neff:.2f}, Val Neff: {val_neff:.2f}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Stopping training.")
            model.load_state_dict(early_stopper.best_weights)
            break
    writer.close()
if __name__ == "__main__":
    trainCNNGRU(X_train, y_train, X_val, y_val)
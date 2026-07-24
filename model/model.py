print("Imports [1/7] (tensorboard)")
import tensorboard
print("Imports [2/7] (torch)")
import torch
print("Imports [3/7] (json)")
import json
print("Imports [4/7] (pandas)")
import pandas as pd
print("Imports [5/7] (random)")
import random
print("Imports [6/7] (sklearn)")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
print("Imports [7/7] (numpy)")
import numpy as np
from datetime import datetime
import copy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
# 3. Ensure deterministic cuDNN algorithms (GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load our data
print("Loading data...")
raw_data = json.load(open("data.json"))
df = pd.DataFrame([{"user_id": user_id, "data": sessions} for user_id, sessions in raw_data.items()])
print("Printing data...")
print(df)
print("Data loaded.\n\n\n\n\n")

# Grabbing users w/ at least 30 cycles
valid_users = df[df["data"].apply(len) >= 30].reset_index(drop=True)
print("Printing dataset size...")
print(valid_users["data"].apply(len).sum())

# Pick random user as yes and rest as no
yes_user_row = valid_users.iloc[random.randint(0, len(valid_users)-1)]
print("\n\nPrinting yes user id...")
yes_user_id = yes_user_row["user_id"]
print(yes_user_id)

cycle_rows = []
for _, row in valid_users.iterrows():
    label = 1 if row["user_id"] == yes_user_id else 0
    for cycle_idx, cycle_data in enumerate(row["data"]):
        cycle_rows.append({
            "user_id": row["user_id"],
            "cycle_id": f"{row['user_id']}_c{cycle_idx}",
            "data": cycle_data,
            "label": label       # 1 = Yes User, 0 = No User
        })

cycle_df = pd.DataFrame(cycle_rows)

# Make data split (75% train, 15% validation, 10% test)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(cycle_df, train_size=0.75, random_state=RANDOM_SEED, stratify=cycle_df['label'])
val_df, test_df = train_test_split(temp_df, train_size=0.60, random_state=RANDOM_SEED, stratify=temp_df['label'])

# Train tensors
X_train = []
y_train = []

# Validation tensors
X_val = []
y_val = []

# Turn train_df into tensors, taking each cycle's data and label and splitting by 128
for cycle in train_df.iterrows():
    # cycle is a tuple with (index, row), where row is a Series
    cycle_data = cycle[1]["data"]
    cycle_label = cycle[1]["label"]

    # Split cycle_data into chunks of 128
    chunks = [cycle_data[i:i+128] for i in range(0, len(cycle_data), 128)]
    for chunk in chunks:
        if len(chunk) < 128:
            continue
        # Convert chunk to tensor and add to train dataset
        # Turn chunk for features into a vector with time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height
        # Each tensor should look like [[time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height], [...], ... 128 times]
        multi_dim_array = []
        for data_point in chunk:
            list = [
                data_point["time"],
                data_point["cursor_x"],
                data_point["cursor_y"],
                data_point["target_x"],
                data_point["target_y"],
                data_point["relative_x"],
                data_point["relative_y"],
                data_point["button_state"],
                data_point["movement_index"],
                data_point["target_width"],
                data_point["target_height"],
                data_point["canvas_width"],
                data_point["canvas_height"]
            ]
            multi_dim_array.append(list)
        X = torch.tensor(multi_dim_array, dtype=torch.float32)
        y = torch.tensor(cycle_label, dtype=torch.long)
        # Add to train dataset (implement this part)
        X_train.append(X)
        y_train.append(y)

# Turn val_df into tensors, taking each cycle's data and label and splitting by 128
for cycle in val_df.iterrows():
    # cycle is a tuple with (index, row), where row is a Series
    cycle_data = cycle[1]["data"]
    cycle_label = cycle[1]["label"]

    # Split cycle_data into chunks of 128
    chunks = [cycle_data[i:i+128] for i in range(0, len(cycle_data), 128)]
    for chunk in chunks:
        if len(chunk) < 128:
            continue
        # Convert chunk to tensor and add to validation dataset
        # Turn chunk for features into a vector with time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height
        # Each tensor should look like [[time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height], [...], ... 128 times]
        multi_dim_array = []
        for data_point in chunk:
            list = [
                data_point["time"],
                data_point["cursor_x"],
                data_point["cursor_y"],
                data_point["target_x"],
                data_point["target_y"],
                data_point["relative_x"],
                data_point["relative_y"],
                data_point["button_state"],
                data_point["movement_index"],
                data_point["target_width"],
                data_point["target_height"],
                data_point["canvas_width"],
                data_point["canvas_height"]
            ]
            multi_dim_array.append(list)
        X = torch.tensor(multi_dim_array, dtype=torch.float32)
        y = torch.tensor(cycle_label, dtype=torch.long)
        # Add to validation dataset (implement this part)
        X_val.append(X)
        y_val.append(y)

# Turn X_train, y_train, X_val, y_val into tensors
X_train = torch.stack(X_train)
y_train = torch.stack(y_train)
X_val = torch.stack(X_val)
y_val = torch.stack(y_val)


class MouseLogistic:
    def __init__(self, n_features):
        self.w = tensor.zeros(n_features)
        self.bias = 0.0
    def sigmoid(self, z):
        return 1/(1+tensor.exp(-z))
    def forward(self, X):
           z = X@ self.w + self.bias
           return self.sigmoid(z)
    

class MouseCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MouseCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_classes, 32, kernel_size=5)
        self.relu = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(32)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3)
        self.norm2 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.maxpool(x)
        return x

# tensorflow units are pytorch hidden_size
units = 64
mouseGRU = torch.nn.GRU(input_size=64, hidden_size=units, batch_first=True)

class MouseCNNGRU(torch.nn.Module):
    def __init__(self, num_classes):
        super(MouseCNNGRU, self).__init__()
        self.cnn = MouseCNN(num_classes)
        self.gru = mouseGRU
        units = 32
        self.dense = torch.nn.Linear(64, units)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.dense2 = torch.nn.Linear(units, 1)

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
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch_x = X_val[i:i+batch_size].to(device)
            batch_y = y_val[i:i+batch_size].to(device)
            batch_x = batch_x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, sequence_length)
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y.float())
            running_loss += loss.item() * len(batch_y)
            predicted = (output.squeeze(-1) > 0).long()
            correct += (predicted == batch_y).sum().item()
            total += len(batch_y)
            total_false_accepts += ((predicted == 1) & (batch_y == 0)).sum().item()
            total_false_rejects += ((predicted == 0) & (batch_y == 1)).sum().item()
            total_negatives += (batch_y == 0).sum().item()
            total_positives += (batch_y == 1).sum().item()
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    far = (100 * total_false_accepts / total_negatives) if total_negatives > 0 else 0.0
    frr = (100 * total_false_rejects / total_positives) if total_positives > 0 else 0.0

    return avg_loss, accuracy, far, frr

def LogisticRegressionEvaluate(model, X_val, y_val):
    probs = model.foward(X_val)
    predicts = tensor.where(probs >= 0.5)
    #Convert PyTorch tensors to NumPy for Scikit-Learn. Taken from AI
    #detach() stops gradient tracking, .cpu() moves data to system memory, .numpy() converts it
    y_numpy = y.detach().cpu().numpy()
    preds_numpy = predictions.detach().cpu().numpy()
    probs_numpy = probabilities.detach().cpu().numpy()
    acc = accuracy_score(y_numpy, preds_numpy)
    loss = log_loss(y_numpy, probs_numpy)
    return acc, loss


# Train the models
def trainLogisticRegression(X_train, y_train, lr= 0.01, steps = 1000):
    n_samples, n_features = X.shape
    model = MouseLogistic(n_features)
    for _ in range(steps):
        ŷ = model.forward(X)
        error = ŷ - y
        dw = (X.T @ error) / n_samples
        db = tensor.mean(error)
        model.w -= lr * dw
        model.b -= lr * db
    return model.w, model.b

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
    writer = SummaryWriter(log_dir=f"runs/neurocursor_experiment_{timestamp}")
    model = MouseCNNGRU(num_classes=13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = torch.tensor([(y_train == 0).sum().item() / (y_train == 1).sum().item()], dtype=torch.float32).to(device)
    print(f"Current negative:positive ratio: {pos_weight.item()}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00067)
    early_stopper = EarlyStopping(patience=10, min_delta=0.0001)
    num_epochs = 50
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
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            batch_x = batch_x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, sequence_length)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
            predicted = (output.squeeze(-1) > 0).long()
            correct += (predicted == batch_y).sum().item()
            total += len(batch_y)
            total_false_accepts += ((predicted == 1) & (batch_y == 0)).sum().item()
            total_false_rejects += ((predicted == 0) & (batch_y == 1)).sum().item()
            total_negatives += (batch_y == 0).sum().item()
            total_positives += (batch_y == 1).sum().item()
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_far = (100 * total_false_accepts / total_negatives) if total_negatives > 0 else 0.0
        train_frr = (100 * total_false_rejects / total_positives) if total_positives > 0 else 0.0
        val_loss, val_acc, val_far, val_frr = CNNGRU_evaluate(model, X_val, y_val, criterion, device, batch_size=batch_size)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar('Loss/train - Training:', train_loss, epoch)
        writer.add_scalar('Accuracy/train - Training:', train_acc, epoch)
        writer.add_scalar('Loss/train - Validation:', val_loss, epoch)
        writer.add_scalar('Accuracy/train - Validation:', val_acc, epoch)
        writer.add_scalar('FAR/train - Validation:', val_far, epoch)
        writer.add_scalar('FRR/train - Validation:', val_frr, epoch)
        writer.add_scalar('FAR/train - Training:', train_far, epoch)
        writer.add_scalar('FRR/train - Training:', train_frr, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Train FAR: {train_far:.2f}%, Train FRR: {train_frr:.2f}%, Val FAR: {val_far:.2f}%, Val FRR: {val_frr:.2f}%")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Stopping training.")
            model.load_state_dict(early_stopper.best_weights)
            break
    writer.close()

trainCNNGRU(X_train, y_train, X_val, y_val)

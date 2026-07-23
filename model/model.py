# Update TODO: Scripts.js put on server
print("Imports [1/6] (tensorboard)")
import tensorboard as tb
print("Imports [2/6] (torch)")
import torch
print("Imports [3/6] (json)")
import json
print("Imports [4/6] (pandas)")
import pandas as pd
print("Imports [5/6] (random)")
import random
print("Imports [6/6] (sklearn)")
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Load our data
print("Loading data...")
raw_data = json.load(open("data.json"))
df = pd.DataFrame([{"user_id": user_id, "data": sessions} for user_id, sessions in raw_data.items()])
print(df)
print("Data loaded.")

# TODO:
# - Grab all users that have at least 30 cycles [x]
# - Pick random user as yes and rest as no (use RANDOM_SEED) [x]
# - Make data loader (75% train, 15% validation, 10% test)
# - Try training the model, fix the million errors that show up, make sure to implement tensorboard and see results

# Grabbing users w/ at least 30 cycles
print("Filtering users with at least 30 cycles...")
valid_users = df[df["data"].apply(len) >= 30].reset_index(drop=True)
print(len(valid_users))

# Pick random user as yes and rest as no
yes_user_row = valid_users.iloc[random.randint(0, len(valid_users)-1)]
print("Printing yes user...")
yes_user_id = yes_user_row["user_id"]
print(yes_user_id)
#no_users = valid_users[valid_users["user_id"] != yes_user_row["user_id"]]

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

print("\nFlattened Dataset Summary")
print(len(cycle_df))
print(f"Yes user cycles (label 1): {(cycle_df['label'] == 1).sum()}")
print(f"No user cycles (label 0): {(cycle_df['label'] == 0).sum()}")

# Make data split (75% train, 15% validation, 10% test)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(cycle_df, train_size=0.75, random_state=RANDOM_SEED, stratify=cycle_df['label'])
val_df, test_df = train_test_split(temp_df, train_size=0.60, random_state=RANDOM_SEED, stratify=temp_df['label'])

print(f"\nData Split Summary")
print(f"Train set: {len(train_df)} cycles (Yes: {train_df['label'].sum()}, No: {len(train_df) - train_df['label'].sum()})")
print(f"Val set:   {len(val_df)} cycles (Yes: {val_df['label'].sum()}, No: {len(val_df) - val_df['label'].sum()})")
print(f"Test set:  {len(test_df)} cycles (Yes: {test_df['label'].sum()}, No: {len(test_df) - test_df['label'].sum()})")

#Review----------------------------------------------

# Make data loader
class CursorDataset(Dataset):
    def __init__(self, dataframe):
        self.samples = dataframe["data"].values
        self.labels = dataframe["label"].values

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if isinstance(sample, dict):
            x = torch.tensor(list(sample.values()), dtype=torch.float32)
        else:
            x = torch.tensor(sample, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def pad_collate_fn(batch):
    """Pads variable length cursor sequences within a batch to prevent shape mismatch errors."""
    sequences, labels = zip(*batch)
    if sequences[0].dim() > 0 and any(s.shape != sequences[0].shape for s in sequences):
        padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        return padded_seqs, torch.stack(labels)
    return torch.stack(sequences), torch.stack(labels)

class_counts = train_df["label"].value_counts()
class_weights = {
    0: 1.0 / class_counts[0],
    1: 1.0 / class_counts[1]
}
sample_weights = [class_weights[label] for label in train_df["label"]]

# WeightedRandomSampler oversamples the target 'Yes' user so batches are balanced during training
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

BATCH_SIZE = 32

# Training loader uses the weighted sampler (do NOT set shuffle=True when using sampler)
train_loader = DataLoader(
    CursorDataset(train_df),
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=pad_collate_fn
)

# Val & Test loaders evaluate on natural distribution
val_loader = DataLoader(
    CursorDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pad_collate_fn
)

test_loader = DataLoader(
    CursorDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pad_collate_fn
)

#-------------------------------------

class MouseCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MouseCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(128 * num_classes, 32, kernel_size=5)
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
mouseGRU = torch.nn.GRU(input_size=64, hidden_size=units)

class MouseCNNGRU(torch.nn.Module):
    def __init__(self, num_classes):
        super(MouseCNNGRU, self).__init__()
        self.cnn = MouseCNN(num_classes)
        self.gru = mouseGRU
        units = 32
        self.dense = torch.nn.Linear(units, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        units = 1
        self.dense2 = torch.nn.Linear(units, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.gru(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
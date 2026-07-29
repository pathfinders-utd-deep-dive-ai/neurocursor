print("[Helpers - Imports (1/5)] Importing torch...")
import torch
print("[Helpers - Imports (2/5)] Importing pandas...")
import pandas as pd
print("[Helpers - Imports (3/5)] Importing json...")
import json
print("[Helpers - Imports (4/5)] Importing random...")
import random
print("[Helpers - Imports (5/5)] Importing numpy...")
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
# 3. Ensure deterministic cuDNN algorithms (GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data():
    # Load our data
    print("[Helpers - Data] Loading data...")
    raw_data = json.load(open("master_dataset.json"))
    df = pd.DataFrame([{"user_id": user_id, "data": sessions} for user_id, sessions in raw_data.items()])
    print("[Helpers - Data] Printing data...")
    print(df)
    print("[Helpers - Data] Data loaded.\n\n\n\n\n")

    # Grabbing users w/ at least 20 cycles
    valid_users = df[df["data"].apply(len) >= 20].reset_index(drop=True)
    print("[Helpers - Data] Printing dataset size...")
    print(valid_users["data"].apply(len).sum())
    print("[Helpers - Data] Printing valid users number...")
    print(len(valid_users))

    # Pick random user as yes and rest as no
    yes_user_row = valid_users.iloc[random.randint(0, len(valid_users)-1)]
    print("\n\n[Helpers - Data] Printing yes user id...")
    yes_user_id = yes_user_row["user_id"]
    print(yes_user_id)

    # user labels, increment for each valid user
    user_labels = {}
    for row in valid_users.iterrows():
        user_id = row[1]["user_id"]
        user_labels[user_id] = len(user_labels)

    cycle_rows = []
    for _, row in valid_users.iterrows():
        for cycle_idx, cycle_data in enumerate(row["data"]):
            # actually instead do
            for data_point in cycle_data:
                data_point["cycle_id"] = f"{row['user_id']}_c{cycle_idx}"
            cycle_rows.append({
                "user_id": row["user_id"],
                "data": cycle_data,
                "user_label": user_labels[row["user_id"]]
            })

    cycle_df = pd.DataFrame(cycle_rows)

    # Make data split (75% train, 15% validation, 10% test)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(cycle_df, train_size=0.75, random_state=RANDOM_SEED, stratify=cycle_df['user_label'])
    val_df, test_df = train_test_split(temp_df, train_size=0.60, random_state=RANDOM_SEED, stratify=temp_df['user_label'])

    # Train tensors
    X_train = []
    y_train = []

    # Validation tensors
    X_val = []
    y_val = []
    def process_cycle_data(cyclesdata):
        X = []
        y = []
        cycleids = []
        for cycle in cyclesdata.iterrows():
            # cycle is a tuple with (index, row), where row is a Series
            cycle_data = cycle[1]["data"]
            cycle_label = cycle[1]["user_label"]
    
            # Split cycle_data into chunks of 128
            chunks = [cycle_data[i:i+128] for i in range(0, len(cycle_data), 128)]
            for chunk in chunks:
                if len(chunk) < 128:
                    continue
                cycleid = chunk[0]["cycle_id"]
                cycleids.append(cycleid)
                # Convert chunk to tensor and add to train dataset
                # Turn chunk for features into a vector with time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height
                # Each tensor should look like [[time, relative_x, relative_y, button_state], [...], ... 128 times]
                multi_dim_chunk_array = []
                for data_point in chunk:
                    if len(data_point) < 13:
                        list = [
                            data_point["time"],
                            data_point["coords"][0],
                            data_point["coords"][1],
                            data_point["coords"][2]
                        ]
                    else:
                        list = [
                            data_point["time"],
                            data_point["relative_x"],
                            data_point["relative_y"],
                            data_point["button_state"]
                        ]
                    multi_dim_chunk_array.append(list)
                raw_chunk_arr = np.array(multi_dim_chunk_array, dtype=np.float32)
    
                # calculate kinematic features
                vel_x = np.zeros(128, dtype=np.float32); vel_x[1:] = np.diff(raw_chunk_arr[:, 1])
                vel_y = np.zeros(128, dtype=np.float32); vel_y[1:] = np.diff(raw_chunk_arr[:, 2])
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
                acc_x = np.zeros(128, dtype=np.float32); acc_x[1:] = np.diff(vel_x)
                acc_y = np.zeros(128, dtype=np.float32); acc_y[1:] = np.diff(vel_y)
                jerk_x = np.zeros(128, dtype=np.float32); jerk_x[1:] = np.diff(acc_x)
                jerk_y = np.zeros(128, dtype=np.float32); jerk_y[1:] = np.diff(acc_y)
                dist  = np.sqrt(raw_chunk_arr[:, 1] ** 2 + raw_chunk_arr[:, 2] ** 2)
                heading = np.arctan2(vel_y, vel_x)
                btn_diff = np.zeros(128, dtype=np.float32); btn_diff[1:] = np.diff(raw_chunk_arr[:, 3]) # button transition
    
                final_chunk_arr = np.column_stack([
                    raw_chunk_arr,
                    vel_x,
                    vel_y,
                    speed,
                    acc_x,
                    acc_y,
                    jerk_x,
                    jerk_y,
                    dist,
                    heading,
                    btn_diff
                ]).astype(np.float32)
                X.append(torch.tensor(final_chunk_arr, dtype=torch.float32))
                y.append(torch.tensor(cycle_label, dtype=torch.long))
        return X, y, cycleids
    # Turn train_df into tensors, taking each cycle's data and label and splitting by 128
    X_train, y_train, train_cycleids = process_cycle_data(train_df)

    # Turn val_df into tensors, taking each cycle's data and label and splitting by 128
    X_val, y_val, val_cycleids = process_cycle_data(val_df)

    # Turn test_df into tensors, taking each cycle's data and label and splitting by 128
    X_test, y_test, test_cycleids = process_cycle_data(test_df)

    # Turn X_train, y_train, X_val, y_val into tensors
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_val = torch.stack(X_val)
    y_val = torch.stack(y_val)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)
    mean = X_train.mean(dim=(0, 1))
    std = X_train.std(dim=(0, 1), unbiased=False)

    # Apply z-score normalization
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return X_train, y_train, X_val, y_val, yes_user_id, user_labels, train_cycleids, val_cycleids, X_test, y_test, test_cycleids
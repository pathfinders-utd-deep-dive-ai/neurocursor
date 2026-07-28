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
            cycle_rows.append({
                "user_id": row["user_id"],
                "cycle_id": f"{row['user_id']}_c{cycle_idx}",
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
        for cycle in cyclesdata.iterrows():
            # cycle is a tuple with (index, row), where row is a Series
            cycle_data = cycle[1]["data"]
            cycle_label = cycle[1]["user_label"]
    
            # Split cycle_data into chunks of 128
            chunks = [cycle_data[i:i+128] for i in range(0, len(cycle_data), 128)]
            for chunk in chunks:
                if len(chunk) < 128:
                    continue
                # Convert chunk to tensor and add to train dataset
                # Turn chunk for features into a vector with time, cursor_x, cursor_y, target_x, target_y, relative_x, relative_y, button_state, movement_index, target_width, target_height, canvas_width, canvas_height
                # Each tensor should look like [[time, relative_x, relative_y, button_state], [...], ... 128 times]
                multi_dim_array = []
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
                    multi_dim_array.append(list)
                raw_arr = np.array(multi_dim_array, dtype=np.float32)
    
                # calculate kinematic features
                vel_x = np.zeros(128, dtype=np.float32); vel_x[1:] = np.diff(raw_arr[:, 1])
                vel_y = np.zeros(128, dtype=np.float32); vel_y[1:] = np.diff(raw_arr[:, 2])
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
                acc_x = np.zeros(128, dtype=np.float32); acc_x[1:] = np.diff(vel_x)
                acc_y = np.zeros(128, dtype=np.float32); acc_y[1:] = np.diff(vel_y)
                jerk_x = np.zeros(128, dtype=np.float32); jerk_x[1:] = np.diff(acc_x)
                jerk_y = np.zeros(128, dtype=np.float32); jerk_y[1:] = np.diff(acc_y)
                dist  = np.sqrt(raw_arr[:, 1] ** 2 + raw_arr[:, 2] ** 2)
                heading = np.arctan2(vel_y, vel_x)
                btn_diff = np.zeros(128, dtype=np.float32); btn_diff[1:] = np.diff(raw_arr[:, 3]) # button transition
    
                final_arr = np.column_stack([
                    raw_arr,
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
                X.append(torch.tensor(final_arr, dtype=torch.float32))
                y.append(torch.tensor(cycle_label, dtype=torch.long))
        return X, y
    # Turn train_df into tensors, taking each cycle's data and label and splitting by 128
    X_train, y_train = process_cycle_data(train_df)

    # Turn val_df into tensors, taking each cycle's data and label and splitting by 128
    X_val, y_val = process_cycle_data(val_df)

    # Turn X_train, y_train, X_val, y_val into tensors
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_val = torch.stack(X_val)
    y_val = torch.stack(y_val)

    return X_train, y_train, X_val, y_val, yes_user_id, user_labels
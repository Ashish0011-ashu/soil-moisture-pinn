import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# PROJECT PATH SETUP
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(PROJECT_ROOT)

from models.pinn_model import PINN


# LOAD DATA
data_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_plant_vase1.csv"
)

data = pd.read_csv(data_path)
print("=" * 60)
print("AGRICULTURE PINN TRAINING (TANH VERSION)")
print("=" * 60)


# CREATE LAG FEATURE
data["moisture0_prev"] = data["moisture0"].shift(1)
data = data.dropna().reset_index(drop=True)


# INPUTS & TARGETS
X = data[[
    "time",
    "moisture1",
    "moisture2",
    "moisture3",
    "moisture4",
    "moisture0_prev"
]].values

y = data["moisture0"].values.reshape(-1, 1)
print(f"Total Samples: {len(X)}")


# CHRONOLOGICAL SPLIT (Prevents Validation Leakage)
split_index = int(len(X) * 0.8)

X_train_raw = X[:split_index]
X_val_raw = X[split_index:]

y_train_raw = y[:split_index]
y_val_raw = y[split_index:]


# NORMALIZATION (Fitted strictly on Training set)
# Time scaling
time_min = X_train_raw[:, 0:1].min()
time_max = X_train_raw[:, 0:1].max()

time_train_norm = (X_train_raw[:, 0:1] - time_min) / (time_max - time_min + 1e-8)
time_val_norm = (X_val_raw[:, 0:1] - time_min) / (time_max - time_min + 1e-8)

# Features scaling
feat_min = X_train_raw[:, 1:].min(axis=0)
feat_max = X_train_raw[:, 1:].max(axis=0)

features_train_norm = (X_train_raw[:, 1:] - feat_min) / (feat_max - feat_min + 1e-8)
features_val_norm = (X_val_raw[:, 1:] - feat_min) / (feat_max - feat_min + 1e-8)

# Recombining components
X_train = np.hstack([time_train_norm, features_train_norm])
X_val = np.hstack([time_val_norm, features_val_norm])

# Target scaling
y_min = y_train_raw.min()
y_max = y_train_raw.max()

y_train = (y_train_raw - y_min) / (y_max - y_min + 1e-8)
y_val = (y_val_raw - y_min) / (y_max - y_min + 1e-8)

print(f"Train Samples : {len(X_train)}")
print(f"Val Samples   : {len(X_val)}")


# TENSORS
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


# MODEL SETUP
model = PINN()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
mse_loss = nn.MSELoss()


# TRAINING CONFIG
epochs = 3000
lambda_phy = 0.05  # Active weight for temporal continuity physics loss
best_val_loss = float("inf")
patience = 200
counter = 0
history = []


# TRAINING LOOP
print("\nTraining Started...\n")
for epoch in range(epochs):
    model.train()
    pred = model(X_train_tensor)

    # DATA LOSS
    data_loss = mse_loss(pred, y_train_tensor)

    
    # REVISED PHYSICS LOSS [Temporal Continuity]
    # Index 5 maps to- 'moisture0_prev'. 
    # Enforces the physical law that moisture levels cannot instantly teleport or drift wildly.
    prev_moisture = X_train_tensor[:, 5:6]
    physics_residual = (pred - prev_moisture) ** 2
    physics_loss = torch.mean(physics_residual)

    total_loss = data_loss + (lambda_phy * physics_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    
    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = mse_loss(val_pred, y_val_tensor)

    scheduler.step(val_loss)

 
    # SAVE HISTORY
    history.append([
        epoch,
        total_loss.item(),
        data_loss.item(),
        physics_loss.item(),
        val_loss.item()
    ])


    # BEST MODEL CHECKPOINT
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        counter = 0
        torch.save(
            model.state_dict(),
            os.path.join(PROJECT_ROOT, "models", "best_model.pth")
        )
    else:
        counter += 1


    # PRINT PROGRESS
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"Total: {total_loss.item():.6f} | "
            f"Data: {data_loss.item():.6f} | "
            f"Physics: {physics_loss.item():.6f} | "
            f"Val: {val_loss.item():.6f}"
        )

    
    # EARLY STOPPING
    if counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break


# SAVE FINAL MODEL PACK
save_path = os.path.join(PROJECT_ROOT, "models", "pinn_final.pth")

torch.save({
    "model": model.state_dict(),
    "feat_min": feat_min,
    "feat_max": feat_max,
    "time_min": time_min,
    "time_max": time_max,
    "y_min": y_min,
    "y_max": y_max,
    "lambda_phy": lambda_phy
}, save_path)

print("\nModel Saved successfully!")


# SAVE TRAINING HISTORY
history_df = pd.DataFrame(
    history,
    columns=["epoch", "total_loss", "data_loss", "physics_loss", "val_loss"]
)

history_path = os.path.join(PROJECT_ROOT, "results", "training_history.csv")
os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
history_df.to_csv(history_path, index=False)

print(f"Training history saved: {history_path}")
print("\nTraining Finished!!")
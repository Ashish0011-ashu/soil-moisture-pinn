import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# PROJECT PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.pinn_model import PINN

# LOAD DATA
data_path = os.path.join(PROJECT_ROOT, "data", "processed_plant_vase1.csv")
data = pd.read_csv(data_path)

x = data[[
    "time",
    "moisture1",
    "moisture2",
    "moisture3",
    "moisture4"
]].values

y = data["moisture0"].values.reshape(-1, 1)

# NORMALIZATION
time = x[:, 0:1]
features = x[:, 1:]

feat_min = features.min(axis=0)
feat_max = features.max(axis=0)

features = (features - feat_min) / (feat_max - feat_min + 1e-8)

x = np.hstack([time, features])

y_min = y.min()
y_max = y.max()

y = (y - y_min) / (y_max - y_min + 1e-8)

# TRAIN / VAL SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# TENSORS
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# TIME needs gradient for du/dt, FEATURES also need gradient
t_train = torch.tensor(
    X_train[:, 0:1],
    dtype=torch.float32,
    requires_grad=True
)

features_train = torch.tensor(
    X_train[:, 1:],
    dtype=torch.float32,
    requires_grad=False
)

X_train_input = torch.cat([t_train, features_train], dim=1)

t_val = torch.tensor(X_val[:, 0:1], dtype=torch.float32, requires_grad=False)
features_val = torch.tensor(X_val[:, 1:], dtype=torch.float32, requires_grad=False)

X_val_input = torch.cat([t_val, features_val], dim=1)

# MODEL
model = PINN()

k = torch.nn.Parameter(torch.tensor(0.5))

optimizer = optim.Adam(list(model.parameters()) + [k], lr=0.001)
mse_loss = nn.MSELoss()

# TRAINING SETUP
epochs = 2000
best_val = float("inf")
patience = 150
counter = 0

val_loss = torch.tensor(0.0)

print("\nTraining started...\n")

# TRAINING LOOP
for epoch in range(epochs):

    # Forward pass
    pred = model(X_train_input)

    # DATA LOSS - how well model fits the data
    data_loss = mse_loss(pred, y_train_tensor)

    # PHYSICS LOSS - du/dt + k*(u - u_mean) = 0
    # This says: moisture at location 0 decays toward average of surrounding sensors
    # Physical meaning: water tends to equilibrate across the soil
    
    u_t = torch.autograd.grad(
        outputs=pred,
        inputs=t_train,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True
    )[0]

    # Mean of sensor readings (moisture1, moisture2, moisture3, moisture4)
    u_mean = torch.mean(features_train, dim=1, keepdim=True)

    # Physics residual: du/dt + k*(u - u_mean) = 0
    # Better than pure decay because it uses sensor data
    physics_residual = u_t + k * (pred - u_mean)
    physics_loss = torch.mean(physics_residual ** 2)

    # Loss balance: start with physics, gradually reduce weight
    lambda_phy = max(0.1, 1.0 - epoch / epochs)

    loss = data_loss + lambda_phy * physics_loss

    # BACKPROP
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Constrain k to physical range
    with torch.no_grad():
        k.data = torch.clamp(k.data, 0.01, 5.0)

    # VALIDATION
    if epoch % 50 == 0:
        with torch.no_grad():
            val_pred = model(X_val_input)
            val_loss = mse_loss(val_pred, y_val_tensor)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            counter = 0
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "models", "best_model.pth"))
        else:
            counter += 1

    # LOGGING
    if epoch % 200 == 0:
        print(
            f"Epoch {epoch} | "
            f"Total: {loss.item():.6f} | "
            f"Data: {data_loss.item():.6f} | "
            f"Physics: {physics_loss.item():.6f} | "
            f"k: {k.item():.4f} | "
            f"Val: {val_loss.item():.6f}"
        )

    # EARLY STOPPING
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# SAVE MODEL
final_path = os.path.join(PROJECT_ROOT, "models", "pinn_final.pth")
torch.save({
    "model": model.state_dict(),
    "k": k.item(),
    "feat_min": feat_min,
    "feat_max": feat_max,
    "y_min": y_min,
    "y_max": y_max
}, final_path)

print("\nTraining complete!")
print("Saved at:", final_path)
print("Final k:", k.item())
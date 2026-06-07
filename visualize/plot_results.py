import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# SECTION 1: PROJECT SETUP AND IMPORTS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.pinn_model import PINN


# SECTION 2: DEFINE FILE PATHS
data_path = os.path.join(PROJECT_ROOT, "data", "processed_plant_vase1.csv")
model_path = os.path.join(PROJECT_ROOT, "models", "pinn_final.pth")


# SECTION 3: LOAD AND PREPARE DATA
data = pd.read_csv(data_path)

# Create lag feature matching training scheme exactly
data["moisture0_prev"] = data["moisture0"].shift(1)
data = data.dropna().reset_index(drop=True)

# Order matching training features explicitly:
# index 0: time, index 1-4: moisture sensors, index 5: moisture0_prev
X = data[[
    "time",
    "moisture1",
    "moisture2",
    "moisture3",
    "moisture4",
    "moisture0_prev"
]].values

y = data["moisture0"].values.reshape(-1, 1)


# SECTION 4: LOAD TRAINED MODEL CHECKPOINT
checkpoint = torch.load(
    model_path,
    map_location=torch.device("cpu"),
    weights_only=False
)

feat_min = checkpoint["feat_min"]
feat_max = checkpoint["feat_max"]
time_min = checkpoint["time_min"]
time_max = checkpoint["time_max"]
y_min = checkpoint["y_min"]
y_max = checkpoint["y_max"]
lambda_phy = checkpoint["lambda_phy"]


# SECTION 5: NORMALIZE INPUT DATA (FITTED TRAINING BOUNDS)
time_col = X[:, 0:1]
features_cols = X[:, 1:]

time_norm = (time_col - time_min) / (time_max - time_min + 1e-8)
features_norm = (features_cols - feat_min) / (feat_max - feat_min + 1e-8)

X_norm = np.hstack([time_norm, features_norm])


# SECTION 6: LOAD TRAINED MODEL
model = PINN()
model.load_state_dict(checkpoint["model"])
model.eval()


# SECTION 7: MAKE PREDICTIONS
X_tensor = torch.tensor(X_norm, dtype=torch.float32)

with torch.no_grad():
    pred = model(X_tensor).numpy()

# Reverse scaling back to original physical units
pred_real = pred * (y_max - y_min) + y_min
y_real = y


# SECTION 9: CALCULATE EVALUATION METRICS
error = y_real - pred_real

# 1. MAE
mae = np.mean(np.abs(error))

# 2. RMSE
rmse = np.sqrt(np.mean(error ** 2))

# 3. MAPE (using small epsilon to protect division by zero)
mape = np.mean(np.abs(error / (y_real + 1e-8))) * 100

# 4. R² Score (Coefficient of Determination)
y_mean = np.mean(y_real)
ss_res = np.sum(error ** 2)
ss_tot = np.sum((y_real - y_mean) ** 2)
r_squared = 1 - (ss_res / (ss_tot + 1e-8))


# SECTION 10: PRINT EVALUATION RESULTS
print("=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"Mean Absolute Error (MAE):              {mae:.6f}")
print(f"Root Mean Squared Error (RMSE):         {rmse:.6f}")
print(f"Mean Absolute Percentage Error (MAPE):  {mape:.4f}%")
print(f"R² Score (Coefficient of Determination): {r_squared:.6f}")
print(f"Physics weight :                      {lambda_phy:.4f}")
print("=" * 50)


# SECTION 11: CREATE VISUALIZATIONS
fig = plt.figure(figsize=(18, 10))

# SUBPLOT 1: Real vs Predicted (Full Dataset)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(y_real, label="Real Moisture", color="blue", linewidth=1.5, alpha=0.8)
ax1.plot(pred_real, label="Predicted", color="red", linewidth=1.5, alpha=0.8)
ax1.set_title("Real vs Predicted (Full)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Moisture")
ax1.set_xlabel("Sample")
ax1.legend()
ax1.grid(True, alpha=0.3)

# SUBPLOT 2: Real vs Predicted (Zoomed)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(y_real[:500], label="Real Moisture", color="blue", linewidth=1.5)
ax2.plot(pred_real[:500], label="Predicted", color="red", linewidth=1.5)
ax2.set_title("Real vs Predicted (Zoomed - First 500)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Moisture")
ax2.set_xlabel("Sample")
ax2.legend()
ax2.grid(True, alpha=0.3)

# SUBPLOT 3: Prediction Error
ax3 = plt.subplot(2, 3, 3)
ax3.plot(error, color="black", linewidth=0.8, alpha=0.7)
ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
ax3.axhline(y=mae, color="green", linestyle="--", alpha=0.5, label=f"MAE: {mae:.4f}")
ax3.axhline(y=-mae, color="green", linestyle="--", alpha=0.5)
ax3.set_title("Prediction Error", fontsize=12, fontweight="bold")
ax3.set_ylabel("Error (Real - Predicted)")
ax3.set_xlabel("Sample")
ax3.legend()
ax3.grid(True, alpha=0.3)

# SUBPLOT 4: Error Distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(error, bins=50, color="green", alpha=0.7, edgecolor="black")
ax4.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
ax4.set_title("Error Distribution", fontsize=12, fontweight="bold")
ax4.set_xlabel("Error Value")
ax4.set_ylabel("Frequency")
ax4.legend()
ax4.grid(True, alpha=0.3)

# SUBPLOT 5: Residuals vs Predicted Value
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(pred_real, error, alpha=0.3, s=10, color="purple")
ax5.axhline(y=0, color="red", linestyle="--", linewidth=2)
ax5.set_title("Residuals vs Predicted Value", fontsize=12, fontweight="bold")
ax5.set_xlabel("Predicted Moisture")
ax5.set_ylabel("Error")
ax5.grid(True, alpha=0.3)

# SUBPLOT 6: Comparison of All Moisture Values
ax6 = plt.subplot(2, 3, 6)
# Indexes 1 to 4 map exactly to features 'moisture1' through 'moisture4'
sensor_avg = np.mean(X[:, 1:5], axis=1, keepdims=True)

ax6.plot(y_real, label="moisture0 (Real)", color="blue", linewidth=1.5, alpha=0.8)
ax6.plot(sensor_avg, label="Sensor Average (1-4)", color="orange", linewidth=1.5, alpha=0.8)
ax6.plot(pred_real, label="Predicted", color="red", linewidth=1.5, alpha=0.8)
ax6.set_title("Moisture Comparison", fontsize=12, fontweight="bold")
ax6.set_ylabel("Moisture (Original Scale)")
ax6.set_xlabel("Sample")
ax6.legend()
ax6.grid(True, alpha=0.3)


# SECTION 12: DISPLAY VISUALIZATION
plt.tight_layout()
plt.show()
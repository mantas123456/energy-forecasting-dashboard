"""
Script: 04_train_xgboost.py
Description: Trains an XGBoost model using pre-engineered features.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))  # Project root
from scripts.utils.load_config import load_config

logging.basicConfig(
    filename='logs/train_xgboost.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Started XGBoost training script.")

config = load_config()
input_file = os.path.join(config["data_paths"]["processed"], "processed_data_features.csv")
df = pd.read_csv(input_file, parse_dates=["timestamp"])
logging.info(f"Loaded feature-engineered data with shape {df.shape}")

features = ["hour", "dayofweek", "month", "lag_1h", "lag_24h", "rolling_3h", "rolling_24h"]
target = "energy_kwh"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(
    n_estimators=config["xgboost"]["n_estimators"],
    max_depth=config["xgboost"]["max_depth"],
    learning_rate=config["xgboost"]["learning_rate"],
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train, y_train)
logging.info("XGBoost model trained.")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
logging.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred
results_df["timestamp"] = df.loc[X_test.index, "timestamp"]

pred_path = config["model_paths"]["predictions"]
plot_path = config["model_paths"]["plots"]
os.makedirs(pred_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

results_df.to_csv(os.path.join(pred_path, "predictions_xgboost.csv"), index=False)

plt.figure(figsize=(12, 4))
plt.plot(results_df["timestamp"], results_df["actual"], label="Actual")
plt.plot(results_df["timestamp"], results_df["predicted"], label="Predicted")
plt.title("XGBoost: Prediction vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "plot_forecast_xgboost.png"))

logging.info("XGBoost results saved and plotted.")

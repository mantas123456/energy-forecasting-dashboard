"""
Script: 06_evaluate_models.py
Description: Compares RMSE and MAE across all trained models using their predictions.
Author: Mantas Valantinavičius
Created: 2025-05-21
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add root to path
sys.path.append(os.path.abspath("."))

from scripts.utils.load_config import load_config

# --- Logging ---
logging.basicConfig(
    filename='logs/evaluate_models.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Started model evaluation script.")

# --- Load Config ---
config = load_config()
pred_path = config["model_paths"]["predictions"]
plot_path = config["model_paths"]["plots"]
os.makedirs(plot_path, exist_ok=True)

# --- Model prediction files ---
model_files = {
    "Prophet": "prophet_predictions.csv",
    "XGBoost": "xgboost_predictions.csv",
    "LinearRegression": "linear_predictions.csv"
}

results = []

for model_name, filename in model_files.items():
    file_path = os.path.join(pred_path, filename)

    try:
        df = pd.read_csv(file_path)

        if "actual" not in df.columns or "predicted" not in df.columns:
            raise ValueError("Missing 'actual' or 'predicted' columns.")
        
        df = df.dropna(subset=["actual", "predicted"])
        if df.empty:
            raise ValueError("No valid data after dropping NaNs.")
        
        rmse = np.sqrt(mean_squared_error(df["actual"], df["predicted"]))
        mae = mean_absolute_error(df["actual"], df["predicted"])
        results.append({"Model": model_name, "RMSE": round(rmse, 3), "MAE": round(mae, 3)})
        logging.info(f"{model_name} evaluated: RMSE={rmse:.2f}, MAE={mae:.2f}")

    except Exception as e:
        logging.warning(f"Could not evaluate {model_name}: {e}")

# --- Save Summary ---
results_df = pd.DataFrame(results)
summary_path = os.path.join(pred_path, "model_evaluation_summary.csv")
results_df.to_csv(summary_path, index=False)
logging.info(f"Saved summary to {summary_path}")

# --- Plot Summary ---
plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(len(results_df))

plt.bar(x - bar_width/2, results_df["RMSE"], bar_width, label="RMSE")
plt.bar(x + bar_width/2, results_df["MAE"], bar_width, label="MAE")
plt.xticks(x, results_df["Model"])
plt.ylabel("Error")
plt.title("Model Comparison: RMSE vs MAE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "model_comparison.png"))
logging.info("Comparison plot saved.")

print("✅ Model evaluation completed. See logs and output folder for results.")

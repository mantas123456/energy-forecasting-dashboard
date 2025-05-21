"""
Script: 03_train_prophet.py
Description: Trains a Prophet forecasting model using preprocessed data and saves the forecast and plot.
Author: Mantas Valantinavičius
Created: 2025-05-21
"""

import os
import sys
import logging
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Add root to path
sys.path.append(os.path.abspath("."))

from scripts.utils.load_config import load_config

# --- Setup Logging ---
logging.basicConfig(
    filename='logs/train_prophet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Started Prophet training script.")

# --- Load Config ---
config = load_config()
logging.info("Configuration loaded.")

# --- Load Data ---
processed_path = config['data_paths']['processed']
input_file = os.path.join(processed_path, "processed_data_features.csv")

try:
    df = pd.read_csv(input_file)
    logging.info(f"Loaded data from {input_file} with shape {df.shape}.")
except Exception as e:
    logging.error(f"Failed to load processed data: {e}")
    raise

# --- Prepare Data for Prophet ---
df_prophet = df.rename(columns={"timestamp": "ds", "energy_kwh": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

# --- Train Prophet Model ---
model = Prophet(
    daily_seasonality=config['prophet']['daily_seasonality'],
    yearly_seasonality=config['prophet']['yearly_seasonality'],
    changepoint_prior_scale=config['prophet']['changepoint_prior_scale']
)
model.fit(df_prophet)
logging.info("Prophet model training completed.")

# --- Forecast ---
horizon = config['modeling']['horizon_hours']
future = model.make_future_dataframe(periods=horizon, freq='h')
forecast = model.predict(future)
forecast["ds"] = pd.to_datetime(forecast["ds"])  # Ensure datetime format
logging.info("Forecast generated.")

# --- Merge for Evaluation ---
merged = pd.merge(forecast, df_prophet, how="left", on="ds", suffixes=("", "_actual"))
merged["actual"] = merged["y"]
merged["predicted"] = merged["yhat"]

# --- Save Forecast and Plot ---
predictions_path = config['model_paths']['predictions']
plots_path = config['model_paths']['plots']
os.makedirs(predictions_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

forecast_file = os.path.join(predictions_path, "prophet_predictions.csv")
plot_file = os.path.join(plots_path, "prophet_forecast_plot.png")

# Save only necessary columns
merged[["ds", "actual", "predicted"]].to_csv(forecast_file, index=False)
logging.info(f"Forecast saved to {forecast_file}")

# Save plot
fig = model.plot(forecast)
fig.savefig(plot_file)
logging.info(f"Plot saved to {plot_file}")
logging.info("Prophet pipeline completed successfully.")

"""
Script: 02_feature_engineering.py
Description: Adds lag and time-based features to energy data for modeling.
Author: Mantas Valantinavičius
Created: 2025-05-21
"""

import os
import sys
import logging
import pandas as pd

# Add root
sys.path.append(os.path.abspath("."))

from scripts.utils.load_config import load_config

# --- Logging ---
logging.basicConfig(
    filename='logs/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Started feature engineering script.")

# --- Load Config ---
config = load_config()
input_file = os.path.join(config['data_paths']['processed'], "processed_data.csv")
output_file = os.path.join(config['data_paths']['processed'], "processed_data_features.csv")

# --- Load Data ---
df = pd.read_csv(input_file, parse_dates=["timestamp"])
logging.info(f"Loaded data with shape {df.shape}")

# --- Feature Engineering ---
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# Lag features
df["lag_1h"] = df["energy_kwh"].shift(1)
df["lag_24h"] = df["energy_kwh"].shift(24)

# Rolling average
df["rolling_3h"] = df["energy_kwh"].rolling(3).mean()
df["rolling_24h"] = df["energy_kwh"].rolling(24).mean()

df.dropna(inplace=True)
logging.info(f"Feature-engineered data shape: {df.shape}")

# --- Save ---
df.to_csv(output_file, index=False)
logging.info(f"Feature-engineered data saved to {output_file}")
print(f"✅ Feature-engineered data saved to {output_file}")

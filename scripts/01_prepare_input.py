"""
Script: 01_prepare_input.py
Description: Creates a sample processed dataset with hourly timestamps and energy values.
Author: Mantas Valantinavičius
Created: 2025-05-21
"""

import os
import sys
sys.path.append(os.path.abspath("."))  # Allow imports from project root
import pandas as pd
import numpy as np
from scripts.utils.load_config import load_config





# --- Load Config ---
config = load_config()

# --- Settings ---
output_dir = config['data_paths']['processed']
os.makedirs(output_dir, exist_ok=True)

# --- Create 1 Year of Hourly Timestamps ---
date_range = pd.date_range(start="2024-01-01", end="2024-12-31 23:00", freq="H")

# --- Generate Synthetic Energy Consumption Pattern ---
np.random.seed(42)
base_demand = 3 + 2 * np.sin(2 * np.pi * date_range.hour / 24)  # daily cycle
seasonal_effect = 1 + 0.5 * np.cos(2 * np.pi * date_range.dayofyear / 365)  # yearly seasonality
noise = np.random.normal(0, 0.2, len(date_range))
energy_kwh = (base_demand * seasonal_effect + noise).round(2)

# --- Build DataFrame ---
df = pd.DataFrame({
    "timestamp": date_range,
    "energy_kwh": energy_kwh
})

# --- Save to CSV ---
output_file = os.path.join(output_dir, "processed_data.csv")
df.to_csv(output_file, index=False)
print(f"✅ Sample processed_data.csv saved to {output_file}")

import pandas as pd
import numpy as np

# Create 7 days of hourly timestamps
date_range = pd.date_range(start="2025-01-01", periods=24*7, freq="H")

# Generate realistic energy values with daily patterns + noise
base = 10 + 3 * np.sin(2 * np.pi * date_range.hour / 24)  # diurnal pattern
noise = np.random.normal(0, 0.5, size=len(date_range))    # random noise
energy = base + noise

# Create DataFrame
df = pd.DataFrame({
    "timestamp": date_range,
    "energy_kwh": energy.round(2)
})

# Save CSV
df.to_csv("sample_energy_data_7d.csv", index=False)
print("✅ Generated: sample_energy_data_7d.csv")

"""
Script: main.py
Description: Unified pipeline runner for energy forecasting dashboard
Author: Mantas Valantinavičius
Created: 2025-05-21
"""

import os
import subprocess
import logging

# --- Setup Logging ---
logging.basicConfig(
    filename='logs/main_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Starting full pipeline...")

# --- Ordered pipeline steps ---
PIPELINE_STEPS = [
    "scripts/01_prepare_input.py",
    "scripts/03_train_prophet.py",
    "scripts/04_train_xgboost.py",
    "scripts/05_train_linear.py",
    "scripts/06_evaluate_models.py"
]

# --- Run each script ---
for step in PIPELINE_STEPS:
    logging.info(f"Running step: {step}")
    print(f"\n🚀 Running: {step}")
    
    result = subprocess.run(["python", step], capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"❌ Failed at step: {step}\n{result.stderr}")
        print(f"❌ Error in {step}:\n{result.stderr}")
        break
    else:
        logging.info(f"✅ Completed: {step}")
        print(f"✅ Completed: {step}")

logging.info("Pipeline finished.")
print("\n✅ All steps completed. Check /logs for full details.")

"""
Script: main.py
Description: Unified pipeline runner for energy forecasting dashboard
Author: Mantas Valantinavičius
"""

import os
import subprocess
import logging

# --- Logging ---
logging.basicConfig(
    filename='logs/log_main_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Starting full pipeline...")

# --- Step Sequence ---
PIPELINE_STEPS = [
    "scripts/01_prepare_input.py",
    "scripts/02_feature_engineering.py",
    "scripts/03_train_prophet.py",
    "scripts/04_train_xgboost.py",
    "scripts/05_train_linear.py",
    "scripts/06_evaluate_models.py",
    "scripts/07_generate_phase2_report.py"
]

# --- Execute ---
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

logging.info("Pipeline completed.")
print("\n✅ All steps completed. Check /logs for details.")

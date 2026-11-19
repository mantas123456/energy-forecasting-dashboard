from pathlib import Path

import pytest
from synthetic import generate

from energy_forecast.config import load_config

SYNTHETIC_CONFIG = Path(__file__).with_name("synthetic_config.yaml")


@pytest.fixture()
def synthetic_cfg():
    """Small synthetic data set: 90 days, 4 sectors, quick XGBoost."""
    cfg = load_config(SYNTHETIC_CONFIG)
    cfg["days"] = 90
    cfg["xgboost"]["n_estimators"] = 30
    return cfg


@pytest.fixture()
def synthetic_data(synthetic_cfg):
    return generate(synthetic_cfg)

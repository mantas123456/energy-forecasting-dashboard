"""Forecasting models with a common fit/predict interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import REFERENCE


class Forecaster(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive:
    """Repeats an earlier value: the same hour `lag_hours` ago.

    Any model worth using has to beat it.
    """

    kind = "naive"

    def __init__(self, lag_hours: int, name: str):
        self.col = f"lag_{lag_hours}h"
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


class TSOForecast:
    """The TSO's own published day-ahead forecast, read from the data, not fitted.

    It is scored like a model so the real models can be compared against it.
    """

    kind = "reference"
    name = "TSO day-ahead forecast"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TSOForecast:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[REFERENCE].to_numpy(dtype=float)


class RidgeModel:
    kind = "model"
    name = "Ridge regression"

    def __init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, None)


class XGBModel:
    kind = "model"
    name = "XGBoost"

    def __init__(self, params: dict[str, Any], seed: int):
        self.model = XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=0, **params
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, None)


def naive_name(lag_hours: int) -> str:
    if lag_hours == 24:
        return "Naive (yesterday)"
    if lag_hours == 168:
        return "Naive (last week)"
    return f"Naive ({lag_hours // 24} days ago)"


def make_models(cfg: dict[str, Any], tso: bool = False) -> list[Forecaster]:
    """Naive baselines, ridge and XGBoost; plus the TSO forecast when `tso` is true."""
    h = cfg["horizon_hours"]
    if h % 24 or not 24 <= h < 168:
        raise ValueError("the naive baselines need a horizon of 24, 48, ... or 144 hours")
    models: list[Forecaster] = [
        SeasonalNaive(h, naive_name(h)),
        SeasonalNaive(168, naive_name(168)),
        RidgeModel(),
        XGBModel(cfg["xgboost"], cfg["seed"]),
    ]
    if tso:
        models.append(TSOForecast())
    return models

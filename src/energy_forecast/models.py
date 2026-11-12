"""Forecasting models with a common fit/predict interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class Forecaster(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive:
    """Repeats an earlier value: same hour yesterday, or same hour last week.

    Any model worth using has to beat both.
    """

    def __init__(self, lag_hours: int, name: str):
        self.col = f"lag_{lag_hours}h"
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


class RidgeModel:
    name = "Ridge regression"

    def __init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, None)


class XGBModel:
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


def make_models(cfg: dict[str, Any]) -> list[Forecaster]:
    h = cfg["horizon_hours"]
    if h != 24:
        raise ValueError("the naive baselines assume a 24-hour horizon")
    return [
        SeasonalNaive(h, "Naive (yesterday)"),
        SeasonalNaive(168, "Naive (last week)"),
        RidgeModel(),
        XGBModel(cfg["xgboost"], cfg["seed"]),
    ]

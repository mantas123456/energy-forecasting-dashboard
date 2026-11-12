"""Feature engineering for day-ahead forecasting without look-ahead leakage.

A forecast for hour t is made `horizon` hours earlier, so every feature built
from the target series is shifted by at least `horizon` hours. Lags and
rolling windows are computed per sector so one sector's history never leaks
into another's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "energy_kwh"


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (dow >= 5).astype(int),
        },
        index=ts.index,
    )


def lag_features(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Lags and rolling statistics using only values at least `horizon` hours old."""
    past = y.shift(horizon)
    return pd.DataFrame(
        {
            f"lag_{horizon}h": past,
            f"lag_{horizon + 24}h": y.shift(horizon + 24),
            "lag_168h": y.shift(max(168, horizon)),
            "roll_mean_24h": past.rolling(24).mean(),
            "roll_mean_168h": past.rolling(168).mean(),
            "roll_std_24h": past.rolling(24).std(),
        },
        index=y.index,
    )


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add features to a long table with columns timestamp, sector, energy_kwh, ...

    Rows without a full feature history are dropped.
    """
    out = []
    for _, g in df.sort_values(["sector", "timestamp"]).groupby("sector", sort=False):
        g = g.copy()
        feats = pd.concat(
            [
                calendar_features(g["timestamp"]),
                lag_features(g[TARGET], horizon),
                # Weather at the target hour. In production this would be a weather
                # forecast; here the simulated value stands in for a perfect forecast.
                g[["temperature_c", "holiday"]],
            ],
            axis=1,
        )
        feats["temp_dev_c"] = (g["temperature_c"] - 21.0).abs()
        out.append(pd.concat([g[["timestamp", "sector", TARGET, "is_anomaly"]], feats], axis=1))
    return pd.concat(out).dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"timestamp", "sector", TARGET, "is_anomaly"}
    return [c for c in df.columns if c not in skip]

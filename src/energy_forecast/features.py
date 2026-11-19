"""Feature engineering for day-ahead forecasting without look-ahead leakage.

A forecast for hour t is made `horizon` hours earlier, so every feature built
from the target series is shifted by at least `horizon` hours. Lags and
rolling windows are computed per sector so one sector's history never leaks
into another's.

The TSO's day-ahead forecast, when the data has one, is carried along as a
reference to compare against. It is not a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "demand"  # hourly energy: MWh for the real data, kWh for the synthetic fixture
REFERENCE = "tso_forecast"
PASS_THROUGH = ["timestamp", "sector", TARGET, "is_anomaly", REFERENCE]


def calendar_features(ts: pd.Series, tz: str | None = None) -> pd.DataFrame:
    """Hour, weekday and season. Timezone-aware timestamps are converted to `tz` first."""
    local = ts.dt.tz_convert(tz) if tz and ts.dt.tz is not None else ts
    hour = local.dt.hour
    dow = local.dt.day_of_week
    doy = local.dt.day_of_year
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


def build_features(df: pd.DataFrame, horizon: int, tz: str | None = None) -> pd.DataFrame:
    """Add features to a long table with columns timestamp, sector, demand, holiday, ...

    `temperature_c` is used when present (the synthetic data has it, the real data does
    not). Rows without the target or a full feature history are dropped; a missing TSO
    forecast does not drop a row.
    """
    has_temp = "temperature_c" in df and df["temperature_c"].notna().any()
    out = []
    for _, g in df.sort_values(["sector", "timestamp"]).groupby("sector", sort=False):
        g = g.copy()
        parts = [
            calendar_features(g["timestamp"], tz),
            lag_features(g[TARGET], horizon),
            g[["holiday"]],
        ]
        if has_temp:
            # Weather at the target hour. In production this would be a weather forecast;
            # in the synthetic data the simulated value stands in for a perfect forecast.
            parts.append(g[["temperature_c"]])
            parts.append((g["temperature_c"] - 21.0).abs().rename("temp_dev_c"))
        feats = pd.concat(parts, axis=1)
        keep = [c for c in PASS_THROUGH if c in g]
        out.append(pd.concat([g[keep], feats], axis=1))
    result = pd.concat(out)
    required = [TARGET, *feature_columns(result)]
    return result.dropna(subset=required).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in PASS_THROUGH]

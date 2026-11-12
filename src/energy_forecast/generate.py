"""Generate realistic synthetic hourly demand for several building sectors.

Each sector gets its own daily load shape, a weekend and holiday response, a
temperature response (heating and cooling around a comfort temperature),
random noise, and a few labelled anomalies. Carbon emissions use an
hour-of-day grid emission factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Relative hourly load shapes (hour 0..23), scaled so each averages 1.0.
_RAW_SHAPES: dict[str, list[float]] = {
    "Residential": [
        0.5,
        0.4,
        0.4,
        0.4,
        0.4,
        0.5,
        0.8,
        1.2,
        1.1,
        0.8,
        0.7,
        0.7,
        0.8,
        0.8,
        0.7,
        0.7,
        0.9,
        1.3,
        1.7,
        1.9,
        1.8,
        1.5,
        1.1,
        0.7,
    ],
    "School": [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.3,
        0.9,
        1.8,
        2.1,
        2.2,
        2.2,
        2.0,
        2.1,
        2.0,
        1.5,
        0.8,
        0.5,
        0.3,
        0.3,
        0.2,
        0.2,
        0.2,
        0.2,
    ],
    "Factory": [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.7,
        1.0,
        1.3,
        1.4,
        1.4,
        1.4,
        1.4,
        1.2,
        1.4,
        1.4,
        1.4,
        1.3,
        1.1,
        0.9,
        0.8,
        0.7,
        0.7,
        0.6,
        0.6,
    ],
    "EV_Charging": [
        0.4,
        0.3,
        0.3,
        0.3,
        0.3,
        0.4,
        0.6,
        1.0,
        1.4,
        1.5,
        1.4,
        1.3,
        1.3,
        1.3,
        1.3,
        1.4,
        1.6,
        1.9,
        1.8,
        1.5,
        1.1,
        0.8,
        0.6,
        0.5,
    ],
}
DAILY_SHAPES = {k: np.array(v) / np.mean(v) for k, v in _RAW_SHAPES.items()}

# Monthly mean air temperature, degC (Malta-like climate).
MONTHLY_MEAN_C = {
    1: 12.5,
    2: 12.5,
    3: 14.0,
    4: 16.5,
    5: 20.0,
    6: 24.5,
    7: 27.5,
    8: 28.0,
    9: 25.5,
    10: 22.0,
    11: 18.0,
    12: 14.5,
}

# Fixed-date Maltese public holidays as (month, day).
HOLIDAYS_MD = {
    (1, 1),
    (2, 10),
    (3, 19),
    (3, 31),
    (5, 1),
    (6, 7),
    (6, 29),
    (8, 15),
    (9, 8),
    (9, 21),
    (12, 8),
    (12, 13),
    (12, 25),
}


def simulate_temperature(
    ts: pd.DatetimeIndex, cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Monthly mean plus a daily cycle peaking mid-afternoon, plus noise."""
    t = cfg["temperature"]
    base = np.array([MONTHLY_MEAN_C[m] for m in ts.month], dtype=float)
    daily = t["daily_swing_c"] * np.cos((np.asarray(ts.hour) - 15) / 24 * 2 * np.pi)
    return base + daily + rng.normal(0, t["noise_std_c"], len(ts))


def holiday_flags(ts: pd.DatetimeIndex) -> np.ndarray:
    pairs = zip(ts.month, ts.day, strict=True)
    return np.array([(m, d) in HOLIDAYS_MD for m, d in pairs], dtype=int)


def emission_factor(hours: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Grid carbon intensity in gCO2e/kWh for each hour of day."""
    ef = cfg["emission_factor_g_per_kwh"]
    out = np.full(hours.shape, float(ef["default"]))
    out[hours < 6] = ef["night"]
    out[(hours >= 10) & (hours <= 16)] = ef["midday"]
    return out


def inject_anomalies(
    energy: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Add spikes, dropouts and short level shifts. Returns (energy, flags)."""
    energy = energy.copy()
    flags = np.zeros(len(energy), dtype=int)
    starts = rng.choice(np.arange(48, len(energy) - 48), size=n, replace=False)
    for i, start in enumerate(starts):
        kind = ("spike", "dropout", "shift")[i % 3]
        if kind == "spike":
            energy[start] *= rng.uniform(2.0, 3.0)
            flags[start] = 1
        elif kind == "dropout":
            length = int(rng.integers(2, 6))
            energy[start : start + length] = 0.0
            flags[start : start + length] = 1
        else:
            length = int(rng.integers(12, 36))
            energy[start : start + length] *= rng.uniform(1.3, 1.6)
            flags[start : start + length] = 1
    return energy, flags


def generate(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a long table: one row per sector per hour."""
    rng = np.random.default_rng(cfg["seed"])
    ts = pd.date_range(cfg["start_date"], periods=cfg["days"] * 24, freq="h")
    temp = simulate_temperature(ts, cfg, rng)
    hol = holiday_flags(ts)
    weekend = np.asarray(ts.dayofweek) >= 5
    hours = np.asarray(ts.hour)
    comfort = cfg["temperature"]["comfort_c"]

    frames = []
    for sector, p in cfg["sectors"].items():
        shape = DAILY_SHAPES[sector][hours]
        demand = p["base_kw"] * shape
        demand = demand * np.where(weekend, p["weekend_factor"], 1.0)
        demand = demand * np.where(hol == 1, p["holiday_factor"], 1.0)
        demand = demand * (1 + p["temp_slope"] * np.abs(temp - comfort))
        demand = demand * (1 + rng.normal(0, cfg["noise_std_fraction"], len(ts)))
        demand = np.clip(demand, 0.0, None)
        demand, flags = inject_anomalies(demand, cfg["anomalies"]["per_sector"], rng)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "sector": sector,
                    "energy_kwh": demand.round(3),  # hourly average kW == kWh per hour
                    "temperature_c": temp.round(1),
                    "holiday": hol,
                    "carbon_kgco2e": (demand * emission_factor(hours, cfg) / 1000).round(3),
                    "is_anomaly": flags,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

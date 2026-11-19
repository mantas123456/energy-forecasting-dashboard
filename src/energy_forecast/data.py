"""Hourly load and the TSO day-ahead forecast from the energy-data-pipeline project.

Sources (config `data.source`, or `--source` on the command line):

- ``postgres``: the pipeline's ``mart_hourly`` table. The database URL is read from the
  environment variable named in ``data.database_url_env``.
- ``csv``: ``data.csv_path`` is either an hourly CSV with columns ``ts_hour``, ``load_mw``
  and ``tso_forecast_mw`` (UTC), like the committed sample ``data/lt_hourly.csv.gz``, or
  the pipeline's ``data/observations/<zone>`` folder of 15-minute files, which are
  averaged to hours the same way the pipeline's ``mart_hourly`` does.

All sources give the same table, so results do not depend on where the data came from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import holidays
import pandas as pd

from .features import TARGET

HOURLY_COLUMNS = ["ts_hour", "load_mw", "tso_forecast_mw"]
ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_SERIES = {"load_actual": "load_mw", "load_forecast_tso": "tso_forecast_mw"}


class DataSourceError(RuntimeError):
    """The configured data source could not be read."""


def _hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df[HOURLY_COLUMNS].copy()
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True)
    for c in HOURLY_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.sort_values("ts_hour").drop_duplicates("ts_hour").reset_index(drop=True)


def read_postgres(url: str, zone: str) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:
        raise DataSourceError("install the database extra: pip install -e '.[db]'") from e
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT ts_hour, load_mw, tso_forecast_mw FROM mart_hourly "
                    "WHERE zone = :zone ORDER BY ts_hour"
                ),
                conn,
                params={"zone": zone},
            )
        engine.dispose()
    except Exception as e:  # connection refused, missing table, bad credentials, ...
        raise DataSourceError(f"could not read mart_hourly: {type(e).__name__}") from e
    return _hourly(df)


def read_observations(folder: Path) -> pd.DataFrame:
    """Average the pipeline's 15-minute observation files to hours."""
    cols = {}
    for series, col in OBSERVATION_SERIES.items():
        files = sorted((folder / series).glob("*.csv.gz"))
        if not files:
            raise DataSourceError(f"no files in {folder / series}")
        raw = pd.concat(pd.read_csv(f) for f in files)
        raw["ts_utc"] = pd.to_datetime(raw["ts_utc"], utc=True)
        cols[col] = raw.set_index("ts_utc")["value"].resample("h").mean()
    return _hourly(pd.DataFrame(cols).rename_axis("ts_hour").reset_index())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataSourceError(f"{path} does not exist")
    if path.is_dir():
        return read_observations(path)
    df = pd.read_csv(path)
    missing = set(HOURLY_COLUMNS) - set(df.columns)
    if missing:
        raise DataSourceError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return _hourly(df)


def read_hourly(cfg: dict[str, Any], source: str | None = None) -> pd.DataFrame:
    """Hourly table with columns ts_hour (UTC), load_mw, tso_forecast_mw."""
    d = cfg["data"]
    source = source or d["source"]
    if source == "postgres":
        url = os.environ.get(d["database_url_env"], "")
        if not url:
            raise DataSourceError(f"environment variable {d['database_url_env']} is not set")
        return read_postgres(url, d["zone"])
    if source == "csv":
        path = Path(d["csv_path"])
        return read_csv(path if path.is_absolute() else ROOT / path)
    raise DataSourceError(f"unknown data source {source!r} (use csv or postgres)")


def to_long(hourly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """The long table the features expect, on a complete hourly UTC index.

    Missing hours stay as NaN, so lags never jump across a gap. Holidays use the local
    date. Hours after the last actual load (the TSO forecast for tomorrow) are dropped.
    """
    d = cfg["data"]
    h = hourly[hourly["ts_hour"] >= pd.Timestamp(d["start"], tz=d["timezone"])]
    h = h.set_index("ts_hour")
    last = h["load_mw"].last_valid_index()
    if last is None:
        raise DataSourceError("no actual load in the data")
    idx = pd.date_range(h.index.min(), last, freq="h", name="ts_hour")
    h = h.reindex(idx)
    local = idx.tz_convert(d["timezone"])
    hol = holidays.country_holidays(
        d["country"], years=range(local.year.min(), local.year.max() + 1)
    )
    return pd.DataFrame(
        {
            "timestamp": idx,
            "sector": d["zone"],
            TARGET: h["load_mw"].to_numpy(),
            "tso_forecast": h["tso_forecast_mw"].to_numpy(),
            "holiday": [int(day in hol) for day in local.date],
            "is_anomaly": 0,
        }
    )


def load(cfg: dict[str, Any], source: str | None = None) -> pd.DataFrame:
    return to_long(read_hourly(cfg, source), cfg)

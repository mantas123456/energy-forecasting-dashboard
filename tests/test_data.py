import numpy as np
import pandas as pd
import pytest

from energy_forecast import data
from energy_forecast.config import load_config
from energy_forecast.features import TARGET


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c["data"]["start"] = "2025-12-30"
    c["data"]["csv_path"] = str(tmp_path / "hourly.csv")
    return c


def hourly_frame():
    ts = pd.date_range("2025-12-30 00:00", "2026-01-02 23:00", freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_hour": ts,
            "load_mw": np.arange(len(ts), dtype=float) + 1000,
            "tso_forecast_mw": 1200.0,
        }
    )


def test_to_long_fills_gaps_drops_future_hours_and_flags_local_holidays(cfg):
    h = hourly_frame()
    h = h.drop(index=[10, 11])  # two hours missing from the source
    h.loc[h.index[-3:], "load_mw"] = np.nan  # TSO forecast only: tomorrow
    long = data.to_long(data._hourly(h), cfg)
    assert long["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert long[TARGET].isna().sum() == 2  # the gap stays visible as NaN
    assert long["timestamp"].max() == pd.Timestamp("2026-01-02 20:00", tz="UTC")
    assert set(long.columns) >= {"timestamp", "sector", TARGET, "tso_forecast", "holiday"}
    assert (long["sector"] == "LT").all()
    # New Year's Day starts at 22:00 UTC on 31 December in Vilnius (UTC+2).
    by_ts = long.set_index("timestamp")["holiday"]
    assert by_ts[pd.Timestamp("2025-12-31 21:00", tz="UTC")] == 0
    assert by_ts[pd.Timestamp("2025-12-31 22:00", tz="UTC")] == 1
    assert by_ts[pd.Timestamp("2026-01-01 21:00", tz="UTC")] == 1
    assert by_ts[pd.Timestamp("2026-01-01 22:00", tz="UTC")] == 0


def test_start_date_filters_in_local_time(cfg):
    cfg["data"]["start"] = "2026-01-01"  # 2025-12-31 22:00 UTC
    long = data.to_long(data._hourly(hourly_frame()), cfg)
    assert long["timestamp"].min() == pd.Timestamp("2025-12-31 22:00", tz="UTC")


def test_read_csv_hourly_file(cfg, tmp_path):
    h = hourly_frame()
    out = h.assign(ts_hour=h["ts_hour"].dt.strftime("%Y-%m-%dT%H:%MZ"))
    out.to_csv(tmp_path / "hourly.csv", index=False)
    got = data.read_hourly(cfg, "csv")
    pd.testing.assert_frame_equal(got, data._hourly(h))


def test_read_csv_reports_missing_columns(cfg, tmp_path):
    pd.DataFrame({"ts_hour": ["2026-01-01T00:00Z"], "load_mw": [1.0]}).to_csv(
        tmp_path / "hourly.csv", index=False
    )
    with pytest.raises(data.DataSourceError, match="tso_forecast_mw"):
        data.read_hourly(cfg, "csv")


def test_read_pipeline_observations_folder_averages_to_hours(cfg, tmp_path):
    folder = tmp_path / "observations" / "LT"
    q = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    for series, values in [
        ("load_actual", [100, 200, 300, 400, 10, 20, 30, 40]),
        ("load_forecast_tso", [1, 1, 1, 1, 2, 2, 2, 2]),
    ]:
        (folder / series).mkdir(parents=True)
        pd.DataFrame(
            {"ts_utc": q.strftime("%Y-%m-%dT%H:%MZ"), "resolution_min": 15, "value": values}
        ).to_csv(folder / series / "2026.csv.gz", index=False)
    cfg["data"]["csv_path"] = str(folder)
    got = data.read_hourly(cfg, "csv")
    assert got["load_mw"].tolist() == [250.0, 25.0]
    assert got["tso_forecast_mw"].tolist() == [1.0, 2.0]


def test_read_postgres_table(cfg, tmp_path, monkeypatch):
    """The SQL is plain enough to check against SQLite with the same table."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    url = f"sqlite:///{tmp_path / 'mart.db'}"
    h = hourly_frame().assign(zone="LT")
    other = hourly_frame().assign(zone="LV", load_mw=1.0)
    engine = sqlalchemy.create_engine(url)
    pd.concat([h, other]).assign(
        ts_hour=lambda d: d["ts_hour"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    ).to_sql("mart_hourly", engine, index=False)
    engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    got = data.read_hourly(cfg, "postgres")
    pd.testing.assert_frame_equal(got, data._hourly(h))


def test_postgres_errors_are_clear(cfg, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(data.DataSourceError, match="DATABASE_URL is not set"):
        data.read_hourly(cfg, "postgres")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///does/not/exist/x.db")
    with pytest.raises(data.DataSourceError, match="could not read mart_hourly"):
        data.read_hourly(cfg, "postgres")
    with pytest.raises(data.DataSourceError, match="unknown data source"):
        data.read_hourly(cfg, "excel")


def test_committed_sample_is_valid():
    """Streamlit Community Cloud and CI rely on this file."""
    c = load_config()
    h = data.read_hourly(c, "csv")
    assert len(h) > 365 * 24
    assert h["ts_hour"].dt.tz is not None
    long = data.load(c, "csv")
    assert long[TARGET].notna().mean() > 0.99
    assert long["tso_forecast"].notna().mean() > 0.99
    assert long[TARGET].between(100, 5000).all()

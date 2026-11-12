from functools import partial

import numpy as np
import pandas as pd

from energy_forecast.config import load_config
from energy_forecast.evaluate import backtest, mae, rmse, rolling_origin_splits, summarise
from energy_forecast.features import build_features
from energy_forecast.generate import emission_factor, generate
from energy_forecast.models import make_models


def small_config():
    cfg = load_config()
    cfg["days"] = 90
    cfg["xgboost"]["n_estimators"] = 30
    return cfg


def test_generate_shape_and_values():
    cfg = small_config()
    df = generate(cfg)
    assert len(df) == cfg["days"] * 24 * len(cfg["sectors"])
    assert (df["energy_kwh"] >= 0).all()
    assert df["is_anomaly"].sum() > 0


def test_generate_is_reproducible():
    cfg = small_config()
    pd.testing.assert_frame_equal(generate(cfg), generate(cfg))


def test_emission_factor_by_hour():
    cfg = load_config()
    ef = emission_factor(np.array([3, 8, 12, 20]), cfg)
    expected = cfg["emission_factor_g_per_kwh"]
    assert list(ef) == [
        expected["night"],
        expected["default"],
        expected["midday"],
        expected["default"],
    ]


def test_metrics():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 6.0])
    assert mae(y, p) == 1.0
    assert rmse(y, p) == np.sqrt(3.0)


def test_splits_are_consecutive_and_end_at_last_hour():
    ts = pd.Series(pd.date_range("2024-01-01", periods=24 * 100, freq="h"))
    splits = list(rolling_origin_splits(ts, 3, 7))
    assert len(splits) == 3
    for (_, end), (start, _) in zip(splits, splits[1:], strict=False):
        assert end == start
    assert splits[-1][1] == ts.max() + pd.Timedelta(hours=1)


def test_backtest_trains_only_on_the_past():
    cfg = small_config()
    feats = build_features(generate(cfg), cfg["horizon_hours"])
    metrics, preds = backtest(feats, partial(make_models, cfg), 2, 7)
    assert set(metrics["model"]) == {
        "Naive (yesterday)",
        "Naive (last week)",
        "Ridge regression",
        "XGBoost",
    }
    assert summarise(metrics)["skill_vs_naive"].notna().all()
    assert preds["timestamp"].min() >= feats["timestamp"].max() - pd.Timedelta(days=14)

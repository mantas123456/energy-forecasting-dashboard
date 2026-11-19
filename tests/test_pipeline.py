import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
import yaml
from synthetic import emission_factor, generate

from energy_forecast.config import load_config
from energy_forecast.evaluate import (
    backtest,
    mae,
    mape,
    rmse,
    rolling_origin_splits,
    summarise,
)
from energy_forecast.features import TARGET, build_features
from energy_forecast.models import TSOForecast, make_models, naive_name
from energy_forecast.pipeline import run


def test_generate_shape_and_values(synthetic_cfg, synthetic_data):
    cfg = synthetic_cfg
    assert len(synthetic_data) == cfg["days"] * 24 * len(cfg["sectors"])
    assert (synthetic_data[TARGET] >= 0).all()
    assert synthetic_data["is_anomaly"].sum() > 0


def test_generate_is_reproducible(synthetic_cfg):
    pd.testing.assert_frame_equal(generate(synthetic_cfg), generate(synthetic_cfg))


def test_emission_factor_by_hour(synthetic_cfg):
    ef = emission_factor(np.array([3, 8, 12, 20]), synthetic_cfg)
    expected = synthetic_cfg["emission_factor_g_per_kwh"]
    assert list(ef) == [
        expected["night"],
        expected["default"],
        expected["midday"],
        expected["default"],
    ]


def test_metrics():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 7.0])
    assert mae(y, p) == 1.0
    assert rmse(y, p) == np.sqrt(3.0)
    assert mape(y, p) == pytest.approx(25.0)


def test_splits_are_consecutive_and_end_at_last_hour():
    ts = pd.Series(pd.date_range("2024-01-01", periods=24 * 100, freq="h"))
    splits = list(rolling_origin_splits(ts, 3, 7))
    assert len(splits) == 3
    for (_, end), (start, _) in zip(splits, splits[1:], strict=False):
        assert end == start
    assert splits[-1][1] == ts.max() + pd.Timedelta(hours=1)


def test_backtest_trains_only_on_the_past(synthetic_cfg, synthetic_data):
    cfg = synthetic_cfg
    feats = build_features(synthetic_data, cfg["horizon_hours"])
    metrics, preds = backtest(feats, partial(make_models, cfg), 2, 7)
    assert set(metrics["model"]) == {
        "Naive (yesterday)",
        "Naive (last week)",
        "Ridge regression",
        "XGBoost",
    }
    assert "skill_vs_tso" not in metrics  # no TSO forecast in the synthetic data
    assert summarise(metrics)["skill_vs_naive"].notna().all()
    assert preds["timestamp"].min() >= feats["timestamp"].max() - pd.Timedelta(days=14)


def test_naive_names_and_horizon_check(synthetic_cfg):
    assert naive_name(24) == "Naive (yesterday)"
    assert naive_name(48) == "Naive (2 days ago)"
    assert naive_name(168) == "Naive (last week)"
    cfg = dict(synthetic_cfg, horizon_hours=48)
    assert [m.name for m in make_models(cfg, tso=True)][0] == "Naive (2 days ago)"
    assert make_models(cfg, tso=True)[-1].name == TSOForecast.name
    with pytest.raises(ValueError):
        make_models(dict(synthetic_cfg, horizon_hours=36))


def with_tso(data, factor=1.0):
    return data.assign(tso_forecast=data[TARGET] * factor)


def test_tso_reference_is_read_not_fitted_and_skill_is_relative_to_it(
    synthetic_cfg, synthetic_data
):
    cfg = synthetic_cfg
    one = synthetic_data[synthetic_data["sector"] == "School"]
    feats = build_features(with_tso(one, 1.10), cfg["horizon_hours"])  # always 10 % high
    metrics, preds = backtest(feats, partial(make_models, cfg, tso=True), 2, 7)
    tso = metrics[metrics["model"] == TSOForecast.name]
    assert (tso["kind"] == "reference").all()
    assert tso["mape_pct"].to_numpy() == pytest.approx([10.0, 10.0])
    assert tso["skill_vs_tso"].to_numpy() == pytest.approx([0.0, 0.0])
    xgb = metrics[metrics["model"] == "XGBoost"]
    np.testing.assert_allclose(
        xgb["skill_vs_tso"], 1 - xgb["mae"].to_numpy() / tso["mae"].to_numpy()
    )
    p = preds[preds["model"] == TSOForecast.name]
    np.testing.assert_allclose(p["predicted"], (p["actual"] * 1.10).round(3), atol=1e-9)


def test_models_scored_on_the_same_hours_when_tso_has_gaps(synthetic_cfg, synthetic_data):
    cfg = synthetic_cfg
    one = with_tso(synthetic_data[synthetic_data["sector"] == "Factory"])
    one.loc[one.index[-30:-10], "tso_forecast"] = np.nan  # 20 hours missing in the last week
    feats = build_features(one, cfg["horizon_hours"])
    assert feats["timestamp"].max() == one["timestamp"].max()  # rows kept despite the gap
    metrics, _ = backtest(feats, partial(make_models, cfg, tso=True), 2, 7)
    last = metrics[metrics["fold"] == 2]
    assert (last["hours"] == 7 * 24 - 20).all()
    assert last.loc[last["model"] == TSOForecast.name, "mae"].iloc[0] == pytest.approx(0.0)


def test_summary_counts_windows_better_than_tso():
    rows = []
    for fold, (model_mae, tso_mae) in enumerate([(5, 10), (12, 10), (8, 10)], start=1):
        for name, kind, value in [("XGBoost", "model", model_mae), ("TSO", "reference", tso_mae)]:
            rows.append(
                dict(
                    sector="LT",
                    fold=fold,
                    model=name,
                    kind=kind,
                    mae=value,
                    rmse=value,
                    mape_pct=value,
                    skill_vs_naive=0.0,
                    skill_vs_tso=1 - value / tso_mae,
                )
            )
    s = summarise(pd.DataFrame(rows)).set_index("model")
    assert s.loc["XGBoost", "folds_better_than_tso"] == 2
    assert s.loc["XGBoost", "folds"] == 3
    assert s.loc["XGBoost", "skill_vs_tso"] == pytest.approx((0.5 - 0.2 + 0.2) / 3, abs=1e-3)


def test_run_on_committed_sample(tmp_path):
    """End to end on the real committed sample, with a short backtest."""
    cfg = load_config()
    cfg["backtest"] = {"n_splits": 2, "test_days": 7}
    cfg["xgboost"]["n_estimators"] = 20
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    summary = run(str(path), str(tmp_path / "out"), source="csv")
    assert TSOForecast.name in set(summary["model"])
    assert "Naive (2 days ago)" in set(summary["model"])
    for f in [
        "metrics_by_fold.csv",
        "metrics_summary.csv",
        "predictions.csv.gz",
        "mape_by_fold.png",
        "example_week.png",
        "run_info.json",
    ]:
        assert (tmp_path / "out" / f).exists(), f
    info = json.loads((tmp_path / "out" / "run_info.json").read_text())
    assert info["zone"] == "LT" and info["source"] == "csv" and info["folds"] == 2

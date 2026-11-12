import numpy as np
import pandas as pd
import pytest

from energy_forecast.features import TARGET, build_features, feature_columns


def make_frame(n_hours=24 * 30, sectors=("A", "B")):
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    frames = []
    for k, s in enumerate(sectors):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "sector": s,
                    # Distinct, strictly increasing values per sector make leaks easy to spot.
                    TARGET: np.arange(n_hours, dtype=float) + 100_000 * k,
                    "temperature_c": 20.0,
                    "holiday": 0,
                    "is_anomaly": 0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("horizon", [1, 24])
def test_target_features_only_use_values_at_least_horizon_old(horizon):
    feats = build_features(make_frame(), horizon)
    target_cols = [c for c in feature_columns(feats) if c.startswith(("lag_", "roll_mean"))]
    # The target increases by 1 per hour, so any feature built from values at
    # least `horizon` hours old must be <= target - horizon.
    for col in target_cols:
        assert (feats[col] <= feats[TARGET] - horizon).all(), col


def test_lags_do_not_cross_sectors():
    feats = build_features(make_frame(), 24)
    b = feats[feats["sector"] == "B"]
    assert (b["lag_24h"] >= 100_000).all()


def test_rows_without_history_are_dropped():
    feats = build_features(make_frame(), 24)
    assert not feats.isna().any().any()
    assert feats.groupby("sector")["timestamp"].min().nunique() == 1

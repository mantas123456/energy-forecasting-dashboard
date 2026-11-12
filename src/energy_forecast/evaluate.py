"""Rolling-origin backtest and error metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from .features import TARGET, feature_columns
from .models import Forecaster


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rolling_origin_splits(
    timestamps: pd.Series, n_splits: int, test_days: int
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (test_start, test_end) windows covering the end of the series.

    For each window the model trains on everything before test_start, as it
    would in real use, and is scored on the following `test_days` days.
    """
    end = timestamps.max() + pd.Timedelta(hours=1)
    for k in range(n_splits, 0, -1):
        start = end - pd.Timedelta(days=test_days * k)
        yield start, start + pd.Timedelta(days=test_days)


def backtest(
    features: pd.DataFrame,
    model_factory: Callable[[], list[Forecaster]],
    n_splits: int,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and score every model per sector and fold.

    Returns (metrics, predictions). Metrics include skill against the better
    of the two naive baselines in each fold: 1 - MAE_model / MAE_naive
    (above 0 means the model beats the baseline).
    """
    cols = feature_columns(features)
    rows, preds = [], []
    for sector, g in features.groupby("sector"):
        for fold, (start, end) in enumerate(
            rolling_origin_splits(g["timestamp"], n_splits, test_days), start=1
        ):
            train = g[g["timestamp"] < start]
            test = g[(g["timestamp"] >= start) & (g["timestamp"] < end)]
            y_test = test[TARGET].to_numpy()
            for model in model_factory():
                model.fit(train[cols], train[TARGET])
                p = model.predict(test[cols])
                rows.append(
                    {
                        "sector": sector,
                        "fold": fold,
                        "model": model.name,
                        "mae": mae(y_test, p),
                        "rmse": rmse(y_test, p),
                    }
                )
                preds.append(
                    pd.DataFrame(
                        {
                            "timestamp": test["timestamp"].to_numpy(),
                            "sector": sector,
                            "fold": fold,
                            "model": model.name,
                            "actual": y_test,
                            "predicted": p.round(3),
                        }
                    )
                )

    metrics = pd.DataFrame(rows)
    naive = (
        metrics[metrics["model"].str.startswith("Naive")]
        .groupby(["sector", "fold"], as_index=False)["mae"]
        .min()
    )
    metrics = metrics.merge(naive.rename(columns={"mae": "naive_mae"}), on=["sector", "fold"])
    metrics["skill_vs_naive"] = 1 - metrics["mae"] / metrics["naive_mae"]
    return metrics.drop(columns="naive_mae"), pd.concat(preds, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per sector and model across folds."""
    return (
        metrics.groupby(["sector", "model"], as_index=False)[["mae", "rmse", "skill_vs_naive"]]
        .mean()
        .round(3)
        .sort_values(["sector", "mae"])
    )

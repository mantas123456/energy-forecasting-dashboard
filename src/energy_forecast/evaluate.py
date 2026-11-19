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


def mape(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute percentage error over hours with non-zero actual demand."""
    nz = y != 0
    return float(np.mean(np.abs(y[nz] - p[nz]) / np.abs(y[nz])) * 100) if nz.any() else np.nan


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

    All models in a fold are scored on the same hours: those where every model has a
    forecast (the TSO forecast can have gaps). Returns (metrics, predictions). Metrics
    include skill against the better naive baseline in each fold,
    1 - MAE_model / MAE_naive, and, when the TSO forecast is one of the models, skill
    against it, 1 - MAE_model / MAE_TSO. Above 0 means the model did better.
    """
    cols = feature_columns(features)
    rows, preds = [], []
    for sector, g in features.groupby("sector"):
        for fold, (start, end) in enumerate(
            rolling_origin_splits(g["timestamp"], n_splits, test_days), start=1
        ):
            train = g[g["timestamp"] < start]
            test = g[(g["timestamp"] >= start) & (g["timestamp"] < end)]
            if train.empty or test.empty:
                continue
            y_test = test[TARGET].to_numpy(dtype=float)
            fold_preds: dict[str, np.ndarray] = {}
            kinds: dict[str, str] = {}
            for model in model_factory():
                kind = getattr(model, "kind", "model")
                model.fit(train[cols], train[TARGET])
                # A reference forecast reads its own column; models see only features.
                p = model.predict(test if kind == "reference" else test[cols])
                fold_preds[model.name] = np.asarray(p, dtype=float)
                kinds[model.name] = kind
            ok = np.all([np.isfinite(p) for p in fold_preds.values()], axis=0)
            for name, p in fold_preds.items():
                rows.append(
                    {
                        "sector": sector,
                        "fold": fold,
                        "test_start": start,
                        "test_end": end,
                        "model": name,
                        "kind": kinds[name],
                        "hours": int(ok.sum()),
                        "mae": mae(y_test[ok], p[ok]),
                        "rmse": rmse(y_test[ok], p[ok]),
                        "mape_pct": mape(y_test[ok], p[ok]),
                    }
                )
                preds.append(
                    pd.DataFrame(
                        {
                            "timestamp": test["timestamp"].to_numpy(),
                            "sector": sector,
                            "fold": fold,
                            "model": name,
                            "actual": y_test,
                            "predicted": p.round(3),
                        }
                    )
                )

    metrics = pd.DataFrame(rows)
    keys = ["sector", "fold"]
    naive = metrics[metrics["kind"] == "naive"].groupby(keys, as_index=False)["mae"].min()
    metrics = metrics.merge(naive.rename(columns={"mae": "naive_mae"}), on=keys)
    metrics["skill_vs_naive"] = 1 - metrics["mae"] / metrics["naive_mae"]
    metrics = metrics.drop(columns="naive_mae")
    ref = metrics[metrics["kind"] == "reference"][[*keys, "mae"]]
    if not ref.empty:
        metrics = metrics.merge(ref.rename(columns={"mae": "tso_mae"}), on=keys)
        metrics["skill_vs_tso"] = 1 - metrics["mae"] / metrics["tso_mae"]
        metrics = metrics.drop(columns="tso_mae")
    return metrics, pd.concat(preds, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per sector and model across folds.

    With a TSO reference, `folds_better_than_tso` counts the folds in which the model's
    MAE was lower than the TSO forecast's, out of `folds`.
    """
    value_cols = ["mae", "rmse", "mape_pct", "skill_vs_naive"]
    has_tso = "skill_vs_tso" in metrics
    if has_tso:
        value_cols.append("skill_vs_tso")
    g = metrics.groupby(["sector", "model"], as_index=False)
    out = g[value_cols].mean()
    out.insert(2, "folds", g["fold"].count()["fold"])
    if has_tso:
        out["folds_better_than_tso"] = g["skill_vs_tso"].agg(lambda s: int((s > 0).sum()))[
            "skill_vs_tso"
        ]
    return out.round(3).sort_values(["sector", "mae"]).reset_index(drop=True)

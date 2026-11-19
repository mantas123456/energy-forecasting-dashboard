"""Run the backtest on Lithuanian hourly load and save the results.

Usage:
    energy-forecast [--config config/config.yaml] [--source csv|postgres] [--out results]
    energy-forecast --source postgres --export data/lt_hourly.csv.gz

The second form refreshes the committed CSV sample from the energy-data-pipeline database.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import data as data_mod  # noqa: E402
from .config import load_config  # noqa: E402
from .evaluate import backtest, summarise  # noqa: E402
from .features import REFERENCE, build_features  # noqa: E402
from .models import TSOForecast, make_models  # noqa: E402

log = logging.getLogger("energy_forecast")
UNIT = "MWh"  # hourly energy; equal to the hour's average load in MW
STYLES = {
    "XGBoost": dict(color="#1f77b4", marker="o"),
    "Ridge regression": dict(color="#9467bd", marker="s"),
    "Naive (last week)": dict(color="#7f7f7f", marker="^", linestyle=":"),
    TSOForecast.name: dict(color="#d62728", marker="D", linestyle="--"),
}


def plot_mape_by_fold(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, style in STYLES.items():
        m = metrics[metrics["model"] == model].sort_values("fold")
        if m.empty:
            continue
        ax.plot(pd.to_datetime(m["test_start"]), m["mape_pct"], label=model, **style)
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Day-ahead load forecast error per 28-day test window (Lithuania)", loc="left", fontsize=11
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example(preds: pd.DataFrame, path: Path, tz: str) -> None:
    last = preds[preds["fold"] == preds["fold"].max()]
    week = last[last["timestamp"] >= last["timestamp"].max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    actual = week[week["model"] == "XGBoost"]
    ts = pd.to_datetime(actual["timestamp"], utc=True).dt.tz_convert(tz)
    ax.plot(ts, actual["actual"], color="black", linewidth=1.4, label="Actual")
    for m in ["XGBoost", TSOForecast.name]:
        d = week[week["model"] == m]
        if not d.empty:
            t = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(tz)
            ax.plot(
                t,
                d["predicted"],
                linewidth=1.1,
                label=m,
                linestyle=STYLES[m].get("linestyle", "-"),
                color=STYLES[m]["color"],
            )
    ax.set_ylabel("Load (MW)")
    ax.set_title("Lithuania: last week of the backtest", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(config_path: str | None, out_dir: str, source: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    d = cfg["data"]
    source = source or d["source"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = data_mod.load(cfg, source)
    log.info(
        "Loaded %d hours for %s from %s (%s to %s UTC), %d without actual load",
        len(data),
        d["zone"],
        source,
        f"{data['timestamp'].min():%Y-%m-%d %H:%M}",
        f"{data['timestamp'].max():%Y-%m-%d %H:%M}",
        int(data["demand"].isna().sum()),
    )
    feats = build_features(data, cfg["horizon_hours"], d["timezone"])
    has_tso = feats[REFERENCE].notna().any()
    log.info("Built %d feature rows; TSO forecast %s", len(feats), "found" if has_tso else "absent")

    bt = cfg["backtest"]
    factory = partial(make_models, cfg, tso=bool(has_tso))
    metrics, preds = backtest(feats, factory, bt["n_splits"], bt["test_days"])
    summary = summarise(metrics)
    metrics.to_csv(out / "metrics_by_fold.csv", index=False, float_format="%.4f")
    summary.to_csv(out / "metrics_summary.csv", index=False)
    preds.to_csv(out / "predictions.csv.gz", index=False)
    info = {
        "zone": d["zone"],
        "source": source,
        "unit": UNIT,
        "timezone": d["timezone"],
        "horizon_hours": cfg["horizon_hours"],
        "data_start_utc": f"{data['timestamp'].min():%Y-%m-%d %H:%M}",
        "data_end_utc": f"{data['timestamp'].max():%Y-%m-%d %H:%M}",
        "hours": int(len(data)),
        "folds": int(metrics["fold"].nunique()),
        "test_days": bt["test_days"],
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    plot_mape_by_fold(metrics, out / "mape_by_fold.png")
    plot_example(preds, out / "example_week.png", d["timezone"])
    log.info("Results written to %s\n%s", out, summary.to_string(index=False))
    return summary


def export(config_path: str | None, path: str, source: str | None = None) -> Path:
    """Write the hourly table (from `start` on) as a CSV, for the committed sample."""
    cfg = load_config(config_path)
    hourly = data_mod.read_hourly(cfg, source)
    start = pd.Timestamp(cfg["data"]["start"], tz=cfg["data"]["timezone"])
    hourly = hourly[hourly["ts_hour"] >= start].copy()
    hourly["ts_hour"] = hourly["ts_hour"].dt.strftime("%Y-%m-%dT%H:%MZ")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.round(2).to_csv(out, index=False)
    log.info("Wrote %d hours to %s", len(hourly), out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="path to YAML config")
    ap.add_argument("--source", choices=["csv", "postgres"], help="override data.source")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--export", metavar="PATH", help="write the hourly data to PATH and stop")
    args = ap.parse_args()
    try:
        if args.export:
            export(args.config, args.export, args.source)
        else:
            run(args.config, args.out, args.source)
    except data_mod.DataSourceError as e:
        raise SystemExit(f"Data source error: {e}") from None


if __name__ == "__main__":
    main()

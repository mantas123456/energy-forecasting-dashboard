"""Run the full pipeline: generate data, build features, backtest, save results.

Usage:
    python -m energy_forecast.pipeline [--config config/config.yaml] [--out results]
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_config  # noqa: E402
from .evaluate import backtest, summarise  # noqa: E402
from .features import build_features  # noqa: E402
from .generate import generate  # noqa: E402
from .models import make_models  # noqa: E402

log = logging.getLogger("energy_forecast")


def plot_skill(summary, path: Path) -> None:
    models = [m for m in summary["model"].unique() if not m.startswith("Naive")]
    sectors = list(summary["sector"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [
            summary.query("sector == @s and model == @m")["skill_vs_naive"].iloc[0] for s in sectors
        ]
        ax.bar(
            [x + i * width for x in range(len(sectors))], [v * 100 for v in vals], width, label=m
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(sectors))], sectors)
    ax.set_ylabel("MAE reduction vs best naive (%)")
    ax.set_title("Day-ahead forecast skill by sector (5-fold rolling backtest)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example(preds, sector: str, path: Path) -> None:
    last = preds[(preds["sector"] == sector) & (preds["fold"] == preds["fold"].max())]
    first_week = last[last["timestamp"] < last["timestamp"].min() + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    actual = first_week[first_week["model"] == "XGBoost"]
    ax.plot(actual["timestamp"], actual["actual"], color="black", linewidth=1.4, label="Actual")
    for m, style in [("XGBoost", "-"), ("Naive (last week)", ":")]:
        d = first_week[first_week["model"] == m]
        ax.plot(d["timestamp"], d["predicted"], style, linewidth=1.2, label=m)
    ax.set_ylabel("Energy (kWh per hour)")
    ax.set_title(f"{sector}: one week of day-ahead forecasts", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(config_path: str | None, out_dir: str) -> None:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d days of hourly data for %d sectors", cfg["days"], len(cfg["sectors"]))
    data = generate(cfg)
    data.to_csv(out / "synthetic_data.csv.gz", index=False)

    feats = build_features(data, cfg["horizon_hours"])
    log.info("Built %d feature rows", len(feats))

    bt = cfg["backtest"]
    metrics, preds = backtest(feats, partial(make_models, cfg), bt["n_splits"], bt["test_days"])
    summary = summarise(metrics)
    metrics.round(4).to_csv(out / "metrics_by_fold.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    preds.to_csv(out / "predictions.csv.gz", index=False)

    plot_skill(summary, out / "skill_by_sector.png")
    plot_example(preds, "School", out / "example_school_week.png")
    log.info("Results written to %s\n%s", out, summary.to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="path to YAML config")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()

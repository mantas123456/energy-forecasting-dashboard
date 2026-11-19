"""Streamlit dashboard for the backtest results.

Run from the repository root:
    streamlit run app/dashboard.py

The results tab reads the committed files in results/. The data tab reads the
energy-data-pipeline database when DATABASE_URL is set (environment or Streamlit
secrets) and reachable, and the committed sample data/lt_hourly.csv.gz otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_forecast import data as data_mod  # noqa: E402
from energy_forecast.config import load_config  # noqa: E402
from energy_forecast.evaluate import backtest, summarise  # noqa: E402
from energy_forecast.features import TARGET, build_features  # noqa: E402
from energy_forecast.models import TSOForecast, make_models  # noqa: E402

RESULTS = ROOT / "results"
CFG = load_config()
TZ = CFG["data"]["timezone"]

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
st.title("Day-ahead electricity load forecasting: Lithuania")


@st.cache_data
def load_results():
    info = json.loads((RESULTS / "run_info.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(RESULTS / "predictions.csv.gz")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert(TZ)
    return (
        pd.read_csv(RESULTS / "metrics_summary.csv"),
        pd.read_csv(RESULTS / "metrics_by_fold.csv"),
        preds,
        info,
    )


def database_url() -> str:
    url = os.environ.get(CFG["data"]["database_url_env"], "")
    if not url:
        try:
            url = st.secrets.get(CFG["data"]["database_url_env"], "")
        except Exception:  # no secrets file
            url = ""
    return url


@st.cache_data(ttl=3600)
def load_data() -> tuple[pd.DataFrame, str]:
    """Hourly data from the database if reachable, else the committed sample."""
    url = database_url()
    if url:
        os.environ.setdefault(CFG["data"]["database_url_env"], url)
        try:
            return data_mod.read_hourly(CFG, "postgres"), "the energy-data-pipeline database"
        except data_mod.DataSourceError:
            pass
    return data_mod.read_hourly(CFG, "csv"), f"the committed sample ({CFG['data']['csv_path']})"


summary, by_fold, preds, info = load_results()
st.caption(
    f"Hourly load from the ENTSO-E Transparency Platform via energy-data-pipeline, "
    f"{info['data_start_utc'][:10]} to {info['data_end_utc'][:10]}. Models forecast "
    f"{info['horizon_hours']} hours ahead and are scored in a {info['folds']}-window rolling "
    f"backtest ({info['test_days']} days each) against naive baselines and the TSO's own "
    "day-ahead forecast."
)
tab_results, tab_data, tab_upload = st.tabs(
    ["Model results", "Explore the data", "Try your own data"]
)

with tab_results:
    s = summary.copy()
    for c in ["skill_vs_naive", "skill_vs_tso"]:
        if c in s:
            s[c] = (s[c] * 100).round(1)
    st.dataframe(
        s.drop(columns="sector").rename(
            columns={
                "model": "Method",
                "folds": "Windows",
                "mae": "MAE (MWh)",
                "rmse": "RMSE (MWh)",
                "mape_pct": "MAPE (%)",
                "skill_vs_naive": "Skill vs best naive (%)",
                "skill_vs_tso": "Skill vs TSO (%)",
                "folds_better_than_tso": "Windows better than TSO",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Means over all test windows. Skill is the mean of 1 - MAE / MAE of the reference "
        "in each window; above 0 means better than the reference."
    )
    w = by_fold.copy()
    w["window"] = pd.to_datetime(w["test_start"], utc=True).dt.tz_convert(TZ).dt.date
    st.plotly_chart(
        px.line(
            w[w["kind"] != "naive"],
            x="window",
            y="mape_pct",
            color="model",
            markers=True,
            labels={"window": "Test window starting", "mape_pct": "MAPE (%)", "model": ""},
            title="Error per test window",
        ),
        width="stretch",
    )

    folds = sorted(preds["fold"].unique())
    fold = st.select_slider("Test window", folds, value=folds[-1])
    default = [m for m in ["XGBoost", TSOForecast.name] if m in set(preds["model"])]
    models = st.multiselect("Forecasts", sorted(preds["model"].unique()), default=default)
    p = preds[preds["fold"] == fold]
    actual = (
        p.drop_duplicates("timestamp")[["timestamp", "actual"]]
        .rename(columns={"actual": "MW"})
        .assign(series="Actual")
    )
    pred = p[p["model"].isin(models)][["timestamp", "predicted", "model"]].rename(
        columns={"predicted": "MW", "model": "series"}
    )
    fig = px.line(
        pd.concat([actual, pred]),
        x="timestamp",
        y="MW",
        color="series",
        labels={"timestamp": "", "MW": "Load (MW)", "series": ""},
    )
    st.plotly_chart(fig, width="stretch")

with tab_data:
    hourly, source = load_data()
    st.caption(f"Data from {source}.")
    local = hourly.assign(ts=hourly["ts_hour"].dt.tz_convert(TZ)).set_index("ts")
    daily = (
        local[["load_mw", "tso_forecast_mw"]]
        .resample("D")
        .sum(min_count=20)
        .div(1000)
        .rename(columns={"load_mw": "Actual", "tso_forecast_mw": "TSO day-ahead forecast"})
        .reset_index()
        .melt(id_vars="ts", var_name="series", value_name="GWh")
    )
    st.plotly_chart(
        px.line(
            daily,
            x="ts",
            y="GWh",
            color="series",
            labels={"ts": "", "GWh": "Consumption (GWh/day)", "series": ""},
        ),
        width="stretch",
    )
    missing = int(hourly["load_mw"].isna().sum())
    st.caption(
        f"{len(hourly):,} hours, latest actual load "
        f"{hourly.loc[hourly['load_mw'].notna(), 'ts_hour'].max().tz_convert(TZ):%Y-%m-%d %H:%M} "
        f"local time; {missing} hours without actual load (mostly tomorrow's TSO forecast)."
    )

with tab_upload:
    st.write(
        "Upload an hourly CSV with columns `timestamp` and `demand` "
        "(optionally `temperature_c`). At least 120 days are needed for the backtest. "
        "The models forecast "
        f"{CFG['horizon_hours']} hours ahead and are compared with the naive baselines."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file, parse_dates=["timestamp"])
        if TARGET not in df and "energy_kwh" in df:  # the format used before v4
            df = df.rename(columns={"energy_kwh": TARGET})
        missing = {"timestamp", TARGET} - set(df.columns)
        if missing:
            st.error(f"The file is missing these columns: {', '.join(sorted(missing))}")
        else:
            df = df.assign(sector="Uploaded", holiday=0, is_anomaly=0)
            feats = build_features(df, CFG["horizon_hours"])
            if feats.empty or feats["timestamp"].max() - feats["timestamp"].min() < pd.Timedelta(
                days=120
            ):
                st.error("Not enough history: the backtest needs at least 120 days.")
            else:
                with st.spinner("Running the backtest"):
                    m, _ = backtest(feats, partial(make_models, CFG), 3, 14)
                st.dataframe(summarise(m), hide_index=True, width="stretch")

"""Streamlit dashboard for the backtest results.

Run from the repository root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_forecast.config import load_config  # noqa: E402
from energy_forecast.evaluate import backtest, summarise  # noqa: E402
from energy_forecast.features import build_features  # noqa: E402
from energy_forecast.models import make_models  # noqa: E402

RESULTS = ROOT / "results"

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
st.title("Day-ahead energy forecasting")
st.caption(
    "Synthetic hourly demand for four sectors. Models are scored in a 5-fold "
    "rolling backtest against two naive baselines."
)


@st.cache_data
def load_results():
    return (
        pd.read_csv(RESULTS / "metrics_summary.csv"),
        pd.read_csv(RESULTS / "predictions.csv.gz", parse_dates=["timestamp"]),
        pd.read_csv(RESULTS / "synthetic_data.csv.gz", parse_dates=["timestamp"]),
    )


summary, preds, data = load_results()
tab_results, tab_data, tab_upload = st.tabs(
    ["Model results", "Explore the data", "Try your own data"]
)

with tab_results:
    sector = st.selectbox("Sector", sorted(summary["sector"].unique()))
    s = summary[summary["sector"] == sector].copy()
    s["skill_vs_naive"] = (s["skill_vs_naive"] * 100).round(1)
    st.dataframe(
        s.rename(
            columns={
                "mae": "MAE (kWh)",
                "rmse": "RMSE (kWh)",
                "skill_vs_naive": "Skill vs best naive (%)",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    folds = sorted(preds["fold"].unique())
    fold = st.select_slider("Backtest fold (28-day test window)", folds, value=folds[-1])
    models = st.multiselect(
        "Models", sorted(preds["model"].unique()), default=["XGBoost", "Naive (last week)"]
    )
    p = preds[(preds["sector"] == sector) & (preds["fold"] == fold)]
    actual = (
        p.drop_duplicates("timestamp")[["timestamp", "actual"]]
        .rename(columns={"actual": "kWh"})
        .assign(series="Actual")
    )
    pred = p[p["model"].isin(models)][["timestamp", "predicted", "model"]].rename(
        columns={"predicted": "kWh", "model": "series"}
    )
    fig = px.line(
        pd.concat([actual, pred]),
        x="timestamp",
        y="kWh",
        color="series",
        labels={"timestamp": "", "kWh": "Energy (kWh per hour)", "series": ""},
    )
    st.plotly_chart(fig, width="stretch")

with tab_data:
    sectors = st.multiselect(
        "Sectors", sorted(data["sector"].unique()), default=sorted(data["sector"].unique())
    )
    daily = (
        data[data["sector"].isin(sectors)]
        .set_index("timestamp")
        .groupby("sector")
        .resample("D")[["energy_kwh", "carbon_kgco2e"]]
        .sum()
        .reset_index()
    )
    metric = st.radio(
        "Show",
        ["energy_kwh", "carbon_kgco2e"],
        horizontal=True,
        format_func={
            "energy_kwh": "Energy (kWh/day)",
            "carbon_kgco2e": "Emissions (kg CO2e/day)",
        }.get,
    )
    st.plotly_chart(
        px.line(
            daily, x="timestamp", y=metric, color="sector", labels={"timestamp": "", metric: ""}
        ),
        width="stretch",
    )
    st.caption(f"{int(data['is_anomaly'].sum())} hours are labelled as injected anomalies.")

with tab_upload:
    st.write(
        "Upload an hourly CSV with columns `timestamp` and `energy_kwh` "
        "(optionally `temperature_c`). At least 120 days are needed for the backtest."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file, parse_dates=["timestamp"])
        missing = {"timestamp", "energy_kwh"} - set(df.columns)
        if missing:
            st.error(f"The file is missing these columns: {', '.join(sorted(missing))}")
        else:
            df = df.assign(sector="Uploaded", holiday=0, is_anomaly=0)
            if "temperature_c" not in df:
                df["temperature_c"] = 21.0
            cfg = load_config()
            feats = build_features(df, cfg["horizon_hours"])
            if feats["timestamp"].max() - feats["timestamp"].min() < pd.Timedelta(days=120):
                st.error("Not enough history: the backtest needs at least 120 days.")
            else:
                with st.spinner("Running the backtest"):
                    m, _ = backtest(feats, partial(make_models, cfg), 3, 14)
                st.dataframe(summarise(m), hide_index=True, width="stretch")

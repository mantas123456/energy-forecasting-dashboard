"""
dashboard.py — Energy Forecasting Dashboard (Tabs + RMSE)
Author: Mantas Valantinavičius
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# === 1. CONFIG ===
PREDICTIONS_DIR = "results/predictions"
EVAL_FILE = os.path.join(PREDICTIONS_DIR, "model_evaluation_summary.csv")
MODEL_FILES = {
    "Prophet": "predictions_prophet.csv",
    "XGBoost": "predictions_xgboost.csv",
    "Linear Regression": "predictions_linear_regression.csv"
}

# === 2. TITLE ===
st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
st.title("⚡ Energy Forecasting Dashboard")

# === 3. TABS ===
tab1, tab2 = st.tabs(["📊 Model Forecasts", "📤 Upload Your CSV"])

# ===================================
# 📊 TAB 1: Pretrained Model Forecasts
# ===================================
with tab1:
    st.subheader("📈 View Forecasts from Trained Models")

    selected_model = st.selectbox("Choose a model to display:", list(MODEL_FILES.keys()))
    pred_file = os.path.join(PREDICTIONS_DIR, MODEL_FILES[selected_model])

    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        if "timestamp" in df.columns:
            df["ds"] = pd.to_datetime(df["timestamp"])
        elif "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        else:
            st.warning("Missing 'timestamp' or 'ds' column in prediction file.")
            st.stop()

        fig = px.line(
            df,
            x="ds",
            y=["actual", "predicted"],
            labels={"ds": "Time", "value": "Energy (kWh)", "variable": "Legend"},
            title=f"{selected_model} Forecast",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No prediction file found for {selected_model}.")

    st.subheader("📊 Model Evaluation Summary")
    if os.path.exists(EVAL_FILE):
        eval_df = pd.read_csv(EVAL_FILE)
        st.dataframe(eval_df.set_index("Model"))
    else:
        st.warning("Evaluation summary CSV not found.")

# ===================================
# 📤 TAB 2: Upload and Forecast
# ===================================
with tab2:
    st.subheader("📥 Upload Your Own CSV for Forecasting")

    uploaded_file = st.file_uploader("Upload a CSV file with 'timestamp' and 'energy_kwh'", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)

        if "timestamp" not in user_df.columns or "energy_kwh" not in user_df.columns:
            st.error("Uploaded CSV must contain 'timestamp' and 'energy_kwh'.")
            st.stop()

        user_df["ds"] = pd.to_datetime(user_df["timestamp"])
        user_df["y"] = user_df["energy_kwh"]

        st.markdown("#### 🧪 Preview Uploaded Data")
        st.dataframe(user_df[["ds", "y"]].head())

        st.markdown("### 🔧 Forecast Settings")
        horizon_hours = st.slider("Select forecast horizon (in hours):", 6, 168, 24, step=6)

        # Forecast with Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(user_df[["ds", "y"]])
        future = model.make_future_dataframe(periods=horizon_hours, freq="H")
        forecast = model.predict(future)

        # Plot
        fig = px.line()
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast")
        fig.add_scatter(x=user_df["ds"], y=user_df["y"], name="Actual")
        fig.update_layout(
            title=f"Forecast on Uploaded Data (+{horizon_hours}h)",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Evaluate if actuals are available in forecast range
        merged = pd.merge(forecast, user_df, how="inner", on="ds")
        merged = merged.dropna(subset=["y", "yhat"])

        if not merged.empty:
            rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
            mae = mean_absolute_error(merged["y"], merged["yhat"])

            st.success(f"📉 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        else:
            st.info("⚠️ No overlapping actual data in forecast horizon for evaluation.")

        # Download Forecast CSV
        st.subheader("📄 Download Forecast")
        download_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        download_df = download_df.rename(columns={
            "ds": "timestamp",
            "yhat": "predicted_energy_kwh",
            "yhat_lower": "prediction_lower_bound",
            "yhat_upper": "prediction_upper_bound"
        })
        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download forecast as CSV",
            data=csv,
            file_name="uploaded_forecast_results.csv",
            mime="text/csv"
        )

# ⚡ Energy Forecasting Dashboard (v2.0)

A full-featured synthetic data pipeline and analytics dashboard for modeling and evaluating hourly energy consumption across multiple sectors. 
Built for analysis, validation, and deployment readiness.

---

## 🔍 Project Overview

This project simulates realistic energy consumption patterns using synthetic data and enables model training, evaluation, and visualization through a Streamlit dashboard. It supports multiple sectors, temperature influence, feature engineering, and per-sector model comparisons.

---

## 📦 Features in v2.0

### 🧪 Synthetic Data Generation
- Realistic hourly energy + temperature profiles
- Configurable sector templates (Residential, School, Factory, EV, etc.)
- Seasonal and diurnal temperature trends

### 🧠 Feature Engineering
- Time-based: hour, day of week, month
- Cyclical encoding (sin/cos)
- Lag and rolling window features
- One-hot encoded sector labels

### 🤖 Model Training (Per Sector)
- `XGBoost` and `Linear Regression`
- Trained and evaluated separately for each sector
- Stored predictions, models, and feature importances

### 📊 Evaluation Summary
- RMSE and MAE comparison table
- Highlighted best-performing model per sector
- Bar plot of RMSEs across models

### 📈 Interactive Dashboard (Streamlit)
- Sector filter and raw data preview
- Plots: energy vs time, temperature vs time, scatter, correlation heatmap
- Download filtered data or metrics

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Scripts
```bash
python scripts/00_generate_synthetic.py
python scripts/01_prepare_input.py
python scripts/02_feature_engineering.py
python scripts/04_train_xgboost.py
python scripts/05_train_linear.py
python scripts/06_summarize_results.py
```

### 3. Launch Dashboard
```bash
streamlit run scripts/dashboard_pipeline.py
```

---

## 📁 Folder Structure
```
energy_forecasting_dashboard/
├── config/                     # Sector profiles YAML
├── data/
│   ├── synthetic/             # Generated synthetic data
│   └── processed/             # Features after engineering
├── models/                    # Trained model artifacts
├── results/
│   ├── predictions/           # Model predictions
│   ├── plots/                 # Visualizations
│   └── summary_model_metrics.csv
├── scripts/                   # All Python scripts
├── README.md
└── requirements.txt
```

---

## 📣 What's Next?
- 📤 Streamlit Cloud deployment
- 📁 Real vs synthetic data comparison module
- 📄 Export PDF reports
- 🔍 Anomaly detection

---

## 🧑‍💻 Author
Created by [Your Name] – powered by Python, Streamlit, XGBoost, and curiosity.

---

## 📄 License
MIT License

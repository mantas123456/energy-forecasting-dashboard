# ⚡ Energy Forecasting Dashboard

A modular, end-to-end time-series forecasting project for hourly building energy consumption across sectors (residential, factory, school, EV charging), powered by synthetic data and deployed with an interactive Streamlit dashboard.

![dashboard-preview](https://user-images.githubusercontent.com/your-screenshot.png) <sub><i>Optional: add a screenshot from your dashboard here.</i></sub>

---

## 🚀 Live App

👉 **[Launch the Streamlit Dashboard](https://your-username.streamlit.app)**
*(Update this after deploying to Streamlit Cloud)*

---

## 📆 Features

* 🔧 Synthetic data generator (1 year of hourly data with seasonality + noise)
* 📊 Multi-model training (Prophet, XGBoost, Linear Regression)
* 🧠 Feature engineering (lag, rolling, cyclical time features)
* 🦪u Model evaluation (RMSE, MAE + visual comparison)
* 🖥️ Streamlit dashboard:

  * Interactive forecast plots (Plotly)
  * Upload your own data
  * Select forecast horizon (6–168h)
  * Download forecast results
  * RMSE & MAE for uploaded data
  * Tabbed layout for clean UX

---

## 📁 Folder Structure

```text
energy_forecasting_dashboard/
├── scripts/                  # All modular pipeline scripts
│   ├── 00_generate_synthetic.py
│   ├── 01_prepare_input.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_prophet.py
│   ├── 04_train_xgboost.py
│   ├── 05_train_linear.py
│   ├── 06_evaluate_models.py
│   ├── 07_generate_phase2_report.py
│   └── dashboard.py
├── results/
│   └── predictions/          # Model outputs
├── data/
│   └── processed/            # Processed and engineered data
├── config/                   # (Optional) global_config.yaml
├── requirements.txt
├── main.py                   # Unified pipeline runner
└── README.md
```

---

## ▶️ Run Locally

```bash
# 1. Clone repo
git clone https://github.com/mantas123456/energy-forecasting-dashboard.git
cd energy-forecasting-dashboard

# 2. Create virtual environment
conda create -n energy_env python=3.10
conda activate energy_env

# 3. Install requirements
pip install -r requirements.txt

# 4. Run pipeline (optional)
python main.py

# 5. Launch dashboard
streamlit run scripts/dashboard.py
```

---

## 📊 Example Uploaded CSV

To test the upload feature, use this structure:

```csv
timestamp,energy_kwh
2025-01-01 00:00:00,12.1
2025-01-01 01:00:00,11.5
...
```

---

## 🗕️ Roadmap

* [x] Forecast horizon slider
* [x] Upload forecasting + RMSE/MAE
* [x] Dashboard tab layout
* [ ] Streamlit Cloud deployment
* [ ] Add XGBoost support to uploaded data
* [ ] Export PDF reports from dashboard
* [ ] Anomaly detection

---

## 🧠 Author

**Mantas Valantinavičius**
📍 Based in Malta | 🔗 [LinkedIn](https://www.linkedin.com/in/mantasvalantinavicius/)
🧪 Focused on data science, energy systems, AI + sustainability

---

## 📜 License

MIT License — feel free to use, extend, and share!

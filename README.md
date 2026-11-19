# Energy Forecasting Dashboard

[![CI](https://github.com/mantas123456/energy-forecasting-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/mantas123456/energy-forecasting-dashboard/actions/workflows/ci.yml)

Day-ahead forecasting of Lithuania's hourly electricity load, with a rolling-origin
backtest, naive baselines, the transmission system operator's (TSO's) own day-ahead
forecast as a benchmark, and a Streamlit dashboard.

The question this project answers: **can a model built only from past load and the
calendar beat the forecast the grid operator publishes, and where does it fail?**

The data is real: hourly load and the TSO day-ahead forecast for Lithuania from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu), collected by my
[energy-data-pipeline](https://github.com/mantas123456/energy-data-pipeline) project.

![Forecast error per test window](results/mape_by_fold.png)

## Results

Twelve 28-day test windows, 18 December 2025 to 19 November 2026. In each window, models
train only on data before it. Forecasts are made 48 hours ahead (see
[How it works](#how-it-works)). MAE is in MWh per hour, which equals the hour's average
load in MW.

| Method | MAE | MAPE | Better than the TSO in |
|---|---:|---:|---:|
| XGBoost | 113.7 | 8.8 % | 6 of 12 windows |
| Ridge regression | 115.2 | 8.8 % | 6 of 12 windows |
| **TSO day-ahead forecast** | 131.6 | 10.1 % | – |
| Naive (same hour last week) | 139.3 | 10.6 % | 5 of 12 windows |
| Naive (same hour 2 days ago) | 159.3 | 11.9 % | 5 of 12 windows |

**Does the model beat the TSO? Not reliably.** On the averages above, XGBoost looks about
14 % better. That figure comes entirely from six windows when the TSO's forecast was
badly off. Window by window, XGBoost wins 6 of 12. Its mean skill against the TSO
(1 − MAE / MAE of the TSO, averaged over windows) is **−2 %**, so it is slightly worse.

| Test window | XGBoost MAPE | TSO MAPE | Skill vs TSO | Better |
|---|---:|---:|---:|---|
| 18 Dec 2025 – 15 Jan 2026 | 10.5 % | 6.0 % | −98 % | TSO |
| 15 Jan – 12 Feb 2026 | 5.4 % | 3.4 % | −65 % | TSO |
| 12 Feb – 12 Mar 2026 | 10.7 % | 6.8 % | −72 % | TSO |
| 12 Mar – 9 Apr 2026 | 14.2 % | 10.7 % | −34 % | TSO |
| 9 Apr – 7 May 2026 | 20.1 % | 16.3 % | −18 % | TSO |
| 7 May – 4 Jun 2026 | 8.7 % | 10.7 % | +22 % | Model |
| 4 Jun – 2 Jul 2026 | 7.3 % | 12.2 % | +39 % | Model |
| 2 Jul – 30 Jul 2026 | 6.5 % | 12.4 % | +50 % | Model |
| 30 Jul – 27 Aug 2026 | 7.1 % | 13.4 % | +48 % | Model |
| 27 Aug – 24 Sep 2026 | 6.3 % | 13.4 % | +52 % | Model |
| 24 Sep – 22 Oct 2026 | 4.7 % | 11.2 % | +57 % | Model |
| 22 Oct – 19 Nov 2026 | 4.2 % | 4.1 % | −4 % | TSO |

What the numbers show:

- **The TSO wins whenever its forecast is unbiased.** From December to April it beats every
  model in every window, often by a wide margin. In the latest window it is level with
  XGBoost.
- **The model's wins come from a period when the TSO forecast was biased low.** From May
  to October 2026 the TSO forecast was on average 100–190 MW below actual load every
  month. Before May and since November it was within about ±100 MW. A model that learns
  recent load levels does not have that problem. The data does not show why the TSO
  forecast drifted.
- **Spring middays are the weak spot for everyone, and worst for the model.** From March
  to early May the error is concentrated between 10:00 and 16:00. Midday load drops to
  as little as 286 MW on sunny days, most likely because rooftop solar covers part of
  consumption behind the meter. XGBoost's midday error reaches about 48 %, against 34 %
  for the TSO. The model has no weather input and cannot see a sunny day coming. The TSO
  presumably can.
- **Against naive baselines the model is useful but not dramatic.** XGBoost is on average
  11 % better than the better naive forecast in each window. Ridge regression does about
  as well as XGBoost.

Full numbers: [`results/metrics_summary.csv`](results/metrics_summary.csv) and
[`results/metrics_by_fold.csv`](results/metrics_by_fold.csv) (per window, with MAE, RMSE,
MAPE, skill against the best naive forecast and against the TSO).

![The last week of the backtest](results/example_week.png)

## How it works

1. **Data** ([`data.py`](src/energy_forecast/data.py)). Hourly load and the TSO day-ahead
   forecast for Lithuania from January 2025. The source is set in
   [`config/config.yaml`](config/config.yaml) or with `--source`:
   - `csv` (the default): the committed sample
     [`data/lt_hourly.csv.gz`](data/lt_hourly.csv.gz), or the pipeline's
     `data/observations/LT` folder of 15-minute files, averaged to hours.
   - `postgres`: the pipeline's `mart_hourly` table. The connection URL is read from the
     `DATABASE_URL` environment variable, never from a file in this repository.

   All three give the same hours and values (the sample is rounded to 0.01 MW). Missing
   hours stay as gaps, so lags never jump across them.
2. **Leak-free features** ([`features.py`](src/energy_forecast/features.py)). Lags of
   48, 72 and 168 hours; 24-hour and 168-hour rolling means of load at least 48 hours
   old; hour, weekday and season encodings; weekend and Lithuanian public-holiday flags.
   All calendar features use Vilnius local time. There is no weather input.
3. **48-hour horizon.** The TSO publishes its day-ahead forecast in the morning before
   the delivery day. At that point, actual load for the last hours of the delivery day
   is up to about 40 hours away. A 24-hour lag would give the model information the TSO
   did not have. Using 48 hours keeps the comparison fair, and slightly favours the TSO.
4. **Rolling-origin backtest** ([`evaluate.py`](src/energy_forecast/evaluate.py)). For
   each 28-day window, models train only on data before it. All methods in a window are
   scored on the same hours.
5. **Models and benchmarks** ([`models.py`](src/energy_forecast/models.py)). Two naive
   forecasts (same hour 2 days ago, same hour last week), ridge regression, XGBoost, and
   the TSO forecast, which is read from the data, not fitted.

The synthetic multi-sector generator from v3 is now a test fixture
([`tests/synthetic.py`](tests/synthetic.py)), so the tests run without data or a database.

## Limitations

- No weather. Temperature and solar irradiance forecasts would most likely close much of
  the gap to the TSO, especially on spring middays. That is the obvious next step.
- One country, under two years of data, and twelve test windows. The result depends on
  the period: the same backtest in summer 2026 would favour the model, in winter the TSO.
- The TSO forecast is taken as published on ENTSO-E. Its error in May–October 2026 may
  reflect a change in what it or the actual-load figure covers, which this data cannot
  show.

## Run it

```bash
git clone https://github.com/mantas123456/energy-forecasting-dashboard
cd energy-forecasting-dashboard
pip install -e ".[app,dev]"

energy-forecast --out results     # backtest on the committed sample (about 15 s)
streamlit run app/dashboard.py    # explore results, or upload your own hourly CSV
pytest                            # tests, including checks that no feature leaks the future
```

To use the pipeline's database instead (with the pipeline's PostgreSQL running):

```bash
pip install -e ".[db]"
export DATABASE_URL=postgresql+psycopg://user:password@localhost:5433/energy
energy-forecast --source postgres --out results
energy-forecast --source postgres --export data/lt_hourly.csv.gz   # refresh the sample
```

Settings such as the horizon, number of windows and XGBoost parameters are in
[`config/config.yaml`](config/config.yaml). `requirements.txt` pins the exact versions that
produced the committed results.

## Deploying on Streamlit Community Cloud

The dashboard does not need the database. The results tab reads the committed files in
`results/`. The data tab reads `DATABASE_URL` when it is set and reachable, and the
committed sample `data/lt_hourly.csv.gz` otherwise. A database on your own machine is not
reachable from Streamlit Cloud, so the deployed app uses the sample.

1. Push the repository to GitHub, including `results/` and `data/lt_hourly.csv.gz`.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from this repository:
   branch `main`, main file `app/dashboard.py`. Under advanced settings, choose Python
   3.11 or newer. The app installs `requirements.txt` and adds `src/` to its own path, so
   the package does not need installing.
3. Optional: to show live data from a database that is reachable from the internet, add
   it under the app's **Secrets** as `DATABASE_URL = "postgresql+psycopg://..."`. Never
   commit it.
4. To update the deployed results: refresh the sample with `--export`, rerun
   `energy-forecast --out results`, and commit both. The app updates on the next push.

## Project layout

```
config/config.yaml        all settings, including the data source
data/lt_hourly.csv.gz     committed sample: hourly load and TSO forecast, Lithuania
src/energy_forecast/      data, features, models, evaluate, pipeline
app/dashboard.py          Streamlit dashboard
tests/                    pytest suite and the synthetic test data
results/                  metrics, predictions and plots from the last run
```

## Changes in v4

v3 ran on synthetic data for four building sectors, so its errors said nothing about real
demand. v4 switches to real Lithuanian load from energy-data-pipeline, with a
configurable source (CSV or PostgreSQL). It also:

- adds the TSO's day-ahead forecast as a benchmark;
- moves to a 48-hour horizon so that comparison is fair;
- uses local time and Lithuanian holidays for the calendar features;
- extends the backtest to twelve windows;
- keeps the synthetic generator as a test fixture.

## Changes in v3

v2 computed rolling averages that included the hour being predicted, which leaked the
answer into the features and made the reported errors look better than they were. v3
rebuilds the pipeline with horizon-aware lags, per-sector features, synthetic multi-sector data,
naive baselines and a rolling backtest, and adds tests that would catch the leak.

## How this was built

I designed the analysis and checked the results. I used Claude Code as
a coding assistant.

---
Mantas Valantinavičius · [github.com/mantas123456](https://github.com/mantas123456)

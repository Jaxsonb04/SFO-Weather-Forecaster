# SFO Weather Forecaster

10 years of NOAA weather data, an XGBoost model, an LSTM, and a question: can I forecast tomorrow's temperature in the Bay Area better than just guessing "same as today"?

Turns out yes, but the interesting part is how the two models fail differently.

## What's in here

I pulled hourly weather data from NOAA's GHCNh archive (2016-2026, missing 2020 because the station went offline), cleaned it, loaded it into SQLite, engineered ~85 features, and trained two models on two prediction targets:

- **Spot temperature 24 hours from now** (what the temp will be at 3pm tomorrow if it's 3pm now)
- **Tomorrow's daily high** (the harder, more useful one)

Both models compete against two baselines: persistence ("same as now") and climatology ("the historical average for this day").

## Results

| Target | XGBoost MAE | LSTM MAE | Baseline MAE |
|---|---|---|---|
| Spot temp 24h ahead | 2.46°F | 2.46°F | 2.60°F |
| Tomorrow's daily high | 4.04°F | **3.47°F** | 4.19°F |

The headline finding: **on next-day highs, the LSTM beats XGBoost by 0.57°F MAE** even though XGBoost gets 69 hand-engineered features and the LSTM only gets 11 raw ones. The LSTM figures out the temporal structure on its own.

On spot temperature they're essentially tied, which is itself interesting - it means hand-engineered lag features were doing most of XGBoost's work.

## What's honest about it

The model is good at average days and bad at heatwaves. Stratified eval shows MAE explodes from ~4°F on normal days to ~13°F on 80-90°F days. There aren't enough hot days in the test set for the model to learn them well, and the sample weighting I added (alpha=2.0 on z-scored temps) helps but doesn't fully close the gap. This is the next problem to solve.

Other limitations worth flagging:
- Single station (SFO airport), so the model doesn't generalize to microclimates 5 miles away
- 2020 data missing, so the model has a discontinuity in its training history
- Test set is 2024-11 to 2026-05, which happened to be a mild stretch - fewer extremes to evaluate against

## File map
combine_psv.py       NOAA pipe-delimited files -> single clean CSV
load_to_db.py        CSV -> SQLite with proper indexes
explore.py           data quality checks, gap detection
sql_analysis.py      10 SQL queries from basic SELECT to window functions
features.py          feature engineering pipeline (lags, rolling stats, etc.)
eda.py               6 plots: trends, seasonality, correlations, distributions
xgboost_model.py     XGBoost with time-series CV, sample weighting, MAE loss
lstm_model.py        PyTorch LSTM, 48h sequences, 11 raw features
compare_models.py    head-to-head + calibration (P(temp > threshold))

## How to run

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch scipy

python combine_psv.py --dir "2016-2026 weather data" --out combined_weather.csv
python load_to_db.py
python features.py
python eda.py
python xgboost_model.py
python lstm_model.py
python compare_models.py
```

Each step writes to disk so you don't have to rerun upstream stages.

## What's next

1. **A/B test the model comparison.** "LSTM beats XGBoost by 0.57°F" sounds confident but I haven't actually proven it's statistically significant. Phase 4 is a paired t-test on per-row errors with a bootstrapped 95% CI on the MAE difference.
2. **Streamlit dashboard.** A simple page showing tomorrow's forecast with the calibrated uncertainty bands from `compare_models.py`. Same model, but presented as a decision tool instead of a notebook.

## Why I built this

I'm teaching myself ML end-to-end. Weather has all the things that make real ML problems hard: time-series structure, missing data, distribution shift, rare events, and a clear notion of what "good" means. Also the results are testable - I can check the model against actual weather tomorrow.

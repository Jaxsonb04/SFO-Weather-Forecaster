# features.py
# feature engineering.
# single responsibility: take raw hourly weather, produce a feature matrix
# ready for modeling. no plotting, no analysis -- those live in eda.py.

import sqlite3
import pandas as pd
import numpy as np

DB_PATH      = "weather.db"
FEATURES_OUT = "weather_features.csv"


def load_data(db_path=DB_PATH):
    """pull raw hourly weather from sqlite and return a clean indexed frame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM weather ORDER BY timestamp", conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    df = df[~df.index.duplicated(keep="last")]

    # enforce a regular hourly grid -- missing hours become NaN rows,
    # which keeps lag features aligned to the correct time offset
    df = df.asfreq("h")
    return df


def engineer_features(df):
    """add time, lag, rolling, and derived features. returns a new dataframe.

    naming convention:
      _lag_Nh    -> value from N hours ago
      _roll_X_Nh -> rolling stat X over the past N hours
      _change_Nh -> current value minus value N hours ago
      target_*   -> prediction target (shifted forward in time)
    """
    df = df.copy()

    # time features
    # sin/cos encoding tells the model that hour=23 is adjacent to hour=0
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # wind direction cyclical encoding (359 degrees is adjacent to 0)
    if "wind_dir" in df.columns:
        df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
        df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))

    # temperature lags -- autocorrelation is the strongest signal in the series
    for lag in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]:
        df[f"temp_lag_{lag}h"] = df["temp_f"].shift(lag)

    for lag in [3, 6, 24, 48]:
        df[f"humidity_lag_{lag}h"] = df["humidity"].shift(lag)
        df[f"pressure_lag_{lag}h"] = df["pressure_hpa"].shift(lag)
    df["wind_lag_24h"]     = df["wind_speed_mph"].shift(24)
    df["dewpoint_lag_24h"] = df["dew_point_f"].shift(24)

    # rolling stats -- smoothed signals reduce noise for the model
    for window in [3, 6, 24, 48]:
        df[f"temp_roll_mean_{window}h"] = df["temp_f"].rolling(window).mean()
    df["temp_roll_std_6h"]  = df["temp_f"].rolling(6).std()
    df["temp_roll_std_24h"] = df["temp_f"].rolling(24).std()

    df["humidity_roll_mean_6h"]  = df["humidity"].rolling(6).mean()
    df["humidity_roll_mean_24h"] = df["humidity"].rolling(24).mean()
    df["wind_roll_mean_6h"]      = df["wind_speed_mph"].rolling(6).mean()
    df["pressure_roll_mean_24h"] = df["pressure_hpa"].rolling(24).mean()

    # pressure tendency -- falling pressure precedes storms, rising precedes clearing
    # recomputed from raw data as the NOAA pressure_change_3h column was 49% null
    df["pressure_change_3h"]  = df["pressure_hpa"] - df["pressure_hpa"].shift(3)
    df["pressure_change_6h"]  = df["pressure_hpa"] - df["pressure_hpa"].shift(6)
    df["pressure_change_24h"] = df["pressure_hpa"] - df["pressure_hpa"].shift(24)

    # dew point depression: distance from saturation. 0 -> fog, large -> dry
    if "dew_point_f" in df.columns:
        df["dewpoint_depression"] = df["temp_f"] - df["dew_point_f"]

    # cloud cover converted from sky_oktas (0-8) to percent (0-100)
    if "sky_oktas" in df.columns:
        df["cloud_cover_pct"] = df["sky_oktas"] * 12.5

    # interaction: clear + dry + calm air -> largest diurnal swing
    if "cloud_cover_pct" in df.columns:
        df["clear_dry_calm"] = (
            (100 - df["cloud_cover_pct"]) / 100
            * (100 - df["humidity"]) / 100
            * np.maximum(0, 15 - df["wind_speed_mph"]) / 15
        )

    # heat momentum: how much warmer than the same hour N days ago
    df["temp_vs_24h_ago"]  = df["temp_f"] - df["temp_lag_24h"]
    df["temp_vs_48h_ago"]  = df["temp_f"] - df["temp_lag_48h"]
    df["temp_vs_168h_ago"] = df["temp_f"] - df["temp_lag_168h"]

    # daily summaries
    df["temp_daily_high"]  = df["temp_f"].rolling(24).max()
    df["temp_daily_low"]   = df["temp_f"].rolling(24).min()
    df["temp_daily_range"] = df["temp_daily_high"] - df["temp_daily_low"]
    df["precip_sum_24h"]   = df["precip_mm"].rolling(24).sum()
    df["precip_sum_72h"]   = df["precip_mm"].rolling(72).sum()
    df["temp_max_72h"]     = df["temp_f"].rolling(72).max()
    df["temp_max_168h"]    = df["temp_f"].rolling(168).max()

    # targets
    df["target_temp_next_24h"] = df["temp_f"].shift(-24)
    df["target_temp_next_48h"] = df["temp_f"].shift(-48)

    # next-day high: max temp in the 24h window starting 24h from now
    df["target_daily_high_next_day"] = (
        df["temp_f"].shift(-47).rolling(24).max()
    )

    # drop rows missing the target or longest lag -- xgboost handles
    # intermediate NaNs internally so we preserve as many rows as possible
    required = [
        "target_temp_next_24h",
        "target_temp_next_48h",
        "target_daily_high_next_day",
        "temp_lag_168h",
    ]
    before = len(df)
    df = df.dropna(subset=required)
    after  = len(df)
    print(f"  dropped {before - after:,} rows missing required features "
          f"({(before - after) / before:.1%})")

    return df


if __name__ == "__main__":
    print("loading raw data from weather.db...")
    df_raw = load_data()
    print(f"  {len(df_raw):,} rows on hourly grid "
          f"({df_raw.index.min()} to {df_raw.index.max()})")

    print("\nengineering features...")
    df = engineer_features(df_raw)
    print(f"  final shape: {df.shape}")

    df.to_csv(FEATURES_OUT)
    print(f"\nsaved: {FEATURES_OUT}")
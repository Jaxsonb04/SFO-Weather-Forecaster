# load combined_weather.csv into SQLite.


import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = "combined_weather.csv"
DB_PATH  = "weather.db"

def load_csv_to_sqlite(csv_path, db_path):
    print(f"reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    # drop columns that are not useful
    drop_cols = ["snow_depth_mm", "wind_gust_ms", "wind_gust_mph", "pressure_change_3h", "source_file"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # sqlite stores datetimes as TEXT. Use ISO format so SQL date functions work correctly.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    Path(db_path).unlink(missing_ok=True)   
    conn = sqlite3.connect(db_path)

    print(f"writing {len(df):,} rows to {db_path}...")
    df.to_sql("weather", conn, index=False)

    print("creating indexes...")
    cur = conn.cursor()
    cur.execute("CREATE INDEX idx_timestamp  ON weather(timestamp)")
    cur.execute("CREATE INDEX idx_station_ts ON weather(station_id, timestamp)")
    conn.commit()

    n = cur.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    cols = [r[1] for r in cur.execute("PRAGMA table_info(weather)").fetchall()]
    print(f"done. {n:,} rows, {len(cols)} columns: {cols}")

    conn.close()

if __name__ == "__main__":
    load_csv_to_sqlite(CSV_PATH, DB_PATH)
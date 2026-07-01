"""Preprocess KDD Cup 2018 PM2.5 data from Monash TSF format to a NumPy/CSV file.

Usage::

    python scripts/experiments/kdd/preprocess_kdd.py \\
        --input data/kdd_2018_no_missing_values.tsf \\
        --output data/kdd_processed.npy \\
        [--station_name aotizhongxin --city beijing --measurement PM2.5]

Parses the raw TSF file, aligns multivariate series, optionally filters by
station/city/measurement, and writes a (K, T) float32 array.
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import pandas as pd


def preprocess(
    tsf_path: str,
    out_path: str,
    station_name: str = None,
    city: str = None,
    measurement: str = None,
):
    print(f"Reading {tsf_path}...")
    series_data = []
    with open(tsf_path, encoding="utf-8") as f:
        data_started = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("@data"):
                data_started = True
                continue
            if not data_started or line.startswith("@") or line.startswith("#"):
                continue

            parts = line.split(":", 5)
            if len(parts) != 6:
                continue

            # Filter by station if specified
            if station_name and parts[2] != station_name:
                continue

            # Filter by city if specified
            if city and parts[1] != city:
                continue

            # Filter by measurement if specified
            if measurement and parts[3] != measurement:
                continue

            vals = [
                np.nan if v in ("?", "NaN", "", "nan") else float(v)
                for v in parts[5].split(",")
            ]
            series_data.append({
                "feature_name": parts[0],
                "city": parts[1],
                "station": parts[2],
                "measurement": parts[3],
                "start_time_raw": parts[4].strip().strip("'\""),
                "values": vals,
            })

    print("Aligning time series...")
    df_meta = pd.DataFrame([
        {k: v for k, v in d.items() if k != "values"} for d in series_data
    ])
    df_meta["start_time"] = pd.to_datetime(df_meta["start_time_raw"], utc=True)

    # Clean and find common start
    valid_mask = df_meta["start_time"] < pd.Timestamp("2017-02-01", tz="UTC")
    df_clean = df_meta[valid_mask].copy()
    common_start = df_clean["start_time"].max()

    aligned_series_dict = {}
    for idx in df_clean.index:
        start = df_clean.loc[idx, "start_time"]
        vals = series_data[idx]["values"]
        name = df_clean.loc[idx, "feature_name"]
        ts = pd.Series(
            vals, index=pd.date_range(start=start, periods=len(vals), freq="h")
        )
        aligned_series_dict[name] = ts[common_start:]

    min_len = min(len(s) for s in aligned_series_dict.values())
    series_list = [s.iloc[:min_len] for s in aligned_series_dict.values()]

    print("Generating covariates...")
    common_index = series_list[0].index
    v_time = np.stack(
        [
            common_index.hour.values / 23.0,
            common_index.dayofweek.values / 6.0,
            (common_index.month.values - 1) / 11.0,
        ],
        axis=0,
    ).astype(np.float32)

    df_clean["city_code"] = df_clean["city"].astype("category").cat.codes
    df_clean["station_code"] = df_clean["station"].astype("category").cat.codes
    df_clean["measurement_code"] = df_clean["measurement"].astype("category").cat.codes
    v_static = df_clean[
        ["city_code", "station_code", "measurement_code"]
    ].values.astype(np.int64)

    payload = {
        "series_list": series_list,
        "covariates_list": [v_time],
        "static_covariates": v_static,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)
    print(f"Preprocessed data saved to {out_file} (D={len(series_list)}, T={min_len})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tsf_path", type=str, required=True)
    p.add_argument("--out_path", type=str, default="data/kdd_processed.pt")
    p.add_argument("--station_name", type=str, default=None)
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--measurement", type=str, default=None)
    args = p.parse_args()
    preprocess(
        args.tsf_path, args.out_path, args.station_name, args.city, args.measurement
    )

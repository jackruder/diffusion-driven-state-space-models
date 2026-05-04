import os
from typing import List, Tuple
import pandas as pd
import numpy as np
import torch

from ddssm.data.dataload import build_loaders_for_expt


def parse_kdd_tsf(filepath: str) -> pd.DataFrame:
    """Parses Monash .tsf KDD file and returns aligned multivariate DataFrame."""
    series_data = []
    with open(filepath, "r", encoding="utf-8") as f:
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

            start_time_raw = parts[4].strip().strip("'\"")
            vals = [
                np.nan if v in ("?", "NaN", "", "nan") else float(v)
                for v in parts[5].split(",")
            ]

            series_data.append({
                "feature_name": parts[0],
                "city": parts[1],
                "station": parts[2],
                "measurement": parts[3],
                "start_time_raw": start_time_raw,
                "values": vals,
            })

    df_meta = pd.DataFrame([
        {k: v for k, v in d.items() if k != "values"} for d in series_data
    ])
    df_meta["start_time"] = pd.to_datetime(
        df_meta["start_time_raw"],
        format="%Y-%m-%d %H-%M-%S",
        utc=True,
        errors="coerce",
    )

    valid_mask = df_meta["start_time"] < pd.Timestamp("2017-02-01", tz="UTC")
    df_clean = df_meta[valid_mask].copy()

    # Create categorical codes
    df_clean["city_code"] = df_clean["city"].astype("category").cat.codes
    df_clean["station_code"] = df_clean["station"].astype("category").cat.codes
    df_clean["measurement_code"] = df_clean["measurement"].astype("category").cat.codes

    common_start = df_clean["start_time"].max()
    aligned_series_dict = {}

    for idx in df_clean.index:
        start = df_clean.loc[idx, "start_time"]
        vals = series_data[idx]["values"]
        ts = pd.Series(
            vals, index=pd.date_range(start=start, periods=len(vals), freq="h")
        )
        aligned_series_dict[df_clean.loc[idx, "feature_name"]] = ts[common_start:]

    min_len = min(len(s) for s in aligned_series_dict.values())

    df_multi = pd.DataFrame({
        name: s.iloc[:min_len] for name, s in aligned_series_dict.items()
    })

    # Ensure df_clean index aligns with df_multi columns
    df_clean = df_clean.set_index("feature_name").loc[df_multi.columns]

    return df_multi, df_clean


def setup_kdd_loaders(
    filepath: str,
    eval_step_size: int = 24,  # 1 for hourly, 24 for daily
    batch_size: int = 64,
    num_train_batches_per_epoch: int = 200,
    train_instances_per_series: float = 32.0,
    device: torch.device | None = None,
    backend: str = "torch",
):
    df_multi, df_clean = parse_kdd_tsf(filepath)

    # Extract covariates
    common_index = df_multi.index
    hour_cov = (common_index.hour.to_numpy(dtype=np.float32) / 23.0) - 0.5
    day_cov = (common_index.dayofweek.to_numpy(dtype=np.float32) / 6.0) - 0.5
    month_cov = ((common_index.month.to_numpy(dtype=np.float32) - 1) / 11.0) - 0.5

    global_covariates = np.stack([hour_cov, day_cov, month_cov], axis=0)  # (3, T)

    series_list = [df_multi[col] for col in df_multi.columns]
    covariates_list = [global_covariates.copy() for _ in range(len(series_list))]

    if eval_step_size == 1:
        test_windows = 697
        val_windows = 625
    elif eval_step_size == 24:
        test_windows = 29
        val_windows = 27
    else:
        test_windows = 744 // 48
        val_windows = 672 // 48

    static_covariates = df_clean[
        ["city_code", "station_code", "measurement_code"]
    ].values

    loaders = build_loaders_for_expt(
        series_list=series_list,
        L1=72,
        L2=48,
        test_windows=test_windows,
        val_windows=val_windows,
        batch_size=batch_size,
        normalize=True,
        num_train_batches_per_epoch=num_train_batches_per_epoch,
        train_instances_per_series=train_instances_per_series,
        device=device,
        covariates_list=covariates_list,
        static_covariates=static_covariates,
        eval_step_size=eval_step_size,
        backend=backend,
    )

    static_covariates = torch.tensor(
        df_clean[["city_code", "station_code", "measurement_code"]].values,
        dtype=torch.long,
    )

    meta = {
        "feature_names": list(df_multi.columns),
        "covariate_names": ["hour", "dayofweek", "month"],
        "feature_to_idx": {name: i for i, name in enumerate(df_multi.columns)},
        "static_covariates": static_covariates,
        "static_covariate_names": ["city_code", "station_code", "measurement_code"],
        "num_classes_per_static": [
            df_clean["city_code"].max() + 1,
            df_clean["station_code"].max() + 1,
            df_clean["measurement_code"].max() + 1,
        ],
    }

    return (*loaders, meta)

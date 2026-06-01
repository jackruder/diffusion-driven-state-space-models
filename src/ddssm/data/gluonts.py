"""GluonTS repository dataset loaders (solar, electricity, traffic, taxi, wiki)."""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset
from .dataload import build_loaders_for_expt

DATASETS: Dict[str, Dict] = {
    "solar": dict(L1=168, L2=24, test_windows=7, val_windows=5),
    "electricity": dict(L1=168, L2=24, test_windows=7, val_windows=5),
    "traffic": dict(L1=168, L2=24, test_windows=7, val_windows=5),
    "taxi": dict(L1=48, L2=24, test_windows=56, val_windows=5),
    "wiki": dict(L1=90, L2=30, test_windows=5, val_windows=5),
}

REPO_NAMES = {
    "solar": "solar-energy",
    "electricity": "electricity",
    "traffic": "traffic",
    "taxi": "taxi_30min",
    "wiki": "wiki-rolling_nips",
}

EXPECTED_K = {
    "solar": 137, "electricity": 370, "traffic": 963, "taxi": 1214, "wiki": 2000,
}

def _period_to_timestamp(x):
    return x.to_timestamp() if hasattr(x, 'to_timestamp') else x

def _repo_to_series_list(repo_name: str, expected_k: Optional[int] = None, force_fresh: bool = False) -> Tuple[List[pd.Series], str]:
    repo = get_dataset(repo_name, regenerate=force_fresh)
    freq = repo.metadata.freq
    unique, seen, idx = {}, set(), 0

    for entry in repo.test:
        item_id = entry.get("item_id", None)
        key = item_id if item_id is not None else idx
        if key in seen:
            if item_id is None:
                idx += 1
            continue

        start = _period_to_timestamp(entry["start"])
        target = entry["target"].astype("float32")
        unique[key] = pd.Series(
            target, index=pd.date_range(start=start, periods=len(target), freq=freq)
        )
        seen.add(key)
        if item_id is None:
            idx += 1
        if expected_k is not None and len(unique) >= expected_k:
            break

    return list(unique.values()), freq

def get_loaders_for(
    name: str, *, batch_size: int = 64, train_instances_per_series: int | None = None,
    num_train_batches_per_epoch: int | None = None, force_fresh_repo: bool = False,
    covariates_list: List[np.ndarray] | None = None,
):
    """Build windowed loaders for a named GluonTS repository dataset.

    Looks up the per-dataset window spec in ``DATASETS``, fetches the
    series via the GluonTS repository, and delegates to
    :func:`build_loaders_for_expt`. Defaults for the epoch size and
    instances-per-series are derived from the number of available train
    windows when not given.

    Args:
        name: Dataset key in ``DATASETS`` (``solar``, ``electricity``, …).
        batch_size: Loader batch size.
        train_instances_per_series: Override sampler intensity.
        num_train_batches_per_epoch: Override train epoch size.
        force_fresh_repo: Regenerate the cached GluonTS repository.
        covariates_list: Optional per-series dynamic covariates.

    Returns:
        ``(train_loader, val_loader, test_loader, (means, stds))``.

    Raises:
        AssertionError: ``name`` is not a known dataset key.
    """
    assert name in DATASETS, f"Unknown dataset key: {name}"
    series_list, _ = _repo_to_series_list(REPO_NAMES[name], expected_k=EXPECTED_K[name], force_fresh=force_fresh_repo)
    
    spec = DATASETS[name]
    K, T_total = len(series_list), min(len(s) for s in series_list)
    T_train = T_total - (spec["val_windows"] * spec["L2"] + spec["test_windows"] * spec["L2"])
    windows_per_series = max(0, T_train - (spec["L1"] + spec["L2"]) + 1)
    
    return build_loaders_for_expt(
        series_list=series_list, L1=spec["L1"], L2=spec["L2"],
        test_windows=spec["test_windows"], val_windows=spec["val_windows"],
        batch_size=batch_size,
        num_train_batches_per_epoch=num_train_batches_per_epoch or int((K * windows_per_series + batch_size - 1) // batch_size),
        train_instances_per_series=train_instances_per_series or windows_per_series,
        covariates_list=covariates_list,
    )

def get_solar_loaders(**kw): return get_loaders_for("solar", **kw)
def get_electricity_loaders(**kw): return get_loaders_for("electricity", **kw)
def get_traffic_loaders(**kw): return get_loaders_for("traffic", **kw)

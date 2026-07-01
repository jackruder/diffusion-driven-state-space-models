"""GluonTS-based data loading utilities: sliding-window loaders, batch parsing, and z-score scaling."""

from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pandas.tseries.frequencies import to_offset

_GLUONTS_IMPORT_ERROR: Exception | None = None
try:
    from gluonts.transform import InstanceSplitter, AddObservedValuesIndicator
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.loader import TrainDataLoader, as_stacked_batches
    from gluonts.torch.batchify import batchify
    from gluonts.transform.sampler import InstanceSampler, ExpectedNumInstanceSampler
    from gluonts.dataset.field_names import FieldName
    from gluonts.dataset.multivariate_grouper import MultivariateGrouper
except (ImportError, ModuleNotFoundError) as err:  # pragma: no cover - GluonTS optional
    _GLUONTS_IMPORT_ERROR = err
    ListDataset = None
    TrainDataLoader = None
    as_stacked_batches = None
    MultivariateGrouper = None
    batchify = None
    AddObservedValuesIndicator = None
    InstanceSplitter = None
    ExpectedNumInstanceSampler = None

    class _FieldNameFallback:
        FEAT_DYNAMIC_REAL = "feat_dynamic_real"

    FieldName = _FieldNameFallback

    class InstanceSampler:
        min_future = 1

        def _get_bounds(self, ts):
            n = len(ts)
            return 0, n - int(self.min_future)


class FixedLastKSampler(InstanceSampler):
    """Deterministic sampler taking the last ``k_last`` window starts.

    Walks backwards from the latest valid start in strides of
    ``step_size`` (defaulting to ``min_future``) and returns the last
    ``k_last`` starts in ascending order. Used to build fixed eval
    windows for the GluonTS backend.
    """

    k_last: int = 1
    step_size: int | None = None

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=np.int64)
        step = (
            self.step_size
            if self.step_size is not None
            else max(int(self.min_future), 1)
        )
        starts = np.arange(b, a - 1, -step, dtype=np.int64)
        return np.sort(starts[: self.k_last])


def _stack_series(series_list: list[pd.Series]) -> np.ndarray:
    K, T = len(series_list), len(series_list[0])
    arr = np.zeros((K, T), dtype=np.float32)
    for k, s in enumerate(series_list):
        arr[k] = s.to_numpy(dtype=np.float32, copy=False)
    return arr


def compute_per_series_zscore(series_list: list[pd.Series], train_end: int):
    """Per-series mean/std over the train tail (NaN-safe).

    Statistics are computed on each series up to ``train_end`` so eval
    windows are never used to normalize. Near-zero stds are clamped to
    1.0 and any NaN mean/std falls back to 0.0 / 1.0.

    Args:
        series_list: One pandas Series per channel.
        train_end: Exclusive end index of the training region.

    Returns:
        ``(means, stds)`` float32 arrays of shape ``(len(series_list),)``.
    """
    arr = _stack_series(series_list)[:, :train_end]
    means = np.nanmean(arr, axis=1)
    stds = np.nanstd(arr, axis=1)
    stds = np.where(stds < 1e-6, 1.0, stds)

    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def apply_per_series_zscore(
    series_list: list[pd.Series], means: np.ndarray, stds: np.ndarray
):
    """Z-score each series with the given per-series ``means``/``stds``.

    Returns a new list of Series (original index preserved); inputs are
    not mutated.
    """
    scaled = []
    for k, s in enumerate(series_list):
        v = (s.to_numpy(dtype=np.float32, copy=False) - means[k]) / stds[k]
        scaled.append(pd.Series(v, index=s.index))
    return scaled


@dataclass
class _WindowSpec:
    end: int  # exclusive end index of full window [end-(L1+L2), end)
    L1: int
    L2: int


class _GroupedWindowDataset(Dataset):
    """Sliding-window view over stacked multivariate series.

    Each item is the canonical model-ready dict with keys:

    * ``observed_data``: ``(D, L1+L2)`` (NaNs zero-filled).
    * ``observation_mask``: ``(D, L1+L2)`` (1 where finite, else 0).
    * ``timepoints``: ``(L1+L2,)`` (local index ``0..L1+L2-1``).
    * ``covariates``: ``(V, L1+L2)`` or ``None``.
    * ``static_covariates``: ``(D, V_static)`` or ``None``.
    """

    def __init__(
        self,
        data: np.ndarray,  # (D, T)
        windows: list[_WindowSpec],
        covariates: np.ndarray | None,  # (V, T)
        static_covariates: np.ndarray | None = None,  # (D, V_static)
    ):
        self.data = data.astype(np.float32, copy=False)
        self.windows = windows
        self.covariates = (
            covariates.astype(np.float32, copy=False)
            if covariates is not None
            else None
        )
        self.static_covariates = (
            static_covariates.astype(np.int64, copy=False)
            if static_covariates is not None
            else None
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i: int):
        w = self.windows[i]
        T = w.L1 + w.L2
        s = w.end - T
        e = w.end

        x = self.data[:, s:e]  # (D, T)
        mask = np.isfinite(x).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32, copy=False)

        t = np.arange(T, dtype=np.float32)

        out = {
            "observed_data": torch.from_numpy(x),
            "observation_mask": torch.from_numpy(mask),
            "timepoints": torch.from_numpy(t),
            "covariates": None,
            "static_covariates": None,
        }
        if self.covariates is not None:
            out["covariates"] = torch.from_numpy(self.covariates[:, s:e])  # (V, T)
        if self.static_covariates is not None:
            out["static_covariates"] = torch.from_numpy(
                self.static_covariates
            )  # (D, V_static)
        return out


def _make_window_ends(
    start_end: int,  # first valid end index (>= L1+L2)
    last_end: int,  # last valid end index (inclusive)
    step: int,
    k_last: int | None = None,
):
    if last_end < start_end:
        return []
    # Anchor the stride grid backward from ``last_end`` so the final window
    # ends exactly at the region boundary. Anchoring forward from
    # ``start_end`` left the grid short by ``(last_end - start_end) % step``,
    # shifting every eval window earlier — far enough that val/test forecast
    # targets could overlap the training region.
    offset = (last_end - start_end) % step
    ends = list(range(start_end + offset, last_end + 1, step))
    if k_last is not None:
        ends = ends[-k_last:]
    return ends


def _collate_model_ready(batch: list[dict]):
    keys = ["observed_data", "observation_mask", "timepoints"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    if batch[0]["covariates"] is None:
        out["covariates"] = None
    else:
        out["covariates"] = torch.stack([b["covariates"] for b in batch], dim=0)

    if batch[0].get("static_covariates", None) is None:
        out["static_covariates"] = None
    else:
        out["static_covariates"] = torch.stack(
            [b["static_covariates"] for b in batch], dim=0
        )

    return out


def build_loaders_for_expt(
    series_list: list[pd.Series],
    L1: int,
    L2: int,
    test_windows: int,
    val_windows: int = 5,
    batch_size: int = 64,
    normalize: bool = True,
    num_train_batches_per_epoch: int | None = None,
    train_instances_per_series: float = 64.0,
    device: torch.device | None = None,
    covariates_list: list[np.ndarray] | None = None,
    static_covariates: np.ndarray | None = None,
    eval_step_size: int | None = None,
    backend: str = "torch",
):
    """Build train/val/test loaders of past/future windows from raw series.

    Splits chronologically into train / val / test regions sized from
    the window counts and stride, optionally z-scores each series using
    train-tail statistics, and constructs sliding-window loaders. The
    ``"torch"`` backend yields the canonical model-ready dict directly
    (see :class:`_GroupedWindowDataset`); the ``"gluonts"`` backend uses
    a GluonTS ``InstanceSplitter`` pipeline whose batches are mapped by
    :func:`parse_batch`.

    Args:
        series_list: One pandas Series per channel, shared time index.
        L1: Past (conditioning) window length.
        L2: Future (forecast) window length.
        test_windows: Number of eval windows in the test region.
        val_windows: Number of eval windows in the validation region.
        batch_size: Loader batch size.
        normalize: Apply per-series z-score (train-tail statistics).
        num_train_batches_per_epoch: ``None`` walks every train window
            (torch backend); otherwise samples a fixed-size epoch.
        train_instances_per_series: GluonTS sampler intensity (and the
            torch fallback epoch size); ignored when walking all windows.
        device: Device for the returned ``means``/``stds`` tensors.
        covariates_list: Optional per-channel dynamic covariates ``(V, T)``.
        static_covariates: Optional static covariates ``(D, V_static)``.
        eval_step_size: Stride between consecutive eval windows; defaults
            to ``L2`` (non-overlapping).
        backend: ``"torch"`` or ``"gluonts"``.

    Returns:
        ``(train_loader, val_loader, test_loader, (means, stds))`` where
        ``means``/``stds`` are tensors when ``normalize`` else ``None``.

    Raises:
        ImportError: ``backend="gluonts"`` but GluonTS failed to import.
        ValueError: Unknown ``backend``, or no train windows available.
    """
    freq = pd.infer_freq(series_list[0].index)
    T_total = min(len(s) for s in series_list)

    step = eval_step_size if eval_step_size is not None else L2
    test_block = (test_windows - 1) * step + L2
    val_block = (val_windows - 1) * step + L2
    train_end = T_total - (val_block + test_block)

    means_t, stds_t = None, None
    if normalize:
        means, stds = compute_per_series_zscore(series_list, train_end)
        series_list = apply_per_series_zscore(series_list, means, stds)
        dev = device or torch.device("cpu")
        means_t, stds_t = (
            torch.from_numpy(means).to(dev),
            torch.from_numpy(stds).to(dev),
        )

    if backend == "gluonts":
        if _GLUONTS_IMPORT_ERROR is not None:
            raise ImportError(
                "GluonTS backend requested but GluonTS imports failed."
            ) from _GLUONTS_IMPORT_ERROR

        def to_listdataset(series, end):
            return ListDataset(
                [
                    {
                        "start": s.index[0],
                        "target": s.values[:end].astype("float32"),
                        FieldName.FEAT_DYNAMIC_REAL: covariates_list[k][:, :end].astype(
                            "float32"
                        )
                        if covariates_list is not None
                        else None,
                    }
                    for k, s in enumerate(series)
                ],
                freq=freq,
            )

        grouper = MultivariateGrouper(max_target_dim=None)
        train_ds = grouper(to_listdataset(series_list, train_end))
        val_ds = grouper(to_listdataset(series_list, T_total - test_block))
        test_ds = grouper(to_listdataset(series_list, T_total))

        mask_obs = AddObservedValuesIndicator(
            target_field=FieldName.TARGET, output_field=FieldName.OBSERVED_VALUES
        )

        train_split = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=train_instances_per_series, min_future=L2
            ),
            past_length=L1,
            future_length=L2,
            time_series_fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_DYNAMIC_REAL],
        )
        train_loader = TrainDataLoader(
            dataset=train_ds,
            transform=mask_obs + train_split,
            batch_size=batch_size,
            stack_fn=batchify,
            num_batches_per_epoch=num_train_batches_per_epoch,
            shuffle_buffer_length=1000,
        )

        def _make_eval(dataset, k_last):
            split = InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=FixedLastKSampler(
                    k_last=k_last, min_past=L1, min_future=L2, step_size=eval_step_size
                ),
                past_length=L1,
                future_length=L2,
                time_series_fields=[
                    FieldName.OBSERVED_VALUES,
                    FieldName.FEAT_DYNAMIC_REAL,
                ],
            )
            return as_stacked_batches(
                (mask_obs + split).apply(dataset, is_train=False),
                batch_size=batch_size,
                output_type=None,
            )

        return (
            train_loader,
            _make_eval(val_ds, val_windows),
            _make_eval(test_ds, test_windows),
            (means_t, stds_t),
        )

    if backend != "torch":
        raise ValueError(f"Unknown backend: {backend}")

    # torch backend
    data = _stack_series(series_list)  # (D, T_total)
    cov_global = covariates_list[0] if covariates_list is not None else None

    T = L1 + L2
    min_end = T

    train_last_end = train_end
    train_ends_all = _make_window_ends(min_end, train_last_end, step=1, k_last=None)
    if len(train_ends_all) == 0:
        raise ValueError("No train windows available. Check split sizes.")

    if num_train_batches_per_epoch is None:
        # Use all available hourly train windows
        train_ends = train_ends_all
    else:
        # Keep old behavior: fixed-size sampled epoch
        n_train = int(num_train_batches_per_epoch * batch_size)
        rng = np.random.default_rng(0)
        train_ends = rng.choice(train_ends_all, size=n_train, replace=True).tolist()

    val_last_end = T_total - test_block
    val_ends = _make_window_ends(min_end, val_last_end, step=step, k_last=val_windows)

    test_last_end = T_total
    test_ends = _make_window_ends(
        min_end, test_last_end, step=step, k_last=test_windows
    )

    train_ds = _GroupedWindowDataset(
        data=data,
        windows=[_WindowSpec(end=e, L1=L1, L2=L2) for e in train_ends],
        covariates=cov_global,
        static_covariates=static_covariates,
    )
    val_ds = _GroupedWindowDataset(
        data=data,
        windows=[_WindowSpec(end=e, L1=L1, L2=L2) for e in val_ends],
        covariates=cov_global,
        static_covariates=static_covariates,
    )
    test_ds = _GroupedWindowDataset(
        data=data,
        windows=[_WindowSpec(end=e, L1=L1, L2=L2) for e in test_ends],
        covariates=cov_global,
        static_covariates=static_covariates,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_model_ready,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_model_ready,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_model_ready,
    )

    return train_loader, val_loader, test_loader, (means_t, stds_t)


def series_ticks_from_batch(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover integer time ticks for a GluonTS batch's past/future windows.

    Uses each item's ``start`` / ``forecast_start`` timestamps and the
    inferred frequency to compute the offset of the forecast start, then
    lays out per-item ``arange`` ticks.

    Returns:
        ``(past_ticks, future_ticks)`` of shape ``(B, L1)`` and
        ``(B, L2)``.
    """
    B, L1 = batch["past_target"].shape[0], batch["past_target"].shape[1]
    L2 = batch["future_target"].shape[1]

    def _to_ts(x) -> pd.Timestamp:
        return x.to_timestamp() if isinstance(x, pd.Period) else pd.Timestamp(x)

    def _freq_str(i) -> str:
        fs, st = batch["forecast_start"][i], batch["start"][i]
        if isinstance(fs, pd.Period) and fs.freq is not None:
            return fs.freqstr
        if isinstance(st, pd.Period) and st.freq is not None:
            return st.freqstr
        return "h"

    past_list, fut_list = [], []
    for i in range(B):
        fs_ts, st_ts = _to_ts(batch["forecast_start"][i]), _to_ts(batch["start"][i])
        off = to_offset(_freq_str(i))
        tick_ns = (
            off.nanos
            if hasattr(off, "nanos")
            else int(pd.to_timedelta(off.delta).value)
        )
        n_fs = int(round((fs_ts.value - st_ts.value) / tick_ns))

        past_list.append(
            torch.arange(n_fs - L1, n_fs, device=device, dtype=torch.float32)
        )
        fut_list.append(
            torch.arange(n_fs, n_fs + L2, device=device, dtype=torch.float32)
        )

    return torch.stack(past_list, dim=0), torch.stack(fut_list, dim=0)


def parse_batch(batch: dict, device: torch.device):
    """Normalize a raw loader batch into the canonical model-ready dict.

    Serves as the ``batch_transform`` for every DataModule. A batch
    already carrying ``observed_data``/``observation_mask``/``timepoints``
    (torch backend, synthetic, KDD) is moved to ``device`` as-is, with
    optional ``covariates``, ``static_covariates`` and ``gt_latent``
    passed through. A GluonTS batch is assembled from its
    ``past_*``/``future_*`` fields: past and future are concatenated
    along time into ``(B, D, L1+L2)``, masks combine observed indicators
    with past padding, and timepoints come from
    :func:`series_ticks_from_batch` (rebased to start at 0).

    Returns:
        A dict with ``observed_data``, ``observation_mask``,
        ``timepoints`` and ``covariates`` (plus pass-through keys on the
        torch path), all on ``device``.
    """
    # Native torch backend path (already model-ready)
    if (
        "observed_data" in batch
        and "observation_mask" in batch
        and "timepoints" in batch
    ):
        out = {
            "observed_data": torch.as_tensor(
                batch["observed_data"], device=device, dtype=torch.float32
            ),
            "observation_mask": torch.as_tensor(
                batch["observation_mask"], device=device, dtype=torch.float32
            ),
            "timepoints": torch.as_tensor(
                batch["timepoints"], device=device, dtype=torch.float32
            ),
            "covariates": None,
        }
        if batch.get("covariates") is not None:
            out["covariates"] = torch.as_tensor(
                batch["covariates"], device=device, dtype=torch.float32
            )

        if batch.get("static_covariates") is not None:
            out["static_covariates"] = torch.as_tensor(
                batch["static_covariates"], device=device, dtype=torch.long
            )

        # Optional ground-truth latents (synthetic data with
        # ``expose_gt_latents=True``; used by the model-v2 evaluation
        # metrics ``gt_latent_jsd`` and ``crps_sum_latent``).
        if batch.get("gt_latent") is not None:
            out["gt_latent"] = torch.as_tensor(
                batch["gt_latent"], device=device, dtype=torch.float32
            )

        return out

    # GluonTS path
    past = torch.as_tensor(
        batch["past_target"], device=device, dtype=torch.float32
    ).transpose(1, 2)
    fut = torch.as_tensor(
        batch["future_target"], device=device, dtype=torch.float32
    ).transpose(1, 2)
    pobs = torch.as_tensor(
        batch["past_observed_values"], device=device, dtype=torch.float32
    ).transpose(1, 2)
    fobs = torch.as_tensor(
        batch["future_observed_values"], device=device, dtype=torch.float32
    ).transpose(1, 2)

    past = torch.nan_to_num(past, nan=0.0)
    fut = torch.nan_to_num(fut, nan=0.0)

    ppad = torch.as_tensor(
        batch.get("past_is_pad", torch.zeros_like(past[:, 0, :])),
        device=device,
        dtype=torch.float32,
    )
    past_mask = pobs * (1.0 - ppad.unsqueeze(1).expand(-1, past.shape[1], -1))

    past_ticks, future_ticks = series_ticks_from_batch(batch, device=device)
    timepoints = torch.cat([past_ticks, future_ticks], dim=1)
    timepoints = timepoints - timepoints[:, :1]

    past_cov = batch.get(f"past_{FieldName.FEAT_DYNAMIC_REAL}")
    future_cov = batch.get(f"future_{FieldName.FEAT_DYNAMIC_REAL}")
    covariates = None
    if past_cov is not None and future_cov is not None:
        past_cov = torch.as_tensor(past_cov, device=device, dtype=torch.float32)
        future_cov = torch.as_tensor(future_cov, device=device, dtype=torch.float32)
        covariates = torch.cat([past_cov, future_cov], dim=2)  # (B, V, L1+L2)

    return {
        "observed_data": torch.cat([past, fut], dim=2),
        "observation_mask": torch.cat([past_mask, fobs], dim=2),
        "timepoints": timepoints,
        "covariates": covariates,
    }

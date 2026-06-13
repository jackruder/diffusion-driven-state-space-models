"""Unified DataModule layer for DDSSM experiments.

A ``DataModule`` exposes train/val/test ``DataLoader`` objects plus a
``DataMetadata`` block that the experiment uses to wire the model's shape
kwargs (``data_dim``, ``covariate_dim``, ``T``, ``use_observation_mask``).
Every dataset — synthetic, GluonTS repository, KDD — is one of these, so the
experiment/eval/viz stages consume a single interface.

Two batch formats are advertised so the experiment can pick the right
``batch_transform``:

* ``"sequence"``: items are full ``(D, T)`` sequences with all-ones masks
  (``SyntheticDataModule``). ``parse_batch``'s native-torch branch is the
  right transform.
* ``"windowed"``: items are ``(D, L1+L2)`` past/future windows with real
  observation masks and optional covariates — the ``WindowedSeriesDataModule``
  family (``KDDDataModule``, ``GluonTSDataModule``). Same ``parse_batch``
  branch handles them because :class:`_GroupedWindowDataset` already emits the
  canonical model-ready dict.
"""

from __future__ import annotations

import abc
from typing import Literal, Callable
from dataclasses import field, dataclass

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from ddssm.data.dataload import parse_batch, build_loaders_for_expt
from ddssm.data.synthetic import SyntheticDataset

BatchFormat = Literal["sequence", "windowed"]


@dataclass
class DataMetadata:
    """Shape / normalization information published by a DataModule.

    Anything the model or trainer needs to know about the data lives here
    so the experiment can read shapes off a single source of truth.

    ``forecast_split`` is the canonical past/future boundary index used
    by forecasting metrics and visualizations. Windowed datasets (KDD)
    set it to ``L1``; sequence datasets (synthetic) leave it ``None``
    and let the eval/viz spec choose explicitly.
    """

    data_dim: int
    covariate_dim: int = 0
    T: int = 0
    use_observation_mask: bool = True
    static_cardinalities: tuple[int, ...] = field(default_factory=tuple)
    means: torch.Tensor | None = None
    stds: torch.Tensor | None = None
    forecast_split: int | None = None

    def forecast_split_or(self, override: int | None) -> int | None:
        """The explicit ``override`` if given, else the dataset's ``forecast_split``.

        Standalone stages resolve their past/future boundary through
        this: an eval/viz spec may set ``T_split`` explicitly, otherwise
        the dataset's own ``forecast_split`` (``L1`` for windowed KDD,
        ``None`` for sequence data) is used.
        """
        return int(override) if override is not None else self.forecast_split


class DDSSMDataModule(abc.ABC):
    """Abstract base every concrete DataModule extends.

    Subclasses implement the three split loaders plus ``metadata``; the
    base supplies :meth:`loader`, the ``split → loader`` dispatch shared
    by every standalone stage (eval / viz / variance).
    """

    batch_format: BatchFormat
    batch_transform: Callable[[dict, torch.device], dict]

    @abc.abstractmethod
    def train_loader(self) -> DataLoader | None: ...
    @abc.abstractmethod
    def val_loader(self) -> DataLoader | None: ...
    @abc.abstractmethod
    def test_loader(self) -> DataLoader | None: ...

    @property
    @abc.abstractmethod
    def metadata(self) -> DataMetadata: ...

    def loader(self, split: str) -> DataLoader | None:
        """Return the loader for ``split`` ∈ {``train``, ``val``, ``test``}."""
        if split == "train":
            return self.train_loader()
        if split == "val":
            return self.val_loader()
        if split == "test":
            return self.test_loader()
        raise ValueError(f"Unknown split: {split!r}")


class NullDataModule(DDSSMDataModule):
    """No data attached. Replaces the ``dataset=none`` sentinel.

    The experiment treats ``train_loader() is None`` as "build only,
    skip ``trainer.fit``" — useful for smoke tests and interactive use.
    """

    batch_format: BatchFormat = "sequence"
    batch_transform = staticmethod(parse_batch)

    def __init__(self, data_dim: int = 1):
        self._meta = DataMetadata(data_dim=data_dim)

    def train_loader(self) -> DataLoader | None:
        return None

    def val_loader(self) -> DataLoader | None:
        return None

    def test_loader(self) -> DataLoader | None:
        return None

    @property
    def metadata(self) -> DataMetadata:
        return self._meta


class SyntheticDataModule(DDSSMDataModule):
    """Sequence-format DataModule wrapping :class:`SyntheticDataset`.

    Each item is a full ``(D, T)`` sequence. ``train``/``val``/``test``
    are deterministic disjoint slices of the same generated population
    (the underlying ``SyntheticDataset`` allocates ``3 * N_per_split``
    sequences and partitions them).

    Args mirror :class:`SyntheticDataset` plus standard ``DataLoader``
    knobs.
    """

    batch_format: BatchFormat = "sequence"
    batch_transform = staticmethod(parse_batch)

    def __init__(
        self,
        mode: str = "lgssm",
        T: int = 64,
        D: int = 1,
        N_per_split: int = 512,
        dataset_seed: int = 1234,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle_train: bool = True,
        use_observation_mask: bool = False,
        expose_gt_latents: bool = False,
    ):
        self.mode = mode
        self.T = T
        self.D = D
        self.N_per_split = N_per_split
        self.dataset_seed = dataset_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self._use_observation_mask = use_observation_mask
        self.expose_gt_latents = bool(expose_gt_latents)

    def _build(self, split: str) -> SyntheticDataset:
        return SyntheticDataset(
            mode=self.mode,
            split=split,
            N_per_split=self.N_per_split,
            T=self.T,
            D=self.D,
            dataset_seed=self.dataset_seed,
            expose_gt_latents=self.expose_gt_latents,
        )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._build(split),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def train_loader(self) -> DataLoader:
        return self._loader("train", shuffle=self.shuffle_train)

    def val_loader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    @property
    def metadata(self) -> DataMetadata:
        return DataMetadata(
            data_dim=self.D,
            covariate_dim=0,
            T=self.T,
            use_observation_mask=self._use_observation_mask,
        )


class WindowedSeriesDataModule(DDSSMDataModule):
    """Base for windowed-format DataModules built from a ``series_list``.

    Subclasses implement :meth:`_load_series` (per-feature series + optional
    dynamic / static covariates) and set the window-spec attributes (``L1``,
    ``L2``, ``test_windows``, ``val_windows`` and the loader knobs). The base
    lazily windows them via
    :func:`~ddssm.data.dataload.build_loaders_for_expt` and publishes the three
    loaders plus a :class:`DataMetadata` block (``forecast_split = L1``).
    """

    batch_format: BatchFormat = "windowed"
    batch_transform = staticmethod(parse_batch)

    # Loader knobs — subclasses override in __init__.
    L1: int = 72
    L2: int = 48
    test_windows: int = 5
    val_windows: int = 5
    eval_step_size: int | None = None
    batch_size: int = 64
    normalize: bool = True
    num_train_batches_per_epoch: int | None = None
    train_instances_per_series: float = 64.0
    backend: str = "torch"

    def __init__(self, use_observation_mask: bool = True) -> None:
        self._use_observation_mask = use_observation_mask
        self._built = False
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._metadata: DataMetadata | None = None

    @abc.abstractmethod
    def _load_series(
        self,
    ) -> tuple[list[pd.Series], list[np.ndarray] | None, np.ndarray | None]:
        """Return ``(series_list, covariates_list, static_covariates)``.

        ``covariates_list`` is per-series ``(V, T)`` dynamic covariates or
        ``None``; ``static_covariates`` is ``(D, V_static)`` categorical codes
        or ``None``. Subclasses may set loader knobs (e.g. data-derived
        ``train_instances_per_series``) here before returning.
        """
        ...

    @staticmethod
    def _default_temporal_covariates(index: pd.Index) -> np.ndarray:
        """Hour-of-day, day-of-week, month features in [-0.5, 0.5]."""
        hour = (index.hour.to_numpy(dtype=np.float32) / 23.0) - 0.5
        dow = (index.dayofweek.to_numpy(dtype=np.float32) / 6.0) - 0.5
        month = ((index.month.to_numpy(dtype=np.float32) - 1) / 11.0) - 0.5
        return np.stack([hour, dow, month], axis=0)

    def _ensure_built(self) -> None:
        if self._built:
            return

        series_list, covariates_list, static_covariates = self._load_series()
        D = len(series_list)
        covariate_dim = covariates_list[0].shape[0] if covariates_list else 0

        static_cardinalities: tuple[int, ...] = ()
        if static_covariates is not None:
            arr = (
                static_covariates.detach().cpu().numpy()
                if isinstance(static_covariates, torch.Tensor)
                else np.asarray(static_covariates)
            )
            static_cardinalities = tuple(
                int(arr[:, j].max()) + 1 for j in range(arr.shape[1])
            )
            static_covariates = arr

        train_loader, val_loader, test_loader, (means, stds) = build_loaders_for_expt(
            series_list=series_list,
            L1=self.L1,
            L2=self.L2,
            test_windows=self.test_windows,
            val_windows=self.val_windows,
            batch_size=self.batch_size,
            normalize=self.normalize,
            num_train_batches_per_epoch=self.num_train_batches_per_epoch,
            train_instances_per_series=self.train_instances_per_series,
            covariates_list=covariates_list,
            static_covariates=static_covariates,
            eval_step_size=self.eval_step_size,
            backend=self.backend,
        )

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._metadata = DataMetadata(
            data_dim=D,
            covariate_dim=covariate_dim,
            T=self.L1 + self.L2,
            use_observation_mask=self._use_observation_mask,
            static_cardinalities=static_cardinalities,
            means=means,
            stds=stds,
            forecast_split=self.L1,
        )
        self._built = True

    def train_loader(self) -> DataLoader:
        self._ensure_built()
        assert self._train_loader is not None
        return self._train_loader

    def val_loader(self) -> DataLoader:
        self._ensure_built()
        assert self._val_loader is not None
        return self._val_loader

    def test_loader(self) -> DataLoader:
        self._ensure_built()
        assert self._test_loader is not None
        return self._test_loader

    @property
    def metadata(self) -> DataMetadata:
        self._ensure_built()
        assert self._metadata is not None
        return self._metadata


class KDDDataModule(WindowedSeriesDataModule):
    """Windowed-format DataModule for the KDD Cup 2018 PM2.5 dataset.

    Loads a preprocessed ``.pt`` payload produced by
    ``scripts/experiments/kdd/preprocess_kdd.py``. The payload is a dict
    with at least ``series_list`` (a list of pandas Series, one per
    feature). Optional ``covariates_list`` and ``static_covariates``
    (when present) are forwarded to :func:`build_loaders_for_expt`; if
    absent, default temporal covariates (hour/dayofweek/month, normalized
    to [-0.5, 0.5]) are derived from the shared time index.

    Args:
        filepath: Path to the preprocessed ``.pt`` payload. Defaults to
            ``data/kdd.pt`` which is tracked via Git LFS in this repo.
        L1, L2: Past / future window lengths.
        eval_step_size: Stride between consecutive eval windows.
        batch_size: Train / val / test batch size.
        num_train_batches_per_epoch: ``None`` walks every train window;
            otherwise samples a fixed number of batches per epoch.
        train_instances_per_series: Used by the GluonTS backend only.
        normalize: Per-series z-score using train-tail statistics.
        backend: ``"torch"`` (windowed Dataset) or ``"gluonts"`` (gluonts
            ``InstanceSplitter`` pipeline). Default is ``"torch"`` because
            it produces the canonical model-ready dict directly.
    """

    batch_format: BatchFormat = "windowed"
    batch_transform = staticmethod(parse_batch)

    def __init__(
        self,
        filepath: str = "data/kdd.pt",
        L1: int = 72,
        L2: int = 48,
        eval_step_size: int = 24,
        batch_size: int = 64,
        num_train_batches_per_epoch: int | None = None,
        train_instances_per_series: float = 32.0,
        normalize: bool = True,
        backend: str = "torch",
        use_observation_mask: bool = True,
    ):
        super().__init__(use_observation_mask=use_observation_mask)
        self.filepath = filepath
        self.L1 = L1
        self.L2 = L2
        self.eval_step_size = eval_step_size
        self.batch_size = batch_size
        self.num_train_batches_per_epoch = num_train_batches_per_epoch
        self.train_instances_per_series = train_instances_per_series
        self.normalize = normalize
        self.backend = backend
        # Eval-window counts follow the stride (independent of the data).
        if eval_step_size == 1:
            self.test_windows, self.val_windows = 697, 625
        elif eval_step_size == 24:
            self.test_windows, self.val_windows = 29, 27
        else:
            self.test_windows, self.val_windows = 744 // 48, 672 // 48

    def _load_series(self):
        payload = torch.load(self.filepath, weights_only=False, map_location="cpu")
        if not isinstance(payload, dict) or "series_list" not in payload:
            raise ValueError(
                f"KDDDataModule: payload at {self.filepath!r} must be a dict "
                f"with a 'series_list' key (list of pandas Series)."
            )
        series_list: list[pd.Series] = payload["series_list"]
        covariates_list = payload.get("covariates_list", None)
        if covariates_list is None:
            cov = self._default_temporal_covariates(series_list[0].index)
            covariates_list = [cov.copy() for _ in range(len(series_list))]
        static_covariates = payload.get("static_covariates", None)
        return series_list, covariates_list, static_covariates


class GluonTSDataModule(WindowedSeriesDataModule):
    """Windowed DataModule for a GluonTS repository dataset.

    Lazily fetches a named dataset (``solar`` / ``electricity`` / ``traffic`` /
    ``taxi`` / ``wiki``) from the GluonTS repository into a per-series list, then
    windows it like any other :class:`WindowedSeriesDataModule`. The fetch is
    deferred to first loader access (never at import), so construction is cheap
    and network-free. No dynamic/static covariates are attached.
    """

    SPECS = {
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
        "solar": 137, "electricity": 370, "traffic": 963,
        "taxi": 1214, "wiki": 2000,
    }

    def __init__(
        self,
        name: str = "solar",
        batch_size: int = 64,
        normalize: bool = True,
        num_train_batches_per_epoch: int | None = None,
        train_instances_per_series: float | None = None,
        backend: str = "torch",
        force_fresh_repo: bool = False,
        use_observation_mask: bool = True,
    ):
        if name not in self.SPECS:
            raise ValueError(
                f"Unknown GluonTS dataset {name!r}; known: {sorted(self.SPECS)}"
            )
        super().__init__(use_observation_mask=use_observation_mask)
        self.name = name
        spec = self.SPECS[name]
        self.L1, self.L2 = spec["L1"], spec["L2"]
        self.test_windows, self.val_windows = spec["test_windows"], spec["val_windows"]
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_train_batches_per_epoch = num_train_batches_per_epoch
        # ``None`` ⇒ derive from available train windows in ``_load_series``.
        self.train_instances_per_series = train_instances_per_series
        self.backend = backend
        self.force_fresh_repo = force_fresh_repo

    @staticmethod
    def _period_to_timestamp(x):
        return x.to_timestamp() if hasattr(x, "to_timestamp") else x

    def _load_series(self):
        from gluonts.dataset.repository.datasets import get_dataset

        repo = get_dataset(self.REPO_NAMES[self.name], regenerate=self.force_fresh_repo)
        freq = repo.metadata.freq
        expected_k = self.EXPECTED_K[self.name]
        unique: dict = {}
        seen: set = set()
        idx = 0
        for entry in repo.test:
            item_id = entry.get("item_id", None)
            key = item_id if item_id is not None else idx
            if key in seen:
                if item_id is None:
                    idx += 1
                continue
            start = self._period_to_timestamp(entry["start"])
            target = entry["target"].astype("float32")
            unique[key] = pd.Series(
                target,
                index=pd.date_range(start=start, periods=len(target), freq=freq),
            )
            seen.add(key)
            if item_id is None:
                idx += 1
            if expected_k is not None and len(unique) >= expected_k:
                break

        series_list = list(unique.values())
        # Derive loader intensity from the available train windows (historical
        # gluonts.py defaults) when the caller didn't pin them.
        K = len(series_list)
        T_total = min(len(s) for s in series_list)
        T_train = T_total - (self.val_windows + self.test_windows) * self.L2
        windows_per_series = max(0, T_train - (self.L1 + self.L2) + 1)
        if self.train_instances_per_series is None:
            self.train_instances_per_series = float(windows_per_series)
        if self.num_train_batches_per_epoch is None:
            self.num_train_batches_per_epoch = int(
                (K * windows_per_series + self.batch_size - 1) // self.batch_size
            )
        return series_list, None, None


__all__ = [
    "BatchFormat",
    "DataMetadata",
    "DDSSMDataModule",
    "NullDataModule",
    "SyntheticDataModule",
    "WindowedSeriesDataModule",
    "KDDDataModule",
    "GluonTSDataModule",
]

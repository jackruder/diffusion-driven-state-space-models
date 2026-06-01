"""Unified DataModule layer for DDSSM experiments.

A ``DataModule`` exposes train/val/test ``DataLoader`` objects plus a
``DataMetadata`` block that the experiment uses to wire the model's shape
kwargs (``data_dim``, ``covariate_dim``, ``T``, ``use_observation_mask``).
This replaces the ad-hoc mix of ``SyntheticDataset`` (single split) and
``setup_kdd_loaders`` (returns three loaders + a meta dict).

Two batch formats are advertised so the experiment can pick the right
``batch_transform``:

* ``"sequence"``: items are full ``(D, T)`` sequences with all-ones masks
  (synthetic data). ``parse_batch``'s native-torch branch is the right
  transform.
* ``"windowed"``: items are ``(D, L1+L2)`` past/future windows with real
  observation masks and optional covariates (KDD). Same ``parse_batch``
  branch handles them because :class:`_GroupedWindowDataset` already emits
  the canonical model-ready dict.
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


class KDDDataModule(DDSSMDataModule):
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
        self.filepath = filepath
        self.L1 = L1
        self.L2 = L2
        self.eval_step_size = eval_step_size
        self.batch_size = batch_size
        self.num_train_batches_per_epoch = num_train_batches_per_epoch
        self.train_instances_per_series = train_instances_per_series
        self.normalize = normalize
        self.backend = backend
        self._use_observation_mask = use_observation_mask
        self._built = False
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._metadata: DataMetadata | None = None

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

        payload = torch.load(self.filepath, weights_only=False, map_location="cpu")
        if not isinstance(payload, dict) or "series_list" not in payload:
            raise ValueError(
                f"KDDDataModule: payload at {self.filepath!r} must be a dict "
                f"with a 'series_list' key (list of pandas Series)."
            )
        series_list: list[pd.Series] = payload["series_list"]
        D = len(series_list)
        T = min(len(s) for s in series_list)

        covariates_list = payload.get("covariates_list", None)
        if covariates_list is None:
            cov = self._default_temporal_covariates(series_list[0].index)
            covariates_list = [cov.copy() for _ in range(D)]
        covariate_dim = covariates_list[0].shape[0] if covariates_list else 0

        static_covariates = payload.get("static_covariates", None)
        static_cardinalities: tuple[int, ...] = ()
        if static_covariates is not None:
            arr = (
                static_covariates.detach().cpu().numpy()
                if isinstance(static_covariates, torch.Tensor)
                else np.asarray(static_covariates)
            )
            static_cardinalities = tuple(int(arr[:, j].max()) + 1 for j in range(arr.shape[1]))
            static_covariates = arr

        if self.eval_step_size == 1:
            test_windows, val_windows = 697, 625
        elif self.eval_step_size == 24:
            test_windows, val_windows = 29, 27
        else:
            test_windows = 744 // 48
            val_windows = 672 // 48

        train_loader, val_loader, test_loader, (means, stds) = build_loaders_for_expt(
            series_list=series_list,
            L1=self.L1,
            L2=self.L2,
            test_windows=test_windows,
            val_windows=val_windows,
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


__all__ = [
    "BatchFormat",
    "DataMetadata",
    "DDSSMDataModule",
    "NullDataModule",
    "SyntheticDataModule",
    "KDDDataModule",
]

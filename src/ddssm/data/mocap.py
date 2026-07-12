"""CMU MoCap subject-35 walking DataModule.

The benchmark shared by Yildiz 2019 (ODE²VAE), Li et al. 2020 (latent
SDE), Course & Nair 2023 (arlatentsde) and Bartosh 2025 (SDE Matching):
16 train / 3 val / 4 test sequences, length 300, 50-dim joint-angle
observations at ``dt = 0.1``, following the preprocessing in Wang et al.
2007. Shipped as a single ``mocap35.mat`` file with keys ``Xtr``,
``Xval``, ``Xtest``.

On construction the module ensures ``data/mocap35.mat`` exists — if not,
it downloads from the canonical Dropbox mirror used by Course & Nair
(same URL that anchors their published numbers). Loading is eager
(single ``loadmat``), the split tensors are transposed from
``(N, T, D)`` to ``(N, D, T)`` to match the sequence-format contract,
optionally z-scored per feature using train statistics, and wrapped in
``MocapDataset`` — a tiny map-style Dataset that emits the canonical
``{observed_data, observation_mask, timepoints}`` batch dict shared with
``SyntheticDataModule``.
"""

from __future__ import annotations

import os
import logging
import pathlib
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ddssm.data.dataload import parse_batch
from ddssm.data.datamodule import BatchFormat, DataMetadata, DDSSMDataModule

logger = logging.getLogger(__name__)

MOCAP35_URL = "https://www.dropbox.com/s/p75wc1j53itonuo/mocap35.mat?dl=1"


def _ensure_mocap35(filepath: str, url: str = MOCAP35_URL) -> None:
    """Download ``mocap35.mat`` to ``filepath`` if it's not already there.

    Atomic: writes to ``filepath + ".tmp"`` then ``os.replace`` on success,
    so an interrupted download never leaves a truncated file behind.
    """
    if os.path.isfile(filepath):
        return
    parent = pathlib.Path(filepath).parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath + ".tmp"
    logger.info("Downloading mocap35.mat from %s → %s", url, filepath)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 15):
            f.write(chunk)
    os.replace(tmp, filepath)


class MocapDataset(Dataset):
    """Map-style Dataset over pre-loaded ``(N, D, T)`` sequence tensor.

    Each item is the canonical model-ready dict — same shape contract as
    ``SyntheticDataset`` so ``parse_batch`` needs no per-dataset branching.
    """

    def __init__(self, X: torch.Tensor, timepoints: torch.Tensor):
        # X: (N, D, T); timepoints: (T,)
        self.X = X
        self.timepoints = timepoints
        self._mask = torch.ones_like(X[0])  # (D, T), reused per item

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "observed_data": self.X[idx],
            "observation_mask": self._mask,
            "timepoints": self.timepoints,
        }


class MocapDataModule(DDSSMDataModule):
    """Sequence-format DataModule for the CMU MoCap subject-35 benchmark.

    Args:
        filepath: On-disk location of ``mocap35.mat``. Downloaded on demand.
        url: Source URL for the ``.mat`` — anchored to the Course & Nair
            Dropbox mirror so numbers stay comparable across the literature.
        download: When ``True`` (default), fetch the file if missing.
            Set ``False`` in offline/CI contexts; a missing file then raises
            ``FileNotFoundError`` cleanly.
        dt: Timestep spacing used to build the shared ``timepoints`` vector.
        normalize: Per-feature z-score using train statistics.
        batch_size, num_workers, pin_memory, drop_last, shuffle_train:
            Standard DataLoader knobs.
        use_observation_mask: Advertised in :class:`DataMetadata`; the
            dataset always emits an all-ones mask regardless, so the model
            can either honor or ignore the field.
    """

    batch_format: BatchFormat = "sequence"
    batch_transform = staticmethod(parse_batch)

    def __init__(
        self,
        filepath: str = "data/mocap35.mat",
        url: str = MOCAP35_URL,
        download: bool = True,
        dt: float = 0.1,
        normalize: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle_train: bool = True,
        use_observation_mask: bool = False,
    ):
        self.filepath = filepath
        self.url = url
        self.download = download
        self.dt = dt
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self._use_observation_mask = use_observation_mask
        self._built = False
        self._splits: dict[str, MocapDataset] = {}
        self.means: torch.Tensor | None = None
        self.stds: torch.Tensor | None = None
        # Shape is a fixed contract of the benchmark; expose in metadata
        # before the file is ever touched so config-time introspection
        # (Hydra instantiation, sweeps) doesn't trigger I/O.
        self._D = 50
        self._T = 300
        self._timepoints = torch.arange(self._T, dtype=torch.float32) * dt

    def _ensure_built(self) -> None:
        if self._built:
            return
        if self.download:
            _ensure_mocap35(self.filepath, self.url)
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                f"MocapDataModule: {self.filepath!r} does not exist and "
                f"download=False. Either pass download=True or fetch the "
                f"file manually from {self.url}."
            )
        # Lazy import — scipy is not in the base dep set, but the venv has
        # it via transitive deps (matplotlib/torchvision pull it). Keep the
        # import inside the builder so module import stays cheap.
        from scipy.io import loadmat

        mat = loadmat(self.filepath)
        # (N, T, D) on disk → (N, D, T) for the sequence-format contract.
        x_tr = np.asarray(mat["Xtr"], dtype=np.float32).transpose(0, 2, 1)
        x_val = np.asarray(mat["Xval"], dtype=np.float32).transpose(0, 2, 1)
        x_test = np.asarray(mat["Xtest"], dtype=np.float32).transpose(0, 2, 1)

        if self.normalize:
            # Per-feature (D,) statistics from the flattened train tensor
            # (N_train * T points). Matches Course & Nair 2023. Some mocap
            # joint axes are constant (never move during walking); guard the
            # near-zero std so those features stay in raw units instead of
            # exploding rounding noise by 1/eps.
            flat_tr = x_tr.transpose(0, 2, 1).reshape(-1, x_tr.shape[1])
            means = flat_tr.mean(axis=0)
            stds_raw = flat_tr.std(axis=0)
            stds = np.where(stds_raw < 1e-6, np.float32(1.0), stds_raw)
            m = means[None, :, None]
            s = stds[None, :, None]
            x_tr = (x_tr - m) / s
            x_val = (x_val - m) / s
            x_test = (x_test - m) / s
            self.means = torch.from_numpy(means)
            self.stds = torch.from_numpy(stds)

        self._splits = {
            "train": MocapDataset(torch.from_numpy(x_tr), self._timepoints),
            "val": MocapDataset(torch.from_numpy(x_val), self._timepoints),
            "test": MocapDataset(torch.from_numpy(x_test), self._timepoints),
        }
        self._built = True

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        self._ensure_built()
        return DataLoader(
            self._splits[split],
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
        # Trigger build so ``means``/``stds`` are populated when
        # ``normalize=True``; consumers of metadata need those.
        self._ensure_built()
        return DataMetadata(
            data_dim=self._D,
            covariate_dim=0,
            T=self._T,
            use_observation_mask=self._use_observation_mask,
            means=self.means,
            stds=self.stds,
            forecast_split=None,
        )

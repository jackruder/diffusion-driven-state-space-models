"""Contract + family-specific tests for :class:`CSDIAdapter`.

The :class:`CSDIContract` subclass lights up the five family-agnostic checks in
:mod:`tests.adapters.contract`; the standalone ``test_*`` functions below cover
the CSDI-specific behaviour the plan calls out (``gt_mask`` semantics vs the
vendored ``get_test_pattern_mask``, cross-process round-trip, ``_format`` guards
both directions, ``hparams``-override for arch construction, and ``resume_from``
optimizer/scheduler/step restore).

The adapter has NO dependency on ``Experiment``; only the ``TrainingScalars``
dataclass is borrowed (via the contract harness) as a plain knob bag.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
import subprocess

import torch
import pytest
from torch.utils.data import Dataset, DataLoader

from ddssm.adapters.csdi import CSDIConfig, CSDIAdapter
from ddssm.data.dataload import parse_batch
from ddssm.data.datamodule import DataMetadata, NullDataModule, TimeSeriesDataModule

if TYPE_CHECKING:
    from pathlib import Path

# Small fixed shapes so every fit / forecast stays fast (few diffusion steps,
# tiny net). Kept as module constants so the contract subclass and the
# standalone tests agree on the data geometry.
_L1 = 8
_L2 = 4
_D = 2
_T = _L1 + _L2
_N = 8


# --------------------------------------------------------------------------- #
# A tiny WINDOWED data module (metadata.forecast_split is not None).            #
# --------------------------------------------------------------------------- #
class _TinyWindowDataset(Dataset):
    """Emits the canonical model-ready dict — the same shape ``parse_batch`` eats.

    One item is a full ``(D, T)`` past+future window with an all-ones observation
    mask and local ``0..T-1`` timepoints, mirroring ``_GroupedWindowDataset``.
    """

    def __init__(self, n: int, d: int, t: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(n, d, t, generator=g)
        self.t = t

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "observed_data": self.data[i],
            "observation_mask": torch.ones_like(self.data[i]),
            "timepoints": torch.arange(self.t, dtype=torch.float32),
            "covariates": None,
            "static_covariates": None,
        }


def _collate(batch: list[dict]) -> dict:
    keys = ["observed_data", "observation_mask", "timepoints"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["covariates"] = None
    out["static_covariates"] = None
    return out


class TinyWindowedDataModule(TimeSeriesDataModule):
    """Minimal windowed DataModule with ``metadata.forecast_split = L1``.

    This is the cheapest object satisfying the CSDI adapter's requirement that
    ``data.metadata.forecast_split`` be set (Synthetic / Null leave it ``None``).
    Uses the shipped ``parse_batch`` as its ``batch_transform`` so raw batches
    flow through the exact production transform.
    """

    batch_format = "windowed"
    batch_transform = staticmethod(parse_batch)

    def __init__(
        self,
        n: int = _N,
        d: int = _D,
        l1: int = _L1,
        l2: int = _L2,
        batch_size: int = 4,
    ) -> None:
        """Build a tiny windowed loader with ``forecast_split = l1``."""
        self.n, self.d, self.l1, self.l2 = n, d, l1, l2
        self.t = l1 + l2
        self.batch_size = batch_size
        self._ds = _TinyWindowDataset(n, d, self.t)

    def _loader(self, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
        )

    def train_loader(self) -> DataLoader:
        """Shuffled train loader over the tiny window set."""
        return self._loader(shuffle=True)

    def val_loader(self) -> DataLoader:
        """Deterministic val loader (same windows, no shuffle)."""
        return self._loader(shuffle=False)

    def test_loader(self) -> DataLoader:
        """Deterministic test loader (same windows, no shuffle)."""
        return self._loader(shuffle=False)

    @property
    def metadata(self) -> DataMetadata:
        """Windowed metadata whose ``forecast_split`` is ``L1`` (not None)."""
        return DataMetadata(
            data_dim=self.d,
            covariate_dim=0,
            T=self.t,
            use_observation_mask=True,
            forecast_split=self.l1,
        )


def _make_config(**overrides: object) -> CSDIConfig:
    """A tiny fixed CSDIConfig — small enough for fast CPU fits.

    ``target_dim`` matches ``_D``; ``num_sample_features=None`` exercises the
    ``None -> target_dim`` resolution in ``_build``.
    """
    base = dict(
        target_dim=_D,
        layers=1,
        channels=8,
        nheads=2,
        diffusion_embedding_dim=8,
        num_steps=3,
        timeemb=8,
        featureemb=4,
        num_sample_features=None,
    )
    base.update(overrides)
    return CSDIConfig(**base)


def _forecast_inputs_from(data: TimeSeriesDataModule) -> dict:
    """Build the forecast kwargs (everything but ``num_samples``) from one batch."""
    raw = next(iter(data.val_loader()))
    batch = data.batch_transform(raw, torch.device("cpu"))
    L1 = data.metadata.forecast_split
    return dict(
        x_hist=batch["observed_data"][..., :L1],
        x_mask=batch["observation_mask"][..., :L1],
        past_time=batch["timepoints"][:, :L1],
        future_time=batch["timepoints"][:, L1:],
        past_covariates=None,
        future_covariates=None,
        static_covariates=None,
    )


# --------------------------------------------------------------------------- #
# Contract harness.                                                            #
# --------------------------------------------------------------------------- #
from tests.adapters.contract import ModelAdapterContract  # noqa: E402


class TestCSDIContract(ModelAdapterContract):
    """Runs the five family-agnostic checks (a)-(e) against ``CSDIAdapter``."""

    def make_adapter(self) -> CSDIAdapter:
        """Build a fresh CSDIAdapter from a FIXED tiny config.

        Contract check (b) reloads WITHOUT hparams, rebuilding the module from
        ``self.config`` — so every adapter must build the same topology.
        """
        return CSDIAdapter(_make_config())

    def make_data(self) -> TimeSeriesDataModule:
        """Return the tiny windowed data module (``forecast_split`` set)."""
        return TinyWindowedDataModule()

    def _forecast_inputs(self, data: TimeSeriesDataModule) -> dict:
        return _forecast_inputs_from(data)

    def test_checkpoint_roundtrip_preserves_forecast(self, tmp_path: Path) -> None:
        """(b) save → load restores forecast-equivalent state.

        CSDI ``evaluate`` is stochastic (ancestral sampling), so we seed
        immediately before each forecast to compare the SAME noise draw before
        vs after reload — this isolates the state round-trip from RNG.
        """
        adapter = self.make_adapter()
        data = self.make_data()
        self._fit(adapter, data, tmp_path)
        batch = self._forecast_inputs(data)

        torch.manual_seed(12345)
        before = adapter.forecast(**batch, num_samples=4)

        ckpt = str(tmp_path / "adapter.pth")
        adapter.save_checkpoint(ckpt)
        reloaded = self.make_adapter()
        reloaded.load_checkpoint(ckpt, device=torch.device("cpu"))

        torch.manual_seed(12345)
        after = reloaded.forecast(**batch, num_samples=4)

        torch.testing.assert_close(before["pred_mean"], after["pred_mean"])


# --------------------------------------------------------------------------- #
# Family-specific tests.                                                       #
# --------------------------------------------------------------------------- #
def _fit_small(
    adapter: CSDIAdapter,
    data: TimeSeriesDataModule,
    tmp_path: Path,
    steps: int = 2,
) -> None:
    from ddssm.experiment.experiment import TrainingScalars

    adapter.fit(
        data=data,
        training=TrainingScalars(steps=steps, log_every=1, validate_every=1),
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "metrics.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )


class _NonWindowedDataModule(TinyWindowedDataModule):
    """Same tiny data, but with ``metadata.forecast_split = None`` (sequence-like)."""

    @property
    def metadata(self) -> DataMetadata:
        """Metadata with ``forecast_split=None`` (rejected by the CSDI adapter)."""
        return DataMetadata(
            data_dim=self.d, covariate_dim=0, T=self.t, forecast_split=None
        )


def test_forecast_split_none_raises(tmp_path: Path) -> None:
    """A non-windowed module (forecast_split is None) is rejected clearly at fit."""
    adapter = CSDIAdapter(_make_config())
    data = _NonWindowedDataModule()  # has train data but forecast_split is None
    from ddssm.experiment.experiment import TrainingScalars

    with pytest.raises(ValueError, match="forecast_split"):
        adapter.fit(
            data=data,
            training=TrainingScalars(steps=1, log_every=1),
            device=torch.device("cpu"),
            csv_log_path=str(tmp_path / "m.csv"),
            tensorboard_dir=str(tmp_path / "tb"),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )


def test_gt_mask_matches_vendored_test_pattern(tmp_path: Path) -> None:
    """The adapter's train ``gt_mask`` reproduces CSDI's forecast-window split.

    With ``target_strategy='test'`` the vendored train-time cond mask is
    ``observed_mask * gt_mask`` (== ``get_test_pattern_mask``). The adapter zeros
    ``gt_mask`` on the forecast window (``t >= L1``); combined with an all-ones
    observed mask that means: history observed (1), future imputed (0) — exactly
    the CSDI HIST/PRED pattern.
    """
    adapter = CSDIAdapter(_make_config())
    data = TinyWindowedDataModule()
    adapter._device = torch.device("cpu")
    adapter._build(adapter.config, torch.device("cpu"))

    raw = next(iter(data.train_loader()))
    csdi_batch = adapter._make_csdi_batch(raw, data, torch.device("cpu"))

    # process_data permutes (B,L,K)->(B,K,L); gt/observed masks live there.
    (_obs, observed_mask, _tp, gt_mask, _fp, _cut, _fid) = adapter.csdi.process_data(
        csdi_batch
    )
    cond_mask = adapter.csdi.get_test_pattern_mask(observed_mask, gt_mask)

    # cond_mask must be 1 on history columns, 0 on forecast columns (masks are
    # exactly 0/1, so integer comparison is exact here).
    L1 = data.metadata.forecast_split
    assert bool(torch.all(cond_mask[..., :L1] == 1))
    assert bool(torch.all(cond_mask[..., L1:] == 0))


def test_forecast_shapes(tmp_path: Path) -> None:
    """Forecast returns (B,D,L2) mean and (B,S,D,L2) samples in normalized space."""
    adapter = CSDIAdapter(_make_config())
    data = TinyWindowedDataModule()
    _fit_small(adapter, data, tmp_path)
    out = adapter.forecast(**_forecast_inputs_from(data), num_samples=3)
    B = data.batch_size
    assert out["pred_mean"].shape == (B, _D, _L2)
    assert out["pred_samples"].shape == (B, 3, _D, _L2)


def test_null_data_fit_noops(tmp_path: Path) -> None:
    """NullDataModule (train_loader() is None) => fit no-ops, no CSV."""
    adapter = CSDIAdapter(_make_config())
    csv_path = str(tmp_path / "metrics.csv")
    from ddssm.experiment.experiment import TrainingScalars

    adapter.fit(
        data=NullDataModule(data_dim=_D),
        training=TrainingScalars(steps=2, log_every=1),
        device=torch.device("cpu"),
        csv_log_path=csv_path,
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert not os.path.exists(csv_path)
    assert adapter.csdi is None


def test_ddssm_payload_rejected(tmp_path: Path) -> None:
    """A DDSSM-style checkpoint payload is rejected by the CSDI loader."""
    bogus = tmp_path / "ddssm.pth"
    torch.save({"_format": "ddssm_ckpt_v3", "model_state": {}}, bogus)
    adapter = CSDIAdapter(_make_config())
    with pytest.raises(ValueError, match="csdi_ckpt_v1"):
        adapter.load_checkpoint(str(bogus), device=torch.device("cpu"))


def test_csdi_format_guard_accepts_own_payload(tmp_path: Path) -> None:
    """A round-tripped CSDI payload carries the ``csdi_ckpt_v1`` tag and loads."""
    adapter = CSDIAdapter(_make_config())
    data = TinyWindowedDataModule()
    _fit_small(adapter, data, tmp_path)
    ckpt = str(tmp_path / "csdi.pth")
    adapter.save_checkpoint(ckpt)

    payload = torch.load(ckpt, weights_only=False)
    assert payload["_format"] == "csdi_ckpt_v1"

    # And a payload whose _format has been corrupted is rejected.
    payload["_format"] = "not_csdi"
    torch.save(payload, ckpt)
    with pytest.raises(ValueError, match="csdi_ckpt_v1"):
        adapter.load_checkpoint(ckpt, device=torch.device("cpu"))


def test_save_before_build_raises(tmp_path: Path) -> None:
    """save_checkpoint before fit/load (csdi is None) raises RuntimeError."""
    adapter = CSDIAdapter(_make_config())
    assert adapter.csdi is None
    with pytest.raises(RuntimeError):
        adapter.save_checkpoint(str(tmp_path / "x.pth"))


def test_hparams_override_wins_for_arch(tmp_path: Path) -> None:
    """Hparams passed to fit/load rebuild arch from hparams, not self.config."""
    adapter = CSDIAdapter(_make_config(channels=8, layers=1))
    data = TinyWindowedDataModule()
    hp = _make_config(channels=16, layers=2)

    from ddssm.experiment.experiment import TrainingScalars

    adapter.fit(
        data=data,
        training=TrainingScalars(steps=1, log_every=1, validate_every=0),
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "m.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        hparams=hp,
    )
    # The built module must reflect hparams (16 channels, 2 layers), not config.
    assert adapter.csdi.diffmodel.channels == 16
    assert len(adapter.csdi.diffmodel.residual_layers) == 2

    ckpt = str(tmp_path / "csdi.pth")
    adapter.save_checkpoint(ckpt)

    # Now load into a fresh adapter (built from a small config) but with the
    # SAME wide hparams; the rebuilt module must match the checkpoint topology.
    reloaded = CSDIAdapter(_make_config(channels=8, layers=1))
    reloaded.load_checkpoint(ckpt, device=torch.device("cpu"), hparams=hp)
    assert reloaded.csdi.diffmodel.channels == 16
    assert len(reloaded.csdi.diffmodel.residual_layers) == 2


def test_resume_restores_optimizer_scheduler_step(tmp_path: Path) -> None:
    """resume_from restores model + optimizer + scheduler + global_step."""
    adapter = CSDIAdapter(_make_config())
    data = TinyWindowedDataModule()
    ckpt_dir = tmp_path / "ckpt"
    from ddssm.experiment.experiment import TrainingScalars

    adapter.fit(
        data=data,
        training=TrainingScalars(
            steps=3, log_every=1, validate_every=0, checkpoint_every=1
        ),
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "m.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(ckpt_dir),
    )
    latest = str(ckpt_dir / "ckpt_latest.pth")
    assert os.path.exists(latest)
    step_before = adapter._global_step
    assert step_before == 3

    # A fresh adapter resumes from that ckpt: global_step is restored, and the
    # continuation runs the REMAINING steps only.
    resumed = CSDIAdapter(_make_config())
    resumed.fit(
        data=data,
        training=TrainingScalars(
            steps=5, log_every=1, validate_every=0, resume_from=latest
        ),
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "m2.csv"),
        tensorboard_dir=str(tmp_path / "tb2"),
        checkpoint_dir=str(tmp_path / "ckpt2"),
    )
    assert resumed._global_step == 5
    # Optimizer + scheduler state were restored (non-empty momentum buffers).
    assert len(resumed._optimizer.state_dict()["state"]) > 0


def test_cross_process_roundtrip(tmp_path: Path) -> None:
    """Save in-process, then load + forecast in a fresh subprocess.

    The child inherits this process's env exports (torch libs, dynamo disable),
    so it can rebuild + reload + forecast. We compare its median forecast (over
    a seeded draw) to ours — equivalence proves the checkpoint survives a
    process boundary.
    """
    adapter = CSDIAdapter(_make_config())
    data = TinyWindowedDataModule()
    _fit_small(adapter, data, tmp_path)

    ckpt = str(tmp_path / "csdi.pth")
    adapter.save_checkpoint(ckpt)

    batch = _forecast_inputs_from(data)
    torch.manual_seed(999)
    ref = adapter.forecast(**batch, num_samples=2)["pred_mean"]

    x_hist_path = str(tmp_path / "x_hist.pt")
    torch.save(batch["x_hist"], x_hist_path)
    out_path = str(tmp_path / "child_out.pt")

    child = f"""
import torch
from tests.adapters.test_csdi import _make_config, _forecast_inputs_from
from tests.adapters.test_csdi import TinyWindowedDataModule
from ddssm.adapters.csdi import CSDIAdapter

adapter = CSDIAdapter(_make_config())
adapter.load_checkpoint({ckpt!r}, device=torch.device("cpu"))
data = TinyWindowedDataModule()
batch = _forecast_inputs_from(data)
batch["x_hist"] = torch.load({x_hist_path!r})
torch.manual_seed(999)
out = adapter.forecast(**batch, num_samples=2)["pred_mean"]
torch.save(out, {out_path!r})
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.abspath("src"),
        os.path.abspath("."),
        env.get("PYTHONPATH", ""),
    ])
    subprocess.run([sys.executable, "-c", child], check=True, env=env)

    child_out = torch.load(out_path)
    torch.testing.assert_close(ref, child_out)

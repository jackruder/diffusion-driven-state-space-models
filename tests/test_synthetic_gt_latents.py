"""Tests for the GT-latent surface on :class:`SyntheticDataset`.

Verifies that ``expose_gt_latents=True`` correctly threads through the
data module and exposes the underlying clean latent ``z`` to consumers
(used by the ``crps_sum_latent`` metric).
"""

from __future__ import annotations

import torch

from ddssm.data.synthetic import SyntheticDataset
from ddssm.data.datamodule import SyntheticDataModule


def test_legacy_no_gt_latent_field_by_default() -> None:
    """Default behaviour: no ``gt_latent`` key in items."""
    ds = SyntheticDataset(
        mode="lgssm",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
    )
    item = ds[0]
    assert "gt_latent" not in item
    assert "observed_data" in item


def test_lgssm_exposes_gt_latents() -> None:
    """LGSSM with the flag on adds ``gt_latent`` of matching shape."""
    ds = SyntheticDataset(
        mode="lgssm",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" in item
    assert item["gt_latent"].shape == item["observed_data"].shape
    # Under LGSSM data = z + noise, so the two should differ.
    diff = (item["observed_data"] - item["gt_latent"]).abs().mean().item()
    assert diff > 0.0


def test_unsupported_modes_dont_expose_gt_latent_even_with_flag() -> None:
    """Modes without a closed-form latent process ignore the flag.

    ``harmonic`` generates ``x_t = sin(...) + noise`` directly — there
    is no underlying latent dynamical system to expose, so the flag is
    a silent no-op for it.
    """
    ds = SyntheticDataset(
        mode="harmonic",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" not in item


def test_nonlinear_bimodal_lift_1d_exposes_gt_latents() -> None:
    """The 1D nonlinear-bimodal-lift mode exposes its scalar latent."""
    from ddssm.data.synthetic import NLBL_SIGMA_X

    ds = SyntheticDataset(
        mode="nonlinear-bimodal-lift",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" in item
    # Latent is scalar (d=1); observation is also D=1 here so shapes match.
    assert item["gt_latent"].shape == (1, 8)
    # Observation = lift(z) + noise: different from z (the lift is nonlinear).
    diff = (item["observed_data"] - item["gt_latent"]).abs().mean().item()
    assert diff > NLBL_SIGMA_X  # bigger than just the obs noise


def test_nonlinear_bimodal_lift_mv_exposes_4d_gt_latent() -> None:
    """The MV variant exposes a 4-D latent and an 8-D observation."""
    from ddssm.data.synthetic import NLBL_MV_OBS_D, NLBL_MV_LATENT_D

    ds = SyntheticDataset(
        mode="nonlinear-bimodal-lift-mv",
        split="val",
        N_per_split=4,
        T=8,
        D=NLBL_MV_OBS_D,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" in item
    assert item["gt_latent"].shape == (NLBL_MV_LATENT_D, 8)
    assert item["observed_data"].shape == (NLBL_MV_OBS_D, 8)


def test_nonlinear_bimodal_lift_mv_rejects_wrong_obs_dim() -> None:
    """MV mode enforces D == NLBL_MV_OBS_D (the lift target)."""
    import pytest

    with pytest.raises(AssertionError, match="nonlinear-bimodal-lift-mv"):
        SyntheticDataset(
            mode="nonlinear-bimodal-lift-mv",
            split="val",
            N_per_split=2,
            T=4,
            D=3,  # wrong
            dataset_seed=0,
        )


def test_henon_lift_exposes_2d_gt_latent() -> None:
    """The chaotic Hénon-lift mode exposes a 2-D latent and an 8-D observation."""
    from ddssm.data.synthetic import HENON_OBS_D, HENON_LATENT_D

    ds = SyntheticDataset(
        mode="henon-lift",
        split="val",
        N_per_split=4,
        T=8,
        D=HENON_OBS_D,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" in item
    assert item["gt_latent"].shape == (HENON_LATENT_D, 8)
    assert item["observed_data"].shape == (HENON_OBS_D, 8)


def test_henon_lift_rejects_wrong_obs_dim() -> None:
    """Hénon-lift enforces D == HENON_OBS_D (the lift target)."""
    import pytest

    with pytest.raises(AssertionError, match="henon-lift"):
        SyntheticDataset(
            mode="henon-lift",
            split="val",
            N_per_split=2,
            T=4,
            D=3,
            dataset_seed=0,
        )


def test_henon_lift_latent_stays_bounded() -> None:
    """Chaos safety: the clamped map + small process noise must never let a
    trajectory escape to the divergent region (which would blow up the lift
    and the per-dim standardisation). The whole latent path stays finite and
    O(1) after standardisation.
    """
    from ddssm.data.synthetic import HENON_OBS_D

    ds = SyntheticDataset(
        mode="henon-lift",
        split="train",
        N_per_split=128,
        T=64,
        D=HENON_OBS_D,
        dataset_seed=1,
        expose_gt_latents=True,
    )
    z = ds.gt_latents
    assert torch.isfinite(z).all()
    assert z.abs().max().item() < 12.0  # standardised ⇒ no divergent spikes


def test_data_module_threads_flag_to_dataset() -> None:
    """``SyntheticDataModule(expose_gt_latents=True)`` propagates the flag."""
    dm = SyntheticDataModule(
        mode="lgssm",
        T=8,
        D=1,
        N_per_split=4,
        batch_size=2,
        expose_gt_latents=True,
    )
    batch = next(iter(dm.val_loader()))
    assert "gt_latent" in batch
    assert batch["gt_latent"].shape == batch["observed_data"].shape


def test_parse_batch_threads_gt_latent_to_device() -> None:
    """``parse_batch`` (the data module's transform) carries ``gt_latent`` through."""
    dm = SyntheticDataModule(
        mode="lgssm",
        T=8,
        D=1,
        N_per_split=4,
        batch_size=2,
        expose_gt_latents=True,
    )
    raw_batch = next(iter(dm.val_loader()))
    parsed = dm.batch_transform(raw_batch, torch.device("cpu"))
    assert "gt_latent" in parsed
    assert parsed["gt_latent"].dtype == torch.float32


def test_dataset_splits_keep_gt_latents_disjoint() -> None:
    """train/val/test slices each retain their own gt_latents."""
    common = dict(
        mode="lgssm", N_per_split=4, T=8, D=1, dataset_seed=0, expose_gt_latents=True
    )
    train = SyntheticDataset(split="train", **common)
    val = SyntheticDataset(split="val", **common)
    test = SyntheticDataset(split="test", **common)

    assert train.gt_latents is not None
    assert val.gt_latents is not None
    assert test.gt_latents is not None
    # Each split has N_per_split sequences.
    assert train.gt_latents.shape[0] == 4
    assert val.gt_latents.shape[0] == 4
    assert test.gt_latents.shape[0] == 4
    # They should be different sequences (drawn from different slice).
    assert not torch.equal(train.gt_latents[0], val.gt_latents[0])


def test_disabled_flag_preserves_legacy_behavior() -> None:
    """``expose_gt_latents=False`` (default) ⇒ no GT-latent state stored."""
    ds = SyntheticDataset(
        mode="lgssm",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
    )
    assert ds.gt_latents is None
    item = ds[0]
    assert "gt_latent" not in item

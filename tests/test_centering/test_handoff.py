"""Unit tests for :func:`ddssm.centering.handoff.perform_centering_handoff`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import pytest

from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.handoff import (
    CenteringHandoffConf,
    perform_centering_handoff,
)
from ddssm.centering.baselines import MLPBaseline
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition

B = 2
D = 4
J = 1
T_MAX = 8


def _module_state(mod: torch.nn.Module) -> list[torch.Tensor]:
    """Snapshot a module's parameter values as cloned tensors."""
    return [p.detach().clone() for p in mod.parameters()]


def _flat_norm(state: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm of the concatenated parameter values."""
    if not state:
        return torch.tensor(0.0)
    return torch.cat([p.reshape(-1) for p in state]).norm()


class _DummyTrainer:
    """Stand-in for :class:`DDSSMTrainer` with only the bits handoff touches."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-3,
        )
        self._rebuild_called_with: Any = None

    def _rebuild_optimizer(self, lrs: Any) -> None:
        self._rebuild_called_with = lrs
        # Mimic real behaviour: discard moments by re-creating the
        # optimizer.  Use a flat parameter group at the supplied lr.
        lr = float(getattr(lrs, "enc_lr", 1e-3))
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr,
        )


def _make_model() -> torch.nn.Module:
    """Build a small module that *looks* like a DDSSM_base for the handoff."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=8, n_layers=2)
    aux = AuxPosterior(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed")
    # The handoff treats encoder / decoder / aux / transition / baseline as
    # opaque submodules.  Build a minimal nn.Module exposing them by
    # attribute name.
    model = SimpleNamespace(
        encoder=torch.nn.Linear(D, D),  # stand-in encoder
        decoder=torch.nn.Linear(D, D),  # stand-in decoder
        aux_posterior=aux,
        baseline=baseline,
        baseline_anchor=None,
        sigma_data=sigma_data,
        transition=BaselineGaussianTransition(
            baseline=baseline, latent_dim=D, j=J, emb_time_dim=4,
        ),
    )
    # Add a fake `.parameters()` that yields union of submodule params.
    nn_module = torch.nn.Module()
    nn_module.encoder = model.encoder
    nn_module.decoder = model.decoder
    nn_module.aux_posterior = model.aux_posterior
    nn_module.baseline = model.baseline
    nn_module.sigma_data = model.sigma_data
    nn_module.transition = model.transition
    nn_module.baseline_anchor = None  # attribute slot
    return nn_module


def test_handoff_perturbs_encoder_by_sigma_pert() -> None:
    """``φ ← φ + σ_pert · ε``: L2 norm of encoder weights moves by ~σ_pert·√n."""
    torch.manual_seed(0)
    model = _make_model()
    pre = _module_state(model.encoder)
    trainer = _DummyTrainer(model)
    spec = CenteringHandoffConf(sigma_pert=1e-1)
    perform_centering_handoff(
        trainer=trainer,
        spec=spec,
        new_lrs=SimpleNamespace(enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3),
    )
    post = _module_state(model.encoder)
    # Each parameter has moved by roughly σ_pert in elementwise L2 (the
    # perturbation is independent N(0, σ_pert²); the expected L2
    # difference per element ≈ σ_pert).
    diffs = [post[i] - pre[i] for i in range(len(pre))]
    delta_norm = _flat_norm(diffs)
    pre_norm = _flat_norm(pre)
    # The perturbation should be non-trivial relative to the original.
    assert float(delta_norm.item()) > 0.0
    # And smaller-than-saturating.
    assert float((delta_norm / max(pre_norm, 1e-9)).item()) < 10.0


def test_handoff_leaves_aux_posterior_decoder_baseline_transition_untouched() -> None:
    """Other submodules' parameters are byte-for-byte unchanged."""
    torch.manual_seed(0)
    model = _make_model()
    pre_aux = _module_state(model.aux_posterior)
    pre_dec = _module_state(model.decoder)
    pre_baseline = _module_state(model.baseline)
    pre_trans = _module_state(model.transition)
    trainer = _DummyTrainer(model)
    perform_centering_handoff(
        trainer=trainer,
        spec=CenteringHandoffConf(sigma_pert=1e-1),
        new_lrs=SimpleNamespace(enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3),
    )
    post_aux = _module_state(model.aux_posterior)
    post_dec = _module_state(model.decoder)
    post_baseline = _module_state(model.baseline)
    post_trans = _module_state(model.transition)
    for pre, post in [
        (pre_aux, post_aux),
        (pre_dec, post_dec),
        (pre_baseline, post_baseline),
        (pre_trans, post_trans),
    ]:
        for a, b in zip(pre, post):
            assert torch.equal(a, b)


def test_handoff_snapshots_baseline_anchor() -> None:
    """``model.baseline_anchor`` is populated, frozen, and parameter-disjoint."""
    model = _make_model()
    trainer = _DummyTrainer(model)
    perform_centering_handoff(
        trainer=trainer,
        spec=CenteringHandoffConf(sigma_pert=0.0),
        new_lrs=SimpleNamespace(enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3),
    )
    anchor = model.baseline_anchor
    assert anchor is not None
    # Frozen.
    for p in anchor.parameters():
        assert not p.requires_grad
    # Parameter-disjoint (deep copy).
    for p_live, p_anchor in zip(model.baseline.parameters(), anchor.parameters()):
        assert p_live.data_ptr() != p_anchor.data_ptr()
        # And contents start identical.
        assert torch.equal(p_live, p_anchor)


def test_handoff_rebuilds_optimizer() -> None:
    """``trainer._rebuild_optimizer`` is called once with the new LRs."""
    model = _make_model()
    trainer = _DummyTrainer(model)
    lrs = SimpleNamespace(
        enc_lr=2e-3, dec_lr=3e-3, zinit_lr=4e-3, trans_lr=5e-3,
    )
    perform_centering_handoff(
        trainer=trainer,
        spec=CenteringHandoffConf(sigma_pert=0.0),
        new_lrs=lrs,
    )
    assert trainer._rebuild_called_with is lrs
    # The optimizer should be a *new* AdamW with empty state.
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert len(trainer.optimizer.state) == 0


def test_handoff_resets_sigma_data_schedule_preserves_values() -> None:
    """ema_step zeros, frozen flag set under 'fixed', buffer values persist."""
    model = _make_model()
    # Advance the buffer state somehow.
    model.sigma_data.sigma_data2 = torch.tensor(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], dtype=torch.float32,
    )
    model.sigma_data.ema_step = torch.full(
        (T_MAX,), 7, dtype=torch.long,
    )
    pre_values = model.sigma_data.sigma_data2.clone()
    trainer = _DummyTrainer(model)
    perform_centering_handoff(
        trainer=trainer,
        spec=CenteringHandoffConf(sigma_pert=0.0),
        new_lrs=SimpleNamespace(enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3),
    )
    # Schedule was reset.
    assert torch.equal(model.sigma_data.ema_step, torch.zeros(T_MAX, dtype=torch.long))
    # The "fixed" tracking_mode sets frozen at reset_schedule.
    assert model.sigma_data.frozen is True
    # Buffer values persist.
    assert torch.equal(model.sigma_data.sigma_data2, pre_values)


def test_handoff_zero_sigma_pert_no_encoder_change() -> None:
    """``σ_pert = 0`` leaves the encoder weights byte-for-byte unchanged."""
    model = _make_model()
    pre = _module_state(model.encoder)
    trainer = _DummyTrainer(model)
    perform_centering_handoff(
        trainer=trainer,
        spec=CenteringHandoffConf(sigma_pert=0.0),
        new_lrs=SimpleNamespace(enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3),
    )
    post = _module_state(model.encoder)
    for a, b in zip(pre, post):
        assert torch.equal(a, b)


def test_handoff_raises_without_baseline() -> None:
    """A model without a Baseline cannot be handed off."""
    model = _make_model()
    model.baseline = None  # type: ignore[assignment]
    trainer = _DummyTrainer(model)
    with pytest.raises(AttributeError):
        perform_centering_handoff(
            trainer=trainer,
            spec=CenteringHandoffConf(sigma_pert=0.0),
            new_lrs=SimpleNamespace(
                enc_lr=1e-3, dec_lr=1e-3, zinit_lr=1e-3, trans_lr=1e-3,
            ),
        )

"""Split-loss training tests: loss objects, grad routing, skip, topology.

Covers the split-loss stack end to end: ``FullELBO``'s ``SplitLoss``
return and shims, λ-independence of the ψ side, the trainer's
two-optimizer topology (``_install_split_topology`` / ``_backward_loss``
/ ``_optimizer_step`` / ``_install_scheduler``), the non-finite grad
skip, the F1/F2/F3 regression contracts (checkpoint mode adoption, live
requires-grad re-filtering of the split caches, single-mode downgrade at
``fit()`` entry), the probe/training parity of the adaptive-IS ``p_k``
clip, and the unit-level ``(phith, psi)`` init-score contract on the
Gaussian transitions.
"""

from __future__ import annotations

import os
import csv as _csv
import copy
import math
import logging
from unittest.mock import patch

import torch
import pytest
from torch.utils.data import Dataset, DataLoader

from ddssm.model.dssd import DDSSMHyperParamsConf
from ddssm.model.losses import FullELBO, SplitLoss, LossComponents
from tests.test_trainer import make_small_model
from ddssm.training.train import DDSSMTrainer
from ddssm.variance.probe import _p_k_for_mode
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.training.checkpoint import Checkpoint
from ddssm.training.train_utils import make_warmup_cosine, split_params_phith_psi
from ddssm.model.centering.baselines import MLPBaseline
from tests.test_integration.conftest import make_vhp_model
from ddssm.model.transitions.diffusion import _adaptive_is_density_meandom
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition


@pytest.fixture(scope="module", autouse=True)
def _eager_models():
    """Build models eagerly (no ``torch.compile``) — fast and deterministic."""
    old = os.environ.get("DDSSM_TORCH_COMPILE")
    os.environ["DDSSM_TORCH_COMPILE"] = "0"
    yield
    if old is None:
        os.environ.pop("DDSSM_TORCH_COMPILE", None)
    else:
        os.environ["DDSSM_TORCH_COMPILE"] = old


@pytest.fixture(scope="module")
def vhp_model(_eager_models):
    """One shared DiffusionTransition model; tests deepcopy before mutating."""
    torch.manual_seed(0)
    return make_vhp_model()


class _DS(Dataset):
    """Tiny deterministic dataset shaped for the DiffusionTransition model."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        return {
            "observed_data": torch.randn(1, 5),
            "observation_mask": torch.ones(1, 5),
            "timepoints": torch.arange(5, dtype=torch.long),
        }


def _make_trainer(model, tmp_path, *, split=True, rate_lambda=None) -> DDSSMTrainer:
    """Trainer (grad_accum=1) with a ``FullELBO`` active loss installed."""
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    lam = rate_lambda if rate_lambda is not None else (lambda _s: 1.0)
    trainer._active_loss = FullELBO(rate_lambda=lam, use_split_loss=split)
    return trainer


def _fit_one_step(trainer) -> None:
    loader = DataLoader(_DS(), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=1,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )


def _diffusion_batch(B: int = 2, T: int = 5) -> dict[str, torch.Tensor]:
    return {
        "observed_data": torch.randn(B, 1, T),
        "observation_mask": torch.ones(B, 1, T),
        "timepoints": torch.arange(T, dtype=torch.long).expand(B, T),
    }


def _forward_split(trainer, *, seed: int):
    """Zero grads, seed, and run one training forward; return the loss."""
    trainer.model.train()
    for opt in trainer._optimizers:
        opt.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    batch = _diffusion_batch()
    loss, _metrics, _weight = trainer._compute_loss_and_metrics(batch=batch, amp=False)
    return loss


def _live(params):
    return [p for p in params if p.requires_grad]


def _fill_grads(model, value: float = 0.01) -> None:
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, value)


def _snapshot_params(model):
    return [p.detach().clone() for p in model.parameters()]


def _components(**overrides) -> LossComponents:
    values = dict(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.tensor(3.0),
        trans_kl_phith=torch.tensor(4.0),
        trans_kl_psi=torch.tensor(5.0),
        r_sigma_p=torch.tensor(6.0),
        r_mu_p=torch.tensor(7.0),
    )
    values.update(overrides)
    return LossComponents(**values)


# ---------------------------------------------------------------------------
# Loss-object level (no model needed)
# ---------------------------------------------------------------------------


def test_full_elbo_split_returns_splitloss_single_returns_tensor():
    """Split mode returns a ``SplitLoss`` pair; single mode a plain tensor."""
    comps = _components()
    single = FullELBO(rate_lambda=lambda _s: 0.5, lambda_sigma_p=2.0, lambda_mu_p=3.0)
    out = single(comps, 1)
    assert isinstance(out, torch.Tensor)
    expected_phith = 1.0 + 0.5 * (2.0 + 4.0) + 2.0 * 6.0 + 3.0 * 7.0
    assert out.item() == pytest.approx(expected_phith)

    split = FullELBO(
        rate_lambda=lambda _s: 0.5,
        lambda_sigma_p=2.0,
        lambda_mu_p=3.0,
        use_split_loss=True,
    )
    out_split = split(comps, 1)
    assert isinstance(out_split, SplitLoss)
    assert out_split.phith.item() == pytest.approx(expected_phith)
    assert out_split.psi.item() == pytest.approx(5.0 + 3.0)


def test_splitloss_shims():
    """``SplitLoss`` mirrors the tensor surface the fit loop touches."""
    phith = torch.tensor(2.0, requires_grad=True) * 1.0
    psi = torch.tensor(3.0, requires_grad=True) * 1.0
    sl = SplitLoss(phith=phith, psi=psi)
    assert sl.total.item() == pytest.approx(5.0)
    detached = sl.detach()
    assert isinstance(detached, SplitLoss)
    assert not detached.phith.requires_grad and not detached.psi.requires_grad
    assert sl.phith.requires_grad, "detach() must not mutate the original"
    assert sl.item() == pytest.approx(5.0)
    assert float(sl) == pytest.approx(5.0)
    halved = sl / 2
    assert isinstance(halved, SplitLoss)
    assert halved.phith.item() == pytest.approx(1.0)
    assert halved.psi.item() == pytest.approx(1.5)


def test_psi_side_ignores_lambda():
    """The ψ side is NOT gated by ``rate_lambda``; φθ at λ=0 is KL-free."""
    comps = _components()
    lam0 = FullELBO(rate_lambda=lambda _s: 0.0, use_split_loss=True)(comps, 1)
    lam1 = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=True)(comps, 1)
    assert torch.equal(lam0.psi, lam1.psi)
    assert lam0.phith.item() == pytest.approx(comps.recon.item())
    assert lam1.phith.item() == pytest.approx(1.0 + 2.0 + 4.0)


# ---------------------------------------------------------------------------
# Split backward: gradient routing
# ---------------------------------------------------------------------------


def test_split_backward_routes_grads(vhp_model, tmp_path):
    """Each side's backward populates only its own parameter set."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    loss = _forward_split(trainer, seed=1)
    assert isinstance(loss, SplitLoss)

    phith_live = _live(trainer._phith_params)
    psi_live = _live(trainer._psi_params)

    # φθ pass first (mirrors _backward_loss): ψ params must stay untouched.
    loss.phith.backward(inputs=phith_live, retain_graph=True)
    assert all(p.grad is None for p in psi_live), "phith backward leaked into psi"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in phith_live)

    # ψ pass second: φθ grads must be bit-identical to their snapshot.
    snap = [None if p.grad is None else p.grad.clone() for p in phith_live]
    loss.psi.backward(inputs=psi_live)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in psi_live)
    for p, s in zip(phith_live, snap):
        if s is None:
            assert p.grad is None, "psi backward leaked into a phith param"
        else:
            assert torch.equal(p.grad, s), "psi backward modified a phith grad"

    # The trainer's own split backward populates both sides in one call.
    loss2 = _forward_split(trainer, seed=2)
    trainer._backward_loss(loss2, scaler=trainer.scaler, amp=False)
    assert any(p.grad is not None for p in phith_live)
    assert any(p.grad is not None for p in psi_live)


def test_split_grads_match_separate_backwards(vhp_model, tmp_path):
    """``_backward_loss`` grads equal per-side ``autograd.grad`` exactly.

    A second identical model (deepcopy) is forwarded under the same seed,
    so both graphs are bit-identical; the split backward's grads must
    match a plain ``autograd.grad`` of each side w.r.t. its own params.
    """
    model_a = copy.deepcopy(vhp_model)
    model_b = copy.deepcopy(vhp_model)
    trainer_a = _make_trainer(model_a, tmp_path / "a")
    trainer_a._install_split_topology()
    trainer_b = _make_trainer(model_b, tmp_path / "b")

    loss_a = _forward_split(trainer_a, seed=11)
    trainer_a._backward_loss(loss_a, scaler=trainer_a.scaler, amp=False)

    trainer_b.model.train()
    torch.manual_seed(11)
    batch = _diffusion_batch()
    loss_b, _m, _w = trainer_b._compute_loss_and_metrics(batch=batch, amp=False)
    assert torch.equal(loss_a.phith.detach(), loss_b.phith.detach())
    assert torch.equal(loss_a.psi.detach(), loss_b.psi.detach())

    phith_b, psi_b = split_params_phith_psi(model_b)
    grads_phith = torch.autograd.grad(
        loss_b.phith, phith_b, retain_graph=True, allow_unused=True
    )
    grads_psi = torch.autograd.grad(loss_b.psi, psi_b, allow_unused=True)

    phith_a = _live(trainer_a._phith_params)
    psi_a = _live(trainer_a._psi_params)
    assert len(phith_a) == len(phith_b) and len(psi_a) == len(psi_b)
    for p, g in zip(phith_a, grads_phith):
        if g is None:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.equal(p.grad, g)
    for p, g in zip(psi_a, grads_psi):
        if g is None:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.equal(p.grad, g)


def test_psi_trains_when_lambda_zero_end_to_end(vhp_model, tmp_path):
    """One split fit step at λ=0 still trains the score net (ψ side)."""
    trainer = _make_trainer(
        copy.deepcopy(vhp_model), tmp_path, rate_lambda=lambda _s: 0.0
    )
    pre = {
        k: v.detach().clone()
        for k, v in trainer.model.transition.diffmodel.state_dict().items()
    }
    _fit_one_step(trainer)
    assert len(trainer._optimizers) == 2 and trainer.opt_psi is not None
    assert len(trainer.opt_psi.state_dict()["state"]) > 0, (
        "psi optimizer accumulated no state — the score net never stepped"
    )
    post = trainer.model.transition.diffmodel.state_dict()
    assert any(not torch.equal(pre[k], post[k]) for k in pre), (
        "no diffmodel param changed despite rate_lambda == 0"
    )


# ---------------------------------------------------------------------------
# Non-finite grad skip
# ---------------------------------------------------------------------------


def test_grad_skip_gates_everything(tmp_path):
    """A NaN grad skips optimizer, scheduler, and EMA; a finite step advances."""
    torch.manual_seed(3)
    trainer = DDSSMTrainer(
        model=make_small_model(),
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    sched = make_warmup_cosine(trainer._optimizers[0], total_steps=100, warmup_steps=5)
    trainer._install_scheduler(sched)
    model = trainer.model
    params = [p for p in model.parameters() if p.requires_grad]

    _fill_grads(model)
    params[0].grad.fill_(float("nan"))
    before = _snapshot_params(model)
    shadow_before = {k: v.clone() for k, v in trainer.ema.shadow.items()}
    epoch_before = sched.last_epoch

    assert trainer._optimizer_step(scaler=trainer.scaler, amp=False) is False
    assert trainer.grad_skip_count == 1
    assert math.isnan(trainer._last_grad_norm)
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p.detach(), b), "param changed on a skipped step"
    assert all(p.grad is None for p in params), "grads not zeroed (set_to_none)"
    assert sched.last_epoch == epoch_before, "scheduler stepped on a skip"
    for k, v in trainer.ema.shadow.items():
        assert torch.equal(v, shadow_before[k]), "EMA shadow moved on a skip"
    assert len(trainer.optimizer.state) == 0, "Adam state created on a skip"

    # A finite step then advances params, scheduler, and the EMA shadow.
    _fill_grads(model)
    assert trainer._optimizer_step(scaler=trainer.scaler, amp=False) is True
    assert trainer.grad_skip_count == 1
    assert math.isfinite(trainer._last_grad_norm)
    assert sched.last_epoch == epoch_before + 1
    assert any(
        not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before)
    )
    assert any(
        not torch.equal(trainer.ema.shadow[k], shadow_before[k]) for k in shadow_before
    )


def test_grad_skip_covers_both_optimizers_split_mode(vhp_model, tmp_path):
    """A NaN on a ψ param grad discards the step on BOTH optimizers."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    _fill_grads(trainer.model)
    _live(trainer._psi_params)[0].grad.fill_(float("nan"))
    before = _snapshot_params(trainer.model)

    assert trainer._optimizer_step(scaler=trainer.scaler, amp=False) is False
    assert trainer.grad_skip_count == 1
    for p, b in zip(trainer.model.parameters(), before):
        assert torch.equal(p.detach(), b), "param changed on a skipped split step"
    assert all(p.grad is None for p in trainer.model.parameters())
    for opt in trainer._optimizers:
        assert len(opt.state) == 0, "optimizer state created on a skipped step"


# ---------------------------------------------------------------------------
# F2 regression: split caches re-filter by the live requires_grad
# ---------------------------------------------------------------------------


def test_split_backward_after_freezing_baseline(vhp_model, tmp_path):
    """REGRESSION F2a: freezing the baseline after install must not crash.

    ``perform_centering_handoff`` flips ``requires_grad=False`` on the
    baseline after the split topology is installed; the split backward
    must re-filter its caches by the live flag (a stale snapshot fed
    frozen tensors to ``backward(inputs=...)`` and crashed).
    """
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()  # baseline trainable at install
    for p in trainer.model.baseline.parameters():
        p.requires_grad = False  # what perform_centering_handoff does
    loss = _forward_split(trainer, seed=5)
    trainer._backward_loss(loss, scaler=trainer.scaler, amp=False)  # must not raise
    assert all(p.grad is None for p in trainer.model.baseline.parameters()), (
        "frozen baseline must accumulate no grads"
    )
    assert any(p.grad is not None for p in trainer.model.encoder.parameters()), (
        "unfrozen phith side must still receive grads"
    )


def test_split_backward_after_unfreezing_encoder(vhp_model, tmp_path):
    """REGRESSION F2b: a module unfrozen after install must receive grads."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    for p in trainer.model.encoder.parameters():
        p.requires_grad = False  # frozen at install time
    trainer._install_split_topology()
    for p in trainer.model.encoder.parameters():
        p.requires_grad = True  # unfrozen by the next stage
    loss = _forward_split(trainer, seed=6)
    trainer._backward_loss(loss, scaler=trainer.scaler, amp=False)
    assert any(p.grad is not None for p in trainer.model.encoder.parameters()), (
        "encoder unfrozen after install was silently starved of grads"
    )


# ---------------------------------------------------------------------------
# F1 regression: undeclared-mode restore adopts the checkpoint's mode
# ---------------------------------------------------------------------------


def test_undeclared_trainer_adopts_split_checkpoint(vhp_model, tmp_path):
    """REGRESSION F1: a no-loss single-topology trainer restores a split ckpt.

    This is the orchestrator's preempt-resume path (restore BEFORE any
    stage loss is installed): the trainer has declared no mode yet, so it
    must adopt the checkpoint's split mode — install the topology and
    load the ψ optimizer state — instead of raising a mode mismatch.
    """
    producer = _make_trainer(copy.deepcopy(vhp_model), tmp_path / "prod")
    producer._install_split_topology()
    # Fabricate ψ Adam state without a fit: zero grads, one step.
    for group in producer.opt_psi.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    producer.opt_psi.step()
    producer.opt_psi.zero_grad(set_to_none=True)
    ckpt_path = str(tmp_path / "split.pth")
    producer.save_checkpoint(ckpt_path)
    saved = Checkpoint.load(ckpt_path, device=torch.device("cpu"))
    assert saved.split_loss is True
    assert saved.optimizer_state_psi is not None

    consumer = DDSSMTrainer(
        model=copy.deepcopy(vhp_model),
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
        tensorboard_dir=str(tmp_path / "tb2"),
        quiet=True,
    )
    assert consumer._active_loss is None and len(consumer._optimizers) == 1

    consumer.restore_from_checkpoint(ckpt_path)  # must NOT raise

    assert len(consumer._optimizers) == 2, "split topology must be adopted"
    assert consumer.opt_psi is not None
    assert len(consumer.opt_psi.state_dict()["state"]) > 0, (
        "psi optimizer state was not actually loaded"
    )


# ---------------------------------------------------------------------------
# F3 regression: single-mode fit() entry downgrades the split topology
# ---------------------------------------------------------------------------


def test_single_mode_fit_downgrades_split_topology(vhp_model, tmp_path):
    """REGRESSION F3: a single-loss stage after split steps drops ``opt_psi``."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    assert len(trainer._optimizers) == 2

    trainer._active_loss = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=False)
    _fit_one_step(trainer)

    assert len(trainer._optimizers) == 1
    assert trainer.opt_psi is None
    assert trainer.optimizer is trainer._optimizers[0]
    assert trainer._phith_params is None and trainer._psi_params is None
    ckpt = Checkpoint.from_trainer(trainer)
    assert ckpt.split_loss is False, "single-loss stage ckpt must not be split"
    assert ckpt.optimizer_state_psi is None


# ---------------------------------------------------------------------------
# Probe / training density parity (p_k clip)
# ---------------------------------------------------------------------------


def test_probe_density_matches_training_density(vhp_model):
    """``_p_k_for_mode`` reproduces the training-time p_k_clip-clipped density."""
    transition = copy.deepcopy(vhp_model.transition)
    sd2 = torch.tensor([1.0], dtype=transition.sigma_tilde.dtype)
    unclipped = _adaptive_is_density_meandom(
        transition.sigma_tilde, sd2, floor=transition.gfloor, p_k_clip=0.0
    ).squeeze(0)
    # Make sure the clip binds so equal-vs-differ is a real check, not 0==0.
    if unclipped.min() >= transition.p_k_clip:
        transition.p_k_clip = float(unclipped.min().item()) * 3.0
    clipped = _adaptive_is_density_meandom(
        transition.sigma_tilde,
        sd2,
        floor=transition.gfloor,
        p_k_clip=transition.p_k_clip,
    ).squeeze(0)
    assert not torch.allclose(clipped, unclipped), "precondition: clip must bind"

    probe_pk = _p_k_for_mode(transition, "adaptive_is")
    assert torch.equal(probe_pk, clipped), (
        "probe density must equal the p_k_clip-clipped training density"
    )
    assert not torch.allclose(probe_pk, unclipped), (
        "probe density must differ from the unclipped one (pre-fix behavior)"
    )


# ---------------------------------------------------------------------------
# Scheduler topology mirror
# ---------------------------------------------------------------------------


def test_scheduler_mirrors_psi_side(vhp_model, tmp_path, caplog):
    """``_install_scheduler`` mirrors ψ; direct assignment warns once."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    sched = make_warmup_cosine(trainer._optimizers[0], total_steps=100, warmup_steps=5)
    trainer._install_scheduler(sched)
    assert len(trainer._schedulers) == 2
    assert trainer._schedulers[0] is sched
    assert trainer._schedulers[1].optimizer is trainer.opt_psi
    epochs_before = [s.last_epoch for s in trainer._schedulers]
    _fill_grads(trainer.model, 0.001)
    assert trainer._optimizer_step(scaler=trainer.scaler, amp=False) is True
    assert [s.last_epoch for s in trainer._schedulers] == [
        e + 1 for e in epochs_before
    ], "both schedulers must step together"

    # Legacy path: ``trainer.scheduler`` assigned directly (no topology
    # mirror) in split mode → the one-time UNSCHEDULED-psi warning fires.
    trainer._schedulers = []
    assert trainer.scheduler is sched  # assigned directly, as legacy code did
    with caplog.at_level(logging.WARNING, logger="ddssm.training.train"):
        _fill_grads(trainer.model, 0.001)
        trainer._optimizer_step(scaler=trainer.scaler, amp=False)
        assert any("UNSCHEDULED" in r.message for r in caplog.records)
        caplog.clear()
        _fill_grads(trainer.model, 0.001)
        trainer._optimizer_step(scaler=trainer.scaler, amp=False)
        assert not any("UNSCHEDULED" in r.message for r in caplog.records), (
            "the split legacy-scheduler warning must fire only once"
        )


# ---------------------------------------------------------------------------
# Transition unit level: (phith, psi) init-score contract
# ---------------------------------------------------------------------------


def _init_kl_inputs(transition, *, B=2, S=2, T=4, emb_time=8, seed=0):
    torch.manual_seed(seed)
    d = transition.latent_dim
    zs = torch.randn(B, S, d, T)
    enc_stats = {
        "mus": 0.3 * torch.randn(B, S, d, T),
        "logvars": -1.0 + 0.2 * torch.randn(B, S, d, T),
    }
    time_embed = torch.randn(B, T, emb_time)
    return zs, enc_stats, time_embed


def test_transition_kl_init_returns_loss_psi():
    """``return_psi=True`` adds a zero ``loss_psi`` on Gaussian transitions."""
    baseline = MLPBaseline(latent_dim=2, j=1, hidden_dim=8, n_layers=1)
    transitions = [
        BaselineGaussianTransition(
            baseline=baseline, latent_dim=2, j=1, emb_time_dim=8
        ),
        make_small_model().transition,  # plain GaussianTransition
    ]
    for transition in transitions:
        aux = AuxPosterior(latent_dim=2, j=1, hidden_dim=8, n_layers=1)
        zs, enc_stats, time_embed = _init_kl_inputs(transition)
        out = transition.transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=aux,
            time_embed=time_embed,
            return_psi=True,
        )
        assert "loss_psi" in out
        assert out["loss_psi"].shape == ()
        assert out["loss_psi"].item() == 0.0, "no score net → zero psi init loss"
        out_default = transition.transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=aux,
            time_embed=time_embed,
        )
        assert "loss_psi" not in out_default, "loss_psi is opt-in via return_psi"


def test_score_init_step_zero_psi_for_nondiffusion():
    """``_score_init_step`` returns a graph-free zero ψ on Gaussian transitions."""
    baseline = MLPBaseline(latent_dim=2, j=1, hidden_dim=8, n_layers=1)
    transitions = [
        make_small_model().transition,  # plain GaussianTransition
        BaselineGaussianTransition(
            baseline=baseline, latent_dim=2, j=1, emb_time_dim=8
        ),
    ]
    B, S, T = 2, 2, 4
    for transition in transitions:
        torch.manual_seed(0)
        d = transition.latent_dim
        z_t = torch.randn(B * S, d)
        z_hist = torch.randn(B * S, d, 1)
        _zs, enc_stats, time_embed = _init_kl_inputs(transition, B=B, S=S, T=T)
        phith, psi = transition._score_init_step(
            step=0,
            z_t=z_t,
            z_hist=z_hist,
            enc_stats=enc_stats,
            time_embed=time_embed,
            sigma_data=None,
            B=B,
            S=S,
            T=T,
        )
        assert phith.shape == () and torch.isfinite(phith)
        assert psi.shape == ()
        assert psi.item() == 0.0
        assert not psi.requires_grad, "psi zero must carry no graph"


# ---------------------------------------------------------------------------
# Ports from the parallel local split-loss implementation (test_local_ prefix).
# ---------------------------------------------------------------------------


def test_local_loss_components_alias_still_works():
    """``components.trans_kl`` is ``components.trans_kl_phith`` (identity)."""
    comps = _components(init_kl_phith=torch.tensor(2.5), trans_kl_phith=torch.tensor(3.5))
    assert comps.trans_kl is comps.trans_kl_phith
    assert comps.init_kl is comps.init_kl_phith


def test_local_full_elbo_single_mode_numerical_parity():
    """``use_split_loss=False`` output is ``recon + λ·phith_KL + regs`` bit-for-bit.

    Nonzero ``*_psi`` fields prove single-mode reads only the ``*_phith`` fields.
    """
    recon, init_phith, init_psi = 1.0, 2.0, 99.0
    trans_phith, trans_psi = 3.0, 88.0
    r_sigma_p, r_mu_p = 4.0, 5.0
    lam, lambda_sigma_p, lambda_mu_p = 0.5, 0.1, 0.01
    comps = LossComponents(
        recon=torch.tensor(recon),
        init_kl_phith=torch.tensor(init_phith),
        init_kl_psi=torch.tensor(init_psi),
        trans_kl_phith=torch.tensor(trans_phith),
        trans_kl_psi=torch.tensor(trans_psi),
        r_sigma_p=torch.tensor(r_sigma_p),
        r_mu_p=torch.tensor(r_mu_p),
    )
    got = FullELBO(
        rate_lambda=lambda _s: lam,
        lambda_sigma_p=lambda_sigma_p,
        lambda_mu_p=lambda_mu_p,
        use_split_loss=False,
    )(comps, 0)
    expected = recon + lam * (init_phith + trans_phith) + lambda_sigma_p * r_sigma_p + lambda_mu_p * r_mu_p
    assert isinstance(got, torch.Tensor)
    assert got.item() == pytest.approx(expected)


def test_local_no_clip_grad_norm_attribute_or_branch(vhp_model, tmp_path):
    """Trainer has no legacy ``clip_grad_norm`` attribute after M5."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    assert not hasattr(trainer, "clip_grad_norm")


def test_local_zero_grad_covers_both_optimizers(vhp_model, tmp_path):
    """Under split, iterating ``_optimizers`` and calling ``zero_grad`` clears
    both φθ and ψ side gradients."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    assert len(trainer._optimizers) == 2
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    for opt in trainer._optimizers:
        opt.zero_grad(set_to_none=True)
    for p in _live(trainer._phith_params) + _live(trainer._psi_params):
        assert p.grad is None


def test_local_split_loss_second_backward_requires_retain_graph(vhp_model, tmp_path):
    """Without ``retain_graph=True``, the ψ-side backward must fail with a
    freed-graph error."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    loss = _forward_split(trainer, seed=2)
    assert isinstance(loss, SplitLoss)
    # First backward WITHOUT retain_graph — the shared subgraph is freed.
    loss.phith.backward(inputs=_live(trainer._phith_params))
    with pytest.raises(RuntimeError, match="backward|graph|freed"):
        loss.psi.backward(inputs=_live(trainer._psi_params))


def test_local_split_loss_amp_scales_both_backwards(vhp_model, tmp_path):
    """Under AMP, ``scaler.scale`` is invoked on both the φθ and ψ scalars."""
    trainer = _make_trainer(copy.deepcopy(vhp_model), tmp_path)
    trainer._install_split_topology()
    loss = _forward_split(trainer, seed=3)
    assert isinstance(loss, SplitLoss)
    scale_calls = []
    orig_scale = trainer.scaler.scale

    def spy_scale(t):
        scale_calls.append(t)
        return orig_scale(t)

    trainer.scaler.scale = spy_scale
    trainer._backward_loss(loss, scaler=trainer.scaler, amp=True)
    assert len(scale_calls) == 2, f"expected 2 scale() calls; got {len(scale_calls)}"


def test_local_split_loss_grad_accum_correct_scaling(vhp_model, tmp_path):
    """``grad_accum_steps=4`` divides both φθ and ψ scalars before backward."""
    model = copy.deepcopy(vhp_model)
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=4),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=True)
    trainer._install_split_topology()
    loss = _forward_split(trainer, seed=5)
    assert isinstance(loss, SplitLoss)
    expected_phith = loss.phith.item() / 4
    expected_psi = loss.psi.item() / 4
    calls: list[float] = []
    orig_backward = torch.Tensor.backward

    def spy_backward(self, *args, **kwargs):
        calls.append(float(self.detach()))
        return orig_backward(self, *args, **kwargs)

    with patch.object(torch.Tensor, "backward", spy_backward):
        trainer._backward_loss(loss, scaler=trainer.scaler, amp=False)
    assert len(calls) == 2, f"expected 2 backward calls; got {calls}"
    got_phith, got_psi = calls
    assert got_phith == pytest.approx(expected_phith, rel=1e-5)
    assert got_psi == pytest.approx(expected_psi, rel=1e-5)


def test_local_finite_grad_steps_normally_and_never_rescales(vhp_model, tmp_path):
    """A large finite gradient triggers a step (no skip); the norm computation
    uses ``max_norm=inf`` so grads are never rescaled."""
    model = copy.deepcopy(vhp_model)
    trainer = _make_trainer(model, tmp_path, split=False)
    _fit_one_step(trainer)
    target = next(p for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    target.grad = torch.full_like(target, 1e6)
    param_pre = target.detach().clone()
    trainer._optimizer_step(scaler=trainer.scaler, amp=False)
    assert not torch.equal(target.detach(), param_pre), "large finite grad should step"
    # Second pass: monkeypatch clip_grad_norm_ to prove it's only called with inf.
    calls: list[float] = []
    import torch.nn.utils as _tnu
    orig_clip = _tnu.clip_grad_norm_

    def spy_clip(params, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return orig_clip(params, max_norm, *args, **kwargs)

    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 1.0)
    with patch.object(_tnu, "clip_grad_norm_", spy_clip):
        trainer._optimizer_step(scaler=trainer.scaler, amp=False)
    assert calls, "clip_grad_norm_ was never called"
    assert all(math.isinf(c) for c in calls), (
        f"clip_grad_norm_ called with finite max_norm (would rescale!): {calls}"
    )


def test_local_validation_logs_scalar_total_under_split(vhp_model, tmp_path):
    """Under split mode, val rows must log ``loss/total`` as a parseable scalar
    (not a SplitLoss repr)."""
    model = copy.deepcopy(vhp_model)
    csv_path = tmp_path / "metrics.csv"
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
        csv_log_path=str(csv_path),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=True)
    loader = DataLoader(_DS(), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=loader,
        total_steps=1,
        validate_every=1,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    val_rows = [r for r in rows if r.get("split") == "val"]
    assert val_rows, f"no val rows in metrics.csv; rows: {rows}"
    tv = val_rows[-1].get("loss/total", "")
    assert tv and "SplitLoss" not in tv, f"loss/total should be scalar, got {tv!r}"
    float(tv)


def test_local_grad_norm_and_skips_logged(tmp_path):
    """``optim/grad_norm`` and ``optim/grad_skips`` land as CSV columns."""
    from tests.test_trainer import _SyntheticBatchDataset

    model = make_small_model()
    csv_path = tmp_path / "metrics.csv"
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
        csv_log_path=str(csv_path),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=2,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    assert rows, "empty metrics.csv"
    cols = set(rows[0].keys())
    assert "optim/grad_norm" in cols, f"optim/grad_norm not in columns: {cols}"
    assert "optim/grad_skips" in cols, f"optim/grad_skips not in columns: {cols}"


def test_local_single_mode_psi_betas_threaded(vhp_model, tmp_path):
    """``hparams.psi_betas`` tags ψ-only groups inside the single-mode optimizer."""
    from ddssm.training.stages import StageLrsConf

    model = copy.deepcopy(vhp_model)
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1, psi_betas=[0.9, 0.99]),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=False)
    _fit_one_step(trainer)
    _, real_psi = split_params_phith_psi(model)
    psi_ids = {id(p) for p in real_psi}

    def _has_psi_group_with_betas(opt):
        for g in opt.param_groups:
            g_ids = {id(p) for p in g["params"]}
            if g_ids and g_ids <= psi_ids and g.get("betas") == (0.9, 0.99):
                return True
        return False

    assert _has_psi_group_with_betas(trainer.optimizer), (
        "no ψ-only group with betas=(0.9, 0.99) in single-mode optimizer"
    )
    trainer._rebuild_optimizer(StageLrsConf(enc_lr=1e-4, dec_lr=2e-4, trans_lr=3e-4))
    assert _has_psi_group_with_betas(trainer.optimizer), (
        "ψ-only group with custom betas lost after _rebuild_optimizer"
    )


def test_local_psi_betas_none_is_default_topology(vhp_model, tmp_path):
    """With ``psi_betas=None`` (default), every group carries the AdamW default betas."""
    model = copy.deepcopy(vhp_model)
    hparams = DDSSMHyperParamsConf(grad_accum_steps=1)
    assert hparams.psi_betas is None
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=hparams,
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=False)
    _fit_one_step(trainer)
    for g in trainer.optimizer.param_groups:
        assert g["betas"] == (0.9, 0.999), (
            f"group carries non-default betas under psi_betas=None: {g['betas']}"
        )

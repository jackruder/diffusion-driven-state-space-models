"""Unit tests for :mod:`ddssm.transitions.baseline_gaussian`."""

from __future__ import annotations

from unittest.mock import patch

import torch
import pytest

from ddssm.nn.gaussians import gaussian_kl_divergence
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import (
    MLPBaseline,
    BaseBaseline,
    ZeroBaseline,
    PersistenceBaseline,
)
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

# ---------------------------------------------------------------------------
# Shapes / helpers
# ---------------------------------------------------------------------------

B = 2
S = 3
D = 2
T = 5
EMB_TIME = 8


def _make_transition(
    baseline: BaseBaseline,
    j: int,
) -> BaselineGaussianTransition:
    return BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=D,
        j=j,
        emb_time_dim=EMB_TIME,
    )


def _make_enc_stats(j: int) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    zs = torch.randn(B, S, D, T)
    mus = 0.3 * torch.randn(B, S, D, T)
    logvars = -1.0 + 0.2 * torch.randn(B, S, D, T)
    enc_stats = {"mus": mus, "logvars": logvars}
    time_embed = torch.randn(B, T, EMB_TIME)
    logq_paths = torch.randn(B, S, T)
    return zs, enc_stats, time_embed, logq_paths


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_rejects_baseline_with_wrong_latent_dim() -> None:
    """Constructor rejects a baseline whose latent_dim disagrees."""
    with pytest.raises(ValueError):
        _make_transition(MLPBaseline(latent_dim=D + 1, j=1), j=1)


def test_rejects_baseline_with_wrong_j() -> None:
    """Constructor rejects a baseline whose j disagrees."""
    with pytest.raises(ValueError):
        _make_transition(MLPBaseline(latent_dim=D, j=1), j=2)


# ---------------------------------------------------------------------------
# transition_kl — closed-form KL matches gaussian_kl_divergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", [1, 2])
def test_transition_kl_matches_analytic(j: int) -> None:
    """``transition_kl`` reproduces ``gaussian_kl_divergence`` summed over t."""
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    transition = _make_transition(baseline, j=j)
    zs, enc_stats, time_embed, logq_paths = _make_enc_stats(j)

    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
    )
    assert "kl" in out
    kl = out["kl"]
    assert kl.shape == ()
    assert torch.isfinite(kl)

    # Recompute the KL directly: for every (b, s, t) call baseline,
    # call gaussian_kl_divergence, sum over t and mean over (b, s).
    n_steps = T - j
    expected_per_b = torch.zeros(B)
    for b in range(B):
        for s in range(S):
            for t in range(j, T):
                z_hist = zs[b, s, :, t - j : t].unsqueeze(0)  # (1, d, j)
                p_mu, p_logvar = baseline.mean_and_logvar(z_hist)
                q_mu = enc_stats["mus"][b, s, :, t].unsqueeze(0)
                q_logvar = enc_stats["logvars"][b, s, :, t].unsqueeze(0)
                kl_step = gaussian_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
                expected_per_b[b] += kl_step.item() / S
    expected = expected_per_b.mean()
    assert torch.allclose(kl, expected, atol=1e-4)
    assert n_steps > 0  # sanity


def test_transition_kl_zero_when_q_equals_p() -> None:
    """KL = 0 when the encoder is identical to the baseline prior."""
    baseline = ZeroBaseline(latent_dim=D, j=1)
    # Override σ_p head to produce constant zero logvars (σ_p ≡ 1).
    with torch.no_grad():
        for layer in baseline.sigma_head.body:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
    transition = _make_transition(baseline, j=1)

    # Build encoder stats that match the prior: mu_q = 0, logvar_q = 0.
    zs = torch.zeros(B, S, D, T)
    enc_stats = {
        "mus": torch.zeros(B, S, D, T),
        "logvars": torch.zeros(B, S, D, T),
    }
    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=torch.zeros(B, S, T),
        time_embed=torch.zeros(B, T, EMB_TIME),
    )
    assert torch.allclose(out["kl"], torch.tensor(0.0), atol=1e-6)


def test_transition_kl_rejects_mc_only_encoder() -> None:
    """No silent MC fallback — Gaussian (mus, logvars) is required."""
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=4, n_layers=1)
    transition = _make_transition(baseline, j=1)
    zs, _, time_embed, logq_paths = _make_enc_stats(1)
    with pytest.raises(ValueError):
        transition.transition_kl(
            enc_stats={},  # missing mus / logvars
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
        )


# ---------------------------------------------------------------------------
# transition_kl — σ_data buffer update
# ---------------------------------------------------------------------------


def test_transition_kl_updates_sigma_data_per_t() -> None:
    """``transition_kl`` updates the σ_data buffer at every visited t."""
    j = 1
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_transition(baseline, j=j)
    zs, enc_stats, time_embed, logq_paths = _make_enc_stats(j)
    buf = SigmaDataBuffer(T_max=T, tracking_mode="per_t", ema_decay=0.0)

    update_calls = []
    real_update = buf.update

    def _spy(t_idx, mu_hat_batch, sigma_t2_batch) -> None:  # noqa: ANN001
        update_calls.append(t_idx.tolist())
        real_update(t_idx, mu_hat_batch, sigma_t2_batch)

    with patch.object(buf, "update", side_effect=_spy):
        transition.transition_kl(
            enc_stats=enc_stats,
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
            sigma_data=buf,
        )

    # Default chunk size is 1, so we expect (T - j) updates, each
    # touching a single timestep, starting at 1-based t = j + 1.
    flat = [t for call in update_calls for t in call]
    assert flat == list(range(j + 1, T + 1))


def test_transition_kl_no_buffer_no_update() -> None:
    """When ``sigma_data=None`` no buffer state is mutated."""
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=4, n_layers=1)
    transition = _make_transition(baseline, j=1)
    zs, enc_stats, time_embed, logq_paths = _make_enc_stats(1)
    transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
    )  # no buffer — should not raise


# ---------------------------------------------------------------------------
# transition_kl_init — VHP init term + mixed-history walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", [1, 2])
def test_transition_kl_init_shape_and_finite(j: int) -> None:
    """Shape sanity for the init-term return dict."""
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    transition = _make_transition(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    zs, enc_stats, time_embed, _ = _make_enc_stats(j)
    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
    )
    assert set(out) == {"loss", "entropy", "vhp", "kl_aux", "loss_init"}
    for k in ("loss", "entropy", "vhp", "kl_aux", "loss_init"):
        assert out[k].shape == ()
        assert torch.isfinite(out[k])
    # BaselineGaussian uses the default entropy policy: loss = -H + vhp.
    assert torch.allclose(out["loss"], out["entropy"] + out["vhp"])
    assert torch.allclose(out["vhp"], out["loss_init"] + out["kl_aux"])


def test_transition_kl_init_return_psi_is_zero_for_baseline() -> None:
    """Non-diffusion transitions have no ψ side; opt-in ``loss_psi`` is zero."""
    j = 1
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    transition = _make_transition(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    zs, enc_stats, time_embed, _ = _make_enc_stats(j)
    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        return_psi=True,
    )
    assert "loss_psi" in out
    assert out["loss_psi"].shape == ()
    assert float(out["loss_psi"]) == 0.0


def test_transition_kl_init_walks_mixed_history() -> None:
    """At j=2 the history transitions all-aux → 1-aux + 1-real."""
    j = 2
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_transition(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_enc_stats(j)

    captured_history = []
    real_mean_and_lv = baseline.mean_and_logvar

    def _capture(z_hist):  # noqa: ANN001
        captured_history.append(z_hist.detach().clone())
        return real_mean_and_lv(z_hist)

    with patch.object(baseline, "mean_and_logvar", side_effect=_capture):
        transition.transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=aux,
            time_embed=time_embed,
        )

    # We see j = 2 calls inside the init loop.
    assert len(captured_history) == 2
    # At step 0 (t=1): z_hist = z_aux (both slots aux).
    # At step 1 (t=2): newest slot is z_1 from the encoder, oldest is
    # the last aux slot.
    BS = B * S
    z1 = zs[:, :, :, 0].reshape(BS, D)
    # The newest slot of the step-1 history should equal z_1.
    assert torch.allclose(captured_history[1][:, :, -1], z1, atol=1e-6)
    # The newest slot of the step-0 history is NOT z_1.
    assert not torch.allclose(captured_history[0][:, :, -1], z1, atol=1e-3)


def test_transition_kl_init_updates_sigma_data_at_init_slots() -> None:
    """``transition_kl_init`` updates buffer slots t = 1 … j."""
    j = 2
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_transition(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_enc_stats(j)
    buf = SigmaDataBuffer(T_max=T, tracking_mode="per_t", ema_decay=0.0)

    update_calls = []
    real_update = buf.update

    def _spy(t_idx, mu_hat_batch, sigma_t2_batch) -> None:  # noqa: ANN001
        update_calls.append(int(t_idx.item()) if t_idx.numel() == 1 else t_idx.tolist())
        real_update(t_idx, mu_hat_batch, sigma_t2_batch)

    with patch.object(buf, "update", side_effect=_spy):
        transition.transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=aux,
            time_embed=time_embed,
            sigma_data=buf,
        )

    assert update_calls == [1, 2]


def test_transition_kl_init_grad_flows_to_aux_posterior() -> None:
    """The init term is differentiable w.r.t. the aux posterior."""
    j = 1
    baseline = PersistenceBaseline(latent_dim=D, j=j)
    transition = _make_transition(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_enc_stats(j)

    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
    )
    (out["loss_init"] + out["kl_aux"]).backward()
    aux_grads = [p.grad for p in aux.parameters() if p.grad is not None]
    assert len(aux_grads) > 0


# ---------------------------------------------------------------------------
# sample_latent_trajectory (inherited from BaseTransition)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", [1, 2])
@pytest.mark.parametrize("with_time_embed", [False, True])
def test_sample_latent_trajectory_shape_and_finite(
    j: int, with_time_embed: bool
) -> None:
    """Autoregressive rollout returns the documented shape and is finite.

    Regression guard for the ``hist_valid_len`` NameError that hid in
    ``BaseTransition.sample_latent_trajectory`` until nothing exercised it.
    """
    baseline = PersistenceBaseline(latent_dim=D, j=j)
    transition = _make_transition(baseline, j=j)
    torch.manual_seed(0)
    z_hist = torch.randn(B, D, j)
    steps = 4
    ctx = None
    if with_time_embed:
        ctx = {"time_embed": torch.randn(B, j + steps, EMB_TIME)}

    traj = transition.sample_latent_trajectory(z_hist, steps=steps, S=S, ctx=ctx)

    assert traj.shape == (B, S, D, steps)
    assert torch.isfinite(traj).all()


# ---------------------------------------------------------------------------
# seq_log_prob — return value is a log-probability, not an NLL
# (regression guard for the total_nll -> total_log_p rename, item 2)
# ---------------------------------------------------------------------------


def test_seq_log_prob_sign_is_log_prob_not_nll() -> None:
    """``seq_log_prob`` accumulates log p (positive for tight Gaussians).

    The variable was previously named ``total_nll`` despite storing a
    *sum of log-probabilities*, not a negated one.  This test pins the
    sign convention: under a well-fitted prior with small variance, the
    log-probability returned should be negative-to-zero (Gaussian log-pdf
    is negative), but crucially the function returns the un-negated value
    — the caller in ``transition_kl`` negates it via ``L_p = -seq_log_prob``.
    """
    j = 1
    baseline = ZeroBaseline(latent_dim=D, j=j)
    transition = _make_transition(baseline, j=j)
    # Encoder samples sitting exactly at the prior mean (zero), so log p is
    # maximal for this prior.  seq_log_prob should be finite and negative
    # (log-pdf of a Gaussian is ≤ 0 for unit-variance priors at σ_p ≡ 1).
    zs = torch.zeros(B, S, D, T)
    time_embed = torch.zeros(B, T, EMB_TIME)
    val = transition.seq_log_prob(zs, time_embed)
    assert val.shape == (B,)
    assert torch.isfinite(val).all()
    # A Gaussian with σ² = 1 has log p(0) = -d/2 · log(2π) < 0 per step.
    assert (val < 0.0).all(), "seq_log_prob must return log-prob (≤ 0 for N(0,I))"

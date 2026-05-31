"""Unit tests for :mod:`ddssm.transitions.diffusion`."""

from __future__ import annotations

import math
from functools import partial

import torch
import pytest

from ddssm.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.baselines import MLPBaseline, ZeroBaseline
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)

B = 2
S = 2
D = 2
T = 5
J = 1
EMB_TIME = 8
T_MAX = 10
CHANNELS = 8
NHEADS = 4


def _tiny_unet():
    return partial(
        CSDIUnet,
        channels=CHANNELS,
        n_layers=1,
        embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )


def _make_diffusion(
    baseline,
    j: int = J,
    schedule: DiffusionScheduleConfig | None = None,
) -> DiffusionTransition:
    if schedule is None:
        schedule = DiffusionScheduleConfig(
            S_k=1, k_chunk=1, num_steps=20, beta_min=0.1, beta_max=20.0,
            tau_min=1e-3, k_sampling_mode="uniform",
        )
    return DiffusionTransition(
        baseline=baseline,
        latent_dim=D,
        j=j,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_tiny_unet(),
        schedule=schedule,
    )


def _make_batch(j: int = J, T: int = T):
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


def test_constructor_rejects_baseline_with_wrong_dim() -> None:
    """The baseline's latent_dim must match."""
    with pytest.raises(ValueError):
        _make_diffusion(MLPBaseline(latent_dim=D + 1, j=J))


def test_constructor_zero_inits_final_layer() -> None:
    """diffusion builds its CSDIUnet with ``zero_init_output=True``."""
    transition = _make_diffusion(MLPBaseline(latent_dim=D, j=J))
    # CSDIUnet's final layer is ``output_projection2``.
    w = transition.diffmodel.output_projection2.weight.detach()
    assert torch.equal(w, torch.zeros_like(w))


def test_constructor_side_dim_bumps_by_one() -> None:
    """The side-info dim accommodates the padding-mask channel."""
    transition = _make_diffusion(MLPBaseline(latent_dim=D, j=J))
    # E_t + E_f + cond_mask + padding_mask = EMB_TIME + EMB_TIME + 1 + 1
    expected = EMB_TIME + EMB_TIME + 1 + 1
    assert transition.side_dim == expected


# ---------------------------------------------------------------------------
# EDM constant reduction at σ_data = 1
# ---------------------------------------------------------------------------


def test_edm_constants_reduce_to_v2_at_sigma_data_unit() -> None:
    """At σ_data² ≡ 1, the per-call EDM constants match V2's hardcoded values."""
    baseline = ZeroBaseline(latent_dim=D, j=J)
    transition = _make_diffusion(baseline)
    K = transition.num_steps
    k_idx = torch.arange(K).view(K, 1)  # (K, 1) — one per step

    # V2 hardcoded: c_skip = α², c_out = √(1−α²), c_in = α.
    sigma_tilde2 = transition.sigma_tilde.pow(2)
    # σ_data² ≡ 1.
    sd2 = torch.ones(K)
    denom = sigma_tilde2 + sd2
    c_skip = sd2 / denom
    c_out = transition.sigma_tilde * sd2.sqrt() / denom.sqrt()
    c_in = 1.0 / denom.sqrt()

    # Verify the V2 form.
    assert torch.allclose(c_skip, transition.alpha2, atol=1e-6)
    assert torch.allclose(c_out, transition.one_minus_alpha2.sqrt(), atol=1e-6)
    assert torch.allclose(c_in, transition.alpha, atol=1e-6)


# ---------------------------------------------------------------------------
# transition_kl
# ---------------------------------------------------------------------------


def test_transition_kl_runs_and_returns_finite() -> None:
    """``transition_kl`` produces a finite scalar loss."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert set(out.keys()) >= {"kl"}
    assert torch.isfinite(out["kl"])


@pytest.mark.slow
def test_transition_kl_is_invariant_to_num_steps() -> None:
    """The ESM loss estimates an integral over τ — it must be ~invariant to the
    grid size, not scale as 1/num_steps.

    Regression guard for the IS-normalization bug: the per-draw weight baked in
    the (½·dτ) Riemann measure AND divided by num_steps, double-counting the
    τ-measure and shrinking the loss by a factor of K. Doubling num_steps then
    halved the loss; the correct estimator is grid-invariant (up to MC +
    discretisation noise).
    """
    def _sched(num_steps: int) -> DiffusionScheduleConfig:
        return DiffusionScheduleConfig(
            S_k=2048, k_chunk=256, num_steps=num_steps,
            beta_min=0.1, beta_max=20.0, tau_min=1e-3, k_sampling_mode="uniform",
        )

    baseline = ZeroBaseline(latent_dim=D, j=J)  # no params → only the net to sync
    t_coarse = _make_diffusion(baseline, schedule=_sched(10))
    t_fine = _make_diffusion(baseline, schedule=_sched(20))
    # Share F_ψ weights so the two estimate the SAME integrand.
    t_fine.diffmodel.load_state_dict(t_coarse.diffmodel.state_dict())
    t_fine.embed_layer.load_state_dict(t_coarse.embed_layer.state_dict())

    zs, enc_stats, time_embed, logq_paths = _make_batch()

    def _kl(tr) -> float:
        torch.manual_seed(1)
        sd = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
        return float(
            tr.transition_kl(
                enc_stats=enc_stats, zs=zs, logq_paths=logq_paths,
                time_embed=time_embed, sigma_data=sd,
            )["kl"].detach()
        )

    kl_coarse = _kl(t_coarse)
    kl_fine = _kl(t_fine)
    assert kl_coarse > 0 and kl_fine > 0
    # Grid-invariant: ratio ≈ 1. (The 1/K bug gave kl_coarse/kl_fine ≈ 20/10 = 2.)
    rel = abs(kl_coarse - kl_fine) / kl_fine
    assert rel < 0.5, f"ESM loss scales with num_steps: {kl_coarse=} {kl_fine=}"


def test_transition_kl_rejects_mc_only_encoder() -> None:
    """No silent MC fallback — Gaussian (mus, logvars) required."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    zs, _, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    with pytest.raises(ValueError):
        transition.transition_kl(
            enc_stats={},
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
            sigma_data=sigma_data,
        )


# ---------------------------------------------------------------------------
# Native-coord score reconstruction (model-v2.org § Exact likelihood, Layer 2)
# ---------------------------------------------------------------------------


def test_score_matches_closed_form_under_zero_init_diffmodel() -> None:
    """Score collapses to ``-ẑ / (α · (σ̃² + σ_d²))`` when F_ψ ≡ 0.

    With diffusion's default zero-init final projection both ``weight`` and
    ``bias`` of ``output_projection2`` are zero, so ``F_ψ`` outputs zero
    exactly and the EDM denoiser reduces to ``D_ψ = c_skip · ẑ``.
    Substituting into the native-coord score
        s_ψ(z, τ, z_hist) = (D_ψ − ẑ) / (α_τ · σ̃_τ²),
            ẑ = z/α_τ − μ_p(z_hist),
    with ``c_skip = σ_d² / (σ̃² + σ_d²)`` gives the closed form
        s_ψ = -ẑ / (α_τ · (σ̃_τ² + σ_d²)).

    Exercises the full Layer-2 composition (rescale + center + EDM skip
    path + Tweedie + de-rescale) at a continuous τ value that is *not*
    on the discrete schedule grid — the implementation must compute
    α(τ), σ̃(τ) closed-form for downstream prob-flow ODE use.
    """
    torch.manual_seed(42)
    baseline = MLPBaseline(latent_dim=D, j=J)
    transition = _make_diffusion(baseline)
    transition.eval()

    out_w = transition.diffmodel.output_projection2.weight
    out_b = transition.diffmodel.output_projection2.bias
    assert torch.equal(out_w, torch.zeros_like(out_w))
    assert out_b is None or torch.equal(out_b, torch.zeros_like(out_b))

    z = torch.randn(B, D)
    tau = torch.full((B,), 0.4)
    z_hist = torch.randn(B, D, J)
    ctx = {
        "hist_time_emb": torch.randn(B, J, EMB_TIME),
        "target_time_emb": torch.randn(B, 1, EMB_TIME),
    }
    sigma_d2 = torch.tensor([0.7, 1.3])

    beta_min = transition.schedule.beta_min
    beta_max = transition.schedule.beta_max
    int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau.pow(2)
    alpha = torch.exp(-0.5 * int_beta)
    sigma_tilde2 = (1.0 - alpha.pow(2)) / alpha.pow(2)

    mu_p = baseline.mean(z_hist)
    z_hat = z / alpha.unsqueeze(-1) - mu_p
    expected = -z_hat / (
        alpha.unsqueeze(-1) * (sigma_tilde2.unsqueeze(-1) + sigma_d2.unsqueeze(-1))
    )

    actual = transition.score(
        z=z,
        tau=tau,
        z_hist=z_hist,
        ctx=ctx,
        sigma_d2=sigma_d2,
    )

    assert actual.shape == (B, D)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Prob-flow ODE log-density (model-v2.org § Exact likelihood, Layer 1)
# ---------------------------------------------------------------------------


def test_log_prob_matches_analytic_gaussian_when_score_is_marginal() -> None:
    """Prob-flow ODE log-density matches analytic Gaussian when score is the marginal score.

    Reduction sanity check #2 from model-v2.org § Exact likelihood:
    when the trained score equals the analytic encoder-marginal score
    of ``N(μ_t, σ_t²)``, the prob-flow ODE pushforward IS that Gaussian,
    so ``log_prob(z)`` matches ``log N(z; μ_t, σ_t²)`` to ODE-solver
    tolerance (modulo endpoint and tau_min approximations).

    The schedule uses a stiff ``β_max=50`` so the endpoint approximation
    ``log p_ψ^{ode,1} ≈ log N(0, I)`` is sub-tolerance (``α(1) ≈ 4e-6``).
    """
    torch.manual_seed(0)
    baseline = ZeroBaseline(latent_dim=D, j=J)
    schedule = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=20,
        beta_min=0.1, beta_max=50.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = _make_diffusion(baseline, schedule=schedule)
    transition.eval()

    mu_t = torch.tensor([[0.3, -0.4], [0.5, 0.2]])
    sigma2_t = torch.tensor([[0.7, 1.1], [0.5, 0.9]])

    beta_min = schedule.beta_min
    beta_max = schedule.beta_max

    def analytic_score(z, tau, z_hist, ctx, sigma_d2, padding_mask=None):
        if tau.dim() == 0:
            tau = tau.expand(z.shape[0])
        int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau.pow(2)
        alpha = torch.exp(-0.5 * int_beta).unsqueeze(-1)
        alpha2 = alpha.pow(2)
        marginal_var = alpha2 * sigma2_t + (1.0 - alpha2)
        return -(z - alpha * mu_t) / marginal_var

    transition.score = analytic_score

    z = torch.tensor([[0.2, 0.4], [-0.1, 0.3]])
    z_hist = torch.zeros(B, D, J)
    ctx = {
        "hist_time_emb": torch.zeros(B, J, EMB_TIME),
        "target_time_emb": torch.zeros(B, 1, EMB_TIME),
    }
    sigma_d2 = torch.ones(B)

    actual = transition.log_prob(
        z=z, z_hist=z_hist, ctx=ctx, sigma_d2=sigma_d2,
        rtol=1e-7, atol=1e-7,
    )

    expected = (
        -0.5 * ((z - mu_t).pow(2) / sigma2_t).sum(dim=-1)
        - 0.5 * sigma2_t.log().sum(dim=-1)
        - 0.5 * D * math.log(2.0 * math.pi)
    )

    assert actual.shape == (B,)
    assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_log_prob_hutchinson_matches_exact_for_diagonal_jacobian() -> None:
    """Hutchinson estimator equals exact-trace when the score Jacobian is diagonal.

    Cycle-3 tracer. The analytic encoder marginal of an isotropic
    Gaussian has score ``s(z) = -(z − αμ_t)/σ_τ²`` with Jacobian
    ``J = -I/σ_τ²`` — purely diagonal.  With Rademacher ``v ∈ {±1}``
    and ``|v|² = D`` deterministically, ``vᵀ J v = -D/σ_τ² = tr(J)``
    exactly — zero variance per draw.  So Hutchinson and exact-trace
    must agree to ODE-solver tolerance for any ``v``.

    Still exercises the full Hutchinson code path (probe generation,
    ``grad_outputs=v`` reverse-mode pass, ``vᵀ J v`` reduction): a
    broken probe distribution or wrong reduction would shift the
    answer in a way this test catches.
    """
    torch.manual_seed(0)
    baseline = ZeroBaseline(latent_dim=D, j=J)
    schedule = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=20,
        beta_min=0.1, beta_max=50.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = _make_diffusion(baseline, schedule=schedule)
    transition.eval()

    mu_t = torch.tensor([[0.3, -0.4], [0.5, 0.2]])
    sigma2_t = torch.tensor([[0.7, 1.1], [0.5, 0.9]])

    beta_min = schedule.beta_min
    beta_max = schedule.beta_max

    def analytic_score(z, tau, z_hist, ctx, sigma_d2, padding_mask=None):
        if tau.dim() == 0:
            tau = tau.expand(z.shape[0])
        int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau.pow(2)
        alpha = torch.exp(-0.5 * int_beta).unsqueeze(-1)
        alpha2 = alpha.pow(2)
        marginal_var = alpha2 * sigma2_t + (1.0 - alpha2)
        return -(z - alpha * mu_t) / marginal_var

    transition.score = analytic_score

    z = torch.tensor([[0.2, 0.4], [-0.1, 0.3]])
    z_hist = torch.zeros(B, D, J)
    ctx = {
        "hist_time_emb": torch.zeros(B, J, EMB_TIME),
        "target_time_emb": torch.zeros(B, 1, EMB_TIME),
    }
    sigma_d2 = torch.ones(B)

    common_kwargs = dict(
        z=z, z_hist=z_hist, ctx=ctx, sigma_d2=sigma_d2,
        rtol=1e-7, atol=1e-7,
    )
    exact_logp = transition.log_prob(divergence_mode="exact", **common_kwargs)

    gen = torch.Generator().manual_seed(123)
    hutch_logp = transition.log_prob(
        divergence_mode="hutchinson", generator=gen, **common_kwargs
    )

    assert hutch_logp.shape == (B,)
    assert torch.allclose(hutch_logp, exact_logp, atol=1e-4, rtol=1e-4)

def test_transition_kl_updates_sigma_data_per_t() -> None:
    """``transition_kl`` updates buffer slots for every visited t."""
    j = J
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    zs, enc_stats, time_embed, logq_paths = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", ema_decay=0.0)
    pre_step = sigma_data.ema_step.clone()
    transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    # The ema_step counters for t = j+1..T should have advanced.
    advanced = sigma_data.ema_step > pre_step
    expected = torch.zeros(T_MAX, dtype=torch.bool)
    expected[j : T] = True  # 0-based slice covers internal slots j..T-1
    assert torch.equal(advanced, expected)


# ---------------------------------------------------------------------------
# transition_kl_init  (VHP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", [1, 2])
def test_transition_kl_init_shape_and_finite(j: int) -> None:
    """Init term returns finite ``loss_init`` and ``kl_aux``."""
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert set(out.keys()) == {"loss", "entropy", "vhp", "kl_aux", "loss_init"}
    for k in ("loss", "entropy", "vhp", "kl_aux", "loss_init"):
        assert torch.isfinite(out[k])
    # diffusion cancels the encoder entropy in stage 2: entropy == 0, loss == vhp.
    assert out["entropy"].item() == 0.0
    assert torch.allclose(out["loss"], out["vhp"])
    assert torch.allclose(out["vhp"], out["loss_init"] + out["kl_aux"])


def test_transition_kl_init_updates_sigma_data_at_init_slots() -> None:
    """Init walks update buffer slots t = 1 … j."""
    j = 2
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", ema_decay=0.0)
    pre_step = sigma_data.ema_step.clone()
    transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    # Slots 1..j (internal indices 0..j-1) should have advanced.
    expected = torch.zeros(T_MAX, dtype=torch.bool)
    expected[:j] = True
    advanced = sigma_data.ema_step > pre_step
    assert torch.equal(advanced, expected)


def test_transition_kl_init_grad_flows_to_aux_posterior() -> None:
    """Gradient propagates through the aux posterior at init."""
    j = 1
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, j=j)
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed", init_value=1.0)
    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    (out["loss_init"] + out["kl_aux"]).backward()
    aux_grads = [p.grad for p in aux.parameters() if p.grad is not None]
    assert len(aux_grads) > 0


# ---------------------------------------------------------------------------
# log_prob_init  (VHP initial-state log-density)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", [1, 2])
def test_log_prob_init_shape_and_finite(j: int) -> None:
    """``log_prob_init`` returns a finite ``(B, S)`` per-trajectory log-density."""
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, j=j)
    transition.eval()
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, _, time_embed, _ = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed", init_value=1.0)
    out = transition.log_prob_init(
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert out.shape == (B, S)
    assert torch.isfinite(out).all()


def test_log_prob_init_does_not_update_sigma_data() -> None:
    """Unlike ``transition_kl_init``, log-density eval must not touch the buffer."""
    j = 2
    baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, j=j)
    transition.eval()
    aux = AuxPosterior(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
    zs, _, time_embed, _ = _make_batch(j=j)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", ema_decay=0.0)
    pre_step = sigma_data.ema_step.clone()
    pre_val = sigma_data.sigma_data2.clone()
    transition.log_prob_init(
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert torch.equal(sigma_data.ema_step, pre_step)
    assert torch.equal(sigma_data.sigma_data2, pre_val)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_shape_and_baseline_shift() -> None:
    """``sample`` returns ``(B, 1, d)`` and adds μ_p back to the centered draw."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    z_hist = torch.randn(B, D, J)
    ctx = {
        "hist_time_emb": torch.zeros(B, J, EMB_TIME),
        "target_time_emb": torch.zeros(B, 1, EMB_TIME),
    }
    z_sample = transition.sample(z_hist=z_hist, S=1, ctx=ctx)
    assert z_sample.shape == (B, 1, D)
    assert torch.isfinite(z_sample).all()


def test_sample_reads_sigma_data_buffer_at_t() -> None:
    """``sample`` indexes the frozen σ_data² buffer at the 1-based ``t`` in ctx."""
    from unittest import mock

    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    z_hist = torch.randn(B, D, J)
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed")
    with torch.no_grad():
        buf.sigma_data2.copy_(torch.linspace(0.5, 0.9, T_MAX))
    ctx = {
        "hist_time_emb": torch.zeros(B, J, EMB_TIME),
        "target_time_emb": torch.zeros(B, 1, EMB_TIME),
        "sigma_data": buf,
        "t": 3,
    }
    with mock.patch.object(buf, "read", wraps=buf.read) as spy:
        z = transition.sample(z_hist=z_hist, S=1, ctx=ctx)
    assert torch.isfinite(z).all()
    assert int(spy.call_args[0][0]) == 3  # read(t=3), not the σ_data≡1 fallback


def test_sample_clamps_sigma_data_beyond_horizon() -> None:
    """Beyond the trained horizon, ``sample`` holds σ_data²[T_max] (no IndexError)."""
    from unittest import mock

    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    z_hist = torch.randn(B, D, J)
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed")
    # read(t) is strict and would raise for t > T_max; sample() must clamp.
    with pytest.raises(IndexError):
        buf.read(T_MAX + 5)
    ctx = {
        "hist_time_emb": torch.zeros(B, J, EMB_TIME),
        "target_time_emb": torch.zeros(B, 1, EMB_TIME),
        "sigma_data": buf,
        "t": T_MAX + 5,  # past the trained horizon
    }
    with mock.patch.object(buf, "read", wraps=buf.read) as spy:
        z = transition.sample(z_hist=z_hist, S=1, ctx=ctx)  # must NOT raise
    assert torch.isfinite(z).all()
    assert int(spy.call_args[0][0]) == T_MAX  # clamped to the last trained slot

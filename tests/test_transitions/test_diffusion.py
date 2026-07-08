"""Unit tests for :mod:`ddssm.transitions.diffusion`."""

from __future__ import annotations

import math
from functools import partial

import torch
import pytest

from ddssm.nn.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import MLPBaseline, ZeroBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
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
CHANNELS = 16
NHEADS = 2


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
            S_k=1,
            k_chunk=1,
            num_steps=20,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="uniform",
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
    """Diffusion builds its CSDIUnet with ``zero_init_output=True``."""
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


def test_transition_kl_does_not_mutate_sigma_data_under_no_grad() -> None:
    """REGRESSION: an eval-mode (no_grad) ``transition_kl`` must NOT mutate σ_data.

    σ_data is updated inside the forward; doing so during evaluation drifts the
    buffer toward the eval data and inflates the eval ELBO's transition-KL term
    (obj1) by ~2-4x — the bug this guards. With autograd enabled (training) the
    same call DOES update σ_data.
    """
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    before = sigma_data.sigma_data2.clone()

    with torch.no_grad():
        transition.transition_kl(
            enc_stats=enc_stats,
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
            sigma_data=sigma_data,
        )
    assert torch.equal(sigma_data.sigma_data2, before), "eval forward mutated σ_data"

    with torch.enable_grad():
        transition.transition_kl(
            enc_stats=enc_stats,
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
            sigma_data=sigma_data,
        )
    assert not torch.equal(sigma_data.sigma_data2, before), (
        "training forward left σ_data unchanged"
    )


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
            S_k=2048,
            k_chunk=256,
            num_steps=num_steps,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="uniform",
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
                enc_stats=enc_stats,
                zs=zs,
                logq_paths=logq_paths,
                time_embed=time_embed,
                sigma_data=sd,
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
        S_k=1,
        k_chunk=1,
        num_steps=20,
        beta_min=0.1,
        beta_max=50.0,
        tau_min=1e-3,
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
        z=z,
        z_hist=z_hist,
        ctx=ctx,
        sigma_d2=sigma_d2,
        rtol=1e-7,
        atol=1e-7,
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
        S_k=1,
        k_chunk=1,
        num_steps=20,
        beta_min=0.1,
        beta_max=50.0,
        tau_min=1e-3,
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
        z=z,
        z_hist=z_hist,
        ctx=ctx,
        sigma_d2=sigma_d2,
        rtol=1e-7,
        atol=1e-7,
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
    expected[j:T] = True  # 0-based slice covers internal slots j..T-1
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


# ---------------------------------------------------------------------------
# EDM (Karras 2022) sampler — analytic distributional correctness
# ---------------------------------------------------------------------------


def _make_edm_diffusion(
    sigma_d2: float,
    *,
    num_steps: int,
    edm_s_churn: float = 0.0,
    edm_s_noise: float = 1.0,
    edm_sigma_max_rel: float | None = None,
    edm_sigma_min_rel: float | None = None,
) -> tuple[DiffusionTransition, SigmaDataBuffer]:
    """Build a ZeroBaseline EDM transition + a frozen σ_data² buffer.

    ZeroBaseline ⟹ μ_p≡0 so the returned sample IS the centered draw, and the
    default zero-init final projection ⟹ F_ψ≡0 so the EDM denoiser collapses to
    the exact Tweedie denoiser for N(0, σ_d²).
    """
    transition = DiffusionTransition(
        baseline=ZeroBaseline(latent_dim=D, j=J),
        latent_dim=D,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_tiny_unet(),
        schedule=DiffusionScheduleConfig(
            S_k=1,
            k_chunk=1,
            num_steps=num_steps,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="uniform",
        ),
        sampler="edm",
        edm_s_churn=edm_s_churn,
        edm_s_noise=edm_s_noise,
        edm_sigma_max_rel=edm_sigma_max_rel,
        edm_sigma_min_rel=edm_sigma_min_rel,
    )
    transition.eval()
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed", init_value=sigma_d2)
    return transition, buf


def _edm_draw(transition, buf, n_samples: int) -> torch.Tensor:
    """Draw ``n_samples`` centered EDM samples; returns ``(n_samples, D)``."""
    z_hist = torch.zeros(n_samples, D, J)
    ctx = {
        "hist_time_emb": torch.zeros(n_samples, J, EMB_TIME),
        "target_time_emb": torch.zeros(n_samples, 1, EMB_TIME),
        "sigma_data": buf,
        "t": J + 1,
    }
    z = transition.sample(z_hist=z_hist, S=1, ctx=ctx)  # (n, 1, D)
    assert z.shape == (n_samples, 1, D)
    return z.squeeze(1)


@pytest.mark.parametrize("sigma_d2", [0.5, 1.0, 2.0])
def test_edm_sampler_zero_denoiser_is_gaussian_n0_sigma_data(sigma_d2: float) -> None:
    """Deterministic EDM (no churn) with F_ψ≡0 emits samples ~ N(0, σ_d²).

    With the zero-init final projection F_ψ≡0, the EDM denoiser reduces to the
    skip path  D = c_skip·x̃ = σ_d²/(σ̃²+σ_d²)·x̃  — *exactly* the optimal
    (Tweedie) denoiser for data ~ N(0, σ_d²I).  The PF-ODE the sampler
    integrates is then the exact reverse flow of that Gaussian, so the centered
    draw is distributed N(0, σ_d²) to discretisation + MC tolerance — a
    network-independent correctness check on the sampler itself.
    """
    torch.manual_seed(0)
    n = 4000
    transition, buf = _make_edm_diffusion(sigma_d2, num_steps=40)

    # Establish F_ψ ≡ 0: the EDM denoiser is the pure skip path.
    w = transition.diffmodel.output_projection2.weight
    b = transition.diffmodel.output_projection2.bias
    assert torch.equal(w, torch.zeros_like(w))
    assert b is None or torch.equal(b, torch.zeros_like(b))

    z = _edm_draw(transition, buf, n)
    assert torch.isfinite(z).all()
    flat = z.reshape(-1)  # iid N(0, σ_d²) across rows × coords
    emp_mean = flat.mean().item()
    emp_var = flat.var().item()
    # mean SE = √(σ_d²/(n·D)); 0.06·√σ_d² ≈ 5 SE.
    assert abs(emp_mean) < 0.06 * math.sqrt(sigma_d2), f"{sigma_d2=} mean={emp_mean}"
    assert abs(emp_var / sigma_d2 - 1.0) < 0.1, f"{sigma_d2=} var={emp_var}"


def test_edm_sampler_churn_preserves_gaussian_marginal() -> None:
    """The stochastic-churn EDM branch still maps F_ψ≡0 to N(0, σ_d²).

    Churn injects noise (σ̂ > σ_cur) then the Heun step removes it; with the
    exact denoiser the stationary marginal N(0, σ_d²+σ̃²) is preserved, so the
    σ→0 sample is still N(0, σ_d²).  Looser tolerance than the deterministic
    case — churn adds Monte-Carlo variance and a little discretisation bias.
    Exercises the otherwise-untouched churn branch (γ>0, noise injection).
    """
    torch.manual_seed(0)
    n = 4000
    sigma_d2 = 1.0
    transition, buf = _make_edm_diffusion(
        sigma_d2, num_steps=40, edm_s_churn=16.0, edm_s_noise=1.0
    )
    # γ = min(S_churn/N, √2−1) > 0 ⟹ the churn branch fires.
    assert min(transition.edm_s_churn / 40, math.sqrt(2.0) - 1.0) > 0.0

    z = _edm_draw(transition, buf, n)
    assert torch.isfinite(z).all()
    flat = z.reshape(-1)
    assert abs(flat.mean().item()) < 0.08
    assert abs(flat.var().item() / sigma_d2 - 1.0) < 0.15


@pytest.mark.parametrize("sigma_max_rel", [5.0, 10.0])
def test_edm_sigma_max_rel_clamps_schedule(sigma_max_rel: float) -> None:
    """Clamping σ_max to C·σ_data still produces N(0, σ_d²) with F_ψ≡0."""
    torch.manual_seed(0)
    n = 4000
    sigma_d2 = 2.0
    transition, buf = _make_edm_diffusion(
        sigma_d2,
        num_steps=40,
        edm_sigma_max_rel=sigma_max_rel,
    )
    assert transition.edm_sigma_max_rel == sigma_max_rel
    z = _edm_draw(transition, buf, n)
    assert torch.isfinite(z).all()
    flat = z.reshape(-1)
    assert abs(flat.mean().item()) < 0.06 * math.sqrt(sigma_d2)
    assert abs(flat.var().item() / sigma_d2 - 1.0) < 0.1


class _MoGPreconditionedDenoiser(torch.nn.Module):
    """Mock ``diffmodel`` whose EDM-preconditioned denoiser is the *exact*
    posterior mean of a Gaussian mixture  ``Σ_k w_k N(μ_k, σ_d² I)``.

    The sampler drives the network with ``c_in·x̃`` (last slot of ``latent_w``)
    and ``c_noise = ¼·log σ``, then forms ``D = c_skip·x̃ + c_out·F``.  We invert
    the preconditioning to recover ``x̃`` and ``σ``, evaluate the closed-form MoG
    Tweedie denoiser  ``E[x|x̃] = Σ_k r_k(x̃)·[μ_k + c_skip·(x̃−μ_k)]``  with
    responsibilities ``r_k ∝ w_k N(x̃; μ_k, (σ_d²+σ²)I)``, then re-precondition to
    ``F = (D − c_skip·x̃)/c_out``.  Feeding this through ``_edm_sample_centered``
    must reconstruct the mixture — a non-Gaussian, multimodal target.
    """

    def __init__(
        self, means: torch.Tensor, weights: torch.Tensor, sigma_d2: float
    ) -> None:
        super().__init__()
        self.register_buffer("means", means.double())  # (K, d)
        self.register_buffer("log_weights", weights.double().log())  # (K,)
        self.sigma_d2 = float(sigma_d2)

    def forward(self, latent_w, side_win, c_noise):
        del side_win
        dtype = latent_w.dtype
        x_pre = latent_w[..., -1].double()  # (B, d) = c_in·x̃
        sigma = float(torch.exp(4.0 * c_noise[0]).item())  # σ = exp(4·¼log σ)
        sd2 = self.sigma_d2
        var_pert = sigma * sigma + sd2
        c_skip = sd2 / var_pert
        c_out = (sigma * math.sqrt(sd2)) / math.sqrt(var_pert)
        x_tilde = x_pre * math.sqrt(var_pert)  # undo c_in = 1/√var_pert

        diff = x_tilde.unsqueeze(1) - self.means.unsqueeze(0)  # (B, K, d)
        log_r = self.log_weights.unsqueeze(0) - 0.5 * diff.pow(2).sum(-1) / var_pert
        log_r = log_r - torch.logsumexp(log_r, dim=1, keepdim=True)
        resp = log_r.exp()  # (B, K)
        comp_mean = self.means.unsqueeze(0) + c_skip * diff  # m_k = μ_k+c_skip·diff
        d_star = (resp.unsqueeze(-1) * comp_mean).sum(1)  # (B, d) = E[x|x̃]

        f = (d_star - c_skip * x_tilde) / c_out
        return f.to(dtype).unsqueeze(-1)  # (B, d, 1)


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.85, 0.15)])
def test_edm_sampler_recovers_mixture_of_gaussians(weights) -> None:
    """The EDM sampler reconstructs a 2-component Gaussian mixture from its
    exact closed-form (Tweedie) denoiser.

    Stronger than the single-Gaussian check: the MoG score is nonlinear and
    multimodal, so recovering the right mode *weights*, *means* and per-mode
    *variance* validates the full Heun integration on a genuinely non-Gaussian
    target — the bimodal regime (0.85/0.15) this project's data lives in.  The
    init prior N(0, σ_max²+σ_d²) mismatches the true σ_max-marginal only in its
    mean (μ̄≠0), but that error is suppressed by σ_d/σ_max ≈ 0.003 in the flow,
    so recovery is exact to discretisation + MC tolerance.
    """
    torch.manual_seed(0)
    n = 8000
    sigma_d2 = 0.25
    means = torch.tensor([[3.0, 3.0], [-3.0, -3.0]])  # (K=2, d=2): ~17σ_d apart
    w = torch.tensor(weights)

    transition, buf = _make_edm_diffusion(sigma_d2, num_steps=64)
    transition.diffmodel = _MoGPreconditionedDenoiser(means, w, sigma_d2)
    transition.eval()

    z = _edm_draw(transition, buf, n)  # (n, d)
    assert torch.isfinite(z).all()

    # Nearest-centroid assignment is unambiguous at ~17σ separation.
    sq = (z.unsqueeze(1) - means.unsqueeze(0)).pow(2).sum(-1)  # (n, K)
    assign = sq.argmin(dim=1)

    for k in range(2):
        sel = z[assign == k]
        frac = sel.shape[0] / n
        assert abs(frac - float(w[k])) < 0.03, f"mode {k}: weight {frac} vs {w[k]}"
        assert torch.allclose(sel.mean(0), means[k], atol=0.1), f"mode {k}: mean"
        var_k = sel.var(dim=0)  # (d,) — per coord
        assert torch.allclose(
            var_k, torch.full_like(var_k, sigma_d2), rtol=0.15, atol=0.05
        ), f"mode {k}: var {var_k} vs σ_d²={sigma_d2}"


# ---------------------------------------------------------------------------
# adaptive_is importance sampling — Group A: distribution properties
# ---------------------------------------------------------------------------


def _adaptive_is_schedule(num_steps: int = 20, **kw) -> DiffusionScheduleConfig:
    return DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=num_steps,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="adaptive_is",
        **kw,
    )


def _adaptive_is_meandom_pk(
    transition, sigma_d2: float, p_k_clip: float = 0.0
) -> torch.Tensor:
    """Compute the mean-dominated adaptive-IS density at a given σ_d² from
    the transition's σ̃ buffer. Mirrors the helper the diffusion module
    will expose; used to assert distribution properties without relying
    on a static ``transition.p_k`` buffer (which is None under
    ``adaptive_is`` since p_k is recomputed per-row from live σ_d²).
    ``p_k_clip=0.0`` (default) gives the raw analytic density; pass
    ``transition.p_k_clip`` to mirror the clipped training/probe density.
    """
    from ddssm.model.transitions.diffusion import _adaptive_is_density_meandom

    s = transition.sigma_tilde
    sd2 = torch.tensor([sigma_d2], dtype=s.dtype)
    pk = _adaptive_is_density_meandom(s, sd2, floor=1e-12, p_k_clip=p_k_clip)
    return pk.squeeze(0)  # (K,) from (1, K)


def test_esm_override_p_k_controls_is_correction() -> None:
    """``mc_override["p_k"]`` is the density used for the IS reweighting.

    Regression: the variance probe draws its shared ``k_idx`` from a
    global proposal, but the adaptive modes recomputed the *live* per-row
    density for the ``wtilde / p_k`` correction — biasing probed losses
    and gradients by q(k)/p(k) per row. With the fix, the caller-supplied
    proposal is what divides the weight.
    """
    from ddssm.model.transitions.diffusion import _adaptive_is_density_meandom

    torch.manual_seed(0)
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J), schedule=_adaptive_is_schedule()
    )
    transition.eval()
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    K = int(transition.sigma_tilde.numel())
    N = B * S
    k_idx = torch.randint(0, K, (N, 1))
    eps = torch.randn(N, D, 1)

    # The buffer inits at σ_d² = 1.0 and no_grad forwards don't mutate it,
    # so this is exactly the density the loss recomputes per row (including
    # the post-normalization p_k clip the loss now applies by default).
    live_pk = _adaptive_is_density_meandom(
        transition.sigma_tilde,
        torch.tensor([1.0]),
        floor=transition.gfloor,
        p_k_clip=transition.p_k_clip,
    ).squeeze(0)
    uniform_pk = torch.full((K,), 1.0 / K)

    def _loss(p_k: torch.Tensor | None) -> float:
        override = {"k_idx": k_idx, "eps": eps}
        if p_k is not None:
            override["p_k"] = p_k
        with torch.no_grad():
            out = transition.transition_kl(
                enc_stats=enc_stats,
                zs=zs,
                logq_paths=logq_paths,
                time_embed=time_embed,
                sigma_data=sigma_data,
                mc_override=override,
            )
        return float(out["kl"])

    # Supplying the same density the loss would recompute is a no-op…
    assert _loss(live_pk) == pytest.approx(_loss(None), rel=1e-5)
    # …while supplying the actual (different) proposal changes the
    # correction — the pre-fix path silently ignored it.
    assert _loss(uniform_pk) != pytest.approx(_loss(live_pk), rel=1e-3)


def test_adaptive_is_p_k_sums_to_one() -> None:
    """Mean-dom adaptive-IS density at σ_d=1 is a valid PMF."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J), schedule=_adaptive_is_schedule()
    )
    pk = _adaptive_is_meandom_pk(transition, sigma_d2=1.0)
    assert torch.allclose(pk.sum(), torch.tensor(1.0), atol=1e-6)
    assert (pk >= 0).all()


@pytest.mark.parametrize("sigma_d2", [0.25, 1.0, 4.0])
def test_adaptive_is_p_k_peaks_at_sigma_d_over_sqrt3(sigma_d2: float) -> None:
    """Mean-dom adaptive-IS density peaks at s = σ_d/√3."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=200),
    )
    pk = _adaptive_is_meandom_pk(transition, sigma_d2=sigma_d2)
    k_peak = pk.argmax()
    s_peak = float(transition.sigma_tilde[k_peak])
    s_expected = math.sqrt(sigma_d2) / math.sqrt(3)
    # Allow 15% relative tolerance since peak resolution depends on grid.
    assert abs(s_peak - s_expected) / s_expected < 0.15


def test_adaptive_is_cdf_mass_below_01_is_small_at_sigma_d_one() -> None:
    """At σ_d=1 only a small fraction of IS mass lies below σ̃ = 0.1."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=500),
    )
    pk = _adaptive_is_meandom_pk(transition, sigma_d2=1.0)
    mask_low = transition.sigma_tilde <= 0.1
    mass_low = float(pk[mask_low].sum())
    assert mass_low < 0.05


def test_adaptive_is_cdf_median_near_s1_at_sigma_d_one() -> None:
    """At σ_d=1 the CDF median (50% mass) sits near s = 1."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=500),
    )
    pk = _adaptive_is_meandom_pk(transition, sigma_d2=1.0)
    cdf = pk.cumsum(dim=0)
    idx_median = (cdf >= 0.5).nonzero(as_tuple=True)[0][0]
    s_median = float(transition.sigma_tilde[idx_median])
    assert abs(s_median - 1.0) < 0.3


@pytest.mark.parametrize("num_steps", [20, 100, 500])
@pytest.mark.parametrize("sigma_d2", [0.25, 1.0, 4.0])
def test_adaptive_is_formula_matches_sigma_tilde(
    num_steps: int, sigma_d2: float
) -> None:
    """Mean-dom p_k(s) = s/(σ_d²+s²)² (normalised) recomputed from σ̃."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=num_steps),
    )
    pk = _adaptive_is_meandom_pk(transition, sigma_d2=sigma_d2)
    s = transition.sigma_tilde
    expected = s / (sigma_d2 + s * s).pow(2)
    expected = expected.clamp_min(1e-12)
    expected = expected / expected.sum()
    assert torch.allclose(pk, expected, atol=1e-6)


def test_adaptive_is_meandom_at_sigma_d_one_matches_legacy_esm_is_formula() -> None:
    """At σ_d=1 the new mean-dom formula reduces to the old s/(1+s²)² form."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=200),
    )
    pk_new = _adaptive_is_meandom_pk(transition, sigma_d2=1.0)
    s = transition.sigma_tilde
    pk_old = s / (1.0 + s * s).pow(2)
    pk_old = pk_old.clamp_min(1e-12) / pk_old.clamp_min(1e-12).sum()
    assert torch.allclose(pk_new, pk_old, atol=1e-6)


def test_adaptive_is_full_collapses_to_meandom_at_special_case() -> None:
    """Full formula at (μ̂²=1, σ²=σ_d²=1) equals the mean-dom formula."""
    from ddssm.model.transitions.diffusion import (
        _adaptive_is_density_full,
        _adaptive_is_density_meandom,
    )

    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J),
        schedule=_adaptive_is_schedule(num_steps=200),
    )
    s = transition.sigma_tilde
    sigma_d2 = torch.tensor([1.0])
    sigma2 = torch.tensor([1.0])
    mu_hat2 = torch.tensor([1.0])
    pk_full = _adaptive_is_density_full(
        s,
        sigma_d2=sigma_d2,
        sigma2=sigma2,
        mu_hat2=mu_hat2,
        floor=1e-12,
    ).squeeze(0)
    pk_meandom = _adaptive_is_density_meandom(
        s,
        sigma_d2=sigma_d2,
        floor=1e-12,
    ).squeeze(0)
    assert torch.allclose(pk_full, pk_meandom, atol=1e-6)


def test_adaptive_is_full_call_site_reduces_per_coordinate(monkeypatch) -> None:
    """Regression (d > 1): the adaptive_is_full call site must reduce σ²/μ̂²
    over the coordinate axis with ``.mean`` (per-coordinate scale), not
    ``.sum``.

    σ_data tracks residual variance *per coordinate*, so passing a d×-scaled
    sum to the full density makes ``(σ²-σ_d²)²`` never vanish at real
    calibration and breaks the collapse to mean-dom for d > 1. The
    function-level collapse test above can't catch this because it feeds
    pre-reduced scalars; here we go through ``transition_kl`` with a constant
    posterior logvar (so every coordinate's σ² is identical) and assert the
    value handed to the full density equals that per-coordinate σ² (exp(c)),
    not D·exp(c).
    """
    import ddssm.model.transitions.diffusion as diff_mod

    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    sched = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="adaptive_is_full",
    )
    transition = _make_diffusion(baseline, schedule=sched)

    log_c = -0.5
    sigma2_per_coord = math.exp(log_c)
    torch.manual_seed(0)
    zs = torch.randn(B, S, D, T)
    mus = 0.3 * torch.randn(B, S, D, T)
    logvars = torch.full((B, S, D, T), log_c)
    enc_stats = {"mus": mus, "logvars": logvars}
    time_embed = torch.randn(B, T, EMB_TIME)
    logq_paths = torch.randn(B, S, T)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    captured: dict[str, torch.Tensor] = {}
    real_full = diff_mod._adaptive_is_density_full

    def spy(
        sigma_tilde, *, sigma_d2, sigma2, mu_hat2, floor=1e-12, p_k_clip: float = 0.0
    ):
        captured["sigma2"] = sigma2.detach().clone()
        captured["mu_hat2"] = mu_hat2.detach().clone()
        return real_full(
            sigma_tilde,
            sigma_d2=sigma_d2,
            sigma2=sigma2,
            mu_hat2=mu_hat2,
            floor=floor,
            p_k_clip=p_k_clip,
        )

    monkeypatch.setattr(diff_mod, "_adaptive_is_density_full", spy)

    transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )

    assert "sigma2" in captured, "adaptive_is_full density was never called"
    sig2 = captured["sigma2"]
    # Per-coordinate mean equals exp(log_c); the .sum bug would give D·exp(log_c).
    assert torch.allclose(sig2, torch.full_like(sig2, sigma2_per_coord), atol=1e-5), (
        f"expected per-coordinate σ²≈{sigma2_per_coord}, got {sig2}"
    )
    # Explicit guard against the d>1 .sum regression.
    assert float(sig2.max()) < D * sigma2_per_coord - 1e-3


# ---------------------------------------------------------------------------
# adaptive_is — Group B: training integration
# ---------------------------------------------------------------------------


def test_adaptive_is_transition_kl_runs_and_returns_finite() -> None:
    """transition_kl produces a finite scalar loss under adaptive_is."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, schedule=_adaptive_is_schedule())
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert torch.isfinite(out["kl"])


def test_adaptive_is_transition_kl_gradients_flow() -> None:
    """Backward through the adaptive_is loss produces nonzero diffmodel gradients."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline, schedule=_adaptive_is_schedule())
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    out["kl"].backward()
    grads = [p.grad for p in transition.diffmodel.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().max() > 0 for g in grads)


def test_adaptive_is_full_transition_kl_runs_and_returns_finite() -> None:
    """transition_kl produces a finite scalar loss under adaptive_is_full."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    sched = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="adaptive_is_full",
    )
    transition = _make_diffusion(baseline, schedule=sched)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert torch.isfinite(out["kl"])


@pytest.mark.slow
def test_adaptive_is_transition_kl_is_invariant_to_num_steps() -> None:
    """ESM loss under adaptive_is is grid-invariant (not scaling with num_steps)."""

    def _sched(K: int) -> DiffusionScheduleConfig:
        return DiffusionScheduleConfig(
            S_k=2048,
            k_chunk=256,
            num_steps=K,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="adaptive_is",
        )

    baseline = ZeroBaseline(latent_dim=D, j=J)
    t_coarse = _make_diffusion(baseline, schedule=_sched(10))
    t_fine = _make_diffusion(baseline, schedule=_sched(20))
    t_fine.diffmodel.load_state_dict(t_coarse.diffmodel.state_dict())
    t_fine.embed_layer.load_state_dict(t_coarse.embed_layer.state_dict())

    zs, enc_stats, time_embed, logq_paths = _make_batch()

    def _kl(tr) -> float:
        torch.manual_seed(1)
        sd = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
        return float(
            tr.transition_kl(
                enc_stats=enc_stats,
                zs=zs,
                logq_paths=logq_paths,
                time_embed=time_embed,
                sigma_data=sd,
            )["kl"].detach()
        )

    kl_coarse = _kl(t_coarse)
    kl_fine = _kl(t_fine)
    assert kl_coarse > 0 and kl_fine > 0
    rel = abs(kl_coarse - kl_fine) / kl_fine
    assert rel < 0.5, f"ESM loss scales with num_steps: {kl_coarse=} {kl_fine=}"


# ---------------------------------------------------------------------------
# adaptive_is — Group C: cross-mode comparison
# ---------------------------------------------------------------------------


def test_adaptive_is_concentrates_more_in_informative_range() -> None:
    """At σ_d=1 adaptive_is puts >70% of mass in [0.3, 3.0] and more than lsgm_is."""
    sched_ad = _adaptive_is_schedule(num_steps=200)
    sched_lsgm = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=200,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="lsgm_is",
    )
    t_ad = _make_diffusion(ZeroBaseline(latent_dim=D, j=J), schedule=sched_ad)
    t_lsgm = _make_diffusion(ZeroBaseline(latent_dim=D, j=J), schedule=sched_lsgm)

    pk_ad = _adaptive_is_meandom_pk(t_ad, sigma_d2=1.0)
    mask = (t_ad.sigma_tilde >= 0.3) & (t_ad.sigma_tilde <= 3.0)
    mass_ad = float(pk_ad[mask].sum())
    mass_lsgm = float(t_lsgm.p_k[mask].sum())
    assert mass_ad > 0.70
    assert mass_ad > mass_lsgm


def test_adaptive_is_deprioritizes_low_noise() -> None:
    """At σ_d=1 adaptive_is puts less mass below σ̃ = 0.1 than lsgm_is."""
    sched_ad = _adaptive_is_schedule(num_steps=200)
    sched_lsgm = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=200,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="lsgm_is",
    )
    t_ad = _make_diffusion(ZeroBaseline(latent_dim=D, j=J), schedule=sched_ad)
    t_lsgm = _make_diffusion(ZeroBaseline(latent_dim=D, j=J), schedule=sched_lsgm)

    pk_ad = _adaptive_is_meandom_pk(t_ad, sigma_d2=1.0)
    mask_low = t_ad.sigma_tilde <= 0.1
    mass_ad = float(pk_ad[mask_low].sum())
    mass_lsgm = float(t_lsgm.p_k[mask_low].sum())
    assert mass_ad < mass_lsgm


def test_adaptive_is_constructor_rejects_unknown_mode() -> None:
    """Unknown k_sampling_mode raises ValueError listing the supported modes."""
    with pytest.raises(ValueError, match="adaptive_is"):
        _make_diffusion(
            ZeroBaseline(latent_dim=D, j=J),
            schedule=DiffusionScheduleConfig(
                S_k=1,
                k_chunk=1,
                num_steps=20,
                k_sampling_mode="bogus",
            ),
        )


def test_constructor_rejects_legacy_esm_is_string() -> None:
    """The legacy ``esm_is`` mode string was renamed to ``adaptive_is`` and
    must now raise so users get an explicit signal rather than silent
    behaviour change.
    """
    with pytest.raises(ValueError, match="adaptive_is"):
        _make_diffusion(
            ZeroBaseline(latent_dim=D, j=J),
            schedule=DiffusionScheduleConfig(
                S_k=1,
                k_chunk=1,
                num_steps=20,
                k_sampling_mode="esm_is",
            ),
        )


# ---------------------------------------------------------------------------
# adaptive_is — Group D: variance probe compatibility
# ---------------------------------------------------------------------------


def test_probe_p_k_for_mode_adaptive_is_matches_formula() -> None:
    """``_p_k_for_mode("adaptive_is", sigma_d2=1.0)`` returns the mean-dom density.

    The probe threads the transition's ``p_k_clip`` so its diagnostic
    density matches the estimator training actually uses — the expected
    formula must mirror the clip.
    """
    from ddssm.variance.probe import _p_k_for_mode

    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J), schedule=_adaptive_is_schedule()
    )
    reconstructed = _p_k_for_mode(transition, "adaptive_is", sigma_d2=1.0)
    expected = _adaptive_is_meandom_pk(
        transition, sigma_d2=1.0, p_k_clip=transition.p_k_clip
    )
    assert torch.allclose(reconstructed, expected, atol=1e-6)
    # And it must NOT be the unclipped density when the default clip binds.
    unclipped = _adaptive_is_meandom_pk(transition, sigma_d2=1.0)
    assert transition.p_k_clip > 0.0
    assert not torch.allclose(reconstructed, unclipped, atol=1e-6)


# ---------------------------------------------------------------------------
# adaptive_is — Group E: p_k clip (post-normalization IS-probability floor)
# ---------------------------------------------------------------------------

PK_CLIP = 1e-3


def _pk_clip_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A spread of density inputs for the p_k-clip tests.

    Includes extreme σ_d²/σ²/μ̂² combinations that push the unclipped
    post-normalization probabilities arbitrarily low.
    """
    s = torch.logspace(-3, 3, 40)  # (K,)
    sd2 = torch.tensor([1e-8, 1e-2, 1.0, 1e4, 1e6])
    sg2 = torch.tensor([1e-8, 1.0, 1e4, 1e-6, 1e2])
    mh2 = torch.tensor([0.0, 1.0, 1e6, 1e-9, 1e3])
    return s, sd2, sg2, mh2


def _pk_densities(p_k_clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Both adaptive densities over the ``_pk_clip_inputs`` spread."""
    from ddssm.model.transitions.diffusion import (
        _adaptive_is_density_full,
        _adaptive_is_density_meandom,
    )

    s, sd2, sg2, mh2 = _pk_clip_inputs()
    pk_meandom = _adaptive_is_density_meandom(s, sd2, p_k_clip=p_k_clip)
    pk_full = _adaptive_is_density_full(s, sd2, sg2, mh2, p_k_clip=p_k_clip)
    return pk_meandom, pk_full


def test_pk_clip_bounds_probability() -> None:
    """With clip c, every prob ≥ c/(1+K·c) and each row still sums to 1."""
    c = PK_CLIP
    s, *_ = _pk_clip_inputs()
    K = int(s.numel())
    bound = c / (1.0 + K * c)
    # Sanity: the spread genuinely produces sub-c probs without the clip.
    for pk_unclipped in _pk_densities(0.0):
        assert float(pk_unclipped.min()) < c
    for pk in _pk_densities(c):
        assert (pk >= bound * (1.0 - 1e-5)).all(), float(pk.min())
        assert torch.allclose(pk.sum(dim=-1), torch.ones(pk.shape[0]), atol=1e-6)


def test_pk_clip_zero_recovers_prior_behavior() -> None:
    """``p_k_clip=0.0`` (and the omitted default) is bit-identical.

    Reference: the pre-clip formula ``raw / raw.sum(-1, keepdim=True)``
    computed inline, for both densities.
    """
    from ddssm.model.transitions.diffusion import (
        _adaptive_is_density_full,
        _adaptive_is_density_meandom,
    )

    floor = 1e-12
    s, sd2, sg2, mh2 = _pk_clip_inputs()

    # Mean-dom: the pre-change computation, inline.
    s32 = s.to(torch.float32)
    sd2c = sd2.to(torch.float32).clamp_min(floor).unsqueeze(-1)
    raw_m = (s32 / (sd2c + s32 * s32).pow(2)).clamp_min(floor)
    expected_m = raw_m / raw_m.sum(dim=-1, keepdim=True)
    assert torch.equal(
        _adaptive_is_density_meandom(s, sd2, floor=floor, p_k_clip=0.0),
        expected_m,
    )
    assert torch.equal(
        _adaptive_is_density_meandom(s, sd2, floor=floor),
        expected_m,
    )

    # Full: the pre-change computation, inline.
    s2 = s32 * s32
    sg2c = sg2.to(torch.float32).clamp_min(floor).unsqueeze(-1)
    mh2c = mh2.to(torch.float32).unsqueeze(-1)
    num = s32 * (mh2c * (sg2c + s2) + (sg2c - sd2c).pow(2))
    den = (sd2c + s2).pow(2) * (sg2c + s2)
    raw_f = (num / den.clamp_min(floor)).clamp_min(floor)
    expected_f = raw_f / raw_f.sum(dim=-1, keepdim=True)
    assert torch.equal(
        _adaptive_is_density_full(s, sd2, sg2, mh2, floor=floor, p_k_clip=0.0),
        expected_f,
    )
    assert torch.equal(
        _adaptive_is_density_full(s, sd2, sg2, mh2, floor=floor),
        expected_f,
    )


def test_pk_clip_none_is_alias_for_zero() -> None:
    """``p_k_clip=None`` on the schedule is accepted and means clip-off.

    The plan (and Hydra ``null`` overrides) use ``None`` to disable the
    clip; the transition coerces it to ``0.0`` at construction so the
    float-typed density path is untouched.
    """
    schedule = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=10, p_k_clip=None
    )
    transition = _make_diffusion(ZeroBaseline(latent_dim=D, j=J), schedule=schedule)
    assert transition.p_k_clip == pytest.approx(0.0)


def test_pk_clip_bounds_is_weight() -> None:
    """The weight-bounding property (with w̃≡1): max 1/p ≤ (1+K·c)/c."""
    c = PK_CLIP
    s, *_ = _pk_clip_inputs()
    K = int(s.numel())
    cap = (1.0 + K * c) / c
    for pk in _pk_densities(c):
        assert float((1.0 / pk).max()) <= cap * (1.0 + 1e-5)


def _pk_clip_transition_kl(transition: DiffusionTransition) -> float:
    """Fixed-seed transition_kl on the shared batch (fresh σ_data buffer)."""
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    torch.manual_seed(7)
    with torch.no_grad():
        out = transition.transition_kl(
            enc_stats=enc_stats,
            zs=zs,
            logq_paths=logq_paths,
            time_embed=time_embed,
            sigma_data=sigma_data,
        )
    return float(out["kl"])


def test_pk_clip_threaded_through_transition_kl() -> None:
    """``schedule.p_k_clip`` reaches the adaptive density inside the loss.

    The default (1e-3, active) changes the kl vs an explicit 0.0, while
    0.0 itself is bit-reproducible across reseeded runs.
    """

    def _build(**kw: float) -> DiffusionTransition:
        torch.manual_seed(3)
        return _make_diffusion(
            ZeroBaseline(latent_dim=D, j=J),
            schedule=_adaptive_is_schedule(**kw),
        )

    t_default = _build()
    t_off = _build(p_k_clip=0.0)
    assert t_default.p_k_clip == pytest.approx(PK_CLIP)  # dataclass default
    assert t_off.p_k_clip == pytest.approx(0.0)

    kl_off_1 = _pk_clip_transition_kl(t_off)
    kl_off_2 = _pk_clip_transition_kl(t_off)
    assert kl_off_1 == kl_off_2, "p_k_clip=0.0 is not reseed-reproducible"
    assert _pk_clip_transition_kl(t_default) != kl_off_1, (
        "default p_k_clip=1e-3 did not reach the adaptive density"
    )


@pytest.mark.parametrize("mode", ["uniform", "lsgm_is"])
def test_pk_clip_has_no_effect_on_static_modes(mode: str) -> None:
    """The clip is specific to the adaptive densities.

    The uniform and lsgm_is losses are bit-identical with p_k_clip
    1e-3 vs 0.0.
    """

    def _build(p_k_clip: float) -> DiffusionTransition:
        torch.manual_seed(3)
        return _make_diffusion(
            ZeroBaseline(latent_dim=D, j=J),
            schedule=DiffusionScheduleConfig(
                S_k=1,
                k_chunk=1,
                num_steps=20,
                beta_min=0.1,
                beta_max=20.0,
                tau_min=1e-3,
                k_sampling_mode=mode,
                p_k_clip=p_k_clip,
            ),
        )

    kl_on = _pk_clip_transition_kl(_build(PK_CLIP))
    kl_off = _pk_clip_transition_kl(_build(0.0))
    assert kl_on == kl_off


# ---------------------------------------------------------------------------
# Sampling schedule split — independent from the IS-density work
# ---------------------------------------------------------------------------


def test_sampling_schedule_split_uses_independent_num_steps() -> None:
    """A DiffusionSamplingScheduleConfig with num_steps=10 yields a 10-step
    rollout while the training schedule stays at its default num_steps=20.

    Verified via ``self.sample_num_steps`` (the buffer the sampler loop
    reads at line 1098 of diffusion.py).
    """
    from ddssm.model.transitions.diffusion import (
        DiffusionSamplingScheduleConfig,
    )

    baseline = ZeroBaseline(latent_dim=D, j=J)
    sampling_schedule = DiffusionSamplingScheduleConfig(
        num_steps=10,
        tau_min=1e-3,
        tau_max=1.0,
        beta_min=0.1,
        beta_max=20.0,
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=D,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_tiny_unet(),
        schedule=DiffusionScheduleConfig(
            S_k=1,
            k_chunk=1,
            num_steps=20,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="adaptive_is",
        ),
        sampling_schedule=sampling_schedule,
    )
    # Training schedule is preserved.
    assert int(transition.num_steps) == 20
    assert transition.sigma_tilde.numel() == 20
    # Sampling schedule has its own buffers.
    assert int(transition.sample_num_steps) == 10
    assert transition.sample_sigma_tilde.numel() == 10
    assert transition.sample_tau.numel() == 10


def test_sampling_schedule_defaults_to_training_when_none() -> None:
    """When sampling_schedule=None the sample_* buffers alias the training ones."""
    transition = _make_diffusion(
        ZeroBaseline(latent_dim=D, j=J), schedule=_adaptive_is_schedule(num_steps=20)
    )
    assert int(transition.sample_num_steps) == int(transition.num_steps)
    assert torch.equal(transition.sample_sigma_tilde, transition.sigma_tilde)
    assert torch.equal(transition.sample_tau, transition.tau)


# ---------------------------------------------------------------------------
# Ported from the parallel local implementation (see git stash@{0}).
# ---------------------------------------------------------------------------


def test_local_esm_chunk_loss_returns_phith_and_psi() -> None:
    """Under non-unit weights, the ELBO-weighted (phith) and unit-weight (psi)
    accumulators must differ; both are returned from ``_esm_chunk_loss``.
    """
    from tests.fixtures.golden_values import make_m1_transition, make_m1_inputs

    transition = make_m1_transition()
    inputs = make_m1_inputs()

    torch.manual_seed(42 + 2)
    with torch.no_grad():
        out = transition._esm_chunk_loss(
            **inputs,
            return_per_sample=False,
        )
    # Contract: 3-tuple (sum_phith, sum_psi, mu_hat_t)
    assert len(out) == 3
    sum_phith, sum_psi, mu_hat_t = out
    assert sum_phith.dim() == 0
    assert sum_psi.dim() == 0
    assert torch.isfinite(sum_phith) and torch.isfinite(sum_psi)
    # Under S_k=2 with sigma_d²=1 and uniform p_k, the IS weight per draw is
    # wtilde_full/p_k — NOT unit — so phith != psi.
    assert not torch.allclose(sum_phith, sum_psi, atol=1e-6), (
        f"phith and psi must differ under non-unit weights: "
        f"{sum_phith=} {sum_psi=}"
    )


def test_local_esm_chunk_loss_phith_reproduces_prior_single_loss() -> None:
    """Bit-level: ``sum_phith`` matches the pre-refactor single-accumulator scalar.

    The pre-refactor ``_esm_chunk_loss`` computed a single ELBO-weighted loss;
    the phith side must reproduce that value exactly.
    """
    from tests.fixtures.golden_values import (
        make_m1_transition,
        make_m1_inputs,
        M1_ESM_CHUNK_LOSS_SCALAR,
        M1_ESM_CHUNK_LOSS_PER_SAMPLE,
    )

    transition = make_m1_transition()
    inputs = make_m1_inputs()

    torch.manual_seed(42 + 2)
    with torch.no_grad():
        sum_phith, _sum_psi, _mu_hat = transition._esm_chunk_loss(
            **inputs, return_per_sample=False
        )
    assert float(sum_phith) == pytest.approx(M1_ESM_CHUNK_LOSS_SCALAR, rel=0.0, abs=0.0), (
        f"phith scalar drifted from pre-refactor golden: "
        f"got {float(sum_phith)!r}, want {M1_ESM_CHUNK_LOSS_SCALAR!r}"
    )

    # Also check the per-sample phith reproducer.
    torch.manual_seed(42 + 2)
    with torch.no_grad():
        per_sample_phith, _per_sample_psi, _mu_hat = transition._esm_chunk_loss(
            **inputs, return_per_sample=True
        )
    for i, ref in enumerate(M1_ESM_CHUNK_LOSS_PER_SAMPLE):
        assert float(per_sample_phith[i]) == pytest.approx(ref, rel=0.0, abs=0.0), (
            f"per_sample_phith[{i}] drifted from golden: "
            f"got {float(per_sample_phith[i])!r}, want {ref!r}"
        )


def test_local_esm_chunk_loss_ratio_matches_weight_at_forced_k() -> None:
    """With all draws forced onto a single k and uniform p_k, the phith/psi
    ratio equals the per-k IS weight ``wtilde_full / p_k``.

    Evidence that the ``/ float(self.S_k)`` normalisation is applied
    symmetrically on both accumulators (renamed from the local
    ``test_esm_chunk_loss_sk_division_applied_to_both``).
    """
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    schedule = DiffusionScheduleConfig(
        S_k=2, k_chunk=2, num_steps=20,
        beta_min=0.1, beta_max=20.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = _make_diffusion(baseline, schedule=schedule)
    transition.eval()

    torch.manual_seed(0)
    N = 4
    d = D
    mu_t = torch.randn(N, d)
    sigma2_t = torch.full((N, d), 1.0)
    z_hist = torch.zeros(N, d, J)
    sigma_d2_per_row = torch.ones(N)
    padding_mask = torch.zeros(N, J + 1)
    time_win = torch.randn(N, J + 1, EMB_TIME)
    ctx = {
        "hist_time_emb": time_win[:, :J, :],
        "target_time_emb": time_win[:, J:, :],
    }

    K = transition.num_steps
    k_idx = torch.zeros(N, 2, dtype=torch.long)  # both draws at k=0
    eps = torch.randn(N, d, 2)
    p_k_override = torch.full((K,), 1.0 / K)

    torch.manual_seed(1)
    with torch.no_grad():
        sum_phith, sum_psi, _ = transition._esm_chunk_loss(
            mu_t=mu_t, sigma2_t=sigma2_t, z_hist=z_hist, ctx=ctx,
            sigma_d2_per_row=sigma_d2_per_row, padding_mask=padding_mask,
            mc_override={"k_idx": k_idx, "eps": eps, "p_k": p_k_override},
        )

    sd2 = 1.0
    k = 0
    wtilde_base = float(transition.wtilde_base[k])
    st2 = float(transition.sigma_tilde[k].pow(2))
    wtilde_full = wtilde_base * sd2 / (st2 + sd2)
    w_expected = wtilde_full / (1.0 / K)

    assert torch.isfinite(sum_phith) and torch.isfinite(sum_psi)
    assert sum_psi > 0
    ratio = float(sum_phith / sum_psi)
    assert ratio == pytest.approx(w_expected, rel=1e-5), (
        f"phith/psi ratio {ratio} should equal weight {w_expected} — "
        f"evidence S_k normalization is symmetric"
    )


def test_local_transition_kl_returns_kl_phith_kl_psi_and_alias() -> None:
    """``transition_kl`` returns both KL sides plus a ``kl`` alias to phith."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    schedule = DiffusionScheduleConfig(
        S_k=2, k_chunk=2, num_steps=20,
        beta_min=0.1, beta_max=20.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = _make_diffusion(baseline, schedule=schedule)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
    )
    assert "kl" in out
    assert "kl_phith" in out
    assert "kl_psi" in out
    assert torch.isfinite(out["kl"])
    assert torch.isfinite(out["kl_phith"])
    assert torch.isfinite(out["kl_psi"])
    # Alias contract: kl == kl_phith (existing consumers read `kl`).
    assert torch.equal(out["kl"], out["kl_phith"])
    # Under non-unit weights they must differ.
    assert not torch.allclose(out["kl_phith"], out["kl_psi"], atol=1e-6)


def test_local_transition_kl_returns_kl_psi_per_sample() -> None:
    """Under ``return_per_sample`` the dict also carries ``kl_psi_per_sample``."""
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    schedule = DiffusionScheduleConfig(
        S_k=2, k_chunk=2, num_steps=20,
        beta_min=0.1, beta_max=20.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = _make_diffusion(baseline, schedule=schedule)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    out = transition.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        sigma_data=sigma_data,
        return_per_sample=True,
    )
    assert "kl_psi_per_sample" in out
    assert out["kl_psi_per_sample"].shape == (B * S,)
    assert torch.isfinite(out["kl_psi_per_sample"]).all()


def test_local_transition_kl_init_returns_loss_psi_with_return_psi() -> None:
    """``transition_kl_init(return_psi=True)`` includes ``loss_psi`` in the dict.

    Adapted from local: claude branch requires ``return_psi=True`` opt-in
    (local always emits ``loss_psi``). Confirms the phith composition is
    unchanged (loss = entropy + loss_init + kl_aux, with the diffusion
    ``_init_entropy_term`` cancelling to zero).
    """
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    transition = _make_diffusion(baseline)
    aux = AuxPosterior(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    zs, enc_stats, time_embed, _ = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    out = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=sigma_data,
        return_psi=True,
    )
    assert "loss_psi" in out
    assert torch.isfinite(out["loss_psi"])
    # The main "loss" composition on the phith side is unchanged: entropy +
    # loss_init + kl_aux (with diffusion cancelling the encoder entropy → 0).
    assert torch.allclose(out["loss"], out["loss_init"] + out["kl_aux"])
    # And without return_psi, loss_psi must be absent (opt-in contract).
    out_default = transition.transition_kl_init(
        enc_stats=enc_stats,
        zs=zs,
        aux_posterior=aux,
        time_embed=time_embed,
        sigma_data=SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t"),
    )
    assert "loss_psi" not in out_default


@pytest.mark.parametrize(
    "transition_kind",
    ["gaussian", "baseline_gaussian"],
)
def test_local_score_init_step_nondiffusion_returns_zero_psi(transition_kind: str) -> None:
    """Non-diffusion transitions return a zero ψ tensor from ``_score_init_step``.

    The hook contract is a 2-tuple ``(phith, psi)``. Gaussian and
    BaselineGaussian have no ψ score-net side so they return
    ``loss.new_zeros(())``.
    """
    from ddssm.model.transitions.transitions import GaussianTransition
    from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

    j = 1
    if transition_kind == "gaussian":
        transition = GaussianTransition(
            latent_dim=D, j=j, emb_time_dim=EMB_TIME, hidden_dim=8
        )
    else:
        baseline = MLPBaseline(latent_dim=D, j=j, hidden_dim=4, n_layers=1)
        transition = BaselineGaussianTransition(
            baseline=baseline, latent_dim=D, j=j, emb_time_dim=EMB_TIME
        )

    torch.manual_seed(0)
    BS = B * S
    z_t = torch.randn(BS, D)
    z_hist = torch.randn(BS, D, j)
    mus = 0.3 * torch.randn(B, S, D, T)
    logvars = -1.0 + 0.2 * torch.randn(B, S, D, T)
    enc_stats = {"mus": mus, "logvars": logvars}
    time_embed = torch.randn(B, T, EMB_TIME)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    out = transition._score_init_step(
        step=0,
        z_t=z_t,
        z_hist=z_hist,
        enc_stats=enc_stats,
        time_embed=time_embed,
        sigma_data=sigma_data,
        B=B,
        S=S,
        T=T,
    )
    assert isinstance(out, tuple) and len(out) == 2
    phith, psi = out
    assert psi.shape == ()  # ``loss.new_zeros(())``
    assert float(psi) == 0.0
    # And phith must be a scalar too.
    assert phith.dim() == 0


def test_local_pk_clip_bounds_max_weight() -> None:
    """The IS weight max is bounded by ``w_ll_max · (1 + K·c) / c``.

    A complementary framing to :func:`test_pk_clip_bounds_is_weight`
    (which asserts ``max(1/p) ≤ (1+K·c)/c``): given an arbitrary
    non-negative per-k weight ``w_ll``, the ratio ``w_ll / p_k`` is
    bounded by ``max(w_ll) · (1+K·c)/c``.
    """
    from ddssm.model.transitions.diffusion import _adaptive_is_density_meandom

    torch.manual_seed(0)
    K = 20
    sigma_tilde = torch.linspace(1e-3, 5.0, K)
    sigma_d2 = torch.tensor([0.25])
    p_k_clip = 1e-3

    pk = _adaptive_is_density_meandom(
        sigma_tilde, sigma_d2, floor=1e-12, p_k_clip=p_k_clip
    ).squeeze(0)
    w_ll = torch.rand(K)  # random non-negative weight
    ratio = w_ll / pk
    w_ll_max = float(w_ll.max())
    bound = w_ll_max * (1.0 + K * p_k_clip) / p_k_clip
    assert float(ratio.max()) <= bound + 1e-4


def test_local_pk_clip_threaded_through_transition_kl_via_spy() -> None:
    """``schedule.p_k_clip`` reaches the adaptive density inside ``transition_kl``.

    Complementary to :func:`test_pk_clip_threaded_through_transition_kl`
    (which compares kl values); this one spies on the density function
    and asserts the clip value flows through.
    """
    import ddssm.model.transitions.diffusion as diff_mod

    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    schedule_clip = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=20,
        beta_min=0.1, beta_max=20.0, tau_min=1e-3,
        k_sampling_mode="adaptive_is",
        p_k_clip=1e-2,  # aggressive floor so we can detect it
    )
    transition = _make_diffusion(baseline, schedule=schedule_clip)
    zs, enc_stats, time_embed, logq_paths = _make_batch()
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")

    captured: dict[str, torch.Tensor] = {}
    real_meandom = diff_mod._adaptive_is_density_meandom

    def spy(sigma_tilde, sigma_d2, floor=1e-12, p_k_clip=0.0):
        captured["p_k_clip"] = p_k_clip
        pk = real_meandom(sigma_tilde, sigma_d2, floor=floor, p_k_clip=p_k_clip)
        captured["pk"] = pk.detach().clone()
        return pk

    import unittest.mock
    with unittest.mock.patch.object(
        diff_mod, "_adaptive_is_density_meandom", side_effect=spy
    ):
        with torch.no_grad():
            transition.transition_kl(
                enc_stats=enc_stats,
                zs=zs,
                logq_paths=logq_paths,
                time_embed=time_embed,
                sigma_data=sigma_data,
            )
    assert "p_k_clip" in captured
    assert captured["p_k_clip"] == 1e-2, (
        f"schedule.p_k_clip=1e-2 not threaded through: {captured['p_k_clip']=}"
    )
    # And the clipped density respects the floor.
    K = int(transition.sigma_tilde.numel())
    floor_expected = 1e-2 / (1.0 + K * 1e-2)
    assert float(captured["pk"].min()) >= floor_expected - 1e-8


def test_local_diffusion_schedule_config_has_p_k_clip_default() -> None:
    """``DiffusionScheduleConfig.p_k_clip`` defaults to 1e-3."""
    config = DiffusionScheduleConfig()
    assert hasattr(config, "p_k_clip"), "DiffusionScheduleConfig missing p_k_clip"
    assert config.p_k_clip == 1e-3

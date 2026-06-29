"""Tests for the parallel AR-flow-on-noise encoder (``ARFlowEncoder``).

Built up alongside the implementation as TDD vertical slices (see the plan).
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn

from ddssm.model.encoder import ARFlowEncoder, arflow_cumsum
from ddssm.nn.futsum import TransformerFutureSummary


def make_encoder(
    *,
    data_dim: int = 12,
    latent_dim: int = 8,
    hidden_dim: int = 16,
    channels: int = 16,
    nheads: int = 2,
    backbone: str = "transformer",
    covariate_dim: int = 0,
    static_covariate_dim: int = 0,
    grad_checkpoint: bool = False,
) -> ARFlowEncoder:
    return ARFlowEncoder(
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=1,
        emb_time_dim=0,
        use_mask=False,
        hidden_dim=hidden_dim,
        covariate_dim=covariate_dim,
        static_covariate_dim=static_covariate_dim,
        channels=channels,
        causal_layers=2,
        nheads=nheads,
        backbone=backbone,
        fut_summary=partial(
            TransformerFutureSummary,
            summary_dim=hidden_dim,
            nheads=nheads,
            transformer_layers=1,
        ),
        grad_checkpoint=grad_checkpoint,
    )


# ---- Slice 1: drop-in contract --------------------------------------------


def test_contract_shapes_and_family() -> None:
    B, D, T, S, d = 2, 12, 5, 3, 8
    enc = make_encoder(data_dim=D, latent_dim=d, hidden_dim=16)
    obs = torch.randn(B, D, T)
    te = torch.zeros(B, T, 0)

    zs, logqs, stats = enc.sample_paths(obs, te, S=S)

    assert zs.shape == (B, S, d, T)
    assert logqs.shape == (B, S, T)
    assert set(stats.keys()) == {"mus", "logvars"}
    assert stats["mus"].shape == (B, S, d, T)
    assert stats["logvars"].shape == (B, S, d, T)
    assert enc.is_gaussian_family is True
    assert torch.isfinite(zs).all() and torch.isfinite(logqs).all()


def test_sample_paths_default_S_is_one() -> None:
    B, D, T, d = 2, 12, 4, 8
    enc = make_encoder(data_dim=D, latent_dim=d, hidden_dim=16)
    zs, logqs, stats = enc.sample_paths(torch.randn(B, D, T), torch.zeros(B, T, 0))
    assert zs.shape == (B, 1, d, T)
    assert logqs.shape == (B, 1, T)


# ---- Slice A+B: strong data path (backbone-input + head-concat) -----------


def test_data_influences_latent_mean_at_init() -> None:
    # The A+B data path + small-nonzero head init on the data branch must make the
    # latent mean depend on the observed data AT INIT. The old zero-init head gave
    # g=0 → mus = shift_right(cumsum(ση)) → purely noise-driven, data-independent →
    # no foothold for recon to grow the signal (the collapse).
    torch.manual_seed(0)
    B, D, T, d = 2, 12, 6, 8
    enc = make_encoder(data_dim=D, latent_dim=d, hidden_dim=16)
    obs = torch.randn(B, D, T, requires_grad=True)
    te = torch.zeros(B, T, 0)
    _zs, _lq, stats = enc.sample_paths(obs, te, S=1)
    stats["mus"].sum().backward()
    assert obs.grad is not None
    assert obs.grad.abs().max() > 0.0  # data reaches the latent mean at init


# ---- Slice 2: additive flow + consistency ---------------------------------


def test_arflow_cumsum_identities() -> None:
    from ddssm.model.encoder import arflow_cumsum, _shift_right_time

    BS, d, T = 4, 8, 6
    g = torch.randn(BS, d, T)
    logvar = torch.randn(BS, d, T) * 0.3
    eta = torch.randn(BS, d, T)
    z, mus, lv = arflow_cumsum(g, eta, logvar, -7.0, 6.0)
    sigma = (0.5 * lv).exp()

    # cumsum of the increment, z_0 = 0
    assert torch.allclose(z, torch.cumsum(g + sigma * eta, dim=-1), atol=1e-6)
    # μ_t = z_{t-1} + g_t, and the first step μ_1 = g_1 (z_0 = 0)
    assert torch.allclose(mus, _shift_right_time(z) + g, atol=1e-6)
    assert torch.allclose(mus[..., 0], g[..., 0], atol=1e-6)
    # reparam residual z_t − μ_t = σ_t⊙η_t
    assert torch.allclose(z - mus, sigma * eta, atol=1e-6)
    # THE σ_data / ESM invariant: mu_hat = μ_t − z_{t-1} = g_t (the residual the
    # diffusion denoises) — NOT z_t. So σ_data tracks Var[g]+E[σ²], not Var[z_t].
    assert torch.allclose(mus[..., 1:] - z[..., :-1], g[..., 1:], atol=1e-6)


def test_logq_self_consistency() -> None:
    from ddssm.nn.gaussians import gaussian_log_prob

    B, D, T, S, d = 2, 12, 5, 3, 8
    enc = make_encoder(data_dim=D, latent_dim=d, hidden_dim=16)
    zs, logqs, stats = enc.sample_paths(torch.randn(B, D, T), torch.zeros(B, T, 0), S=S)
    recomputed = gaussian_log_prob(
        zs.permute(0, 1, 3, 2),
        stats["mus"].permute(0, 1, 3, 2),
        stats["logvars"].permute(0, 1, 3, 2),
    )  # (B, S, T)
    assert torch.allclose(recomputed, logqs, atol=1e-5)


def test_persistence_baseline_matches_z_prev() -> None:
    # The transition forms mu_hat = mus − PersistenceBaseline.mean(z_hist) and relies
    # on it being the innovation g. That holds iff the baseline's z_{t-1} (from
    # unfolding zs) equals the encoder's z_prev = shift_right(zs). Pin the off-by-one.
    from ddssm.model.centering.baselines import PersistenceBaseline
    from ddssm.model.encoder import _shift_right_time

    B, S, d, T, j = 2, 2, 8, 6, 1
    enc = make_encoder(latent_dim=d, hidden_dim=16)
    zs, _, _ = enc.sample_paths(torch.randn(B, 12, T), torch.zeros(B, T, 0), S=S)
    baseline = PersistenceBaseline(latent_dim=d, j=j)
    z_prev = _shift_right_time(zs)  # z_prev[..., t] = zs[..., t-1]
    for t in range(j, T):
        z_hist = zs[..., t - j : t].reshape(-1, d, j)  # (N, d, j)
        mu_p = baseline.mean(z_hist).reshape(B, S, d)
        assert torch.allclose(mu_p, z_prev[..., t], atol=1e-6), f"t={t}"


# ---- Slice 3 + 5: strict causality (both backbones) -----------------------


def _live_encoder(backbone: str, d: int = 4, T: int = 6) -> ARFlowEncoder:
    enc = make_encoder(
        latent_dim=d, hidden_dim=16, channels=16, nheads=2, backbone=backbone
    )
    # Randomize the zero-init head so g/σ actually depend on η — zero-init would make
    # ∂μ/∂η structurally zero and mask an anti-causal wiring (Agent-4's blind spot).
    nn.init.normal_(enc.causal_net.head.weight, std=0.5)
    nn.init.normal_(enc.causal_net.head.bias, std=0.5)
    enc.eval()
    return enc


@pytest.mark.parametrize("backbone", ["conv", "transformer"])
def test_strict_causality_probe(backbone: str) -> None:
    d, T = 4, 6
    enc = _live_encoder(backbone, d=d, T=T)
    h = torch.randn(1, T, enc.summary_dim)
    eta = torch.randn(1, d, T, requires_grad=True)
    g, logvar = enc.causal_net(eta, h)
    _, mus, _ = arflow_cumsum(
        g, eta, logvar, enc.clamp_logvar_min, enc.clamp_logvar_max
    )

    def grad_wrt_eta(scalar: torch.Tensor) -> torch.Tensor:
        if eta.grad is not None:
            eta.grad = None
        scalar.backward(retain_graph=True)
        return eta.grad.clone()

    for s in range(T):
        # μ_s and logσ²_s must NOT depend on η_{≥s} (the Gaussian-conditional claim
        # is about both the mean AND the variance).
        for tensor, name in ((g, "g"), (logvar, "logvar"), (mus, "mu")):
            grad = grad_wrt_eta(tensor[..., s].sum())
            assert grad[..., s:].abs().max() < 1e-6, f"{name} leaks η≥{s} ({backbone})"
        # …and it must actually USE η_{<s} (path alive, not dead-zeroed).
        if s >= 1:
            grad = grad_wrt_eta(g[..., s].sum())
            assert grad[..., :s].abs().max() > 1e-7, f"g_{s} dead path ({backbone})"


@pytest.mark.parametrize("backbone", ["conv", "transformer"])
def test_causality_finite_difference(backbone: str) -> None:
    d, T, t_pert = 4, 6, 2
    enc = _live_encoder(backbone, d=d, T=T)
    h = torch.randn(1, T, enc.summary_dim)
    with torch.no_grad():
        eta0 = torch.randn(1, d, T)
        g0, lv0 = enc.causal_net(eta0, h)
        _, mus0, _ = arflow_cumsum(
            g0, eta0, lv0, enc.clamp_logvar_min, enc.clamp_logvar_max
        )
        eta1 = eta0.clone()
        eta1[..., 0, t_pert] += 1.0
        g1, lv1 = enc.causal_net(eta1, h)
        _, mus1, _ = arflow_cumsum(
            g1, eta1, lv1, enc.clamp_logvar_min, enc.clamp_logvar_max
        )
    # μ_{≤t_pert} unchanged (they depend only on η_{<s} ∌ η_{t_pert}); μ_{>t_pert} must move.
    assert torch.allclose(mus0[..., : t_pert + 1], mus1[..., : t_pert + 1], atol=1e-6)
    assert (mus0[..., t_pert + 1 :] - mus1[..., t_pert + 1 :]).abs().max() > 1e-6


# ---- Slice 4: feature mixer + h_t side-info + parallelism ------------------


@pytest.mark.parametrize("backbone", ["conv", "transformer"])
def test_h_affects_g(backbone: str) -> None:
    # The per-position h_t side-info must actually steer g (otherwise the evidence is
    # ignored). With η fixed, two different h → different g.
    enc = _live_encoder(backbone, d=4, T=6)
    eta = torch.randn(1, 4, 6)
    h1 = torch.randn(1, 6, enc.summary_dim)
    h2 = torch.randn(1, 6, enc.summary_dim)
    g1, _ = enc.causal_net(eta, h1)
    g2, _ = enc.causal_net(eta, h2)
    assert (g1 - g2).abs().max() > 1e-6


def test_covariates_reach_summary() -> None:
    # dssd passes `covariates` into sample_paths and gluonts carries temporal covariates;
    # they must reach the future-summary (else h silently drops them).
    B, D, T, V, d = 2, 12, 5, 3, 4
    enc = make_encoder(data_dim=D, latent_dim=d, hidden_dim=16, covariate_dim=V)
    nn.init.normal_(enc.causal_net.head.weight, std=0.5)
    nn.init.normal_(enc.causal_net.head.bias, std=0.5)
    enc.eval()
    obs, te = torch.randn(B, D, T), torch.zeros(B, T, 0)
    cov0, cov1 = torch.zeros(B, V, T), torch.randn(B, V, T)
    torch.manual_seed(0)
    zs0, _, _ = enc.sample_paths(obs, te, covariates=cov0)
    torch.manual_seed(0)
    zs1, _, _ = enc.sample_paths(obs, te, covariates=cov1)
    assert (zs0 - zs1).abs().max() > 1e-6


def test_single_parallel_forward() -> None:
    # The whole point: one parallel pass over all T, not a per-step Python loop.
    enc = make_encoder(latent_dim=4, hidden_dim=16)
    calls = {"n": 0}
    enc.causal_net.register_forward_hook(
        lambda *a: calls.__setitem__("n", calls["n"] + 1)
    )
    enc.sample_paths(torch.randn(2, 12, 32), torch.zeros(2, 32, 0), S=2)
    assert calls["n"] == 1


# ---- Slice 6: identity-baseline guard -------------------------------------


def test_identity_baseline_guard() -> None:
    from ddssm.model.dssd import _require_persistence_baseline
    from ddssm.model.centering.baselines import PersistenceBaseline, ZeroBaseline

    enc = make_encoder(latent_dim=4, hidden_dim=16)
    assert enc.requires_persistence_baseline is True
    # persistence/identity is accepted
    _require_persistence_baseline(enc, PersistenceBaseline(latent_dim=4, j=1))
    # any other baseline (or None) is rejected — the additive frame would mis-center
    with pytest.raises(NotImplementedError):
        _require_persistence_baseline(
            enc, ZeroBaseline(latent_dim=4, j=1, hidden_dim=8, n_layers=2)
        )
    with pytest.raises(NotImplementedError):
        _require_persistence_baseline(enc, None)


# ---- Slice 7: closed-form entropy matches MC ------------------------------


def test_entropy_closed_form_matches_mc() -> None:
    d, T, j = 4, 5, 1
    enc = make_encoder(latent_dim=d, hidden_dim=16, channels=16, nheads=2)
    # small random head so logvar varies (zero-init would make this trivially constant)
    nn.init.normal_(enc.causal_net.head.weight, std=0.2)
    nn.init.normal_(enc.causal_net.head.bias, std=0.2)
    enc.eval()
    _, logqs, stats = enc.sample_paths(
        torch.randn(1, 12, T), torch.zeros(1, T, 0), S=6000
    )
    closed = enc.entropy_transition(stats, j)
    mc = enc.mc_entropy_transition(logqs, j)
    assert torch.allclose(closed, mc, rtol=0.05, atol=0.05), (closed.item(), mc.item())


# ---- Slice 8: end-to-end ELBO + σ_data residual tracking -------------------


def test_end_to_end_arflow_elbo_and_sigma_data() -> None:
    from experiments.gluonts_forecast.model import build_gluonts_model

    torch.manual_seed(0)
    B, D, T, latent = 2, 12, 6, 8
    model = build_gluonts_model(
        data_dim=D,
        latent_dim=latent,
        T_max=T,
        nheads=2,
        channels=16,
        summary_layers=1,
        diffusion_layers=2,
        num_steps=4,
        grad_checkpoint=False,
        sigma_data_ema_decay=0.0,  # one update lands exactly on the batch estimate
        encoder_type="arflow",
    )
    assert isinstance(model.encoder, ARFlowEncoder)
    model.train()  # σ_data updates are training-only

    x = torch.randn(B, D, T)
    mask = torch.ones(B, D, T)
    timepoints = torch.arange(T).unsqueeze(0).expand(B, -1)
    components, metrics, _ = model(
        observed_data=x, observation_mask=mask, timepoints=timepoints
    )
    loss = components.total()
    assert torch.isfinite(loss), loss
    loss.backward()
    # Gradient reaches the encoder's IAF (μ, logσ²) head.
    hgrad = model.encoder.causal_net.head.weight.grad
    assert hgrad is not None and torch.isfinite(hgrad).all() and hgrad.abs().sum() > 0

    # σ_data tracks the IAF residual Var[μ_t − z_{t-1}] + E[σ²]. Since z_t = μ_t + σ_t·η_t is
    # NOT accumulated (no cumsum), the residual stays BOUNDED and roughly flat in t — the
    # whole point vs the random-walk Var[z_t] ∝ t. Slot 0 (t=1) folds in the VHP-init aux
    # variance, excluded here.
    sd2 = model.sigma_data.sigma_data2.detach()
    main = sd2[1:]
    assert torch.isfinite(main).all() and (main > 0).all()
    assert (main < 50).all()  # bounded — no ∝t random-walk blow-up
    assert (main.max() / main.min()).item() < 6.0  # roughly flat in t


# ---- Forward-message variants (o1_flow / fb_mf / fb_flow) ------------------


def make_fwd_encoder(
    forward_message: str,
    stochastic_state: bool,
    *,
    data_dim: int = 12,
    latent_dim: int = 8,
    hidden_dim: int = 16,
    channels: int = 16,
    nheads: int = 2,
) -> ARFlowEncoder:
    fwd = partial(
        TransformerFutureSummary, summary_dim=hidden_dim, nheads=nheads,
        transformer_layers=1, reverse_time=False,  # forward-causal f_t
    )
    return ARFlowEncoder(
        data_dim=data_dim, latent_dim=latent_dim, j=1, emb_time_dim=0,
        use_mask=False, hidden_dim=hidden_dim, channels=channels, causal_layers=2,
        nheads=nheads, backbone="transformer",
        fut_summary=partial(
            TransformerFutureSummary, summary_dim=hidden_dim, nheads=nheads,
            transformer_layers=1,
        ),
        stochastic_state=stochastic_state, forward_message=forward_message,
        fwd_summary=fwd, fwd_layers=2,
    )


# (forward_message, stochastic_state) for the 3 shipped variants.
_FWD_VARIANTS = [
    ("fwd_summary", True),   # o1_flow
    ("fwd_data", False),     # fb_mf
    ("fwd_data", True),      # fb_flow
]


@pytest.mark.parametrize("fm,ss", _FWD_VARIANTS)
def test_fwd_message_shapes_and_logq(fm: str, ss: bool) -> None:
    from ddssm.nn.gaussians import gaussian_log_prob

    B, D, T, S, d = 2, 12, 5, 3, 8
    enc = make_fwd_encoder(fm, ss, data_dim=D, latent_dim=d)
    zs, logqs, stats = enc.sample_paths(torch.randn(B, D, T), torch.zeros(B, T, 0), S=S)
    assert zs.shape == (B, S, d, T)
    assert logqs.shape == (B, S, T)
    assert stats["mus"].shape == (B, S, d, T)
    # logq must equal the per-step Gaussian density at the realized (z; μ, logσ²).
    recomputed = gaussian_log_prob(
        zs.permute(0, 1, 3, 2),
        stats["mus"].permute(0, 1, 3, 2),
        stats["logvars"].permute(0, 1, 3, 2),
    )
    assert torch.allclose(recomputed, logqs, atol=1e-5)
    assert torch.isfinite(zs).all() and torch.isfinite(logqs).all()


@pytest.mark.parametrize("fm,ss", _FWD_VARIANTS)
def test_fwd_message_data_reaches_mean(fm: str, ss: bool) -> None:
    # The forward-message path must let the observed data reach the latent mean.
    B, D, T, d = 2, 12, 6, 8
    enc = make_fwd_encoder(fm, ss, data_dim=D, latent_dim=d)
    obs = torch.randn(B, D, T, requires_grad=True)
    _zs, _lq, stats = enc.sample_paths(obs, torch.zeros(B, T, 0), S=1)
    stats["mus"].sum().backward()
    assert obs.grad is not None and obs.grad.abs().max() > 0.0


def test_fwd_data_message_is_live() -> None:
    # The forward summary module (f_t) must carry gradient — not dead-wired.
    B, D, T, d = 2, 12, 6, 8
    enc = make_fwd_encoder("fwd_data", True, data_dim=D, latent_dim=d)
    _zs, _lq, stats = enc.sample_paths(torch.randn(B, D, T), torch.zeros(B, T, 0), S=1)
    stats["mus"].sum().backward()
    grads = [p.grad for p in enc.fwd_sum_module.parameters() if p.grad is not None]
    assert grads and max(g.abs().max() for g in grads) > 0.0


def test_fwd_summary_refiner_is_live() -> None:
    # The forward refiner producing o_t must carry gradient.
    B, D, T, d = 2, 12, 6, 8
    enc = make_fwd_encoder("fwd_summary", True, data_dim=D, latent_dim=d)
    _zs, _lq, stats = enc.sample_paths(torch.randn(B, D, T), torch.zeros(B, T, 0), S=1)
    stats["mus"].sum().backward()
    grads = [p.grad for p in enc.fwd_refiner.parameters() if p.grad is not None]
    assert grads and max(g.abs().max() for g in grads) > 0.0


@pytest.mark.parametrize("reverse_time", [True, False])
def test_future_summary_time_direction(reverse_time: bool) -> None:
    # reverse_time toggles the data-message direction: backward b_t = F(x_{t:T})
    # vs forward f_t = F(x_{1:t}). Perturb x at t_pert and check which steps move.
    T, D, t_pert = 6, 5, 2
    fs = TransformerFutureSummary(
        data_dim=D, emb_time_dim=0, use_mask=False, summary_dim=16, nheads=2,
        transformer_layers=1, reverse_time=reverse_time,
    )
    fs.eval()
    te = torch.zeros(1, T, 0)
    with torch.no_grad():
        x0 = torch.randn(1, T, D)
        h0 = fs(observed_data=x0, observed_mask=None, t_emb=te)
        x1 = x0.clone()
        x1[:, t_pert, :] += 1.0
        h1 = fs(observed_data=x1, observed_mask=None, t_emb=te)
    diff = (h0 - h1).abs().amax(dim=-1)[0]  # (T,)
    assert diff[t_pert] > 1e-6
    if reverse_time:
        # b_t includes x_{t_pert} iff t ≤ t_pert → strictly-later steps unchanged.
        assert diff[t_pert + 1 :].max() < 1e-6
    else:
        # f_t includes x_{t_pert} iff t ≥ t_pert → strictly-earlier steps unchanged.
        assert diff[:t_pert].max() < 1e-6


def test_fwd_flow_strict_noise_causality() -> None:
    # Adding a (forward/backward) DATA context must NOT break the IAF's noise
    # causality: μ_s, logσ²_s ⟂ η_{≥s} (the exact-log-prob guarantee). Probe the
    # conditioner at the augmented context dim used by the fwd_data flow.
    d, T = 4, 6
    enc = make_fwd_encoder("fwd_data", True, latent_dim=d)
    nn.init.normal_(enc.causal_net.head.weight, std=0.5)
    nn.init.normal_(enc.causal_net.head.bias, std=0.5)
    enc.eval()
    ctx_dim = enc.causal_net.summary_dim  # 2·hidden for fwd_data
    c = torch.randn(1, T, ctx_dim)
    eta = torch.randn(1, d, T, requires_grad=True)
    g, logvar = enc.causal_net(eta, c)

    def grad_wrt_eta(scalar: torch.Tensor) -> torch.Tensor:
        if eta.grad is not None:
            eta.grad = None
        scalar.backward(retain_graph=True)
        return eta.grad.clone()

    for s in range(T):
        for tensor, name in ((g, "mu"), (logvar, "logvar")):
            grad = grad_wrt_eta(tensor[..., s].sum())
            assert grad[..., s:].abs().max() < 1e-6, f"{name} leaks η≥{s}"

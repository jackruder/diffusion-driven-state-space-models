"""Full-method LGSSM integration test for :meth:`DDSSM_base.log_prob`.

Unlike ``test_lgssm_integration`` (which reimplements the assembly loops
in-test against the prob-flow + IWAE *utilities*), this drives the real
``DDSSM_base.log_prob`` orchestration end-to-end and checks it against the
exact Kalman log-likelihood.

To make the result *exact* the leaf component **values** are set to the
ground-truth LGSSM (their internals are neural nets that wouldn't equal
ground truth otherwise), but the **orchestration is production code**:
the encode call, the per-``t`` decoder loop, the per-``t`` prob-flow ODE
transition loop, the ``sigma_d2`` lookup, and the IWAE assembly all run as
``DDSSM_base.log_prob`` itself.

Mocked to ground truth:
  * encoder  → exact RTS smoother (FFBS samples + analytic log q), so the
    IWAE collapses to the exact value at K=1 (model-v2.org § sanity #1);
  * decoder  → true emission ``N(z_t, r)``;
  * transition score → analytic diffused-conditional score (the prob-flow
    ODE still runs for real on it);
  * initial state → analytic ``log N(z_1; m0, P0)`` (sidesteps the implicit
    VHP-marginal representation; the init *plumbing* is covered separately
    by the log_prob_init unit tests + the K=1 self-consistency test).
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import torch

from ddssm.aggregators import ContextProducerAggregator
from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.baselines import MLPBaseline
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.combiners import CompoundCombiner
from ddssm.decoder import GaussianDecoder
from ddssm.diffnets import (
    CSDIUnet,
    ContextProducer,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.dist_heads import GaussianDistHead
from ddssm.dssd import DDSSM_base
from ddssm.encoder import GaussianEncoder
from ddssm.fusions import ConcatLinearFusion
from ddssm.futsum import GRUFutureSummary
from ddssm.gaussians import GaussianHead
from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition
from ddssm.transitions.diffusion import (
    DiffusionScheduleConfig,
    DiffusionTransition,
)

from tests.test_likelihood.test_lgssm_integration import (
    A,
    BETA_MAX,
    BETA_MIN,
    M0,
    P0,
    Q,
    R,
    _backward_sample,
    _kalman_loglik,
    _normal_logpdf,
)

EMB_TIME = 8
CHANNELS = 8
NHEADS = 4
T_MAX = 10


def _make_scalar_stage2_model() -> DDSSM_base:
    """Tiny d=1, j=1 stage-2 diffusion model.  Leaf modules are placeholders —
    the test monkey-patches encoder/decoder/score/init to ground truth."""
    d, data, j = 1, 1, 1
    ctx = partial(
        ContextProducer, channels=CHANNELS, num_layers=1,
        residual_block=ResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    agg = partial(
        ContextProducerAggregator, channels=CHANNELS, num_layers=1,
        residual_block=ResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    tiny_unet = partial(
        CSDIUnet, channels=CHANNELS, n_layers=1, embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    encoder = GaussianEncoder(
        data_dim=data, latent_dim=d, j=j, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS,
        combiner=partial(
            CompoundCombiner, aggregator=agg, fusion=partial(ConcatLinearFusion),
        ),
        dist_head=partial(GaussianDistHead),
        fut_summary=partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1),
    )
    decoder = GaussianDecoder(
        latent_dim=d, data_dim=data, j=j, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS, context=ctx, gaussian_head=GaussianHead,
    )
    baseline = MLPBaseline(latent_dim=d, j=j, hidden_dim=8, n_layers=2)
    schedule = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=20, beta_min=BETA_MIN, beta_max=BETA_MAX,
        tau_min=1e-3, k_sampling_mode="uniform",
    )
    stage1 = BaselineGaussianTransition(
        baseline=baseline, latent_dim=d, j=j, emb_time_dim=EMB_TIME,
    )
    transition = DiffusionTransition(
        baseline=baseline, latent_dim=d, j=j, emb_time_dim=EMB_TIME,
        T_max=T_MAX, unet=tiny_unet, schedule=schedule,
    )
    aux = AuxPosterior(latent_dim=d, j=j, hidden_dim=8, n_layers=2)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed", init_value=1.0)
    hparams = SimpleNamespace(
        S=1, ema_decay=0.999, weight_decay=1e-2, batch_size=2,
        grad_accum_steps=1, t_chunk=4, clip_grad_norm=None,
        enc_lr=1e-3, dec_lr=1e-3,
        trans_lr=1e-3, logvar_min=-7.0, logvar_max=7.0,
    )
    model = DDSSM_base(
        encoder=encoder, decoder=decoder, transition=transition,
        j=j, data_dim=data, latent_dim=d, emb_time_dim=EMB_TIME,
        aux_posterior=aux, baseline=baseline,
        sigma_data=sigma_data, stage1_transition=stage1,
    )
    model.stage_selector = "stage_2"
    return model


def test_full_model_log_prob_matches_kalman_loglik() -> None:
    """``DDSSM_base.log_prob`` reproduces the exact Kalman log-likelihood.

    Leaf component values are ground-truth (encoder = RTS smoother,
    decoder/init analytic, transition score analytic); the assembly loops
    + prob-flow ODE run as production code.  With the exact posterior as
    proposal, each IWAE weight equals ``log p(x_{1:T})`` to ODE tolerance,
    so the K-trajectory estimate must match the Kalman value.
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dtype = torch.float64

    B, T, K = 2, 4, 4
    x = torch.randn(B, T, dtype=dtype)  # observations

    true_loglik, m_filt, P_filt = _kalman_loglik(x)
    z, total_logq = _backward_sample(m_filt, P_filt, K, gen)  # z (B,K,T), logq (B,K)

    model = _make_scalar_stage2_model()
    model.eval()
    transition = model.transition

    # --- mock encoder: exact RTS smoother (FFBS) ---
    zs = z.unsqueeze(2)  # (B, K, d=1, T)
    logq_paths = torch.zeros(B, K, T, dtype=dtype)
    logq_paths[:, :, 0] = total_logq  # model sums over T → exact log q
    model._encode_latents = lambda **kw: (zs, logq_paths, {})

    # --- mock decoder: true emission N(z_t, r) ---
    R_t = torch.tensor(R, dtype=dtype)

    def mock_decoder(
        x_t, z_hist, time_embed, time_idx,
        observation_mask_t=None, covariates=None, static_embed=None,
    ):
        z_t = z_hist[..., -1]  # (B, d)
        lp = _normal_logpdf(x_t, z_t, R_t).sum(dim=-1)  # (B,)
        return lp, None, None, None

    model.decoder.log_likelihood = mock_decoder

    # --- mock transition score: analytic diffused-conditional score ---
    def analytic_score(z, tau, z_hist, ctx, sigma_d2, padding_mask=None):
        if tau.dim() == 0:
            tau = tau.expand(z.shape[0])
        int_beta = BETA_MIN * tau + 0.5 * (BETA_MAX - BETA_MIN) * tau.pow(2)
        alpha = torch.exp(-0.5 * int_beta).unsqueeze(-1)
        alpha2 = alpha.pow(2)
        var = alpha2 * Q + (1.0 - alpha2)
        z_prev = z_hist[..., -1]  # (B, d) = z_{t-1}
        return -(z - alpha * A * z_prev) / var

    transition.score = analytic_score

    # --- mock initial state: analytic log N(z_1; m0, P0) ---
    M0_t = torch.tensor(M0, dtype=dtype)
    P0_t = torch.tensor(P0, dtype=dtype)

    def mock_log_prob_init(zs, aux_posterior, time_embed, sigma_data=None, covariates=None, **kw):
        z1 = zs[:, :, 0, 0]  # (B, K) — d=1, t=0
        return _normal_logpdf(z1, M0_t, P0_t)

    transition.log_prob_init = mock_log_prob_init

    observed_data = x.unsqueeze(1)  # (B, 1, T)
    observation_mask = torch.ones(B, 1, T, dtype=dtype)
    timepoints = torch.arange(T).expand(B, T).clone().long()

    log_p = model.log_prob(
        observed_data=observed_data,
        observation_mask=observation_mask,
        timepoints=timepoints,
        rtol=1e-9,
        atol=1e-9,
    )

    assert log_p.shape == (B,)
    assert torch.allclose(log_p, true_loglik, atol=2e-3, rtol=0.0)

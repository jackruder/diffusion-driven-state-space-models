"""End-to-end smoke test for :meth:`DDSSM_base.log_prob`.

The component primitives (prob-flow ODE, IWAE assembly, VHP) are
unit-tested in ``test_prob_flow``, ``test_iwae`` and ``test_vhp``;
this test verifies the composition layer wires them into a valid
``(B,)`` log-likelihood on a tiny stage-2 diffusion model.
"""

from __future__ import annotations

from types import SimpleNamespace
from functools import partial

import torch

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.model.dssd import DDSSM_base
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.diffnets import (
    CSDIUnet,
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.gaussians import GaussianHead
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.aggregators import ContextProducerAggregator
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import MLPBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

J = 2
DATA_DIM = 3
LATENT_DIM = 4
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2
T_MAX = 10


def _make_stage2_model() -> DDSSM_base:
    ctx = partial(
        ContextProducer,
        channels=CHANNELS,
        num_layers=1,
        residual_block=ResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    agg = partial(
        ContextProducerAggregator,
        channels=CHANNELS,
        num_layers=1,
        residual_block=ResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    tiny_unet = partial(
        CSDIUnet,
        channels=CHANNELS,
        n_layers=1,
        embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    encoder = GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=partial(
            CompoundCombiner,
            aggregator=agg,
            fusion=partial(ConcatLinearFusion),
        ),
        dist_head=partial(GaussianDistHead),
        fut_summary=partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1),
    )
    decoder = GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=ctx,
        gaussian_head=GaussianHead,
    )
    baseline = MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    schedule = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    stage1 = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=tiny_unet,
        schedule=schedule,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed")
    hparams = SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=2,
        grad_accum_steps=1,
        t_chunk=4,
        clip_grad_norm=None,
        enc_lr=1e-3,
        dec_lr=1e-3,
        trans_lr=1e-3,
        logvar_min=-13.0,
        logvar_max=13.0,
    )
    model = DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        aux_posterior=aux,
        baseline=baseline,
        sigma_data=sigma_data,
        stage1_transition=stage1,
    )
    model.stage_selector = "stage_2"
    return model


def test_log_prob_runs_end_to_end_and_returns_finite_log_likelihood() -> None:
    """``DDSSM_base.log_prob`` composes encoder + decoder + diffusion.log_prob + VHP.

    Cycle-6 tracer.  This is the integration test: the underlying
    primitives are correctness-tested individually (prob-flow, IWAE
    assembly, VHP IS estimator).  Here we verify the composition wires
    them together into a valid ``(B,)`` log-likelihood on a tiny
    stage-2 diffusion model with VHP-via-AuxPosterior.
    """
    torch.manual_seed(0)
    model = _make_stage2_model()
    model.eval()

    B, T = 2, 5
    observed_data = torch.randn(B, DATA_DIM, T)
    observation_mask = torch.ones(B, DATA_DIM, T)
    timepoints = torch.arange(T).expand(B, T).clone().long()

    log_p = model.log_prob(
        observed_data=observed_data,
        observation_mask=observation_mask,
        timepoints=timepoints,
    )

    assert log_p.shape == (B,)
    assert torch.all(torch.isfinite(log_p))


def test_log_prob_at_k1_equals_single_trajectory_iwae_weight() -> None:
    """At K=1 the IWAE reduces to the single-trajectory weight ``log p − log q``.

    Self-consistency check on the assembly + VHP-init wiring: with one
    trajectory, ``logmeanexp`` is the identity, so ``model.log_prob`` must
    equal the per-component log-densities (init + transition + decoder −
    encoder log q) recomputed from the model's own modules on the *same*
    encoded trajectory.  No ground-truth mocking — this catches a wrong
    sign, a dropped term, or a mis-wired init in the composition.
    """
    torch.manual_seed(0)
    model = _make_stage2_model()
    model.eval()

    B, T = 2, 5
    j = J
    observed_data = torch.randn(B, DATA_DIM, T)
    observation_mask = torch.ones(B, DATA_DIM, T)
    timepoints = torch.arange(T).expand(B, T).clone().long()

    from ddssm.nn.net_utils import time_embedding

    # Encode once; reuse the SAME trajectory for both the method and the
    # manual reassembly (seed the model's internal sampling identically).
    torch.manual_seed(123)
    log_p_method = model.log_prob(
        observed_data=observed_data,
        observation_mask=observation_mask,
        timepoints=timepoints,
        K=1,
    )

    torch.manual_seed(123)
    time_embed = time_embedding(
        timepoints, model.emb_time_dim, device=observed_data.device
    )
    zs, logq_paths, _ = model._encode_latents(
        observed_data=observed_data,
        time_embed=time_embed,
        observation_mask=observation_mask,
        covariates=None,
        static_embed=None,
    )
    zs = zs[:, :1]
    logq_paths = logq_paths[:, :1]

    log_q_z = logq_paths.sum(dim=-1)  # (B, 1)

    log_p_dec = torch.zeros(B, 1)
    for t in range(T):
        z_hist = zs[:, 0, :, : t + 1]
        if z_hist.shape[-1] > j:
            z_hist = z_hist[..., -j:]
        logp_t, _, _, _ = model.decoder.log_likelihood(
            x_t=observed_data[:, :, t],
            z_hist=z_hist,
            time_embed=time_embed,
            time_idx=torch.full((B,), t, dtype=torch.long),
            observation_mask_t=observation_mask[:, :, t],
        )
        log_p_dec[:, 0] = log_p_dec[:, 0] + logp_t

    transition = model._active_transition()
    log_p_trans = torch.zeros(B, 1)
    for t in range(j, T):
        sigma_d2 = model.sigma_data.read(t + 1).expand(B).to(dtype=zs.dtype)
        ctx = {
            "hist_time_emb": time_embed[:, t - j : t, :],
            "target_time_emb": time_embed[:, t : t + 1, :],
        }
        logp_t = transition.log_prob(
            z=zs[:, 0, :, t],
            z_hist=zs[:, 0, :, t - j : t],
            ctx=ctx,
            sigma_d2=sigma_d2,
        )
        log_p_trans[:, 0] = log_p_trans[:, 0] + logp_t

    log_p_init = transition.log_prob_init(
        zs=zs,
        aux_posterior=model.aux_posterior,
        time_embed=time_embed,
        sigma_data=model.sigma_data,
    )

    expected = (log_p_init + log_p_trans + log_p_dec - log_q_z).squeeze(-1)
    assert torch.allclose(log_p_method, expected, atol=1e-4, rtol=1e-4)

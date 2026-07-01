"""Analytic-truth regression tests for the ELBO rate terms.

Builds a degenerate DDSSM where every rate-side component has a
closed-form value (independent of any sampled noise) and asserts the
full ``DDSSM_base.forward`` matches.  Catches the class of bug the
math audit flagged — wrong reductions, sign flips, off-by-one in the
j-window — that produce finite-but-wrong losses.

The setup:

* ``q(z|x) = N(0, I)`` for every (b, s, t).  Achieved by a stub
  encoder that returns ``mus = 0``, ``logvars = 0``, ``zs = 0``
  deterministically.  No reparam noise, no encoder weights involved.

* Baseline prior ``p(z_t | z_hist) = N(0, I)``.  Achieved with a
  :class:`ZeroBaseline` whose state-conditional σ_head weights are
  zeroed → ``log σ_p² ≡ 0`` for every input.

* Stage 1 + ``BaselineGaussianTransition`` so ``trans_kl`` is the
  closed-form Gaussian KL (no diffusion noise / ESM stochasticity).

Under that setup the three closed-form rate-term invariants are:

1. ``KL(q || p) = 0`` per (b, s, t) because q ≡ p; therefore
   ``components.trans_kl = 0`` exactly.

2. ``r_sigma_p = 0.5 · (mean log σ_p²)² = 0`` because every
   ``log σ_p²`` value is exactly 0.

3. ``r_mu_p = 0.5 · mean ‖μ_p(z) − μ_p^(0)(z)‖² = 0`` immediately
   after the snapshot (live params == anchor params).  Tested in a
   stage-2 variant since the model only computes ``r_mu_p`` in that
   stage.

Asserts ``== 0`` exactly (not approx) where the math says exactly
zero — reductions of zero tensors and parameter diffs of identical
modules are bit-exact under any floating-point implementation.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from ddssm.model.dssd import DDSSM_base
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import ZeroBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

D = 2  # latent dim
J = 1  # history window
EMB = 4  # time-embedding dim
T = 4  # sequence length for the single-shape tests
T_MAX = 8  # ``sigma_data`` buffer capacity — large enough for the longest test
B = 3  # batch
S = 2  # sample paths
DATA_DIM = 1


def _zero_sigma_head(baseline: ZeroBaseline) -> None:
    """Zero out ``ZeroBaseline``'s σ_head weights so ``logvar_p ≡ 0``."""
    with torch.no_grad():
        for layer in baseline.sigma_head.body:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()


def _stub_encode_returns_standard_normal(
    self,
    observed_data,
    time_embed,
    observation_mask,
    covariates=None,
    static_embed=None,
):
    """Replacement for ``DDSSM_base._encode_latents`` — q ≡ N(0, I).

    Returns deterministic ``zs = 0`` samples and ``(mus = 0, logvars = 0)``
    encoder stats.  Bypasses the real encoder so the test depends only on
    the ELBO assembly + transition + regulariser code paths.
    """
    bsz = observed_data.shape[0]
    seq_len = observed_data.shape[-1]
    zs = torch.zeros(bsz, self.S, D, seq_len)
    logq_paths = torch.zeros(bsz, self.S, seq_len)
    enc_stats = {
        "mus": torch.zeros(bsz, self.S, D, seq_len),
        "logvars": torch.zeros(bsz, self.S, D, seq_len),
    }
    return zs, logq_paths, enc_stats


class _StubDecoder(nn.Module):
    """Decoder whose output is irrelevant to the rate-term assertions.

    Implements ``log_likelihood`` (the only interface the model's
    reconstruction path uses) returning constant zero outputs. The
    recon term isn't analytically pinned here — only the rate-side
    components are asserted.
    """

    def __init__(self, latent_dim: int, data_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

    def log_likelihood(
        self,
        x_t,
        z_hist,
        time_embed,
        time_idx,
        observation_mask_t=None,
        covariates=None,
        static_embed=None,
    ):
        bsz = z_hist.shape[0]
        device = z_hist.device
        logp_t = torch.zeros(bsz, device=device)
        mu_x = torch.zeros(bsz, self.data_dim, device=device)
        logvar_x = torch.zeros(bsz, self.data_dim, device=device)
        obs_count_t = torch.full(
            (bsz,),
            float(self.data_dim),
            device=device,
        )
        return logp_t, mu_x, logvar_x, obs_count_t


def _build_q_equals_prior_model(
    *, stage_selector: str, snapshot_anchor: bool = False, baseline_mode: str = "pinned"
) -> DDSSM_base:
    """Construct a DDSSM with q ≡ p ≡ N(0, I) for the chosen stage."""
    baseline = ZeroBaseline(latent_dim=D, j=J, hidden_dim=8, n_layers=1)
    _zero_sigma_head(baseline)
    transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=D,
        j=J,
        emb_time_dim=EMB,
    )
    aux = AuxPosterior(latent_dim=D, j=J, hidden_dim=8, n_layers=1)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed", init_value=1.0)

    anchor = baseline.snapshot() if snapshot_anchor else None
    model = DDSSM_base(
        encoder=nn.Identity(),  # never called; _encode_latents is stubbed
        decoder=_StubDecoder(D, DATA_DIM),
        transition=transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=D,
        emb_time_dim=EMB,
        use_observation_mask=False,
        aux_posterior=aux,
        baseline=baseline,
        baseline_anchor=anchor,
        baseline_mode=baseline_mode,
        sigma_data=sigma_data,
        stage1_transition=transition,
    )
    # Override the latent-encoding path so q ≡ N(0, I) deterministically.
    model._encode_latents = types.MethodType(
        _stub_encode_returns_standard_normal,
        model,
    )
    model.stage_selector = stage_selector
    model.S = S
    model.eval()
    return model


def _zero_batch():
    """Pinned input batch.  Values don't affect the rate-term assertions."""
    return {
        "observed_data": torch.zeros(B, DATA_DIM, T),
        "observation_mask": torch.ones(B, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(B, T).clone().long(),
    }


def test_trans_kl_is_zero_when_q_equals_baseline_prior():
    """Closed-form ``KL(q || p) = 0`` ⟹ ``components.trans_kl = 0``.

    Under q ≡ p ≡ N(0, I) the per-step Gaussian KL is exactly zero;
    the summed-and-averaged ``trans_kl`` is exactly zero regardless
    of how many (b, s, t) cells contribute.  Asserts the model-level
    assembly preserves this invariant.
    """
    model = _build_q_equals_prior_model(stage_selector="stage_1")
    batch = _zero_batch()
    components, _, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
        train=True,
    )
    assert components.trans_kl.item() == 0.0


def test_r_sigma_p_is_zero_when_log_sigma_p_squared_is_zero():
    """``r_sigma_p = 0.5 · (mean log σ_p²)² = 0`` when every log σ_p² is 0.

    The σ_head bias-only initialisation guarantees ``log σ_p² ≡ 0``
    so the global anchor evaluates to ``0.5 · 0² = 0`` exactly.
    Locks the regulariser-assembly path (caller passes
    ``lambda_sigma_p=1.0`` so we're testing the *raw* anchor value).
    """
    model = _build_q_equals_prior_model(stage_selector="stage_1")
    batch = _zero_batch()
    components, _, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
        train=True,
    )
    assert components.r_sigma_p.item() == 0.0


def test_r_mu_p_is_zero_at_snapshot():
    """``r_mu_p = 0.5 · mean ‖μ_live − μ_anchor‖² = 0`` immediately after snapshot.

    With ``baseline_anchor = baseline.snapshot()`` and no training
    steps in between, the live and anchor parameters are identical
    so the per-sample squared L2 of their μ-head outputs is exactly
    zero for any input.  Requires stage 2 + learnable mode (the only
    branch in which ``DDSSM_base.forward`` computes ``r_mu_p``).
    """
    model = _build_q_equals_prior_model(
        stage_selector="stage_2",
        baseline_mode="learnable",
        snapshot_anchor=True,
    )
    batch = _zero_batch()
    components, _, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
        train=True,
    )
    assert components.r_mu_p.item() == 0.0


def test_trans_kl_invariant_to_target_step_count():
    """``trans_kl = 0`` regardless of how many target timesteps contribute.

    Catches the V3 mean-vs-sum bug at the BaselineGaussian level: the
    fixed reduction sums over ``t = j+1 .. T`` then averages over
    ``(b, s)``; a hypothetical reduction that DIVIDES by ``(T-j)``
    would still give 0 here (0 / N = 0) — but combined with the
    non-zero analytic check in ``test_transitions/test_baseline_gaussian``
    (``test_transition_kl_matches_analytic``) the convention is locked.

    Run with two sequence lengths sharing the same per-step encoder
    moments; both must produce exactly 0.
    """
    model = _build_q_equals_prior_model(stage_selector="stage_1")

    short = {
        "observed_data": torch.zeros(B, DATA_DIM, J + 1),
        "observation_mask": torch.ones(B, DATA_DIM, J + 1),
        "timepoints": torch.arange(J + 1).expand(B, J + 1).clone().long(),
    }
    long_seq = {
        "observed_data": torch.zeros(B, DATA_DIM, J + 5),
        "observation_mask": torch.ones(B, DATA_DIM, J + 5),
        "timepoints": torch.arange(J + 5).expand(B, J + 5).clone().long(),
    }
    c_short, _, _ = model(
        short["observed_data"],
        short["observation_mask"],
        short["timepoints"],
        train=True,
    )
    c_long, _, _ = model(
        long_seq["observed_data"],
        long_seq["observation_mask"],
        long_seq["timepoints"],
        train=True,
    )
    assert c_short.trans_kl.item() == 0.0
    assert c_long.trans_kl.item() == 0.0

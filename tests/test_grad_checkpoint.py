"""Gradient-checkpoint equivalence.

The flag-gated checkpointing added for the gluonts campaign (score-net call in
``DiffusionTransition._esm_chunk_loss`` and the future-summary in
``GaussianEncoder.sample_paths``) must be a pure memory/compute trade: identical
loss and identical gradients vs the non-checkpointed path. Both helpers are
deterministic given inputs (the stochastic sampling lives *outside* the
checkpointed regions), so on CPU the match should be exact to float tolerance.
"""

from __future__ import annotations

from functools import partial

import torch
import pytest

from ddssm.nn.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.futsum import TransformerFutureSummary
from ddssm.model.encoder import GaussianEncoder
from ddssm.model.centering.baselines import MLPBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)

B, S, D, T, J, EMB = 2, 2, 2, 5, 1, 8
T_MAX = 10


def _close(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
        f"{name}: max abs diff {(a - b).abs().max().item():.3e}"
    )


def _make_diffusion(time_chunk_size: int | None = None) -> DiffusionTransition:
    torch.manual_seed(7)
    baseline = MLPBaseline(latent_dim=D, j=J, hidden_dim=4, n_layers=1)
    return DiffusionTransition(
        baseline=baseline,
        latent_dim=D,
        j=J,
        emb_time_dim=EMB,
        T_max=T_MAX,
        unet=partial(
            CSDIUnet,
            channels=16,
            n_layers=2,
            embedding_dim=16,
            residual_block=DiffResidualBlockConfig(
                # dropout=0.0 mirrors the gluonts score-net: the checkpoint uses
                # preserve_rng_state=False, which is only correct for a
                # deterministic forward.
                feature=FeatureMixerConfig(type="transformer", nheads=2, n_layers=1, dropout=0.0)
            ),
        ),
        schedule=DiffusionScheduleConfig(
            S_k=1, k_chunk=1, num_steps=20, time_chunk_size=time_chunk_size
        ),
    )


# time_chunk_size>1 batches several timesteps per checkpointed score-net call;
# the checkpoint recompute must stay grad-exact at any chunk size.
@pytest.mark.parametrize("time_chunk_size", [None, 3])
def test_diffusion_grad_checkpoint_equivalence(time_chunk_size: int | None) -> None:
    tr_off = _make_diffusion(time_chunk_size)
    tr_on = _make_diffusion(time_chunk_size)
    tr_on.load_state_dict(tr_off.state_dict())
    tr_off.train()
    tr_on.train()
    tr_off.grad_checkpoint = False
    tr_on.grad_checkpoint = True

    def run(tr: DiffusionTransition):
        torch.manual_seed(0)
        zs = torch.randn(B, S, D, T, requires_grad=True)
        mus = (0.3 * torch.randn(B, S, D, T)).requires_grad_(True)
        logvars = (-1.0 + 0.2 * torch.randn(B, S, D, T)).requires_grad_(True)
        time_embed = torch.randn(B, T, EMB)
        logq = torch.randn(B, S, T)
        sd = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
        torch.manual_seed(123)  # identical multinomial/randn sampling
        kl = tr.transition_kl(
            enc_stats={"mus": mus, "logvars": logvars},
            zs=zs,
            logq_paths=logq,
            time_embed=time_embed,
            sigma_data=sd,
        )["kl"]
        kl.backward()
        pg = {
            n: p.grad.detach().clone()
            for n, p in tr.named_parameters()
            if p.grad is not None
        }
        return kl.detach(), zs.grad.clone(), mus.grad.clone(), pg

    kl0, zg0, mg0, pg0 = run(tr_off)
    kl1, zg1, mg1, pg1 = run(tr_on)

    _close(kl0, kl1, "kl")
    _close(zg0, zg1, "zs.grad")
    _close(mg0, mg1, "mus.grad")
    assert pg0.keys() == pg1.keys()
    assert len(pg0) > 0  # the score-net params actually received gradients
    for n in pg0:
        _close(pg0[n], pg1[n], f"param {n}")


def test_recon_chunk_invariance_and_grad_checkpoint() -> None:
    """The vectorized decoder recon (deterministic gluonts decoder, dropout=0) is
    invariant to the time-chunk size, and its checkpoint is grad-exact. Run on
    CPU so there's no atomic-reduction noise to confound the comparison."""
    from experiments.gluonts_forecast.model import build_gluonts_model

    torch.manual_seed(0)
    m = build_gluonts_model(
        data_dim=5, latent_dim=4, T_max=8, time_chunk=2, grad_checkpoint=True
    )
    m.train()
    Bb, Ss, dd, Tt, Dd = 3, 1, 4, 8, 5
    torch.manual_seed(1)
    obs = torch.randn(Bb, Dd, Tt)
    mask = torch.ones(Bb, Dd, Tt)
    te = torch.zeros(Bb, Tt, 0)  # emb_time_dim=0 (time-conditioning off)
    zs = torch.randn(Bb, Ss, dd, Tt)

    def recon(chunk: int, ckpt: bool):
        m._recon_time_chunk = chunk
        m._recon_grad_checkpoint = ckpt
        z = zs.clone().requires_grad_(True)
        L, _ = m._reconstruction_loss(obs, te, z, mask)
        m.zero_grad(set_to_none=True)
        L.backward()
        pg = {
            n: p.grad.detach().clone()
            for n, p in m.decoder.named_parameters()
            if p.grad is not None
        }
        return L.detach(), z.grad.clone(), pg

    L1, zg1, _ = recon(1, False)
    Lc, zgc, _ = recon(3, False)  # chunk-invariance: 1 vs 3
    Lk, zgk, pgk = recon(3, True)  # checkpoint exactness at chunk=3

    _close(L1, Lc, "recon loss chunk 1 vs 3")
    _close(zg1, zgc, "recon zs.grad chunk 1 vs 3")
    _close(L1, Lk, "recon loss ckpt")
    _close(zg1, zgk, "recon zs.grad ckpt")
    assert len(pgk) > 0
    _, _, pg1 = recon(1, False)
    for n in pg1:
        _close(pg1[n], pgk[n], f"recon ckpt param {n}")


def test_encoder_grad_checkpoint_equivalence() -> None:
    def build() -> GaussianEncoder:
        torch.manual_seed(11)
        return GaussianEncoder(
            data_dim=D,
            latent_dim=4,
            j=J,
            emb_time_dim=EMB,
            use_mask=False,
            hidden_dim=16,
            fut_summary=partial(
                TransformerFutureSummary,
                summary_dim=16,
                nheads=2,
                transformer_layers=1,
            ),
        )

    enc_off = build()
    enc_on = build()
    enc_on.load_state_dict(enc_off.state_dict())
    enc_off.train()
    enc_on.train()
    enc_off.grad_checkpoint = False
    enc_on.grad_checkpoint = True

    def run(enc: GaussianEncoder):
        torch.manual_seed(0)
        x = torch.randn(B, D, T, requires_grad=True)
        te = torch.randn(B, T, EMB)
        torch.manual_seed(99)  # identical per-step z sampling
        zs, logqs, _ = enc.sample_paths(x, te, S=S)
        loss = zs.pow(2).sum() + logqs.sum()
        loss.backward()
        pg = {
            n: p.grad.detach().clone()
            for n, p in enc.named_parameters()
            if p.grad is not None
        }
        return zs.detach(), logqs.detach(), x.grad.clone(), pg

    z0, l0, xg0, pg0 = run(enc_off)
    z1, l1, xg1, pg1 = run(enc_on)

    _close(z0, z1, "zs")
    _close(l0, l1, "logqs")
    _close(xg0, xg1, "x.grad")
    assert pg0.keys() == pg1.keys()
    assert len(pg0) > 0
    for n in pg0:
        _close(pg0[n], pg1[n], f"param {n}")

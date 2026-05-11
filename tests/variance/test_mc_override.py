from __future__ import annotations

import torch

from ddssm.diffnets import CSDIUnet, DiffResidualBlockConfig, FeatureMixerConfig
from ddssm.transitions.diffusion_v2 import DiffusionV2ScheduleConfig
from ddssm.transitions.diffusion_v2 import DiffusionV2Transition
from functools import partial


def make_transition(schedule: DiffusionV2ScheduleConfig) -> DiffusionV2Transition:
    tiny_unet = partial(
        CSDIUnet,
        channels=8,
        n_layers=1,
        embedding_dim=8,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=4, n_layers=1)
        ),
    )
    return DiffusionV2Transition(
        latent_dim=2,
        j=2,
        emb_time_dim=8,
        unet=tiny_unet,
        schedule=schedule,
    )


def _override_tensors(zs: torch.Tensor, transition) -> dict[str, torch.Tensor]:
    b, s, d, _ = zs.shape
    n = b * s
    sk = int(transition.S_k)
    k_idx = torch.zeros(n, sk, dtype=torch.long)
    eps = torch.zeros(n, d, sk)
    return {"k_idx": k_idx, "eps": eps}


def _fixed_batch():
    torch.manual_seed(123)
    b, s, d, t = 4, 2, 2, 8
    zs = torch.randn(b, s, d, t)
    enc_stats = {
        "mus": 0.5 * torch.randn(b, s, d, t),
        "logvars": -1.0 + 0.3 * torch.randn(b, s, d, t),
    }
    time_embed = torch.randn(b, t, 8)
    logq_paths = torch.randn(b, s, t)
    return zs, enc_stats, time_embed, logq_paths


def test_mc_override_deterministic_with_fixed_inputs() -> None:
    zs, enc_stats, time_embed, logq_paths = _fixed_batch()
    cfg = DiffusionV2ScheduleConfig(S_k=2, k_chunk=1, num_steps=8, objective="esm")
    trans = make_transition(schedule=cfg)
    override = _override_tensors(zs, trans)
    out1 = trans.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        mc_override=override,
    )
    out2 = trans.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        mc_override=override,
    )
    assert torch.allclose(out1["L_p"], out2["L_p"])


def test_mc_override_objective_matches_dsm_transition() -> None:
    zs, enc_stats, time_embed, logq_paths = _fixed_batch()
    cfg_esm = DiffusionV2ScheduleConfig(S_k=1, k_chunk=1, num_steps=8, objective="esm")
    cfg_dsm = DiffusionV2ScheduleConfig(S_k=1, k_chunk=1, num_steps=8, objective="dsm")
    trans_esm = make_transition(schedule=cfg_esm)
    trans_dsm = make_transition(schedule=cfg_dsm)
    trans_dsm.load_state_dict(trans_esm.state_dict())
    override = _override_tensors(zs, trans_esm) | {"objective": "dsm"}
    out_override = trans_esm.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        mc_override=override,
    )
    out_dsm = trans_dsm.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        mc_override=_override_tensors(zs, trans_dsm),
    )
    assert torch.allclose(out_override["L_p"], out_dsm["L_p"], atol=1e-6, rtol=1e-6)


def test_return_per_sample_sums_to_scalar_lp() -> None:
    zs, enc_stats, time_embed, logq_paths = _fixed_batch()
    cfg = DiffusionV2ScheduleConfig(S_k=1, k_chunk=1, num_steps=8, objective="esm")
    trans = make_transition(schedule=cfg)
    override = _override_tensors(zs, trans)
    out = trans.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
        mc_override=override,
        return_per_sample=True,
    )
    assert "L_p_per_sample" in out
    assert torch.allclose(out["L_p_per_sample"].sum(), out["kl"], atol=1e-6, rtol=1e-6)

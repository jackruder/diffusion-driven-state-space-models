"""Locks the gluonts_forecast architecture decisions (from the design drilling).

These are the choices that the synthetic init_centering family does NOT share, so
a regression in the shared factory wouldn't be caught elsewhere: the single
``2×latent`` width rule, the transformer future-summary, persistence/pinned
centering, gradient checkpointing, and a deterministic (dropout-free) score-net.
"""

from __future__ import annotations

import torch

from ddssm.nn.futsum import TransformerFutureSummary
from ddssm.model.centering.baselines import PersistenceBaseline
from experiments.gluonts_forecast.model import build_gluonts_model
from experiments.gluonts_forecast.datasets import GLUONTS_BY_LABEL, GLUONTS_DATASETS


def test_width_rule_is_two_times_latent() -> None:
    latent = 64
    m = build_gluonts_model(data_dim=137, latent_dim=latent, T_max=192)
    assert m.encoder.summary_dim == 2 * latent
    assert m.encoder.hidden_dim == 2 * latent
    assert m.decoder.hidden_dim == 2 * latent


def test_transformer_future_summary_and_additive_frame() -> None:
    m = build_gluonts_model(data_dim=137, latent_dim=32, T_max=192)
    assert isinstance(m.encoder.fut_sum_module, TransformerFutureSummary)
    assert m.encoder.mu_mode == "additive"


def test_persistence_pinned_and_checkpointing() -> None:
    m = build_gluonts_model(data_dim=137, latent_dim=32, T_max=192)
    assert isinstance(m.baseline, PersistenceBaseline)
    assert m.baseline_mode == "pinned"
    assert m.encoder.grad_checkpoint is True
    assert m.transition.grad_checkpoint is True
    # Decoder recon is vectorized over time chunks + checkpointed (default
    # time_chunk=16), and the decoder must be deterministic (dropout=0) for that.
    assert m._recon_time_chunk == 16
    assert m._recon_grad_checkpoint is True
    dec_drops = [
        mod.p for mod in m.decoder.modules() if isinstance(mod, torch.nn.Dropout)
    ]
    assert dec_drops and all(p == 0.0 for p in dec_drops)


def test_score_net_csdi_dims_and_dropout_free() -> None:
    m = build_gluonts_model(data_dim=137, latent_dim=32, T_max=192)
    unet = m.transition.diffmodel
    assert unet.channels == 64
    assert unet.n_layers == 4
    assert m.transition.num_steps == 128
    # Score-net feature transformer must be deterministic (dropout=0) so the
    # gradient checkpoint's preserve_rng_state=False recompute stays exact.
    drops = [mod.p for mod in unet.modules() if isinstance(mod, torch.nn.Dropout)]
    assert drops and all(p == 0.0 for p in drops), f"score-net dropout(s)={drops}"


def test_latent_grid_keeps_summary_heads_divisible() -> None:
    # 2×latent (the summary width) must be divisible by the 8 attention heads.
    for latent in (16, 32, 64, 128, 256, 512):
        assert (2 * latent) % 8 == 0


def test_dataset_axis_dims() -> None:
    assert len(GLUONTS_DATASETS) == 5
    solar = GLUONTS_BY_LABEL["solar"]
    assert solar.data_dim == 137 and solar.T_max == 192
    assert GLUONTS_BY_LABEL["wiki"].data_dim == 2000
    assert GLUONTS_BY_LABEL["electricity"].data_dim == 370

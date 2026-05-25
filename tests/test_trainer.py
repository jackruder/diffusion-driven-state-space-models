# tests/test_trainer.py
from functools import partial

import torch
import pytest
from hydra_zen import builds, instantiate

from ddssm.dssd import DDSSM_base
from ddssm.train import DDSSMTrainer
from ddssm.decoder import GaussianDecoder
from ddssm.encoder import GaussianEncoder, GaussianInitPrior
from ddssm.fusions import ConcatLinearFusion
from ddssm.combiners import CompoundCombiner
from ddssm.dist_heads import GaussianDistHead
from ddssm.aggregators import IdentityAggregator
from ddssm.transitions.transitions import GaussianTransition

DDSSMTrainerConf = builds(DDSSMTrainer, populate_full_signature=True)
from types import SimpleNamespace

from torch.utils.data import Dataset, DataLoader

from ddssm.futsum import GRUFutureSummary
from ddssm.diffnets import ContextProducer, FeatureMixerConfig, ResidualBlockConfig
from ddssm.gaussians import GaussianHead

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4

_CTX = partial(
    ContextProducer,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_GH = GaussianHead  # zen_partial-style: parents call _GH(in_features=..., out_features=...)
_FS = partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1)


def make_small_model():
    # j=1 → identity aggregator (no z-history mixing) + concat-linear fusion
    combiner = partial(
        CompoundCombiner,
        aggregator=partial(IdentityAggregator),
        fusion=partial(ConcatLinearFusion),
    )
    enc = GaussianEncoder(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS,
        combiner=combiner,
        dist_head=partial(GaussianDistHead),
        fut_summary=_FS,
    )
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM, data_dim=DATA_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    zinit = GaussianInitPrior(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, aux_context=_CTX, gaussian_head=_GH, aux_posterior_head=_GH,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    hp = SimpleNamespace(
        S=1, ema_decay=0.999, weight_decay=1e-2, batch_size=1, grad_accum_steps=1,
        t_chunk=4, clip_grad_norm=None, lambda_schedule="none", lambda_start=0.001,
        lambda_end=1.0, lambda_warmup_steps=1, enc_lr=1e-3, dec_lr=1e-3,
        zinit_lr=1e-3, trans_lr=1e-3, logvar_min=-7.0, logvar_max=7.0,
    )
    return DDSSM_base(
        encoder=enc, decoder=dec, z_init=zinit, transition=trans,
        j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM, emb_time_dim=EMB_TIME,
        hyperparams=hp,
    )


@pytest.fixture
def small_model():
    return make_small_model()


def test_trainer_conf_builds_instantiates(small_model, tmp_path):
    trainer = instantiate(
        DDSSMTrainerConf(
            model=small_model,
            device=torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "runs"),
            quiet=True,
        )
    )
    assert isinstance(trainer, DDSSMTrainer)


class _SyntheticBatchDataset(Dataset):
    """One-batch fixture for the CSV logging test."""

    def __init__(self, B: int = 1, T: int = 4):
        self.B = B
        self.T = T

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        return {
            "observed_data": torch.randn(DATA_DIM, self.T),
            "observation_mask": torch.ones(DATA_DIM, self.T),
            "timepoints": torch.arange(self.T, dtype=torch.float32),
        }


def test_trainer_logs_time_elapsed_s_to_csv(small_model, tmp_path):
    """``_log_train_step`` writes ``time/elapsed_s`` as a CSV column.

    This is the Phase-A trainer-side extension that feeds the
    ``wallclock_to_target`` headline metric.
    """
    import csv as _csv

    csv_path = tmp_path / "metrics.csv"
    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        csv_log_path=str(csv_path),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=2,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    assert "time/elapsed_s" in fieldnames, (
        f"missing time/elapsed_s column; got {fieldnames}"
    )
    # Each row's elapsed_s should be finite and non-negative.
    elapsed_values = [float(r["time/elapsed_s"]) for r in rows if r.get("time/elapsed_s")]
    assert elapsed_values, "no time/elapsed_s rows logged"
    for v in elapsed_values:
        assert v >= 0.0
    # And monotonically non-decreasing across steps (real-time progression).
    assert all(b >= a - 1e-6 for a, b in zip(elapsed_values, elapsed_values[1:]))


# ---------------------------------------------------------------------------
# Phase B: ``StageTrainableConf.baseline`` + ELBO-plateau early-stop.
# ---------------------------------------------------------------------------


def _make_model_with_baseline():
    """A tiny DDSSM_base with the model-v2 baseline + aux_posterior slots populated."""
    from ddssm.aux_posterior import AuxPosterior
    from ddssm.centering.baselines import MLPBaseline
    from ddssm.centering.sigma_data import SigmaDataBuffer
    from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition

    base = make_small_model()
    baseline = MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=1)
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=1)
    base.baseline = baseline
    base.aux_posterior = aux
    base.sigma_data = SigmaDataBuffer(T_max=8, tracking_mode="per_t")
    base.stage1_transition = BaselineGaussianTransition(
        baseline=baseline, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
    )
    return base


def test_set_trainable_baseline_field_flips_requires_grad(tmp_path):
    """``StageTrainableConf.baseline=False`` flips ``model.baseline.parameters().requires_grad``."""
    from ddssm.stages import StageTrainableConf

    model = _make_model_with_baseline()
    trainer = DDSSMTrainer(
        model=model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    # Sanity: baseline params start trainable.
    assert all(p.requires_grad for p in model.baseline.parameters())

    trainer._set_trainable(StageTrainableConf(baseline=False))
    assert all(not p.requires_grad for p in model.baseline.parameters())

    trainer._set_trainable(StageTrainableConf(baseline=True))
    assert all(p.requires_grad for p in model.baseline.parameters())


def test_rebuild_optimizer_picks_up_baseline_params(tmp_path):
    """After ``_rebuild_optimizer`` the AdamW groups include baseline params."""
    from ddssm.stages import StageLrsConf

    model = _make_model_with_baseline()
    trainer = DDSSMTrainer(
        model=model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    trainer._rebuild_optimizer(StageLrsConf())
    baseline_param_ids = {id(p) for p in model.baseline.parameters()}
    grouped_param_ids = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert baseline_param_ids & grouped_param_ids, (
        "baseline parameters missing from the rebuilt optimizer"
    )


def test_elbo_plateau_early_stop_triggers(small_model, tmp_path):
    """When the loss is flat, the early-stop spec terminates ``fit`` early."""
    from ddssm.stages import EarlyStopSpec

    trainer = DDSSMTrainer(
        model=small_model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    # ``min_improvement=10.0`` is impossible to meet ⇒ early-stop fires on the
    # first comparison after the warmup window.
    spec = EarlyStopSpec(
        enabled=True, window=4, min_improvement=10.0, warmup_steps=2,
    )
    final_step = trainer.fit(
        train_loader=loader, val_loader=None, total_steps=200,
        validate_every=0, log_every=1, checkpoint_every=None, amp=False,
        early_stop=spec,
    )
    assert final_step < 200, (
        f"early-stop did not trigger; ran the full budget (final_step={final_step})"
    )


def test_elbo_plateau_disabled_runs_full_budget(small_model, tmp_path):
    """``early_stop=None`` leaves the loop running until ``total_steps``."""
    trainer = DDSSMTrainer(
        model=small_model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    final_step = trainer.fit(
        train_loader=loader, val_loader=None, total_steps=6,
        validate_every=0, log_every=1, checkpoint_every=None, amp=False,
        early_stop=None,
    )
    assert final_step == 6

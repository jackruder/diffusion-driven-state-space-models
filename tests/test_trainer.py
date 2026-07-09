# tests/test_trainer.py
from functools import partial

import torch
import pytest
from hydra_zen import builds, instantiate

from ddssm.model.dssd import DDSSM_base
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.combiners import CompoundCombiner
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.aggregators import IdentityAggregator
from ddssm.training.train import DDSSMTrainer
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.transitions.transitions import GaussianTransition

DDSSMTrainerConf = builds(DDSSMTrainer, populate_full_signature=True)

from torch.utils.data import Dataset, DataLoader

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.nn.diffnets import ContextProducer, FeatureMixerConfig, ResidualBlockConfig
from ddssm.nn.gaussians import GaussianHead

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2

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
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=combiner,
        dist_head=partial(GaussianDistHead),
        fut_summary=_FS,
    )
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=CHANNELS, n_layers=1)
    return DDSSM_base(
        encoder=enc,
        decoder=dec,
        transition=trans,
        aux_posterior=aux,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
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
    elapsed_values = [
        float(r["time/elapsed_s"]) for r in rows if r.get("time/elapsed_s")
    ]
    assert elapsed_values, "no time/elapsed_s rows logged"
    for v in elapsed_values:
        assert v >= 0.0
    # And monotonically non-decreasing across steps (real-time progression).
    assert all(b >= a - 1e-6 for a, b in zip(elapsed_values, elapsed_values[1:]))


def test_fit_writes_validation_rows_to_csv(small_model, tmp_path):
    """Validation metrics reach metrics.csv (not just TensorBoard).

    Regression guard: CSVLogger.on_epoch used to be a no-op, so val/* was
    invisible to the CSV that the objective reader and triage tools read.
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
        val_loader=loader,
        total_steps=2,
        validate_every=1,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]
    assert val_rows, "no val rows written to metrics.csv"
    assert all(r["loss/total"] not in ("", None) for r in val_rows)


def test_log_train_step_records_optimized_loss_not_unweighted_elbo(
    small_model, tmp_path
):
    """``loss/total`` in the log is the optimized objective, not the raw ELBO.

    Regression guard: ``_log_train_step`` built ``log_values`` with the
    weighted ``accum_loss`` first and then ``**accum_metrics``, whose own
    (unweighted) ``loss/total`` silently overwrote it. The logged curve then
    disagreed with the early-stop window (which uses ``accum_loss``). The
    unweighted ELBO must survive under ``loss/total_unweighted``.
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
    trainer.global_step = 0
    # accum_metrics carries the model's unweighted ELBO under loss/total;
    # accum_loss is the (different) optimized objective the trainer minimizes.
    accum_metrics = {
        "loss/total": torch.tensor(99.0),
        "loss/distortion/rec": torch.tensor(5.0),
    }
    trainer._log_train_step(
        step=1,
        log_every=1,
        accum_loss=10.0,
        accum_metrics=accum_metrics,
        accum_weight=1,
        device=torch.device("cpu"),
    )
    with open(csv_path, newline="") as f:
        row = next(_csv.DictReader(f))
    assert float(row["loss/total"]) == pytest.approx(10.0 / trainer.grad_accum_steps)
    assert float(row["loss/total_unweighted"]) == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Optimizer-rebuild + ELBO-plateau early-stop coverage.
# ---------------------------------------------------------------------------


# The ``_make_model_with_baseline`` fixture and the tests below relied on
# parametric baselines (MLPBaseline) + ``BaselineGaussianTransition`` +
# ``StageTrainableConf.baseline`` — all removed when the parametric
# baselines were retired. The remaining trainable/rebuild-optimizer coverage
# lives in tests/test_training/test_param_split.py.


def test_rebuild_optimizer_keeps_frozen_params_in_groups(tmp_path):
    """Params frozen at rebuild time still land in the optimizer groups.

    Regression guard: ``param_groups_for_adamw`` used to filter on
    ``requires_grad`` at build time, so a module frozen in stage N but
    unfrozen in stage N+1 could be silently absent from the optimizer
    (when the LRs-unchanged path skips the rebuild) and never train.
    Group membership must be mask-independent; ``requires_grad`` alone
    suppresses updates (AdamW skips grad-None params).
    """
    from ddssm.training.stages import StageLrsConf, StageTrainableConf

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    # Freeze the encoder BEFORE the rebuild — the old filter dropped it here.
    trainer._set_trainable(StageTrainableConf(encoder=False))
    trainer._rebuild_optimizer(StageLrsConf())

    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    grouped_param_ids = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert encoder_param_ids <= grouped_param_ids, (
        "params frozen at rebuild time were dropped from the optimizer groups"
    )


def test_fit_does_not_close_metric_loggers(small_model, tmp_path):
    """``fit`` must leave the metric store open.

    Regression guard: ``fit`` used to close ``self.metrics`` in a
    ``finally``, tearing down CSV/TB/W&B after the FIRST stage of a
    multi-stage run. The run owner (``Experiment.train``) closes loggers.
    """

    class _SentinelLogger:
        def __init__(self):
            self.closed = False

        def on_step(self, split, step, row):
            pass

        def on_epoch(self, split, epoch, row):
            pass

        def close(self):
            self.closed = True

    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    sentinel = _SentinelLogger()
    trainer.metrics.loggers.append(sentinel)
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
    assert not sentinel.closed, "fit() closed the metric loggers"


def test_abort_on_nonfinite_loss_raises_before_backward(small_model, tmp_path):
    """With the guard enabled, a NaN micro-batch loss aborts the step.

    The guard's host-side read is now gated on ``abort_on_nonfinite_loss``
    (the default path accumulates the loss on-device); this pins the
    guard's behaviour so the optimization can't silently disable it.
    """
    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer.abort_on_nonfinite_loss = True
    trainer._compute_loss_and_metrics = lambda batch, amp: (  # type: ignore[assignment]
        torch.tensor(float("nan")),
        {},
        1,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        trainer.fit(
            train_loader=loader,
            val_loader=None,
            total_steps=1,
            validate_every=0,
            log_every=1,
            checkpoint_every=None,
            amp=False,
        )


def test_elbo_plateau_early_stop_triggers(small_model, tmp_path):
    """When the loss is flat, the early-stop spec terminates ``fit`` early."""
    from ddssm.training.stages import EarlyStopSpec

    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    # ``min_improvement=10.0`` is impossible to meet ⇒ early-stop fires on the
    # first comparison after the warmup window.
    spec = EarlyStopSpec(
        enabled=True,
        window=4,
        min_improvement=10.0,
        warmup_steps=2,
    )
    final_step = trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=200,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
        early_stop=spec,
    )
    assert final_step < 200, (
        f"early-stop did not trigger; ran the full budget (final_step={final_step})"
    )


def test_validation_runs_under_ema_swap(small_model, tmp_path):
    """``_run_validation`` enters the EMA swap so val uses the EMA model (ADR-0005)."""
    from contextlib import contextmanager

    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)

    entered: list[int] = []
    real_swap = trainer.ema.swap

    @contextmanager
    def _spy_swap():
        entered.append(1)
        with real_swap():
            yield

    trainer.ema.swap = _spy_swap  # type: ignore[assignment]
    trainer.fit(
        train_loader=loader,
        val_loader=loader,
        total_steps=2,
        validate_every=1,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    assert entered, "validation did not enter the EMA swap"


def test_warn_ema_decay_too_high_for_budget(small_model, tmp_path, caplog):
    """``_warn_if_ema_decay_too_high`` fires when τ/budget > 5%.

    τ = 1 / (1 − decay). Budget 500, decay 0.9999 → τ = 10_000 (2000% of budget) —
    warn. Budget 500, decay 0.9 → τ = 10 (2%) — silent. Also boundary + edges.
    """
    import logging

    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )

    def _run(decay: float, total_steps: int) -> list[str]:
        trainer.ema_decay = decay
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="ddssm.training.train"):
            trainer._warn_if_ema_decay_too_high(total_steps)
        return [r.getMessage() for r in caplog.records if "[ema]" in r.getMessage()]

    # Warn: decay=0.9999 with 500 steps — the footgun the warning targets.
    assert _run(0.9999, 500), "expected an [ema] warning for τ/budget=2000%"
    # Silent: decay=0.9 (τ=10) with 500-step budget (2% of budget).
    assert not _run(0.9, 500), "unexpected warning at τ/budget=2%"
    # Silent edge: decay=1.0 has no defined τ; treat as swap-only, no warn.
    assert not _run(1.0, 500)
    # Silent edge: total_steps=0 (should not divide-by-zero, no warn).
    assert not _run(0.9999, 0)


def test_elbo_plateau_disabled_runs_full_budget(small_model, tmp_path):
    """``early_stop=None`` leaves the loop running until ``total_steps``."""
    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    final_step = trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=6,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
        early_stop=None,
    )
    assert final_step == 6


def test_train_meters_reset_between_stages(small_model, tmp_path):
    """Stage-2 metrics must not include stage-1 accumulation after ``fit()`` resets.

    Regression guard: train SplitStore was never reset across ``fit()`` calls,
    so keys written in stage 1 with MeanMeter semantics carried stale sums into
    stage 2's first flush.  The fix resets the train split at ``fit()`` entry.

    We assert via the SplitStore directly: after ``reset()``, a MeanMeter
    seeded with a stage-1 value and then updated with a stage-2 value must
    report only the stage-2 value.
    """
    from ddssm.training.loggers import MetricSpec, MetricStore

    store = MetricStore(
        spec=[MetricSpec("loss/*", "mean")],
        loggers=[],
    )

    # Stage 1: accumulate without flushing (log_every window not reached yet).
    store.update("train", {"loss/total": torch.tensor(100.0)}, weight=1)

    # fit() entry for stage 2 resets.
    store._split("train").reset()

    # Stage 2: accumulate only stage-2 values.
    store.update("train", {"loss/total": torch.tensor(1.0)}, weight=1)

    row = store.step_end("train", step=1)
    assert row["loss/total"] < 10.0, (
        f"loss/total={row['loss/total']:.2f} was contaminated by stage-1 "
        "accumulation (expected ≈ 1.0)"
    )


def test_fit_resets_train_meters_on_entry(small_model, tmp_path):
    """``fit()`` itself clears the train split so stage boundaries are clean.

    Concrete end-to-end proof: the trainer exposes a stale meter if reset is
    skipped and the loss value changes sharply between stages.  We check the
    raw meter value after stage-2's first fit() call but before any log flush.
    """
    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )

    # Manually prime the train split with a large value to simulate stage-1 leftovers.
    # Use a key that fits the "loss/*" mean spec.
    trainer.metrics._split("train").add("loss/sentinel", 999.0, 1.0)
    sentinel_val = trainer.metrics._split("train").meters["loss/sentinel"].value()
    assert sentinel_val == pytest.approx(999.0)

    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)
    # fit() entry must reset the train split, clearing the sentinel meter.
    trainer.fit(
        train_loader=loader,
        total_steps=1,
        validate_every=0,
        log_every=100,  # no CSV flush — we only care about the reset
        checkpoint_every=None,
        amp=False,
    )
    # After reset, the sentinel meter must be gone / zeroed — the split was
    # rebuilt from scratch via reset(), so the meter's sum/weight are 0.
    sentinel = trainer.metrics._split("train").meters.get("loss/sentinel")
    if sentinel is not None:
        assert sentinel.value() == pytest.approx(0.0), (
            "stage-1 sentinel value survived into stage-2 (reset did not fire)"
        )


def test_validation_enters_autocast_when_amp_enabled(small_model, tmp_path):
    """``_run_validation`` enters torch.amp.autocast when amp=True.

    Regression guard: the validation loop ran in full precision even when AMP
    was enabled, making val/train losses non-comparable and slowing validation.
    """
    import unittest.mock as mock

    trainer = DDSSMTrainer(
        model=small_model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)

    entered: list[bool] = []
    real_autocast = torch.amp.autocast

    def _spy_autocast(device_type, *args, **kwargs):
        if kwargs.get("enabled", True):
            entered.append(True)
        return real_autocast(device_type, *args, **kwargs)

    with mock.patch("torch.amp.autocast", side_effect=_spy_autocast):
        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            total_steps=2,
            validate_every=1,
            log_every=1,
            checkpoint_every=None,
            amp=True,
        )

    assert entered, "validation did not enter torch.amp.autocast with amp=True"


def test_param_groups_for_adamw_accepts_psi_betas_none(small_model):
    """``param_groups_for_adamw`` with ``psi_betas=None`` (default) is backwards-compat.

    Regression guard: the new optional kwarg must default to ``None`` so existing
    call sites that don't pass it continue to work and produce group dicts with
    no ``betas`` key.
    """
    from ddssm.training.train_utils import param_groups_for_adamw

    groups = param_groups_for_adamw(
        small_model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=0.01,
        psi_betas=None,
    )
    assert groups, "param_groups_for_adamw returned empty list"
    for g in groups:
        assert "betas" not in g, (
            f"psi_betas=None must produce no 'betas' key in any group; got: {g}"
        )

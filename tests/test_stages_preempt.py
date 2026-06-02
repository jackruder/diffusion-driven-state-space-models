"""Tests for stage-aware multi-stage resume in :class:`StageOrchestrator`.

Covers ADR-0009 multi-stage resume support: per-stage ``stage_prefix``
embedded in the checkpoint payload, ``StageOrchestrator.run(resume_from=)``
that uses it to skip already-completed stages and suppress the centering
handoff on the resumed stage.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from functools import partial
from unittest.mock import patch

import torch
import pytest
from torch.utils.data import Dataset, DataLoader

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.model.dssd import DDSSM_base
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.diffnets import ContextProducer, FeatureMixerConfig, ResidualBlockConfig
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.gaussians import GaussianHead
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.aggregators import IdentityAggregator
from ddssm.training.train import DDSSMTrainer
from ddssm.training.stages import (
    StagesConf,
    StageLrsConf,
    StageSpecConf,
    StageOrchestrator,
    StageTrainableConf,
)
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.training.checkpoint import Checkpoint
from ddssm.model.centering.handoff import CenteringHandoffConf
from ddssm.model.transitions.transitions import GaussianTransition

# ---------------------------------------------------------------------------
# Mini-model trainer fixture (mirrors test_train_preempt.py)
# ---------------------------------------------------------------------------

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
_GH = GaussianHead
_FS = partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1)


def _make_small_model() -> DDSSM_base:
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
    trans = GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=CHANNELS, n_layers=1)
    return DDSSM_base(
        encoder=enc, decoder=dec, transition=trans, aux_posterior=aux,
        j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM, emb_time_dim=EMB_TIME,
    )


class _SyntheticBatchDataset(Dataset):
    def __init__(self, B: int = 2, T: int = 4):
        self.B = B
        self.T = T

    def __len__(self) -> int:
        return self.B

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "observed_data": torch.randn(DATA_DIM, self.T),
            "observation_mask": torch.ones(DATA_DIM, self.T),
            "timepoints": torch.arange(self.T, dtype=torch.float32),
        }


def _make_trainer(tmp_path, **kwargs: Any) -> DDSSMTrainer:
    return DDSSMTrainer(
        model=_make_small_model(),
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        quiet=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Checkpoint payload: stage_prefix field
# ---------------------------------------------------------------------------


def test_checkpoint_payload_carries_stage_prefix(tmp_path) -> None:
    """A trainer save with a stage prefix embeds ``stage_prefix`` in the payload."""
    trainer = _make_trainer(tmp_path)
    # Drive a periodic save with a stage prefix; checkpoint_dir is the
    # trainer's configured ckpts dir.
    latest_path = trainer._save_periodic_checkpoint(step=5, checkpoint_prefix="stage_1")
    assert os.path.isfile(latest_path)
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert payload.get("stage_prefix") == "stage_1"


def test_checkpoint_loads_with_stage_prefix_field(tmp_path) -> None:
    """``Checkpoint.load`` populates ``stage_prefix`` (and is None for legacy)."""
    trainer = _make_trainer(tmp_path)
    latest_path = trainer._save_periodic_checkpoint(step=5, checkpoint_prefix="stage_1")
    ckpt = Checkpoint.load(latest_path, device=torch.device("cpu"))
    assert ckpt.stage_prefix == "stage_1"

    # Legacy ckpt: hand-build a payload WITHOUT stage_prefix to simulate the
    # pre-ADR-0009-followup format. ``Checkpoint.load`` must return
    # ``stage_prefix=None`` and not crash.
    legacy_payload = {
        "_format": "ddssm_ckpt_v1",
        "model_config_yaml": None,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": None,
        "ema_decay": trainer.ema_decay,
        "ema_state": None,
        "global_step": 5,
        "grad_accum_steps": 1,
    }
    legacy_path = tmp_path / "legacy.pth"
    torch.save(legacy_payload, str(legacy_path))
    legacy = Checkpoint.load(str(legacy_path), device=torch.device("cpu"))
    assert legacy.stage_prefix is None


# ---------------------------------------------------------------------------
# StageOrchestrator.run(resume_from=...)
# ---------------------------------------------------------------------------


class _OrchTrainer:
    """Stub trainer for orchestrator tests; records calls + supports restore."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(stage_selector="stage_0")
        self.global_step: int = 0
        self.device = torch.device("cpu")
        self.calls: list[tuple] = []
        # Captures: ``fit`` kwargs, ``restore`` calls.
        self.fit_calls: list[dict] = []
        self.restore_calls: list[str] = []

    def _rebuild_optimizer(self, lrs) -> None:
        self.calls.append(("rebuild", lrs))

    def _set_trainable(self, t) -> None:
        self.calls.append(("trainable", t))

    def fit(self, **kw) -> None:
        self.calls.append(("fit", kw.get("checkpoint_prefix"), kw.get("resume_from")))
        self.fit_calls.append(dict(kw))

    def restore_from_checkpoint(self, path: str, strict: bool = True) -> dict:
        self.restore_calls.append(path)
        return {"grad_accum_steps": 1}


def _two_stage_config() -> StagesConf:
    stage_1 = StageSpecConf(
        steps=10,
        trainable=StageTrainableConf(),
        lrs=StageLrsConf(enc_lr=1e-3),
        log_every=5, val_every=10, checkpoint_every=10,
    )
    stage_2 = StageSpecConf(
        steps=20,
        trainable=StageTrainableConf(),
        lrs=StageLrsConf(enc_lr=2e-3),
        log_every=5, val_every=10, checkpoint_every=10,
        centering_handoff=CenteringHandoffConf(sigma_pert=0.0),
    )
    return StagesConf(stage_1=stage_1, stage_2=stage_2, run=["stage_1", "stage_2"])


def _write_stage_ckpt(
    tmp_path, *, stage_prefix: str | None, global_step: int = 50
) -> str:
    """Write a minimal ckpt payload with the given stage_prefix and return its path."""
    payload: dict[str, Any] = {
        "_format": "ddssm_ckpt_v1",
        "model_config_yaml": None,
        "model_state": {},
        "optimizer_state": None,
        "ema_decay": 0.999,
        "ema_state": None,
        "global_step": int(global_step),
        "grad_accum_steps": 1,
    }
    if stage_prefix is not None:
        payload["stage_prefix"] = stage_prefix
    path = tmp_path / f"ckpt_{stage_prefix or 'legacy'}_latest.pth"
    torch.save(payload, str(path))
    return str(path)


def test_orchestrator_run_accepts_resume_from(tmp_path) -> None:
    """Smoke: ``run(resume_from=None)`` matches pre-resume behavior; missing path raises."""
    trainer = _OrchTrainer()
    cfg = _two_stage_config()
    orch = StageOrchestrator(trainer, cfg)

    # resume_from=None — both stages run, handoff fires on stage_2.
    with patch("ddssm.training.stages.perform_centering_handoff") as mock_handoff:
        orch.run(train_loader=object(), amp=False, resume_from=None)
    fit_prefixes = [c[1] for c in trainer.calls if c[0] == "fit"]
    assert fit_prefixes == ["stage_1", "stage_2"]
    assert mock_handoff.call_count == 1
    assert trainer.restore_calls == []  # no restore when resume_from is None

    # Missing path — early FileNotFoundError before any stage runs.
    trainer2 = _OrchTrainer()
    orch2 = StageOrchestrator(trainer2, _two_stage_config())
    missing = str(tmp_path / "does_not_exist.pth")
    with pytest.raises(FileNotFoundError):
        orch2.run(train_loader=object(), amp=False, resume_from=missing)
    assert not [c for c in trainer2.calls if c[0] == "fit"]


def test_orchestrator_resumes_into_stage_2_skips_stage_1_and_handoff(
    tmp_path,
) -> None:
    """Resuming a stage_2 ckpt: skip stage_1, skip handoff, restore before stage_2 fit."""
    ckpt_path = _write_stage_ckpt(tmp_path, stage_prefix="stage_2")
    trainer = _OrchTrainer()
    cfg = _two_stage_config()
    orch = StageOrchestrator(trainer, cfg)

    with patch("ddssm.training.stages.perform_centering_handoff") as mock_handoff:
        orch.run(train_loader=object(), amp=False, resume_from=ckpt_path)

    # Handoff MUST NOT have fired.
    assert mock_handoff.call_count == 0
    # Exactly one fit, and it's stage_2.
    fits = [c for c in trainer.calls if c[0] == "fit"]
    assert len(fits) == 1
    assert fits[0][1] == "stage_2"
    # The trainer's restore happened (orchestrator passed resume_from into fit).
    # We trust the orchestrator to pass resume_from to fit, which then triggers
    # the trainer's own restore via _safe_resume.
    assert fits[0][2] == ckpt_path
    # No fit for stage_1.
    assert not any(c[1] == "stage_1" for c in fits)


def test_orchestrator_resumes_into_stage_1_still_runs_stage_2_normally(
    tmp_path,
) -> None:
    """Resuming a stage_1 ckpt: stage_1 fit gets resume_from; stage_2 runs with handoff."""
    ckpt_path = _write_stage_ckpt(tmp_path, stage_prefix="stage_1")
    trainer = _OrchTrainer()
    cfg = _two_stage_config()
    orch = StageOrchestrator(trainer, cfg)

    with patch("ddssm.training.stages.perform_centering_handoff") as mock_handoff:
        orch.run(train_loader=object(), amp=False, resume_from=ckpt_path)

    fits = [c for c in trainer.calls if c[0] == "fit"]
    # Stage_1's fit received resume_from; stage_2's fit did not.
    assert len(fits) == 2
    assert fits[0][1] == "stage_1" and fits[0][2] == ckpt_path
    assert fits[1][1] == "stage_2" and fits[1][2] is None
    # Handoff fires for stage_2 normally.
    assert mock_handoff.call_count == 1


def test_orchestrator_resume_unknown_stage_prefix_raises(tmp_path) -> None:
    """A ckpt referencing a stage not in ``stages.run`` is a config bug → ValueError."""
    ckpt_path = _write_stage_ckpt(tmp_path, stage_prefix="stage_99")
    trainer = _OrchTrainer()
    cfg = _two_stage_config()
    orch = StageOrchestrator(trainer, cfg)
    with pytest.raises(ValueError) as exc:
        orch.run(train_loader=object(), amp=False, resume_from=ckpt_path)
    assert "stage_99" in str(exc.value)


def test_orchestrator_resume_with_no_stage_prefix_starts_stage_1(tmp_path) -> None:
    """Legacy ckpt (no stage_prefix): treat as "resume into first stage", run rest normally."""
    ckpt_path = _write_stage_ckpt(tmp_path, stage_prefix=None)
    trainer = _OrchTrainer()
    cfg = _two_stage_config()
    orch = StageOrchestrator(trainer, cfg)

    with patch("ddssm.training.stages.perform_centering_handoff") as mock_handoff:
        orch.run(train_loader=object(), amp=False, resume_from=ckpt_path)

    fits = [c for c in trainer.calls if c[0] == "fit"]
    assert len(fits) == 2
    # Stage_1 receives resume_from (back-compat for hand-rolled resumes).
    assert fits[0][1] == "stage_1" and fits[0][2] == ckpt_path
    assert fits[1][1] == "stage_2" and fits[1][2] is None
    assert mock_handoff.call_count == 1

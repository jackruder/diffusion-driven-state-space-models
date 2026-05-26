"""End-to-end smoke test for the model-v2 baseline-centering core.

Drives ``init_smoke_simple`` for 5 + 5 steps via the orchestrator,
asserts that:

* The run completes without raising.
* ``metrics.csv`` carries the new model-v2 columns.
* Stage-1 has a non-zero ``loss/rate/init/entropy`` while stage-2 has
  ``entropy == 0`` (the entropy-cancellation invariant from
  ``model-v2.org`` § Entropy cancellation in stage 2).
* ``model.baseline_anchor`` is populated by the end of the run (handoff
  fired before stage 2).
* The σ_data buffer is frozen at the end (the "fixed" tracking mode
  has had its schedule reset by the handoff).

Marked ``slow`` because it constructs the full model + runs the
orchestrator.  Excluded from the default suite; run with::

    pytest tests/test_init_smoke_simple.py -m slow
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import pytest
from hydra_zen import instantiate

pytestmark = pytest.mark.slow


def _get_experiment_cfg(name: str):
    from ddssm._experiment_registry import register_experiments

    register_experiments()  # puts repo root on sys.path + imports experiments
    from conf.registry import store

    for entry in store:
        if entry["group"] == "experiment" and entry["name"] == name:
            return entry["node"]
    raise KeyError(f"Experiment {name!r} not registered")


def test_init_smoke_simple_end_to_end(tmp_path: Path) -> None:
    """5 + 5-step end-to-end run through stage 1 + handoff + stage 2."""
    cfg = _get_experiment_cfg("init_smoke_simple")
    exp = instantiate(cfg)
    # Shrink the stages for a fast smoke run.
    exp.model.config.stages.stage_1.steps = 5
    exp.model.config.stages.stage_2.steps = 5
    exp.model.config.stages.stage_1.log_every = 1
    exp.model.config.stages.stage_2.log_every = 1
    exp.model.config.stages.stage_1.val_every = 0
    exp.model.config.stages.stage_2.val_every = 0
    exp.model.config.stages.stage_1.checkpoint_every = 100
    exp.model.config.stages.stage_2.checkpoint_every = 100

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    exp.train(device=device, run_dir=str(run_dir))

    csv_path = run_dir / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    expected_columns = {
        "loss/rate/init/loss_init",
        "loss/rate/init/kl_aux",
        "loss/rate/trans/r_sigma_p",
        "loss/rate/trans/r_mu_p",
        "loss/rate/trans/log_sigma_p2_mean",
    }
    assert expected_columns.issubset(set(fieldnames)), (
        f"missing columns: {expected_columns - set(fieldnames)}"
    )

    # Identify stage-1 vs stage-2 rows by step.  The trainer's global_step
    # continues across stages; stage 1 covers steps 1..5, stage 2 covers
    # steps 6..10.
    stage1_rows = [r for r in rows if 1 <= int(r["step"]) <= 5]
    stage2_rows = [r for r in rows if 6 <= int(r["step"]) <= 10]
    assert stage1_rows, "no stage-1 rows logged"
    assert stage2_rows, "no stage-2 rows logged"

    # Entropy-cancellation invariant: stage-2 entropy column is exactly 0.
    for r in stage2_rows:
        assert float(r["loss/rate/init/entropy"]) == 0.0, (
            f"stage-2 entropy != 0 at step={r['step']}: {r['loss/rate/init/entropy']}"
        )
    # Stage-1 should have a non-zero (negative) entropy term.
    s1_entropies = [float(r["loss/rate/init/entropy"]) for r in stage1_rows]
    assert any(abs(v) > 1e-3 for v in s1_entropies), (
        "stage-1 entropy column unexpectedly all near zero"
    )

    # The handoff populates ``baseline_anchor``.
    assert exp.model.baseline_anchor is not None
    # The handoff resets the σ_data EMA schedule.  Under the canonical
    # cell's per-t tracking mode the buffer keeps updating after the
    # handoff (``frozen`` stays False); only the per-t step counter
    # resets to zero.  Under "fixed" tracking the buffer freezes.
    if exp.model.sigma_data.tracking_mode == "fixed":
        assert exp.model.sigma_data.frozen is True
    else:
        assert exp.model.sigma_data.frozen is False
        assert int(exp.model.sigma_data.ema_step.max()) == 5  # 5 stage-2 steps


def test_init_smoke_simple_shares_baseline_instance() -> None:
    """Both transitions reference the *same* baseline Python object."""
    cfg = _get_experiment_cfg("init_smoke_simple")
    exp = instantiate(cfg)
    assert exp.model.baseline is exp.model.stage1_transition.baseline
    assert exp.model.baseline is exp.model.transition.baseline

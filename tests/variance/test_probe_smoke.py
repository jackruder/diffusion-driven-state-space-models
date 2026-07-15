"""Smoke test for the variance probe stage on the diffusion path.

Post legacy-purge the probe runs against the init-centering diffusion
transition (the V2 family it was originally written for is gone).
``DiffusionTransition.transition_kl`` gained ``return_per_sample`` so
the probe can collect per-sample ESM losses + score-net gradients.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
from hydra_zen import instantiate

from ddssm.variance.runner import ProbeCell, ProbeSpec, ProbePlotSpec, variance
from ddssm.experiment.registry import register_experiments

register_experiments()

CONF_DIR = (Path(__file__).resolve().parents[2] / "src" / "ddssm" / "conf").as_posix()


def _make_experiment(name: str):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    return instantiate(cfg.experiment)


def test_force_per_k_sets_mode_per_cell_and_restores(monkeypatch) -> None:
    """Every forced-k call runs under its own cell's mode; state is restored.

    ``run_probe`` keys all modes to the SAME shared ``model.transition``.
    Pre-fix, the forced-k sweep never re-set the mode attributes per cell,
    so every cell ran under whichever configuration the replica loop set
    last — and the mutated mode leaked past ``run_probe`` entirely.
    """
    from ddssm.variance.probe import run_probe
    from ddssm.model.transitions.diffusion import DiffusionTransition

    expt = _make_experiment("init_smoke_simple")
    # Shrink the forced-k sweep: the loop bound reads ``num_steps`` off the
    # transition, while the schedule buffers stay full-size (forced indices
    # 0..1 remain valid rows), so this only trims test cost.
    expt.model.module.transition.num_steps = 2

    orig_mode = expt.model.module.transition.k_sampling_mode
    orig_sched_mode = expt.model.module.transition.schedule.k_sampling_mode
    orig_p_k = expt.model.module.transition.p_k

    modes_at_call: list[str] = []
    orig_kl = DiffusionTransition.transition_kl

    def spy(self, *args, **kwargs):
        modes_at_call.append(self.k_sampling_mode)
        return orig_kl(self, *args, **kwargs)

    monkeypatch.setattr(DiffusionTransition, "transition_kl", spy)

    spec = ProbeSpec(
        cells=[ProbeCell("esm", "uniform"), ProbeCell("esm", "lsgm_is")],
        R=1,
        n_batches=1,
        seeds=[0],
        force_per_k=True,
        plots=[],
    )
    run_probe(expt, spec, device=torch.device("cpu"), checkpoint_path=None)

    # Call layout: R × n_cells replica calls, then num_steps × n_cells forced.
    assert len(modes_at_call) == 2 + 2 * 2
    assert modes_at_call[2:] == ["uniform", "lsgm_is", "uniform", "lsgm_is"]

    # The training-time configuration must be restored on the shared module.
    assert expt.model.module.transition.k_sampling_mode == orig_mode
    assert expt.model.module.transition.schedule.k_sampling_mode == orig_sched_mode
    assert expt.model.module.transition.p_k is orig_p_k


def test_variance_runner_smoke(tmp_path: Path) -> None:
    """The probe runner drives a diffusion model end-to-end and writes its artefacts."""
    expt = _make_experiment("init_smoke_simple")
    spec = ProbeSpec(
        cells=[ProbeCell("esm", "uniform"), ProbeCell("dsm", "uniform")],
        R=2,
        n_batches=1,
        seeds=[0],
        force_per_k=False,
        plots=[ProbePlotSpec(name="summary_table", save_filename="summary_table.png")],
    )
    out = variance(expt, spec, device=torch.device("cpu"), run_dir=str(tmp_path))
    assert "summary" in out and "metrics" in out
    assert (tmp_path / "variance_raw.csv").is_file()
    assert (tmp_path / "variance_summary.json").is_file()
    assert (tmp_path / "summary_table.png").is_file()

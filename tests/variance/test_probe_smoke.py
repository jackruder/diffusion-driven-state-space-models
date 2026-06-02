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

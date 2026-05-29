"""Tests for the library launcher: StudyOrchestrator + the ddssm.launch CLI."""

from __future__ import annotations

import pytest

from ddssm import launch as L
from ddssm.launch import (
    LaunchContext,
    OptunaMultiNode,
    OptunaSingleNode,
    PointLaunch,
    SingleJob,
    StudyOrchestrator,
)
from ddssm.study import StudyPoint
from experiments.init_centering.study import INIT_CENTERING_STUDY


def _orch(**kw) -> StudyOrchestrator:
    kw.setdefault("study_prefix", "abl")
    return StudyOrchestrator(INIT_CENTERING_STUDY, **kw)


# --- Strategies -------------------------------------------------------------


def test_single_job_emits_no_sweep() -> None:
    ctx = LaunchContext("pre", "s", "w")
    assert SingleJob().hydra_overrides(StudyPoint("p", {}, {}, {}), PointLaunch(strategy="single_job"), ctx) == []


def test_optuna_single_node_wires_study_and_storage() -> None:
    ctx = LaunchContext("pre", "store", "sweeps")
    ov = OptunaSingleNode().hydra_overrides(
        StudyPoint("p", {}, {}, {}), PointLaunch(strategy="optuna_single_node", sweep="sw", n_trials=7), ctx
    )
    assert "--multirun" in ov and "+sweep=sw" in ov
    assert "hydra.sweeper.n_trials=7" in ov
    assert any(o.startswith("hydra.sweeper.study_name=pre_p") for o in ov)


def test_stub_strategy_raises() -> None:
    with pytest.raises(NotImplementedError):
        OptunaMultiNode().hydra_overrides(StudyPoint("p", {}, {}, {}), PointLaunch(strategy="optuna_multi_node"), ctx=LaunchContext("p", "s", "w"))


# --- Orchestrator render ----------------------------------------------------


def test_render_bakes_data_and_wires_sweep() -> None:
    jobs = _orch().render(INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t"))
    assert [j for j, _ in jobs] == ["init_mlp_pinned_per_t__1d", "init_mlp_pinned_per_t__mv"]
    text = "\n".join(s for _, s in jobs)
    assert "experiment=init_mlp_pinned_per_t__1d" in text
    assert "+sweep=init_ablation_moo" in text
    assert "experiment.data.mode=" not in text          # data baked into the preset
    assert "experiment.model.latent_dim=" not in text    # tiny size: no override


def test_render_paper_size_doubles_latent() -> None:
    text = "\n".join(
        s for _, s in _orch().render(INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t"), size="paper")
    )
    assert "experiment.model.latent_dim=2" in text
    assert "experiment.model.latent_dim=8" in text


def test_render_seed_replication() -> None:
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    job, script = _orch().render(pts, seed=3)[0]
    assert job == "init_mlp_pinned_per_t__1d__seed3"
    assert "experiment.seed=3" in script


# --- launch() + CLI ---------------------------------------------------------


def test_launch_submit_requires_write_dir() -> None:
    with pytest.raises(ValueError):
        _orch().launch(INIT_CENTERING_STUDY.points[:1], submit=True)


def test_cli_dry_run(capsys) -> None:
    rc = L.main(["init_centering", "--select", "cell=init_mlp_pinned_per_t",
                 "--study-prefix", "abl", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "experiment=init_mlp_pinned_per_t__1d" in out
    assert "+sweep=init_ablation_moo" in out


def test_cli_unknown_study_errors() -> None:
    with pytest.raises(SystemExit):
        L.main(["nope", "--dry-run"])


def test_cli_submit_requires_write_dir() -> None:
    with pytest.raises(SystemExit):
        L.main(["init_centering", "--submit"])


def test_submit_shells_out_once_per_file(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("ddssm.launch.submit_sbatch", lambda p: calls.append(p) or "ok")
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t")
    orch = _orch(storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"))
    write_dir = tmp_path / "sbatch"
    orch.launch(pts, write_dir=str(write_dir), submit=True)
    written = sorted(str(p) for p in write_dir.glob("*.sbatch"))
    assert len(written) == 2
    assert sorted(calls) == written

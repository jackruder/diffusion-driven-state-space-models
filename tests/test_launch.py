"""Tests for the library launcher: StudyOrchestrator + the ddssm.launch CLI."""

from __future__ import annotations

import pytest

from ddssm import launch as L
from ddssm.experiment import SBatch
from ddssm.launch import (
    LaunchContext,
    LocalParallel,
    OptunaMultiNode,
    OptunaSingleNode,
    PointLaunch,
    SingleJob,
    SlurmArray,
    StudyOrchestrator,
)
from ddssm.study import StudyPoint
from experiments.init_centering.study import INIT_CENTERING_STUDY

from types import SimpleNamespace
from ddssm.study import Study as _Study, StudyPoint as _StudyPoint


def _single_stage_preempt_study(n_workers: int = 3) -> _Study:
    """A minimal one-point Study with NO multi-stage training, for preempt tests.

    The orchestrator's multi-stage rejection (ADR-0009) blocks INIT_CENTERING_STUDY
    points because they all use StagesB. This fixture sidesteps that constraint
    while still exercising the orchestrator's preempt path end-to-end against
    a registered-looking point.
    """
    point = _StudyPoint(
        name="mock_preempt_point",
        config=SimpleNamespace(training=SimpleNamespace(stages=None)),
        tags={"dataset": "1d", "cell": "mock"},
        coords={},
    )
    return _Study(
        name="mock_preempt_study",
        points=(point,),
        launch=lambda p: PointLaunch(
            strategy="optuna_multi_node",
            sweep="init_ablation_moo",
            n_trials=12,
            n_workers=n_workers,
            preemptive=True,
            preempt_grace_seconds=90,
        ),
    )


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


# --- Preempt fields + strategy placeholder emission (ADR-0009 Phase 3) ------


def test_point_launch_has_preemptive_fields() -> None:
    pl = PointLaunch(preemptive=True, preempt_grace_seconds=90)
    assert pl.preemptive is True
    assert pl.preempt_grace_seconds == 90


def test_point_launch_preemptive_defaults_false_with_180s_grace() -> None:
    pl = PointLaunch()
    assert pl.preemptive is False
    assert pl.preempt_grace_seconds == 180


def test_optuna_single_node_emits_n_per_worker_placeholder_when_preemptive() -> None:
    ctx = LaunchContext("pre", "store", "sweeps")
    pl = PointLaunch(
        strategy="optuna_single_node", sweep="sw", n_trials=7, preemptive=True,
    )
    ov = OptunaSingleNode().hydra_overrides(StudyPoint("p", {}, {}, {}), pl, ctx)
    assert "hydra.sweeper.n_trials=__N_PER_WORKER__" in ov
    assert "hydra.sweeper.n_trials=7" not in ov


def test_optuna_multi_node_emits_n_per_worker_placeholder_when_preemptive() -> None:
    ctx = LaunchContext("pre", "store", "sweeps")
    pl = PointLaunch(
        strategy="optuna_multi_node", sweep="sw", n_trials=12, n_workers=3,
        preemptive=True,
    )
    ov = OptunaMultiNode().hydra_overrides(StudyPoint("p", {}, {}, {}), pl, ctx)
    assert "hydra.sweeper.n_trials=__N_PER_WORKER__" in ov
    assert "hydra.sweeper.n_trials=12" not in ov


def test_local_parallel_emits_n_per_worker_placeholder_when_preemptive() -> None:
    ctx = LaunchContext("pre", "store", "sweeps")
    pl = PointLaunch(
        strategy="local_parallel", sweep="sw", n_trials=12, n_workers=2,
        preemptive=True,
    )
    ov = LocalParallel().hydra_overrides(StudyPoint("p", {}, {}, {}), pl, ctx)
    assert "hydra.sweeper.n_trials=__N_PER_WORKER__" in ov


def test_strategies_emit_literal_n_trials_when_not_preemptive() -> None:
    """Regression: the placeholder must NOT leak into non-preemptive renders."""
    ctx = LaunchContext("pre", "store", "sweeps")
    sp = StudyPoint("p", {}, {}, {})

    pl_single = PointLaunch(strategy="optuna_single_node", sweep="sw", n_trials=7)
    ov_single = OptunaSingleNode().hydra_overrides(sp, pl_single, ctx)
    assert "hydra.sweeper.n_trials=7" in ov_single
    assert "__N_PER_WORKER__" not in " ".join(ov_single)

    pl_multi = PointLaunch(
        strategy="optuna_multi_node", sweep="sw", n_trials=12, n_workers=3,
    )
    ov_multi = OptunaMultiNode().hydra_overrides(sp, pl_multi, ctx)
    assert "hydra.sweeper.n_trials=12" in ov_multi
    assert "__N_PER_WORKER__" not in " ".join(ov_multi)


def test_slurm_array_is_still_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        SlurmArray().hydra_overrides(
            StudyPoint("p", {}, {}, {}),
            PointLaunch(strategy="slurm_array"),
            LaunchContext("p", "s", "w"),
        )


def test_optuna_multi_node_shared_study_per_worker_subdir() -> None:
    ctx = LaunchContext("pre", "store", "sweeps")
    sp = StudyPoint("p", {}, {}, {})
    pl = PointLaunch(strategy="optuna_multi_node", sweep="sw", n_trials=12, n_workers=3)
    strat = OptunaMultiNode()
    assert strat.n_workers_per_point(pl) == 3
    sigs = [strat.hydra_overrides(sp, pl, ctx, worker_idx=i) for i in range(3)]
    # Study name + storage + sweep.dir are shared across workers.
    for ov in sigs:
        assert "hydra.sweeper.study_name=pre_p" in ov
        assert "hydra.sweeper.storage=sqlite:///store/pre_p.db" in ov
        assert "hydra.sweep.dir=sweeps/pre_p" in ov
    # Subdir namespace is per-worker (`hydra.job.num` left for Hydra to expand).
    subdirs = [next(o for o in ov if o.startswith("hydra.sweep.subdir=")) for ov in sigs]
    assert subdirs == [
        "hydra.sweep.subdir=w0_${hydra.job.num}",
        "hydra.sweep.subdir=w1_${hydra.job.num}",
        "hydra.sweep.subdir=w2_${hydra.job.num}",
    ]


def test_optuna_multi_node_rejects_missing_sweep() -> None:
    with pytest.raises(ValueError, match="sweep"):
        OptunaMultiNode().hydra_overrides(
            StudyPoint("p", {}, {}, {}),
            PointLaunch(strategy="optuna_multi_node", n_workers=2),
            LaunchContext("p", "s", "w"),
        )


def test_local_parallel_shares_optuna_multi_node_overrides() -> None:
    """The two strategies emit identical per-worker overrides; only the backend differs."""
    ctx = LaunchContext("pre", "store", "sweeps")
    sp = StudyPoint("p", {}, {}, {})
    pl = PointLaunch(strategy="local_parallel", sweep="sw", n_trials=5, n_workers=2)
    assert LocalParallel().n_workers_per_point(pl) == 2
    multi = OptunaMultiNode().hydra_overrides(sp, pl, ctx, worker_idx=1)
    local = LocalParallel().hydra_overrides(sp, pl, ctx, worker_idx=1)
    assert multi == local


def test_strategy_support_gates() -> None:
    assert OptunaMultiNode.supports_sbatch is True
    assert OptunaMultiNode.supports_local is False
    assert LocalParallel.supports_sbatch is False
    assert LocalParallel.supports_local is True


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
    # The Hydra preset stays unsuffixed (only ``experiment.seed`` differs across
    # seed replicates); the seed suffix lives on ``--job-name`` + the log dir.
    assert "experiment=init_mlp_pinned_per_t__1d " in script
    assert "experiment=init_mlp_pinned_per_t__1d__seed3" not in script
    assert "--job-name=ddssm-init_mlp_pinned_per_t__1d__seed3" in script
    assert "runs/init_mlp_pinned_per_t__1d__seed3/slurm-" in script


# --- Multi-worker render via orchestrator -----------------------------------


def _force_multi_node(n_workers: int):
    """Launch override: rewrite every point's PointLaunch to optuna_multi_node."""

    def _o(point):
        return PointLaunch(
            strategy="optuna_multi_node",
            sweep="init_ablation_moo",
            n_trials=12,
            n_workers=n_workers,
        )

    return _o


def test_render_optuna_multi_node_emits_one_sbatch_per_worker() -> None:
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    jobs = _orch().render(pts, launch_override=_force_multi_node(4))
    names = [j for j, _ in jobs]
    assert names == [
        "init_mlp_pinned_per_t__1d_w0",
        "init_mlp_pinned_per_t__1d_w1",
        "init_mlp_pinned_per_t__1d_w2",
        "init_mlp_pinned_per_t__1d_w3",
    ]
    # Every worker hits the same study + DB; subdirs are per-worker.
    for i, (_, script) in enumerate(jobs):
        # The Hydra preset stays the point name (NOT the worker-suffixed job name).
        assert "experiment=init_mlp_pinned_per_t__1d " in script
        assert "experiment=init_mlp_pinned_per_t__1d_w" not in script
        assert "hydra.sweeper.study_name=abl_init_mlp_pinned_per_t__1d" in script
        assert "hydra.sweeper.storage=sqlite:///" in script
        assert f"hydra.sweep.subdir=w{i}_" in script
        # SBATCH bookkeeping (job_name, log dir) carry the worker suffix.
        assert f"--job-name=ddssm-init_mlp_pinned_per_t__1d_w{i}" in script
        assert f"runs/init_mlp_pinned_per_t__1d_w{i}/slurm-" in script


def test_render_optuna_multi_node_keeps_single_worker_name_when_n_workers_is_1() -> None:
    """``n_workers=1`` (the degenerate case) keeps the base job name — no ``_w0`` suffix."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    jobs = _orch().render(pts, launch_override=_force_multi_node(1))
    assert [j for j, _ in jobs] == ["init_mlp_pinned_per_t__1d"]


def test_render_local_parallel_rejected() -> None:
    """``local_parallel`` cannot be rendered to sbatch."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(strategy="local_parallel", sweep="x", n_workers=2)
    with pytest.raises(ValueError, match="sbatch"):
        _orch().render(pts, launch_override=override)


# --- render_sbatch preempt mode (ADR-0009 Phase 4) --------------------------
#
# These tests exercise ``render_sbatch`` directly with a ``PreemptSpec`` —
# Phase 5 wires the orchestrator to construct + pass the spec; Phase 4 just
# proves the rendering shape. The strategy machinery is used to build a
# realistic ``hydra_overrides`` list (carrying the ``__N_PER_WORKER__``
# placeholder) so that the substitution path is covered end-to-end.


def _preempt_render(
    *,
    n_workers: int = 3,
    n_trials: int = 12,
    grace: int = 90,
    worker_idx: int = 0,
    target: int | None = None,
    extra_flags: tuple[str, ...] = (),
):
    """Build a realistic preemptive sbatch render for the multi-node strategy.

    Returns the rendered script text. Uses the optuna_multi_node strategy to
    generate Hydra overrides (which embed the ``__N_PER_WORKER__`` placeholder
    under ``pl.preemptive=True``), then calls ``render_sbatch`` with a
    ``PreemptSpec`` directly (orchestrator wiring is Phase 5).
    """
    from ddssm.sbatch import PreemptSpec, render_sbatch

    pl = PointLaunch(
        strategy="optuna_multi_node",
        sweep="init_ablation_moo",
        n_trials=n_trials,
        n_workers=n_workers,
        preemptive=True,
        preempt_grace_seconds=grace,
    )
    ctx = LaunchContext("abl", "/tmp/store", "/tmp/sweeps")
    sp = StudyPoint("init_mlp_pinned_per_t__1d", {}, {}, {})
    overrides = OptunaMultiNode().hydra_overrides(sp, pl, ctx, worker_idx=worker_idx)
    study_name = f"abl_{sp.name}"
    storage_url = f"sqlite:////tmp/store/{study_name}.db"
    target_value = pl.n_trials if target is None else target
    ps = PreemptSpec(
        grace_seconds=grace,
        storage_url=storage_url,
        study_name=study_name,
        target=target_value,
        n_workers=n_workers,
        worker_idx=worker_idx,
    )
    exp_sbatch = SBatch(
        partition="gpu", time="04:00:00", gpus=1, cpus=4, mem="32G", nodes=1,
        extra_flags=extra_flags,
    )
    return render_sbatch(
        sp.name,
        exp_sbatch=exp_sbatch,
        hydra_overrides=overrides,
        cli_overrides={"job_name": f"ddssm-{sp.name}_w{worker_idx}"},
        output_pattern=f"runs/{sp.name}_w{worker_idx}/slurm-%j.out",
        preempt=ps,
    )


def test_render_preemptive_emits_three_sbatch_directives() -> None:
    script = _preempt_render(grace=90)
    assert "#SBATCH --requeue" in script
    assert "#SBATCH --signal=B:USR1@90" in script
    assert "#SBATCH --open-mode=append" in script


def test_render_preemptive_injects_launch_remaining_invocation() -> None:
    script = _preempt_render(n_trials=12)
    assert "N_REMAINING=$(python -m ddssm.launch_remaining" in script
    assert "--cleanup-running-older-than 60" in script
    assert "--target 12" in script


def test_render_preemptive_exits_early_if_remaining_is_zero() -> None:
    script = _preempt_render()
    assert 'if [ "$N_REMAINING" -le 0 ]' in script
    assert "exit 0" in script


def test_render_preemptive_computes_n_per_worker_with_ceiling_math() -> None:
    script = _preempt_render(n_workers=3)
    assert "N_PER_WORKER=$(( (N_REMAINING + 3 - 1) / 3 ))" in script


def test_render_preemptive_substitutes_n_per_worker_placeholder() -> None:
    script = _preempt_render()
    assert "hydra.sweeper.n_trials=$N_PER_WORKER" in script
    assert "__N_PER_WORKER__" not in script


def test_render_preemptive_emits_ddssm_invoc_export() -> None:
    script = _preempt_render()
    assert "DDSSM_INVOC=$(date +%s)" in script


def test_render_preemptive_emits_worker_id_and_preemptive_env_exports() -> None:
    script = _preempt_render(worker_idx=2)
    assert "export DDSSM_PREEMPTIVE=1" in script
    assert "DDSSM_WORKER_ID=2" in script


def test_render_preemptive_emits_trap_for_usr1_and_term() -> None:
    script = _preempt_render()
    trap_lines = [ln for ln in script.splitlines() if ln.startswith("trap ")]
    assert trap_lines, "expected a trap directive"
    trap_line = trap_lines[0]
    assert "USR1" in trap_line
    assert "TERM" in trap_line
    assert 'kill -USR1 "$PID"' in trap_line


def test_render_preemptive_runs_child_in_background_with_wait() -> None:
    script = _preempt_render()
    assert "python -m ddssm.app" in script
    # The child runs in the background and the script captures its PID.
    bg_lines = [ln for ln in script.splitlines() if ln.rstrip().endswith("&")]
    assert any("python -m ddssm.app" in ln for ln in bg_lines)
    assert "PID=$!" in script
    assert 'wait "$PID"' in script


def test_render_non_preemptive_unchanged_regression() -> None:
    """Default (non-preemptive) render: none of the preempt-mode artifacts leak in."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    jobs = _orch().render(pts)
    assert jobs, "expected at least one job"
    script = jobs[0][1]
    assert "--requeue" not in script
    assert "N_REMAINING=" not in script
    assert "DDSSM_INVOC=" not in script
    assert "DDSSM_PREEMPTIVE=1" not in script
    # No trap directives in the non-preemptive path.
    assert not any(ln.startswith("trap ") for ln in script.splitlines())


def test_render_preemptive_extra_flags_appear_after_signal() -> None:
    """SLURM is last-line-wins; the user's --signal must come AFTER the injected one."""
    script = _preempt_render(grace=90, extra_flags=("--signal=B:USR2@30",))
    injected_idx = script.index("#SBATCH --signal=B:USR1@90")
    user_idx = script.index("#SBATCH --signal=B:USR2@30")
    assert user_idx > injected_idx, (
        "user-supplied extra_flags must appear AFTER the preempt-injected directives"
    )


# --- run_local with the new strategies --------------------------------------


def test_run_local_rejects_optuna_multi_node(tmp_path) -> None:
    """``optuna_multi_node`` is sbatch-only; local execution must fail loudly."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(strategy="optuna_multi_node", sweep="x", n_workers=2)
    orch = _orch(storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"))
    with pytest.raises(ValueError, match="--local"):
        orch.run_local(pts, out_dir=str(tmp_path / "out"), launch_override=override)


def test_run_local_local_parallel_spawns_one_popen_per_worker(monkeypatch, tmp_path) -> None:
    """``local_parallel`` spawns ``n_workers`` Popens per point and waits for all."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(
        strategy="local_parallel", sweep="init_ablation_moo", n_trials=5, n_workers=3
    )
    orch = _orch(storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"))

    spawned: list[list[str]] = []

    class _FakePopen:
        def __init__(self, cmd, env=None, **kw):
            spawned.append(list(cmd))

        def wait(self):
            return 0

    monkeypatch.setattr("ddssm.launch.subprocess.Popen", _FakePopen)
    # subprocess.run should never be called on the multi-worker path.
    monkeypatch.setattr(
        "ddssm.launch.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("subprocess.run called on multi-worker path")),
    )

    rc = orch.run_local(pts, out_dir=str(tmp_path / "out"), launch_override=override)
    assert rc == 0
    assert len(spawned) == 3
    subdirs = sorted(
        next(a for a in cmd if a.startswith("hydra.sweep.subdir=")) for cmd in spawned
    )
    assert subdirs == [
        "hydra.sweep.subdir=w0_${hydra.job.num}",
        "hydra.sweep.subdir=w1_${hydra.job.num}",
        "hydra.sweep.subdir=w2_${hydra.job.num}",
    ]
    # Every worker invokes ddssm.app with --multirun and the shared study name.
    for cmd in spawned:
        assert "--multirun" in cmd
        assert any(a.startswith("hydra.sweeper.study_name=") for a in cmd)


def test_run_local_local_parallel_propagates_worker_failure(monkeypatch, tmp_path) -> None:
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(
        strategy="local_parallel", sweep="init_ablation_moo", n_workers=2
    )
    orch = _orch(storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"))

    class _FailingSecondPopen:
        n = 0

        def __init__(self, cmd, env=None, **kw):
            type(self).n += 1
            self.rc = 0 if type(self).n == 1 else 1

        def wait(self):
            return self.rc

    monkeypatch.setattr("ddssm.launch.subprocess.Popen", _FailingSecondPopen)
    rc = orch.run_local(
        pts, out_dir=str(tmp_path / "out"), launch_override=override,
    )
    assert rc == 1


# --- Orchestrator preempt wiring (ADR-0009 Phase 5) -------------------------
#
# Phase 4 added ``PreemptSpec`` + ``render_sbatch(..., preempt=...)``; Phase 5
# wires ``StudyOrchestrator.render`` and ``run_local`` to construct the spec,
# pass it through, and substitute the ``__N_PER_WORKER__`` placeholder per
# backend (shell var for sbatch, literal ceil(n/w) for local).


def _force_preemptive_multi_node(n_workers: int, n_trials: int, grace: int = 90):
    """Launch override emitting a preemptive optuna_multi_node PointLaunch."""

    def _o(point):
        return PointLaunch(
            strategy="optuna_multi_node",
            sweep="init_ablation_moo",
            n_trials=n_trials,
            n_workers=n_workers,
            preemptive=True,
            preempt_grace_seconds=grace,
        )

    return _o


def test_orchestrator_render_passes_preemptspec_to_render_sbatch() -> None:
    # Use the single-stage mock study: INIT_CENTERING points are multi-stage
    # (StagesB) and the orchestrator now rejects preemptive+multi-stage (ADR-0009).
    study = _single_stage_preempt_study(n_workers=3)
    orch = StudyOrchestrator(study, study_prefix="abl")
    jobs = orch.render(study.points)
    # Three workers per point.
    assert len(jobs) == 3
    for _, script in jobs:
        # PreemptSpec wired through to render_sbatch produces the directives.
        assert "#SBATCH --requeue" in script
        assert "#SBATCH --signal=B:USR1@90" in script
        # Preamble: launch_remaining CLI.
        assert "N_REMAINING=$(python -m ddssm.launch_remaining" in script
        # Placeholder substituted with the shell var.
        assert "hydra.sweeper.n_trials=$N_PER_WORKER" in script
        assert "__N_PER_WORKER__" not in script


def test_orchestrator_render_sets_worker_idx_into_preemptspec() -> None:
    study = _single_stage_preempt_study(n_workers=3)
    orch = StudyOrchestrator(study, study_prefix="abl")
    jobs = orch.render(study.points)
    # Each worker's script carries DDSSM_WORKER_ID=<idx> from PreemptSpec.worker_idx.
    assert len(jobs) == 3
    for idx, (_, script) in enumerate(jobs):
        assert f"DDSSM_WORKER_ID={idx}" in script


def test_orchestrator_render_threads_storage_and_study_into_preemptspec() -> None:
    study = _single_stage_preempt_study(n_workers=2)
    orch = StudyOrchestrator(
        study, storage_dir="STORE", sweeps_root="SWEEPS", study_prefix="abl",
    )
    jobs = orch.render(study.points)
    # The single mock point's name is ``mock_preempt_point``; study_prefix=abl.
    expected_study = "abl_mock_preempt_point"
    expected_storage = f"sqlite:///STORE/{expected_study}.db"
    for _, script in jobs:
        # The preamble's --storage and --study (from PreemptSpec) match the
        # strategy's hydra_overrides (load_study lookup must hit the SAME DB).
        assert f"--storage {expected_storage}" in script
        assert f"--study {expected_study}" in script
        # Sanity: the same study_name+storage appear in the hydra overrides.
        assert f"hydra.sweeper.study_name={expected_study}" in script
        assert f"hydra.sweeper.storage={expected_storage}" in script


def test_orchestrator_render_single_job_preemptive_raises() -> None:
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(strategy="single_job", preemptive=True)
    with pytest.raises(ValueError) as exc_info:
        _orch().render(pts, launch_override=override)
    msg = str(exc_info.value).lower()
    assert "single_job" in msg or "preemptive" in msg


def test_validate_preempt_compat_allows_multi_stage_now() -> None:
    """ADR-0009 update: multi-stage experiments now support preemptive=True.

    Stage-aware resume (via ``stage_prefix`` in the checkpoint payload + the
    StageOrchestrator's ``resume_from`` path) lifts the v1 multi-stage
    rejection. Rendering a preemptive multi-node sbatch against a real
    multi-stage INIT_CENTERING_STUDY point should now succeed.
    """
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(
        strategy="optuna_multi_node",
        sweep="init_ablation_moo",
        n_trials=12,
        n_workers=2,
        preemptive=True,
    )
    jobs = _orch().render(pts, launch_override=override)
    assert len(jobs) >= 1
    # Sanity: the preempt path emitted its SBATCH directives.
    for _, script in jobs:
        assert "#SBATCH --requeue" in script
        assert "hydra.sweeper.n_trials=$N_PER_WORKER" in script


def test_orchestrator_render_preemptive_accepts_single_stage() -> None:
    """Single-stage points (training.stages is None) must NOT be rejected."""
    study = _single_stage_preempt_study(n_workers=2)
    orch = StudyOrchestrator(study, study_prefix="abl")
    jobs = orch.render(study.points)
    assert len(jobs) >= 1
    # Sanity: the preempt path actually ran (directives present in the script).
    for _, script in jobs:
        assert "#SBATCH --requeue" in script
        assert "hydra.sweeper.n_trials=$N_PER_WORKER" in script


def test_orchestrator_run_local_preemptive_sets_env_and_literal_n_trials(
    monkeypatch, tmp_path,
) -> None:
    """Local path: substitute placeholder with literal ceil(n/w) + set env per Popen."""
    # Use the single-stage mock study and switch its launch to local_parallel.
    base = _single_stage_preempt_study(n_workers=3)
    study = _Study(
        name=base.name,
        points=base.points,
        launch=lambda p: PointLaunch(
            strategy="local_parallel",
            sweep="init_ablation_moo",
            n_trials=12,
            n_workers=3,
            preemptive=True,
        ),
    )
    orch = StudyOrchestrator(
        study, study_prefix="abl",
        storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"),
    )

    spawned: list[tuple[list[str], dict | None]] = []

    class _FakePopen:
        def __init__(self, cmd, env=None, **kw):
            spawned.append((list(cmd), dict(env) if env is not None else None))

        def wait(self):
            return 0

    monkeypatch.setattr("ddssm.launch.subprocess.Popen", _FakePopen)
    monkeypatch.setattr(
        "ddssm.launch.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("subprocess.run called on multi-worker path")),
    )

    rc = orch.run_local(study.points, out_dir=str(tmp_path / "out"))
    assert rc == 0
    assert len(spawned) == 3
    worker_ids = sorted(env["DDSSM_WORKER_ID"] for _, env in spawned)
    assert worker_ids == ["0", "1", "2"]
    for cmd, env in spawned:
        assert env is not None
        assert env.get("DDSSM_PREEMPTIVE") == "1"
        # ceil(12 / 3) == 4.
        assert "hydra.sweeper.n_trials=4" in cmd
        assert "hydra.sweeper.n_trials=__N_PER_WORKER__" not in cmd
        # Placeholder must not survive anywhere else in the cmd.
        assert not any("__N_PER_WORKER__" in a for a in cmd)


def test_orchestrator_run_local_non_preemptive_unchanged(monkeypatch, tmp_path) -> None:
    """Regression: without preemptive=True, no env mutation and no n_trials substitution."""
    pts = INIT_CENTERING_STUDY.select(cell="init_mlp_pinned_per_t", dataset="1d")
    override = lambda p: PointLaunch(
        strategy="local_parallel", sweep="init_ablation_moo", n_trials=6, n_workers=2,
    )
    orch = _orch(storage_dir=str(tmp_path / "o"), sweeps_root=str(tmp_path / "s"))

    spawned: list[tuple[list[str], dict | None]] = []

    class _FakePopen:
        def __init__(self, cmd, env=None, **kw):
            spawned.append((list(cmd), dict(env) if env is not None else None))

        def wait(self):
            return 0

    monkeypatch.setattr("ddssm.launch.subprocess.Popen", _FakePopen)
    monkeypatch.setattr(
        "ddssm.launch.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("subprocess.run called on multi-worker path")),
    )

    rc = orch.run_local(pts, out_dir=str(tmp_path / "out"), launch_override=override)
    assert rc == 0
    assert len(spawned) == 2
    for cmd, env in spawned:
        # No preempt env: either env was not passed (None) or, if passed, lacks
        # the preempt keys.
        if env is not None:
            assert "DDSSM_PREEMPTIVE" not in env
            assert "DDSSM_WORKER_ID" not in env
        # Literal n_trials, no placeholder.
        assert "hydra.sweeper.n_trials=6" in cmd
        assert not any("__N_PER_WORKER__" in a for a in cmd)


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

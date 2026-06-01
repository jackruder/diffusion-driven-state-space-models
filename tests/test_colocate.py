"""Tests for the co-located multi-cell launcher (render_colocated + the
``render_multicell_packed_sbatch`` renderer + the ``ddssm.colocate`` CLI).

The shape under test: ONE sbatch per GPU runs EVERY selected cell with a few
workers apiece on that GPU's single card. Each cell is its own Optuna study (in
the shared DB); the same cell also runs on the other GPUs, so its concurrency is
``workers_per_cell × n_gpus`` and it draws from one shared study.
"""

from __future__ import annotations

import pytest

from ddssm.launch import StudyOrchestrator
from ddssm.cluster.sbatch import SBatch, CellWorker, render_multicell_packed_sbatch
from experiments.init_centering.study import INIT_CENTERING_STUDY

# Two concrete cells (1d) to co-locate in the tests.
_PTS = INIT_CENTERING_STUDY.select(
    cell="init_mlp_pinned_per_t", dataset="1d"
) + INIT_CENTERING_STUDY.select(cell="init_linear_pinned_per_t", dataset="1d")
_RES = SBatch(
    partition="gpuunsafe",
    gpus=1,
    cpus=32,
    mem="80G",
    extra_flags=("--gres=gpu:b6000:1", "--account=group-x"),
)


def _orch():
    return StudyOrchestrator(
        INIT_CENTERING_STUDY, study_prefix="r2", storage_url="postgresql://h/db"
    )


def _render(**kw):
    base = dict(
        n_gpus=2,
        workers_per_cell_per_gpu=2,
        target=96,
        sweep="init_ablation_moo_r2",
        resources=_RES,
    )
    base.update(kw)
    return _orch().render_colocated(_PTS, **base)


# --- shape: one job per GPU, all cells on each ------------------------------


def test_one_sbatch_per_gpu_co_locates_every_cell() -> None:
    jobs = _render(preemptive=False)
    assert [j for j, _ in jobs] == ["r2_colo_g0", "r2_colo_g1"]
    g0 = dict(jobs)["r2_colo_g0"]
    # Both cells, twice each (workers_per_cell_per_gpu=2) -> 4 packed procs.
    assert g0.count("experiment=init_mlp_pinned_per_t__1d") == 2
    assert g0.count("experiment=init_linear_pinned_per_t__1d") == 2
    assert g0.count("PIDS+=($!)") == 4
    assert "--gres=gpu:b6000:1" in g0 and "--cpus-per-task=32" in g0
    assert "#SBATCH --job-name=ddssm-r2_colo_g0" in g0


def test_cpu_pinned_to_share_across_all_packed_procs() -> None:
    # 2 cells x 2 workers = 4 procs on 32 CPU -> 8 threads each.
    g0 = dict(_render(preemptive=False))["r2_colo_g0"]
    assert "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8" in g0


def test_cell_concurrency_spans_gpus_with_distinct_worker_idx() -> None:
    """A cell's per-GPU worker indices are cell-global: GPU0 -> {0,1}, GPU1 ->
    {2,3}, so the SAME cell's ``hydra.sweep.subdir`` never collide across its GPU
    jobs (they share one ``hydra.sweep.dir``).
    """
    jobs = dict(_render(preemptive=True))
    g0, g1 = jobs["r2_colo_g0"], jobs["r2_colo_g1"]
    assert "hydra.sweep.subdir=w0_" in g0 and "hydra.sweep.subdir=w1_" in g0
    assert "hydra.sweep.subdir=w2_" in g1 and "hydra.sweep.subdir=w3_" in g1


def test_same_cell_shares_one_study_across_gpus() -> None:
    jobs = dict(_render(preemptive=True))
    for g in ("r2_colo_g0", "r2_colo_g1"):
        assert (
            jobs[g].count("hydra.sweeper.study_name=r2_init_mlp_pinned_per_t__1d") == 2
        )
        assert "hydra.sweeper.storage=postgresql://h/db" in jobs[g]


# --- preempt: per-cell launch_remaining + budget ----------------------------


def test_preempt_runs_launch_remaining_once_per_cell() -> None:
    g0 = dict(_render(preemptive=True, grace_seconds=120))["r2_colo_g0"]
    assert "#SBATCH --requeue" in g0
    assert "#SBATCH --signal=B:USR1@120" in g0
    # One launch_remaining per cell (2 cells), each its own study + NPW var.
    assert g0.count("python -m ddssm.launch_remaining") == 2
    assert "--study r2_init_mlp_pinned_per_t__1d" in g0
    assert "--study r2_init_linear_pinned_per_t__1d" in g0
    # n_workers = n_gpus * workers_per_cell_per_gpu = 4 (ceiling division).
    assert "NPW_0=$(( (N_REMAINING_0 + 4 - 1) / 4 ))" in g0
    assert "NPW_1=$(( (N_REMAINING_1 + 4 - 1) / 4 ))" in g0


def test_preempt_each_worker_reads_its_cells_npw_and_is_guarded() -> None:
    g0 = dict(_render(preemptive=True))["r2_colo_g0"]
    assert "hydra.sweeper.n_trials=$NPW_0" in g0
    assert "hydra.sweeper.n_trials=$NPW_1" in g0
    # A cell's worker is only launched while that cell still owes trials.
    assert 'if [ "$NPW_0" -gt 0 ]; then' in g0
    assert 'if [ "$NPW_1" -gt 0 ]; then' in g0
    # Shared preempt env + fan-out trap.
    assert "export DDSSM_PREEMPTIVE=1" in g0
    assert 'trap \'for _p in "${PIDS[@]}"; do kill -USR1' in g0


def test_distinct_sampler_seed_per_worker_no_placeholder_leak() -> None:
    g0 = dict(_render(preemptive=True))["r2_colo_g0"]
    for w in (0, 1):
        assert (
            f"hydra.sweeper.sampler.seed=$(( (SLURM_JOB_ID * 100 + {w}) % 2000000000 ))"
            in g0
        )
    assert "__SAMPLER_SEED__" not in g0
    assert "__N_PER_WORKER__" not in g0


# --- renderer guardrails ----------------------------------------------------


def test_render_multicell_empty_raises() -> None:
    with pytest.raises(ValueError):
        render_multicell_packed_sbatch([], spec=_RES, output_pattern="x")


def test_render_multicell_strict_bash_pipefail() -> None:
    cw = CellWorker(
        experiment="e",
        cell_key="c",
        worker_idx=0,
        overrides=["--multirun", "hydra.sweeper.storage=sqlite:///x.db"],
    )
    script = render_multicell_packed_sbatch(
        [cw], spec=SBatch(job_name="ddssm-x"), output_pattern="o"
    )
    assert "set -euo pipefail" in script


# --- CLI --------------------------------------------------------------------


def test_cli_dry_run_renders_per_gpu_scripts(capsys) -> None:
    from ddssm import colocate as C

    rc = C.main([
        "init_centering",
        "--select",
        "cell=init_mlp_pinned_per_t",
        "dataset=1d",
        "--n-gpus",
        "2",
        "--workers-per-cell",
        "2",
        "--target",
        "96",
        "--storage-url",
        "postgresql://h/db",
        "--study-prefix",
        "r2",
        "--no-preempt",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "r2_colo_g0" in out and "r2_colo_g1" in out
    assert "experiment=init_mlp_pinned_per_t__1d" in out


def test_cli_submit_requires_write_dir() -> None:
    from ddssm import colocate as C

    with pytest.raises(SystemExit):
        C.main([
            "init_centering",
            "--n-gpus",
            "1",
            "--target",
            "8",
            "--storage-url",
            "postgresql://h/db",
            "--study-prefix",
            "r2",
            "--submit",
        ])


def test_cli_bad_resources_from_errors() -> None:
    from ddssm import colocate as C

    with pytest.raises(SystemExit):
        C.main([
            "init_centering",
            "--select",
            "cell=init_mlp_pinned_per_t",
            "dataset=1d",
            "--n-gpus",
            "1",
            "--target",
            "8",
            "--storage-url",
            "postgresql://h/db",
            "--study-prefix",
            "r2",
            "--resources-from",
            "not_a_real_cell",
            "--dry-run",
        ])

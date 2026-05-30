---
name: plan-campaign
description: Plan a multi-cell / multi-preset experiment campaign for effective resource utilization. Reads serialized hardware profiles (`compute-resources`) and per-cell cost functions (`compute-budget`), then produces an allocation plan — which cells run where, with what concurrency, in what order — and a runnable launch script (bash multi-cell, SLURM array, or hybrid). Optimises for wallclock under deadline, then for information-per-trial. Catches multi-cell footguns (NFS Optuna visibility, SQLite contention, OOM cascades, preempt-window vs checkpoint cadence). Use when the user has a multi-cell ablation grid, a multi-seed replication, a multi-preset comparison, or says "plan the overnight", "schedule this campaign", "allocate the cluster", or invokes /plan-campaign.
---

<role>
You are the *multi-experiment scheduler*. Where `compute-budget` plans one experiment, you plan a *campaign* — a set of experiments / cells / seeds that should run together to make best use of the resource pool by a deadline. Inputs: serialized hardware profiles, a cost function per cell, a goal (deadline / priority / partial-results-OK). Output: an allocation plan + a launch script.

You do not invent new experiments. You orchestrate ones the user has already specified or built.
</role>

<inputs>

You need three things. Ask only what's missing:

1. **The campaign** — one of:
   - A multi-cell preset with K-axis grid (`baseline_form: "<ablation across {...}>"` in the spec) → enumerate cells via the family's `iter_cells()` helper or by reading the spec.
   - A list of distinct presets to run together ("`init_smoke_high_surface` plus `kdd_diffusion` with 5 seeds each").
   - A reference to a `Study` declaration that defines the campaign shape (e.g. `experiments/init_centering/study.py:INIT_CENTERING_STUDY`, launched via `python -m ddssm.launch <study> --size tiny ...`).
2. **Resource profile(s)** — load from `.claude/resources.yaml` (see `compute-resources` skill). If multiple profiles are usable, ask which to pool, or whether to allocate heterogeneously.
3. **Goal** — deadline (timestamp or duration), priority hint, partial-results policy:
   - Default: minimise total wallclock.
   - Alternative: maximise information gain (cell-priority-weighted) under a hard deadline.
   - Alternative: stage-1 (high-priority cells) gets results by deadline A; rest as bonus.

</inputs>

<phase-1-enumerate>

## Phase 1 — Enumerate the campaign

Produce a flat list of *units of work*. A unit is one (preset, sweep_config, n_trials, seed_set) tuple. Examples:

- K-axis cell grid → N_cells units, each with the same sweep + n_trials.
- N-seed replication → N units of the same preset with different `seed=`.
- Multi-preset comparison → one unit per preset, potentially with different sweeps.

For each unit capture:

```yaml
unit:
  name: <str>                  # used as suffix for run dirs, sbatch names, Optuna study names
  preset: experiment=<name>
  sweep: +sweep=<name>         # optional; null for single-trial units
  trials_planned: <int>
  seeds: <list of ints>        # default: [0]
  priority: <high | medium | low>   # ask the user if non-uniform
  prior_cost: <ref to compute-budget output or "unmeasured">
```

If multi-cell with shared cost characteristics (same preset, just varied cell axes), measure the cost once and reuse. If the units use distinct presets with different costs, you need one measurement per preset — confirm before running multiple benchmarks.

</phase-1-enumerate>

<phase-2-measure>

## Phase 2 — Cost per unit

For each distinct cost class in the campaign, get a per-trial cost. Three sources, same hierarchy as `compute-budget`:

- Parse the most recent run of that preset (`runs/.../metrics.csv`).
- Run a fresh micro-benchmark (see `compute-budget` Phase 2 for the override pattern). Ask before launching multiple benchmarks back-to-back.
- Bound from architecture as a last resort, with `confidence: low`.

For cell grids with parametric axes that affect cost (e.g. `num_diffusion_steps` is in the sweep), the cost function is the same as `compute-budget`'s — parameterise over the tuned axes rather than collapsing.

Build a per-unit aggregate:

```
unit_cost_total_s = trials_planned × E[trial_wallclock_s] × len(seeds)
campaign_cost_total_s = sum(unit_cost_total_s for unit in units)
```

State the assumptions (which seeds, eval overhead factor, etc.) inline.

</phase-2-measure>

<phase-3-allocate>

## Phase 3 — Allocation policy

Decide three things:

### A. Strategy

Match the strategy to the resource shape:

| Resources                              | Strategy                                                                                                                     |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Single device, exclusive               | **In-process workers + cells sequential.** Mirror `scripts/run_overnight_mv_ablation.sh`. One bash driver, one SQLite per cell. |
| Single device, shared / preemptible    | **Same as above, with reduced `workers_per_device`** (memory headroom) and tightened `checkpoint_every` (< preempt window).   |
| Multi-device single node               | **Cells parallel across devices, workers within each cell.** One bash driver per device, or use CUDA_VISIBLE_DEVICES routing. |
| SLURM single partition                 | **SLURM array, one cell per array index.** N_WORKERS within each via independent processes sharing the cell's SQLite.        |
| SLURM multi-partition heterogeneous    | **Per-cell partition assignment.** High-cost cells → fastest partition; low-cost → cheapest.                                  |
| Cluster, multi-node Optuna             | **NFS-shared Optuna DB,** one DB per cell; workers across nodes coordinate via DB. Set SQLite WAL + busy_timeout high.        |

Pick exactly one primary strategy and name it. State the rationale.

### B. Concurrency

Per device pool:

```
workers_per_device = floor(usable_mem × headroom_factor / trial_peak_mem_gb)
```

`headroom_factor` defaults to 0.75 on exclusive devices, 0.50 on shared devices (other tenants).

Cap at 6 unless the user insists — diminishing returns above that from SQLite contention and kernel-launch overhead. **Always use independent processes sharing SQLite**, never Optuna `n_jobs > 1` (joblib-GIL → ~32% GPU util loss; `scripts/run_overnight_mv_ablation.sh` comment #1).

For multi-node SLURM with NFS-shared DB: cap workers per DB at ~8; beyond that, NFS journaling latency starts to dominate. If you need more, shard cells into multiple DBs.

### C. Ordering

Default: priority descending, then cost ascending (cheap cells first so partial results land soon).

Variants the user might want:

- **Information-greedy.** Cells where Optuna's current best is least confident go first.
- **Risk-greedy.** Cells most likely to OOM or fail go first, so failures surface early in the budget rather than at the end.
- **Sequential as declared.** If the user's launcher already defines an order, respect it.

</phase-3-allocate>

<phase-4-fit>

## Phase 4 — Fit to deadline + dial twists

```
parallel_slots = devices_in_pool × workers_per_device
campaign_wallclock_s = ceil(sum(unit_cost_total_s) / parallel_slots) × overhead_factor
overhead_factor = 1.10 (single-device exclusive) | 1.20 (SLURM queue + preempt) | 1.30 (multi-node NFS)
```

If `campaign_wallclock_s > deadline_s`, surface a **quantitative** dial-twist menu. Each option lists the saving in hours, ranked by impact:

```
Over budget by 6h (planned 28h, deadline 22h). Options ranked by impact:

  drop high_surface_smoke (lowest priority)        saves ~3.5h
  cap trials_per_cell 60 → 40                      saves ~9.3h (sweep floor was 30)
  drop a cell-grid axis (3-way → 2-way × 6 cells)  saves ~14h (biggest information loss)
  bump headroom_factor 0.75 → 0.85                 saves ~1.8h (closer to OOM)
  add 1 more H100 to the pool                      saves ~7h (if available)
  flip non-priority cells to AMP                   saves ~5h (needs a smoke first)
```

Let the user choose. Don't auto-cut, especially for priority cells.

If `slack_s` is generous (> 25% of deadline): suggest *strengthening* — more trials, more seeds, a wider sweep range — rather than just finishing early. Surface this as a positive option, not silently absorb the slack.

</phase-4-fit>

<phase-5-materialise>

## Phase 5 — Materialise the launch

Write the campaign plan to `runs/campaigns/<stamp>/plan.yaml` (or wherever the active profile's `runs_dir` points). Generate one of:

### Bash multi-cell driver (single-machine, multi-device)

Mirror `scripts/run_overnight_mv_ablation.sh`:

- Pre-render per-cell sbatch / shell snippets via `python -m ddssm.launch <study> --write-dir ...` (the family's `Study` declares per-point `PointLaunch` intent — ADR-0008).
- Loop over cells; per cell, launch N_WORKERS independent processes sharing one SQLite DB.
- Pre-touch SQLite with `PRAGMA journal_mode=WAL;`.
- Override `hydra.sweep.subdir='w${oc.env:HYDRA_WORKER_ID}_${hydra.job.num}'`.
- Aggregate at the end (`python -m experiments.<family>.report ...`).

### SLURM array (cluster)

- One array submission per *cost class* (or per cell if costs differ).
- `sbatch --array=0-<N> --partition=<from profile> --time=<unit_wallclock + 20%> ...`.
- Each array task targets one cell; workers within share a per-cell SQLite (or NFS-shared if multi-node).
- Per-task `runs_dir` = `<profile.storage.runs_dir>/<campaign>/cell_${SLURM_ARRAY_TASK_ID}/`.
- Submission produces a job ID; record it in the plan.
- Add a `--dependency=afterok:<id>` aggregation step that runs `python -m experiments.<family>.report ...` once all array tasks finish.

### Hybrid

- High-priority cells go to the fast / exclusive resource as bash; low-priority cells go to SLURM queue.
- Both write to the same `runs/campaigns/<stamp>/` tree so aggregation sees everything.

In all cases, generate:

- A `manifest.yaml` listing every cell's preset, sweep, study name, SQLite path, and expected wallclock.
- A `monitor.sh` one-liner: `tail -F runs/campaigns/<stamp>/*/metrics.csv` (or `squeue -u $USER` for SLURM).
- An `aggregate.sh` post-step.

</phase-5-materialise>

<guards>

Before producing the launch:

- [ ] **All cost functions sourced.** If any unit's cost is `unmeasured`, run a benchmark (with confirmation) or document the assumption in `risks:`.
- [ ] **Storage room.** `sum(unit_cost_total_s) × per_trial_storage_estimate` against the active profile's `runs_dir` free space.
- [ ] **NFS visibility.** If multi-node, the Optuna DB path must be on a shared FS (`profile.storage.shared_fs: true`).
- [ ] **SQLite contention.** Workers-per-DB ≤ 6 (local) or ≤ 8 (NFS). If exceeded, shard cells across DBs.
- [ ] **Preempt window.** For preemptible profiles, `checkpoint_every` × `per_step_s` < `profile.slurm.preempt_window_s × 0.5`. Halve gives margin for resume to complete before the next preempt.
- [ ] **Array limits.** `n_cells ≤ profile.slurm.array_limit` per submission. If exceeded, split into multiple arrays.
- [ ] **σ_pert > 0**, **stage-budget shadow**, **no in-process `n_jobs > 1`**, **per-worker subdir prefix** — inherit from `compute-budget` guards. These apply campaign-wide.
- [ ] **Smoke pass.** If the campaign would consume > 5 GPU-hours and no smoke variant of the dominant preset has been run recently, recommend a smoke first. Push back gently.
- [ ] **OOM isolation.** A single OOM should not crash the campaign. For bash driver: trap `set -e` so a failing cell logs and continues. For SLURM array: array tasks are independent by default — confirm `--array` not `--dependency`.

</guards>

<output>

End with:

```yaml
campaign:
  name: <stamp>
  goal: <user-stated>
  deadline: <abs-time or duration>
  resource_profile: <name>
  total_units: <int>
  estimated_wallclock_h: <float>
  estimated_compute_h: <float>            # serial-equivalent
  fits_deadline: <yes | no | tight>
  slack_h: <float>

strategy: <one of the named strategies>

units:
  - name: <str>
    preset: <str>
    sweep: <str or null>
    trials_planned: <int>
    seeds: <list>
    priority: <high | medium | low>
    cost_per_trial_s: <float>
    device_pool: <profile_name>
    workers: <int>
    runs_dir: <path>
    optuna_db: <path>
    checkpoint_every: <int>
    launch: <bash snippet | sbatch line>

risks:
  - <bullet>

monitor:
  command: <one-liner>
  expected_first_result_at: <duration after launch>

post_run:
  aggregate: <command>
  expected_artifacts: [<paths>]
```

Then the **launch entry point** as a concrete command (`bash runs/campaigns/<stamp>/run.sh`, or `sbatch ...`, or a list of sbatch commands). Set env vars explicitly. Confirm with the user before any `sbatch` submission or `nohup` launch — these consume budget the moment they execute.

</output>

<conventions>

- **Use `python -m ddssm.launch <study>`** to render per-point scripts from the family's `Study` declaration. Don't reinvent sbatch generation; if the resource shape doesn't fit any existing `LaunchStrategy`, extend `src/ddssm/launch.py` rather than writing a per-family script.
- **Aggregation belongs in the plan.** Every campaign must have an `aggregate.sh` or equivalent — the user shouldn't have to ask how to read the output.
- **Status surface.** State where the user can see progress without interrupting the campaign — `tail -F <log>`, `sqlite3 <db> 'select count(*) from trials'`, `squeue -u $USER`. Pick the one that fits the strategy.
- **Recoverable shape.** Per-cell SQLite DBs + per-cell `runs_dir` means a partial campaign can be resumed by re-submitting only the cells that didn't finish. Generate a `resume.sh` template alongside `run.sh` when the campaign is large enough that resume is plausible.
- **State all your sources.** Each cost number cites where it came from (measured / extrapolated / bounded). Each strategy choice cites the row of the strategy table that justified it.

</conventions>

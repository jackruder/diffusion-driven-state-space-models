---
name: compute-budget
description: Plan how to run a DDSSM experiment by (a) prompting for available resources, (b) empirically measuring the unit cost via a short benchmark, (c) building a cost function over the experiment's tuned dimensions, and (d) producing a launch command sized to the user's wallclock and compute budget. Scales from tiny smokes on a laptop CPU to massive multi-GPU sweeps. Catches project footguns (stage-budget shadow, joblib-GIL, subdir collisions, σ_pert protocol). Use when the user has a preset (or spec) ready and asks "how should I run this", "size this sweep", "how long will the overnight take", "fit this into N GPU-hours", or invokes /compute-budget.
---

<role>
Plan the launch. You sit downstream of `select-hyperparameters` and `experiment-builder`. Your output is a budget block + a concrete `nohup` / `sbatch` / bare `python -m ddssm.app ...` command. You modify no preset files.

The philosophy is: **measure, don't assume.** Hardware and per-step cost are inputs the user supplies or that you empirically determine — never defaulted. The skill scales naturally from a 60-second CPU smoke run to a multi-day multi-GPU campaign; the right shape of the answer is different at each scale.
</role>

<inputs>

You expect one of:

1. A **preset name** (`experiment=<name>`) — the strong case. You can run the actual code.
2. A **spec YAML** (per `.claude/spec-schema.md`) — the weak case. You can only estimate dimensions and HP counts; you cannot measure per-step cost until the preset exists. If only a spec is given, run `experiment-builder` first or tell the user you'll deliver a structural plan only, with cost left as `unmeasured`.

If neither is supplied, ask.

</inputs>

<phase-1-resources>

## Phase 1 — Resource intake

**Try the resource file first.** Check for `.claude/resources.yaml` (project-local) then `~/.claude/resources.yaml` (user-global). If a profile exists:

- Load `defaults.active_profile` and echo its `description` + key fields (devices, memory, availability).
- Ask the user to confirm or pick a different profile from the registered list.
- If confirmed, skip to *deadline / cost / degradations* below.

**If no profile is registered**, prompt for hardware and offer to persist it via the `compute-resources` skill (so future invocations don't re-ask). Do not assume defaults. Cover:

1. **Compute device.** Free-form. Examples the user might give: "RTX 4090 24 GB, 1 GPU", "A100 80 GB × 4", "H100 cluster, partition `gpu_h100`, can request up to 8", "CPU only, M2 Max 32 GB", "TPU v3-8", "I don't know, what's on this box?". If the user is unsure, offer to run `nvidia-smi --query-gpu=name,memory.total --format=csv` (and `nproc` / `free -h` for CPU-only).

2. **Device availability.** Exclusive / shared / time-sliced / SLURM-queued / preemptible. This changes the parallelism strategy fundamentally — preemptible needs frequent checkpointing; SLURM needs queue-time budget on top of wallclock; shared needs memory headroom for the other tenants.

**Always prompt for these, even when a profile is loaded:**

3. **Wallclock budget.** Free-form: a duration ("3h", "overnight", "by Friday morning"), a compute-hour cap ("≤ 20 GPU-hours"), or "no constraint — tell me what's reasonable". If the deadline is a wall-time, capture the current timestamp so you can compute the slack.

4. **Cost budget**, if relevant. Cloud users may have a hard $ ceiling; cluster users may have GPU-hour quotas. Ask only if the deadline/cost framing was ambiguous.

5. **Allowed degradations.** What can be cut if the budget is tight: trial count, sweep dimensions, ablation axes, AMP, batch size, num_diffusion_steps. The user's answer here defines the slope of the dial-twist menu in Phase 4.

Echo back what you captured before measuring. The user's answers here are inputs to the cost function, not hidden assumptions.

</phase-1-resources>

<phase-2-measure>

## Phase 2 — Measure the unit cost

Determine per-step wallclock and peak memory empirically. Three sources, ranked by accuracy:

### A. Parse a recent run (best)

Look in `runs/` for the most recent run of this preset (or a close variant). For a candidate run:

- Parse `metrics.csv` — column `step` + timestamp give `per_step_s = (last_t - first_t) / (last_step - first_step)`. Use the steady-state portion (drop the first ~10% of steps to skip warmup).
- For peak memory: prefer `submitit/<job>/<job>.out` if SLURM, which logs `MaxRSS`/`MaxVMSize`. Otherwise look for an explicit `torch.cuda.max_memory_allocated()` log line, or a `nvidia-smi --query-gpu=memory.used` dump.
- Report the source: `runs/<dirname>/metrics.csv (N=<steps>)`.

### B. Run a fresh micro-benchmark (when no prior run exists)

Offer to launch a short benchmark with the actual preset and minimal step counts. Use family-appropriate overrides; for multi-stage presets, set both stages tiny. Example shape:

```
python -m ddssm.app experiment=<name> \
    experiment.model.stages.n_pretrain=20 \
    experiment.model.stages.n_stage2=20 \
    experiment.training.log_every=1 \
    hydra.run.dir=runs/bench_<ts>
```

(Adapt to single-stage with `experiment.training.steps=40` instead.)

Then parse the resulting `metrics.csv` exactly as in (A). Run length is a tradeoff: too short and CUDA kernel autotuning skews the throughput; too long and you've burned budget benchmarking. Default 40-80 steps; bump to 200+ for distributed or AMP-toggled regimes where the steady state takes longer to reach.

Always confirm before launching the benchmark — it consumes the user's compute.

### C. Bound from architecture (last resort)

Only if no run can be done (spec-only mode, or hardware doesn't exist yet). Express the estimate as a range with explicit dependencies, not a single number — e.g. "throughput scales as `1 / num_diffusion_steps / residual_layers`; expect 2-5× the cost of the closest existing preset". Mark `confidence: low` in the output.

### What to record

```
unit_cost:
  per_step_s: <float>            # steady-state mean
  per_step_s_stdev: <float>      # over the measurement window
  peak_mem_gb: <float>           # torch.cuda.max_memory_allocated or sacct MaxRSS
  source: measured | extrapolated | bounded
  source_detail: <str>           # which run, or "bench 40 steps on RTX 4090"
```

</phase-2-measure>

<phase-3-cost-function>

## Phase 3 — Build the cost function

The output of this phase is not one number — it is an *expression* over the experiment's structural axes. Compute the dependence on each axis the user is varying, so the dial-twist menu in Phase 4 is quantitative.

### Per-trial cost

```
trial_steps =
  multi_stage:  sum(stage.steps for stage in stages)
  single_stage: training.steps
trial_wallclock_s = trial_steps × per_step_s × eval_overhead_factor
```

`eval_overhead_factor` is empirical from the measured run; if not measured, use a placeholder `× 1.0–1.2` and flag it.

**Footgun guard.** If the preset is multi-stage, `experiment.training.steps` is shadowed by `model.stages.n_<stage>` and is **not** the wallclock-relevant budget. Compute from the stage counts. (CLAUDE.md § Trainer; `scripts/run_overnight_mv_ablation.sh` comment #3.)

### Surface across tuned HPs

If the spec's `hyperparameters.tuned` includes any HP that scales the per-step or per-trial cost, parameterise the cost function rather than collapsing to a point estimate. Common ones:

- `num_diffusion_steps` (linear in per-step cost when it's tuned)
- `residual_channels`, `residual_layers` (roughly linear → quadratic in per-step cost)
- `n_pretrain`, `n_stage2` (linear in trial step count)
- `batch_size` (sub-linear in per-step cost; linear in memory)
- `S` (number of latent samples — linear)

Report as:

```
cost_function:
  trial_wallclock_s: per_step_s × steps_total(n_pretrain, n_stage2) × eval_overhead
    where:
      per_step_s ≈ 0.062 × (num_diffusion_steps / 128) × (residual_layers / 4)
      steps_total = n_pretrain + n_stage2
  trial_peak_mem_gb: 3.5 + 1.1 × log2(latent_dim / 4)     # if measured-fit
```

If most HPs are fixed, the function collapses to a constant — that's fine, just say so.

### Aggregate cost

```
sweep_shape:
  cells:    <int>                              # K-axis grid; 1 if not a grid
  trials_per_cell: <int>                       # Optuna trials
  seeds:    <int>                              # multi-seed replication; 1 if not
total_trials = cells × trials_per_cell × seeds
total_compute_s = total_trials × trial_wallclock_s     # serial-equivalent
```

**Sweep sizing check.** Compare `trials_per_cell` to the dimension of the tuned space. Rules of thumb (state them as such, not facts):

- TPE single-objective: ~20-30 trials per dim as a *plateau* floor. Below that, treat as exploratory.
- NSGA-II multi-objective: needs population × generations on the order of K × 50.
- Categorical-heavy: scale with the product of category counts.

If `trials_per_cell` falls below the floor, surface it as a flag (not a hard reject — exploratory passes are legitimate).

</phase-3-cost-function>

<phase-4-fit>

## Phase 4 — Fit to the resource envelope

### Parallelism strategy

Three independent dimensions of parallelism — decide which apply for the user's hardware:

1. **Workers per device.** `workers_per_device = floor(usable_mem × 0.75 / trial_peak_mem_gb)`. `usable_mem` is the user-supplied device memory (Phase 1) minus any reserved-for-others share if shared. Cap at 6 unless the user insists — diminishing returns from SQLite contention + kernel launch overhead. **Always use independent processes sharing a SQLite Optuna store, never `n_jobs > 1` inside one Optuna process** (joblib-threaded → GIL-bound → observed ~32% GPU util regression; documented in `scripts/run_overnight_mv_ablation.sh` comment #1).

2. **Cells per device.** Default 1 — cells run sequentially on a single device. Concurrent cells × concurrent workers split the device and starve each other. Exception: if `workers_per_device` is already 1, up to ~2 cells in parallel is OK.

3. **Devices in parallel.** For multi-GPU / cluster: one cell per device (or one trial-replica per device for DDP within a trial). For SLURM, that's one job per cell with `gpus_per_node=1`. For DDP within a trial: only worth it for very large models where per-trial wallclock dominates total wallclock.

```
parallel_slots = devices_in_parallel × workers_per_device × cells_in_parallel
```

### Wallclock + slack

```
overhead_factor = 1.10–1.20    # warmup, checkpointing, SQLite contention, scheduler queue time
total_wallclock_s = ceil(total_trials / parallel_slots) × trial_wallclock_s × overhead_factor
slack = deadline_s - total_wallclock_s
```

If `slack < 0`, you're over budget. Surface a **dial-twist menu** with quantitative deltas, ranked by impact. For each option, show the new cost. Example:

```
Over budget by 4.2h (planned 14.2h, budget 10h). Options ranked by impact:

  drop residual_channels 128 → 64    saves ~5.1h (per-step cost halves)
  drop trials_per_cell 60 → 40       saves ~4.7h (1/3 cut; exploratory floor was 20)
  enable AMP                          saves ~3.5h on Ampere+; needs a smoke first
  drop n_stage2 4000 → 3000          saves ~3.5h (and likely loses convergence)
  drop a cell-grid axis (3-way)      saves ~9.5h (most impact; biggest information loss)
```

Let the user pick. Don't auto-cut.

### Scale-appropriate output

The right shape of the launch command varies by scale. Match it:

- **Tiny (single trial, <5 min, CPU or one GPU):** bare `python -m ddssm.app experiment=<name>` — no nohup, no SLURM, no SQLite, no subdir overrides. The user can watch it in the terminal.
- **Medium (sweep, hours, one GPU):** `nohup ... &` with explicit env vars; mirror `scripts/run_overnight_mv_ablation.sh` shape when the preset is grid-shaped, otherwise a single-line `--multirun` command.
- **Large (multi-GPU or multi-node, days):** SLURM `sbatch` array; one cell per array index; resume-on-preempt; periodic checkpointing well below the queue's preempt window. State the partition + array shape + `timeout_min` from `src/ddssm/conf/hydra/launcher/submitit_slurm.yaml`.
- **Massive (cluster-scale):** beyond the project's tested shape. Surface this and ask the user whether they want a sketch or a fully-engineered plan; the latter may need work outside this skill (NFS-shared Optuna DB, distributed schedulers, dynamic resource provisioning).

</phase-4-fit>

<guards>

Before producing a launch command, walk:

- [ ] **Stage-budget shadow.** Confirm multi-stage presets compute total from `model.stages.n_<stage>`, not `training.steps`.
- [ ] **σ_pert > 0** (ADR-0002). If the sweep tunes `sigma_pert`, lower bound must be > 0. If fixed, value must be > 0. Refuse to launch otherwise; point at ADR-0002.
- [ ] **Parametric μ_p needs stage-1 budget.** If `baseline_form ∈ {linear, mlp, ...parametric...}`, `n_pretrain` ≥ a few hundred steps (`project-handoff-protocol-invariants` memory).
- [ ] **No in-process `n_jobs > 1`.** Reject plans that use Optuna joblib threading.
- [ ] **Per-worker subdir prefix.** If N_WORKERS > 1, the command must override `hydra.sweep.subdir` with a per-worker token (e.g. `w${oc.env:HYDRA_WORKER_ID}_${hydra.job.num}`).
- [ ] **WAL on SQLite.** If multiple workers share one Optuna DB, pre-touch with `PRAGMA journal_mode=WAL;`.
- [ ] **Smoke gate.** If the user is about to spend > 1 GPU-hour without having run a smoke variant of this preset recently, recommend a smoke first. (Calibrates the cost function and catches OOMs before they bite at scale.) Push back gently.
- [ ] **Checkpoint cadence.** For runs longer than 30 min on preemptible hardware, `checkpoint_every` must be < expected preempt window. State the assumption.
- [ ] **Disk room.** `total_trials × (metrics.csv + checkpoint footprint)`. Estimate from `checkpoint_every` and an existing checkpoint's size; flag if `runs/` partition is close to full.

</guards>

<output>

End with a single YAML block + a launch command. Shape:

```yaml
resources:
  device: <user-supplied>
  device_mem_gb: <user-supplied>
  devices_available: <user-supplied>
  availability: <exclusive | shared | preemptible | slurm-queued>
  deadline: <user-supplied>
  allowed_degradations: [<list>]

unit_cost:
  per_step_s: <float ± stdev>
  peak_mem_gb: <float>
  source: <measured | extrapolated | bounded>
  source_detail: "<where the numbers came from>"

cost_function:
  trial_wallclock_s: "<expression>"   # parameterised by tuned HPs that scale cost
  trial_peak_mem_gb: "<expression>"

sweep_shape:
  cells: <int>
  trials_per_cell: <int>
  seeds: <int>
  total_trials: <int>
  sweep_sizing_flag: "<exploratory | first-round | converged | over-spec>"

parallelism:
  workers_per_device: <int>
  cells_in_parallel: <int>
  devices_in_parallel: <int>
  store: <none | sqlite-wal | sqlite-wal-nfs>
  subdir: "<hydra.sweep.subdir override or 'default'>"

total:
  wallclock_s: <int>
  wallclock_h: <float>
  compute_h: <float>          # serial-equivalent; useful for cost accounting
  fits_deadline: <yes | no | tight>
  slack_h: <float>

risks:
  - <bullet list>

post_run:
  - "aggregate: <command to run on completion, e.g. python -m experiments.<family>.report ...>"
  - "smoke-result expected at: <path>"
```

Then a clearly-marked launch command:

```bash
# Tiny:
python -m ddssm.app experiment=<name>

# Medium (script-shaped grid):
nohup STAMP=<...> N_TRIALS_PER_WORKER=<...> N_WORKERS=<...> STAGE2_STEPS=<...> \
    bash scripts/run_overnight_mv_ablation.sh > runs/overnight.log 2>&1 &

# Medium (ad-hoc multirun):
python -m ddssm.app --multirun \
    experiment=<name> +sweep=<sweep_name> \
    hydra.sweeper.n_trials=<N> \
    hydra.sweeper.study_name=<study> \
    hydra.sweeper.storage=sqlite:///runs/optuna/<study>.db &

# Large (SLURM array):
sbatch --array=0-<N> --partition=gpu --gpus-per-node=1 --time=<HH:MM:SS> \
    runs/sbatch/<study>/cell_${SLURM_ARRAY_TASK_ID}.sbatch
```

</output>

<conventions>

- **State confidence everywhere.** `measured`, `extrapolated from <run>`, `bounded from architecture`. Don't hide guesses.
- **Don't update SKILL.md heuristics on the fly.** If you measure something surprising, surface it to the user — they may want to save a memory or update a comment in the preset, but the skill itself stays neutral on per-step numbers.
- **Default to the project's existing infrastructure.** Use `scripts/run_overnight_mv_ablation.sh` shape when it fits. Use the launcher preset in `src/ddssm/conf/hydra/launcher/submitit_slurm.yaml` for SLURM unless the user has special requirements.
- **Paths.** Optuna DBs in `runs/optuna/`, sbatch scripts in `runs/sbatch/<study>/`, sweep outputs in `runs/sweeps/<study>/`. Respect the convention.
- **Always point at the aggregation step** after the run. The user shouldn't have to ask how to read the output (`python -m experiments.<family>.report ...`).

</conventions>

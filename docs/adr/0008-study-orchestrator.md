# `StudyOrchestrator` + pluggable `LaunchStrategy` for running studies

A `Study` (ADR-0007) *defines* a family of points; something must *run* them. The
launch **shape** is study-specific: a two-model comparison is one single-GPU job
per point, while a `j=1..16` study runs an Optuna sweep per point and big-`j`
points may want several nodes per cell. A one-size launcher cannot express that,
and three near-identical family launchers (`launch_phase_d` /
`launch_ablation_tiny` / `launch_paper_headline`) plus a separate `smoke_phase_d`
had each re-encoded the sbatch/override mechanics. The `plan-campaign` skill's
"strategy table" (single-device / SLURM array / multi-node Optuna / …) is really
launch *mechanism* that belongs in code.

## Decision

1. **`src/ddssm/launch.py`** provides `ResourceSpec` (an alias of the existing
   `SBatch` dataclass), `PointLaunch`, the `LaunchStrategy` ABC, and
   `StudyOrchestrator`. `experiments/_sbatch.py` moves to `src/ddssm/sbatch.py`
   (the standalone `python -m experiments sbatch` CLI now imports it from there).

2. **Per-point launch intent on the Study, a function of the point.**
   `Study.launch(point) -> PointLaunch(strategy, sweep, n_trials, workers,
   resources)`. Because it is a function of the point, resources can scale with an
   axis value (`j=16 -> nodes=4`). It is overridable at orchestrate time.
   `PointLaunch.resources` **supersedes** `Experiment.sbatch` for study launches;
   `Experiment.sbatch` remains for standalone single-experiment sbatch.

3. **Strategy is explicit and pluggable.** `PointLaunch.strategy` names a
   `LaunchStrategy` (the sbatch shape). Implemented now: `single_job`,
   `optuna_single_node`. Documented extension-point stubs (raise
   `NotImplementedError`): `optuna_multi_node` (NFS-shared DB), `slurm_array`.
   Explicit selection (vs inferring from fields) keeps ambiguous cases — e.g. an
   array of single-node jobs vs N-nodes-per-cell — expressible.

4. **`StudyOrchestrator(study, study_prefix, storage_dir, sweeps_root)` is the
   mechanism, not the policy.** It `select`s points, reads each point's
   `PointLaunch`, dispatches to the named strategy to render, then dry-runs /
   writes / `--submit`s (sbatch) or runs locally (subprocess — replacing
   `smoke_phase_d`). A `seeds=[...]` knob replicates each point with
   `experiment.seed=` overrides. **Cross-point scheduling** — node-pool
   allocation, concurrency caps, ordering, deadline-fitting — is *not* here; that
   is the `plan-campaign` skill, which drives this class. The orchestrator stays
   free of live resource-profile state.

5. **One generic CLI**, `python -m ddssm.launch <study> [--select k=v] [--size
   tiny|paper|smoke] [--seeds ...] [--dry-run|--write-dir|--local] [--submit]`,
   resolving studies from a `register_study(...)` registry. The three family
   launchers + their shims + `smoke_phase_d` are **deleted** (clean break, per the
   `experiment-builder` skill convention); `report.py` consumes `study.points()`.

## Considered alternatives

- **Orchestrator owns cross-point scheduling too** (pool allocation, deadline).
  Rejected: that couples a library class to live resource profiles
  (`.claude/resources.yaml`), free-GPU state, and benchmarking — exactly what
  makes `plan-campaign` a stateful, interactive *skill*. Per-point shape (the
  stated need) is satisfied without it.
- **Infer the strategy from `PointLaunch` fields** (e.g. `nodes>1 ->
  multi_node`). Rejected: ambiguous shapes can't be told apart, and the rules
  become hidden magic.
- **Keep `Experiment.sbatch` as the only resource source.** Rejected: per-point
  resource variation would have to live on the baked preset, coupling resource
  intent into the experiment config rather than the study's launch layer.
- **A `local` strategy alongside the sbatch ones.** Folded into an orchestrator
  *backend* instead (`--local` runs single subprocesses) — the local path is a
  different execution backend, not an sbatch render shape.

## Consequences

- A study's resource shape is code (a function of the point); a new backend is a
  new `LaunchStrategy`; the `plan-campaign` skill shrinks to cost + deadline
  *policy* that drives the orchestrator.
- `python -m ddssm.launch init_centering --size tiny` replaces the three
  launchers; `--size smoke --local` replaces `smoke_phase_d`; `--size paper`
  applies the 2×-latent variant.
- `experiments/_sbatch.py`, `experiments/_launch.py`, `launch_study.py`,
  `launch_phase_d/ablation_tiny/paper_headline.py`, and `smoke_phase_d.py` are
  removed.

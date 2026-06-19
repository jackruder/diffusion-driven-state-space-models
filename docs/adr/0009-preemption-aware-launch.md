# Preemption-aware launch strategies for trial-level resume

Multi-hour Optuna sweeps on preemptible SLURM partitions need three things the
current launching path does not give them: SLURM-driven auto-requeue, per-trial
resume across preempt cycles (a 10-hour trial preempted at hour 4 picks up at
hour 4, not from scratch), and a study budget that stops at the *target* trial
count rather than running additive `n_trials=N` rounds per invocation. Today
the launcher emits sbatch with no `--requeue` / `--signal` plumbing, `train.py`
has no signal handler, abandoned preempted trials linger as `RUNNING` rows
forever, and the Hydra-Optuna sweeper plugin runs `n_trials` *additional*
trials regardless of DB state. Solve behind one opt-in `preemptive=True` knob
on `PointLaunch`.

## Decision

1. **`PointLaunch.preemptive: bool = False`** plus
   `PointLaunch.preempt_grace_seconds: int = 180`. Applies to
   `optuna_single_node`, `optuna_multi_node`, and `local_parallel`. Rejected
   (`ValueError` at orchestrator dispatch) for `single_job` — no trials
   concept. Multi-stage experiments are supported via per-stage
   `ckpt_<prefix>_latest.pth` files and a `stage_prefix` payload field that
   `StageOrchestrator.run(resume_from=...)` reads to skip earlier stages
   and suppress the centering handoff on the resumed stage. No CLI
   override: preemptibility is tied to which partition the point targets
   and is declared on the Study.

2. **SBATCH plumbing.** When `pl.preemptive=True`, `render_sbatch` prepends
   three directives *before* `spec.extra_flags` (so a user's `extra_flags`
   win under SLURM's last-line-wins semantics):

   ```
   #SBATCH --requeue
   #SBATCH --signal=B:USR1@{preempt_grace_seconds}
   #SBATCH --open-mode=append
   ```

   and replaces the plain `exec python -m ddssm.app …` line with a preamble
   that:
   1. Computes `N_REMAINING` via `python -m ddssm.launch_remaining --storage
      … --study … --target … --cleanup-running-older-than 60`.
   2. Exits 0 if the study is done.
   3. Derives `N_PER_WORKER = ceil(N_REMAINING / n_workers)`.
   4. Sets `DDSSM_INVOC=$(date +%s)` so trial sub-dirs do not collide across
      invocations (Hydra-Optuna resets `hydra.job.num` to 0 per multirun).
   5. Exports `DDSSM_PREEMPTIVE=1` and `DDSSM_WORKER_ID`.
   6. Installs `trap '…' USR1 TERM` to forward signals to the trainer PID.
   7. Runs the trainer with `hydra.sweeper.n_trials=$N_PER_WORKER` and
      `hydra.sweep.subdir=w<idx>_${oc.env:DDSSM_INVOC}_${hydra.job.num}` in
      multi-worker strategies, followed by `wait`.

3. **`src/ddssm/launch_remaining.py`** (new).
   `compute_remaining(storage, study_name, target, cleanup_older_than) -> int`.
   Tries `optuna.load_study(load_if_exists=False)`; on `KeyError` returns
   `target` (first-run path, no study stubbing with the wrong config). If
   `cleanup_older_than` is set, sweeps `RUNNING` trials whose `datetime_start`
   is older than the threshold and flips them to `FAILED` via the storage's
   `set_trial_state_values`. The trainer's graceful preempt path will have
   *already* enqueued a retry trial (see §5) for the latest in-flight trial;
   stale `RUNNING` rows reaped by `--cleanup-running-older-than` come from
   ungraceful kills (SIGKILL before the trainer could save) and start fresh
   on next pickup (no `resume_from` to inherit). Budget counts
   `COMPLETE + PRUNED` only; FAILED slots do not consume budget.

4. **Trainer signal path (`src/ddssm/train.py`).** A new
   `PreemptError(RuntimeError)` carries `resume_from: str`. The trainer
   registers handlers for `SIGUSR1` and `SIGTERM` in `__init__` (and
   `SIGINT` iff `DDSSM_PREEMPTIVE=1`); each handler does only
   `self._preempt_pending = True` — CUDA-safe, no torch ops in the
   handler. The step loop checks the flag after `_maybe_save_checkpoint`
   and before any `early_stop_triggered` break; on set it calls
   `_save_periodic_checkpoint` (returning the saved path) and raises
   `PreemptError(resume_from=<path>)`. Clean return would mark the trial
   COMPLETE and consume a budget slot; raising → FAILED via the sweeper's
   normal exception path → the app-layer catches and enqueues a retry
   (see §5). The existing `try/except KeyboardInterrupt` path stays
   untouched for non-preemptive Ctrl-C; under `DDSSM_PREEMPTIVE=1` SIGINT
   routes through the new flag-handler instead.

5. **App-level retry + resume hand-off (`src/ddssm/app.py`).** Inside the
   task function under `DDSSM_PREEMPTIVE=1`:
   - **Load the study** from `cfg.hydra.sweeper.{study_name, storage}` and
     **find the current trial by param-match** against the sampled cfg
     (best-effort; `hydra-optuna-sweeper` does not expose the trial number
     to the task function — log a warning and skip resume on ambiguity).
   - If the trial's `user_attrs` carry `resume_from` (inherited from an
     earlier preempt cycle, see below), inject it into
     `cfg.experiment.training.resume_from`.
   - Wrap training in `try/except PreemptError as e: …`. On preempt:
     **explicitly call `study.add_trial(...)`** to enqueue a new WAITING
     trial that copies the failed trial's `params` and adds
     `user_attrs={"resume_from": e.resume_from, "retried_from":
     current_trial.number}`. Then re-raise so the sweeper marks the
     current trial FAILED via its normal exception path. The next
     `study.ask()` (this invocation or a subsequent requeue) picks up the
     enqueued WAITING trial and the chain walks forward indefinitely.

   Note: relying on Optuna's `RetryFailedTrialCallback` was considered but
   rejected — empirical testing showed the callback does *not* fire on
   `study.tell(trial, FAILED)` from the sweeper's exception-catch path,
   only from the heartbeat-reaper loop inside `study.optimize`. Explicit
   `study.add_trial` covers the graceful-preempt path that
   `RetryFailedTrialCallback` would miss.

6. **Orchestrator wiring (`src/ddssm/launch.py`).** Strategies emit
   `hydra.sweeper.n_trials=__N_PER_WORKER__` as a placeholder when
   `pl.preemptive`; `render()` substitutes `$N_PER_WORKER` for sbatch and
   the literal `math.ceil(n_trials / n_workers)` for `run_local()`.
   `run_local()` also exports `DDSSM_PREEMPTIVE=1` and `DDSSM_WORKER_ID`
   on each `Popen`; the local path skips the DB-cleanup preamble (smoke
   convenience, not budget-aware). Per-worker over-run is bounded by
   `n_workers − 1`.

## Considered alternatives

- **Study-level resume only** (restart the full trial on each preempt).
  Loses per-trial work on every cycle; insufficient for multi-hour trials
  on 4-hour preemptible partitions.
- **`RetryFailedTrialCallback` + monkey-patched `RDBStorage`.** Considered
  as the canonical mechanism; rejected after a gate test showed the
  callback does not fire on `study.tell(trial, FAILED)` from the
  sweeper's catch-block (only from the heartbeat-reaper inside
  `study.optimize`). Graceful preempt is the common case, so the
  callback would miss it.
- **Fork `hydra-optuna-sweeper` as a plugin (`ddssm_optuna_preempt`).**
  Architecturally cleaner but ~20× the code; the upstream sweeper would
  be duplicated to expose two constructor kwargs.
- **Replace the sweeper with `ddssm.optuna_loop.py`.** Maximum control,
  largest scope; only worth it if we are also changing non-preempt
  behaviour, which we are not.
- **`MaxTrialsCallback` for budget enforcement.** The sweeper does not
  expose `callbacks`; would force a fork. Per-worker
  `ceil(remaining / n_workers)` over-runs by at most `n_workers − 1`,
  acceptable.
- **`resume_from` via side-channel files** (per-worker handoff TXTs).
  Race-prone across workers; the Optuna `user_attrs` round-trip is
  atomic by construction.

## Consequences

- One bool on `PointLaunch` flips preempt-safe sbatch, trainer signals,
  and retry-on-resume together. Non-preemptive runs are untouched: the
  `#SBATCH` directives are not injected, no signal handlers fire because
  `--signal` is not requested, `ddssm.app` skips the preempt logic when
  `DDSSM_PREEMPTIVE` is unset.
- Trial-level resume requires loading the Optuna study from inside the
  task function on every preemptive run, plus a param-match lookup
  against the cfg's sampled hparams — small extra IO on the warm path,
  negligible vs training.
- SIGKILL bypass (SLURM kill before the grace period expires) loses the
  in-flight trial: no checkpoint is saved, no retry is enqueued by the
  trainer. The DB cleanup at next preamble flips the stale `RUNNING` row
  to `FAILED` but does not start a fresh retry; the study budget simply
  runs more trials to hit the COMPLETE target. Acceptable: the user
  chose this trade-off over fragile heartbeat-based reaping.
- FAILED retry trials accumulate `run_dir`s on disk. The DB stays tidy
  via the preamble cleanup; disk-side cleanup is a follow-up.
- **Operator constraint:** single-trial wallclock ≤ `--time` −
  `preempt_grace_seconds`. A trial that exceeds the time budget loses at
  most one checkpoint-interval of work per cycle but never completes;
  documented in the operator manual.
- The retry-chain checkpoint walks forward naturally: each trainer
  writes to its own `run_dir`'s `ckpt_<prefix>_latest.pth`,
  `PreemptError` carries that path, the retry's `resume_from` points at
  it, and if the retry preempts the chain advances again. No special
  handling needed.
- Multi-stage experiments resume correctly via the `stage_prefix` field
  embedded in each checkpoint payload: `StageOrchestrator.run(resume_from=...)`
  reads it, iterates `stages.run` from the resumed stage onwards, and
  suppresses `perform_centering_handoff` on the resumed stage (the
  handoff already fired before the saved ckpt was written).

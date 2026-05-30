---
name: monitor-cluster-sweep
description: Monitor a DDSSM Optuna sweep running on a remote SLURM cluster (e.g. MSU Tempest) over SSH. Self-configuring — if no study is named it discovers running experiments and suggests the most active; it then checks whether it has a cached display profile for that experiment ("seen this before?") and, if not, builds one by introspecting the study + a trial's resolved_config.yaml so the table columns/labels/derived metrics come from the experiment, not hardcoded. One round-trip then reports per-cell trial counts + best per objective, any derived informativeness columns (e.g. fixed-target hit%), per-trial duration, the SLURM queue, and an ETA-to-target projection; scp's the Optuna DBs locally, merges them, and (re)launches optuna-dashboard for a single inspection link; and prints an at-a-glance summary table with a delta vs the previous check. Use when the user says "check the cluster sweep", "how's the overnight run", "is the sweep done on Tempest", "monitor the run", "give me a dashboard link", or drives this on a /loop. For LOCAL runs use check-training-progress instead.
---

<role>
Report the health and progress of a DDSSM Optuna sweep that is running on a
remote SLURM cluster. You answer four standing questions:

  1. Is training proceeding and producing the right artifacts?
  2. What objective minima (ELBO etc.) are trials reaching, and is the
     time-to-target objective informative or mostly censored?
  3. Pull the Optuna DB(s) and give a dashboard link to inspect.
  4. Which workers are blocked, and what's the ETA to completion?

This is the *remote* sibling of [[check-training-progress]] (which triages
local `runs/`). Same read-only, schema-aware, high-signal stance — but over SSH,
plus a dashboard and an ETA projection. It is a diagnostic, not the paper-figure
pipeline (`experiments.init_centering.report` owns that).
</role>

<inputs>
Driven entirely by env vars on `monitor.sh` — **all optional**:

- `STUDY_PREFIX` — which experiment to show (common stem of the per-cell DBs at
  `<REMOTE_DIR>/optuna/<STUDY_PREFIX>_<cell><SUFFIX>.db`). **If unset, the driver
  discovers running experiments and auto-picks the most active**, printing the
  ranked alternatives so you can re-run with a different one.
- `SUFFIX` — dataset suffix on DB names. Resolved from discovery / the cached
  profile if unset; default `__mv`.
- `TARGET` — trials/cell target for the ETA projection. Default 128.
- `REBUILD=1` — force a context rebuild (re-introspect, overwrite the profile).
- `HOST` — ssh target. Default `z89p425@tempest-login.msu.montana.edu`.
- `REMOTE_DIR` — project dir on the cluster. Default `~/diffusion-driven-state-space-models`.
- `PORT` / `PULL_DIR` / `NO_DASH` — dashboard port, local pull dir, skip-dashboard.

The display is **not hardcoded**. Per experiment, the driver caches a profile at
`profiles/<study_prefix>.json` (gitignored) holding the introspected context:
the objectives (names + MIN/MAX from the study directions, short labels), the
headline objective for the summary's "best X" column, and any derived columns
(e.g. a `*_to_target_seconds` metric → a hit% column). So a CRPS sweep and an
ELBO MOO sweep render different, correct tables with no code change.
</inputs>

<workflow>

## 1. Run the driver

```bash
# let it discover + suggest the active experiment:
bash .claude/skills/monitor-cluster-sweep/monitor.sh
# or name one explicitly:
STUDY_PREFIX=<prefix> TARGET=<n> bash .claude/skills/monitor-cluster-sweep/monitor.sh
```

It runs the whole pipeline of small cluster scripts (scp'd to `/tmp` and run in
the project venv) plus local steps:

1. **reachability** check (exit 3 → VPN down, see §2).
2. **discover** (`discover.py`) — if no `STUDY_PREFIX`, group the Optuna DBs into
   experiments by matching SLURM job names, rank by live activity, auto-pick the
   top, and print the alternatives. If the user named one, skip this.
3. **profile** — "have I seen this experiment before?" If `profiles/<prefix>.json`
   exists, reuse it ("✓ Seen … before"); else **build context** (`build_context.py`):
   introspect the study (directions, metric_names, params) + a trial's
   `resolved_config.yaml` (objective specs, eval metrics, target) + sniff
   `metrics.json` keys → derive objective labels + derived columns; save the profile.
4. **probe** (`probe.py`) — gather per-cell stats *per the context only*: states,
   best per objective, derived columns, trial duration, queue, ETA projection.
5. **render** — print the diagnostic table + the at-a-glance summary (§5), with a
   delta vs the last snapshot.
6. **pull + merge + dashboard** — scp DBs → `<prefix>_combined.db` → optuna-dashboard.

**Read its output first and quote those numbers** — don't re-derive with ad-hoc
ssh/awk. If discovery surfaced several experiments and the auto-pick isn't the
one the user means, re-run with `STUDY_PREFIX=…`. Force a fresh context (after a
schema/objective change) with `REBUILD=1`.
Pick the freshest set (check mtimes — stale 4 KB DBs are abandoned stubs; a
second copy under a different dir like `~/ddssm/optuna/` may be a stale mirror).

## 2. Reachability

The login node resolves to a private `10.x` campus address. If `monitor.sh`
exits 3 (UNREACHABLE), the MSU VPN is down. Do **not** keep retrying — tell the
user to bring up the VPN (`gp-saml-gui` / GlobalProtect / openconnect) or to run
the probe themselves via `! ssh ...`, then retry once they confirm `ping` works.

## 3. Read the per-cell table

Columns: `COMP RUN best_obj0 best_obj1 hit% xstep50/90 trial_m q(r/p) tl_h proj/target`.

- **best_obj1** is the ELBO depth (MINIMIZE → more negative is better). This is
  the headline "ELBO minima achieved". **best_obj0** is `wallclock_to_target_seconds`.
- **hit%** = fraction of trials that actually reached the ELBO target before the
  step budget. This is the *informativeness* read for the time-to-target axis:
  - 50–65% with `xstep` spread across the budget → informative (real speed gradient).
  - <35% → the axis is mostly **censored**: `obj0` collapses to the
    `penalty="csv_tail_time"` (full runtime) and just re-encodes "didn't
    converge", redundant with obj1. Recommend a softer target next round.
  - `xstep_p90` hugging the step budget → target is near the edge of difficulty.
- **obj0 seconds are NOT comparable across cells** of different model size / GPU
  type (a slow cell's penalty and hit-times are inflated). Compare within a cell.
- **trial_m** = median per-trial wall-clock (minutes). **proj** = projected final
  trial count if no new jobs are added — flag cells landing well under `target`.

## 4. SLURM queue semantics (for the ETA + "is it stuck" question)

- A job that hits its `--time` wall ends in **TIMEOUT and is NOT requeued** —
  `--requeue` governs *preemption/node-failure only*, not time-limit. So each
  running job has a hard end at its `TIME_LEFT`; the campaign drains, it does not
  self-extend. The one in-flight trial is lost; completed trials are safe in the DB.
- **Preemption** requeues *only* on partitions with `PreemptMode=REQUEUE`
  (e.g. `gpuunsafe`, PriorityTier low). `gpupriority` is usually `PreemptMode=OFF`
  (non-preemptible — it's the preemptor), so jobs there run uninterrupted to their wall.
  Check with `scontrol show partition <p> | grep -i preempt`.
- A SIGKILLed worker (wall or preempt) can't mark its trial FAILED → it leaves a
  **zombie RUNNING trial** in the DB. So the `RUN` column over-counts live work;
  trust `COMPLETE` and the best values. NSGA-II ignores incompletes.
- `per-worker n_trials` is a *ceiling the worker won't reach* in one window
  (a ~90-min trial → ~10/worker in 16h; a packed 8-on-1-GPU cell is much slower
  per trial from contention). The per-cell target is reached by accumulating
  across *all* workers that ever run against the study — not by one wave.

## 5. Report (keep it tight)

- **Artifacts (Q1)**: confirm trial dirs have `metrics.csv` + `checkpoints/` and
  the live DB is being written (recent mtime). Note any FAILED-fast churn.
- **Objective minima + informativeness (Q2)**: the best-ELBO table + hit% read.
- **Dashboard (Q3)**: the `http://127.0.0.1:$PORT/dashboard` link (one combined
  study dropdown). Mention it's a snapshot; re-run to refresh.
- **Blocked + ETA (Q4)**: pending reasons (Priority/Resources = queued, not
  broken), min/max `TIME_LEFT`, and the proj-vs-target shortfall.
- **Summary table**: `monitor.sh` prints an at-a-glance markdown table
  (`cell | completed | Δ | best <obj> … | <derived>`) — **one best-column per
  objective** so a MOO study shows every axis (e.g. both the time objective and
  the ELBO), the headline objective starred. The header names the time window
  (`Δ over last 47 min`) with a trials/hr rate, and a `⬇` marks a new best on any
  axis this window. Surface it as-is — it's the operator's preferred headline view.

## 6. Top-up under-sampled cells (only when asked / on a babysitting loop)

When `proj` lands well under `target` for some cells, resubmit *more workers
against the same study* — trials accumulate in the shared DB. Safe pattern
(prevents clobbering first-wave artifacts and auto-fires at the wall):

1. Copy each thin cell's sbatch, `sed` **only** `hydra.sweep.dir=...` to a new
   `_topup` path; leave `hydra.sweeper.study_name` and `...storage` UNCHANGED
   (so the DB keeps accumulating). Verify that invariant before submitting.
2. Pre-arm with a dependency so it launches when the first wave finishes and
   idles no GPU and survives your session dying:
   `sbatch --dependency=afterany:<first-wave-jobids> <topup.sbatch>`.
   Use `afterany` (fires on TIMEOUT too), not `afterok`. List current job ids
   per cell with `squeue -u $USER -h -t RUNNING,PENDING -n <job> -O JobID`.
3. The top-up writes to the `_topup` sweep dir; `probe.py` already folds
   `*_topup/metrics.json` and `metrics.csv` into the cell's stats.

## 7. /loop integration

For overnight babysitting the user typically wires this into `/loop` (dynamic
mode). Each iteration: run `monitor.sh`, report the delta, note any top-up
Dependency→RUNNING transitions. **Stop condition**: no RUNNING *and* no PENDING
jobs matching `init_` for the user (covers first-wave `init_*` and top-up
`TU_init_*`). On stop, PushNotification the final per-cell counts + best ELBO.
SLURM dependencies run independent of the loop, so the resubmission is
guaranteed even if the session dies — the loop is only the reporter.

</workflow>

<conventions>
- **Read-only triage + a dashboard.** The only writes are: local DB pull/merge,
  the dashboard process, and (only when asked) top-up sbatch submission. No new
  plots/CSVs — those belong in `experiments/<family>/report.py`.
- **Trust COMPLETE + best, not RUN** (zombie RUNNING trials, see §4).
- **Same study_name + storage = same DB.** Any override that keeps those two
  identical accumulates trials; only `hydra.sweep.dir` should differ for a top-up.
- **σ_pert > 0** protocol invariant carries over — a study with σ_pert pinned to
  0 / its log-uniform floor is misconfigured (see [[handoff_protocol_invariants]]).
- **DB filename ↔ study_name**: the per-cell study_name equals the DB basename
  without `.db`. The cell label is that minus the `STUDY_PREFIX_` stem and
  `SUFFIX`.
</conventions>

<brainstorming-improvements>
Surface as suggestions if relevant; don't build unprompted:
1. `--watch N` on `monitor.sh` to re-probe every N min without the /loop harness.
2. Pareto-front extract for MOO studies (`study.best_trials` + their params) so the
   user sees the fast-shallow vs slow-deep frontier, not just per-axis minima.
3. Auto-detect the partition preempt mode per cell and annotate which cells can be
   requeued vs which run to the wall, in the queue report.
4. A `--topup thin --target N` flag that generates + dependency-arms top-ups for
   every cell under N automatically (codifies §6).
5. Slack/markdown digest mode for pasting an overnight status into a channel.
6. Cross-snapshot history (append each JSON snapshot) → a trials-over-time and
   best-ELBO-over-time sparkline per cell.
7. Detect zombie RUNNING trials (DB RUNNING but no live squeue worker + stale
   metrics.csv) and report the count so RUN is interpretable.
</brainstorming-improvements>

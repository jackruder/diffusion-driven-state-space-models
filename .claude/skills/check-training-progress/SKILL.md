---
name: check-training-progress
description: Auto-discover and triage active DDSSM training runs and sweeps. Identifies the metrics schema dumped by the current run, extracts head→tail diagnostics, flags λ-warmup / σ_data² / loss-direction red flags, and joins Optuna DBs for sweep status. Use when the user says "check progress", "how's training going", "is the sweep done", "any runs broken", or invokes /check-training-progress.
---

<role>
Triage *currently-running or just-finished* DDSSM training output and report a short, structured health summary. You are a diagnostic — not a paper-figure pipeline (`experiments.init_centering.report` already owns that) and not a config-builder (`select-hyperparameters` owns that).

Your goal is to answer in ~30s: *Is this run healthy? Should the user intervene?* Surface a small number of high-signal numbers and flag the red lights. Defer deep inspection to follow-up questions.
</role>

<inputs>
- **Optional path argument** the user may pass: a single run, a sweep dir, or a parent that holds either. If none given, auto-pick the newest under `runs/`.
- **Repo layout context** — `runs/` contains:
  - `runs/<study>_<cell>__<dataset>/{0,1,...}/` — Hydra-multirun sweep (init-centering family layout)
  - `runs/headline_YYYYMMDD_HHMM/<cell_name>/` — multi-cell control sweeps
  - `runs/<single>/` — single-job runs (`gpu_test/`, `smoke_phase_d/`, ad-hoc Hydra outputs)
  - `runs/optuna/<study>_<cell>__<dataset>.db` — per-cell Optuna sqlite DBs
  - `runs/report/` — finished aggregation artifacts (NOT a place to triage)
  - `runs/sbatch/` — SLURM scripts + slurm-*.out logs
</inputs>

<workflow>

## 1. Discover

Run the helper from anywhere in the repo:

```bash
python .claude/skills/check-training-progress/progress.py [<path>] [--max-trials N] [--no-slurm]
```

It auto-detects one of five layouts (`SINGLE_RUN`, `SWEEP`, `MULTI_CELL`, `MULTI_SWEEP`, `EMPTY`), prints a structured report, and finishes with SLURM/GPU/process scans. **Read what it printed first** — it has already extracted the column families dumped by the run, the lambda-warmup state, σ_data² drift, and per-loss head→tail deltas. Quote those numbers; do not re-derive them with ad-hoc `tail`/`awk`.

If the user named a specific study or cell, pass it as the path. Otherwise let auto-pick run — it picks the newest subdir of `runs/` excluding `optuna/`, `report/`, `sbatch/`.

## 2. Identify what the run is actually dumping

The metric schema evolves per experiment family — never assume columns. The helper script groups columns by prefix family (`loss/`, `diag/`, `calib/`, `optim/`, `time/`) and prints the count per family. Use that to decide which secondary diagnostics are even available before commenting on them. For example: no `diag/sigma_data2/t=*` columns ⇒ this isn't a centering run, so skip σ_data² commentary.

## 3. Apply the health heuristics

Score the run with these heuristics (they match the operator's habits — see [[handoff_protocol_invariants]]):

| Signal                               | Healthy                  | Watch                   | Bad                              |
|--------------------------------------|--------------------------|-------------------------|----------------------------------|
| `loss/total` rel Δ (head→tail)       | < −5 %                   | flat (−5 %…+5 %)        | > +5 % (rising)                  |
| `optim/lambda` last value            | ≥ 0.99 by ~10 % of steps | < 0.99 mid-run          | < 0.5 after 200+ rows            |
| `diag/sigma_data2/t=*` last-row mean | \|μ−1\| < 0.15            | 0.15–0.5                | > 0.5  (centering broken)        |
| `calib/ratio_res2_to_sigma2` tail    | → 1.0                    | 0.5–2.0                 | > 5 or → 0                       |
| NaN/Inf in any column                | none                     | a few late              | growing or in `loss/*`           |
| Sweep: % trials COMPLETE             | ≥ expected               | many RUNNING long       | many FAIL/PRUNED                 |
| `metrics.csv` mtime                  | < 5 min ago (live)       | < 1 hr ago              | > 1 hr ago + state=RUNNING       |

Always cross-check `optuna_state.running` against `metrics.csv` mtime: a stale CSV + DB-says-running ⇒ a dead worker that Optuna hasn't reaped.

## 4. Report

Keep the user-facing summary to roughly:

- **Layout + scope** (1 line). "1 active sweep, 6 trials, 5 done, 1 still writing."
- **Headline numbers** (2–4 lines). Best-trial objective, λ state, σ_data² drift, any NaNs.
- **Red flags** (bulleted, only if present). Name the trial / cell + the specific number.
- **Suggested action** (1 line). One of: *keep going*, *kill trial N*, *check log X*, *aggregate now* (`python -m experiments.init_centering.report all ...`).

If the helper printed concrete `⚠` warnings, surface them verbatim — don't re-paraphrase.

## 5. Follow-ups (don't do unless asked)

The helper's output should answer 90 % of "is it healthy" questions. If the user drills down:

- *"What does trial N look like?"* — call the helper with that trial dir as the path.
- *"Tail the live log."* — `tail -F <path>/stdout.log` or `<path>/app.log`; for SLURM, `runs/sbatch/<study>/slurm-*.out`.
- *"Plot loss curve."* — emit a 6-line matplotlib snippet against `metrics.csv`; **do not** invoke `experiments.init_centering.report`, that's the paper pipeline.
- *"Compare cells."* — run on the `MULTI_CELL` parent or use `experiments.init_centering.report aggregate` for the finished-sweep aggregation flow.

</workflow>

<conventions>
- **Don't write new plots / summary CSVs from this skill.** Persistent artifacts belong in `experiments/<family>/report.py`. This skill is read-only triage.
- **Schema-agnostic.** The helper groups columns by prefix and only comments on what's present. If the user adds a new column family next week, the helper will list it without code changes.
- **σ_pert > 0.** Protocol invariant — see [[handoff_protocol_invariants]]. If a sweep's `params` show σ_pert pinned to 0 or its log-uniform lower bound, that's a misconfigured study, not a bad run.
- **Don't conflate `runs/report/` with active runs.** Auto-pick excludes it; if a user passes it explicitly, the layout will be EMPTY or MULTI_CELL but represents archived state, not training.
</conventions>

<brainstorming-improvements>
Things this skill *could* grow into — surface as suggestions if relevant, don't build unprompted:

1. **Live tail mode** — `--watch` flag that re-runs every N seconds (cheap; CSVs are small). Useful while babysitting a sweep.
2. **Slack-friendly digest** — one-liner-per-cell summary suitable for pasting into a status update.
3. **Cross-run diffing** — given two run dirs (today's vs yesterday's same-cell sweep), report which trials regressed in `stage2_elbo_surrogate`.
4. **Schedule drift inspection** — for multi-stage runs (`StageOrchestrator`), per-stage λ-ramp and step-budget compliance check (did stage_2 actually run as long as `n_stage2` says?).
5. **Trainable-mask audit** — open `resolved_config.yaml`, surface which submodules were frozen this run. Catches "I meant trans-only but trained encoder too" mistakes (matches the warning in `TrainingScalars`).
6. **GPU-utilization sample** — extend the slurm scan to `nvidia-smi -lms 1000 -c 5` and report mean util across 5 samples. Catches under-utilized SLURM jobs that look "active" but the GPU is idle.
7. **Optuna trial sampler-state** — surface `study.best_trials` for MOO studies (currently only first-objective best is reported) and the *params* of the best trial so the user can decide whether to keep that range.
8. **Diff against `params` in Optuna DB** — for a SWEEP layout, fold per-trial sampled params into the table so the user can see *why* trial 3 is the best (e.g. base_lr=3e-4 vs the rest at 1e-4).
9. **Auto-resume hint** — if a stale RUNNING trial is detected, suggest the exact `--multirun` resume command (Hydra path + `hydra.sweeper.continue_unfinished=true`).
10. **`metrics.json` schema warning** — if a finished trial is missing keys the report pipeline expects (`stage2_elbo_surrogate`, `crps_sum_latent_*`), flag it before the user kicks off `experiments.init_centering.report` and gets a half-empty `summary.csv`.

These all share the same "schema-agnostic, read-only triage" stance — none of them require pulling deep model-specific logic into the helper script.
</brainstorming-improvements>

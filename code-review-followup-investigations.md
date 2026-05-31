# Code-review follow-up — investigative plan (#18, #20–26)

Detailed write-up of the remaining review items. P0 (bugs + dead code), P1 (config
hygiene), and the math investigations (#13–15, #27) are already committed; #16
(prune builders.py) was dropped and #17 (strip `v3` → `diffusion`) is done. This
document explains each *remaining* item — the problem, evidence, proposed
approach, risk, and how to verify — so they can be scheduled and executed
deliberately. Nothing here is implemented yet.

Conventions: anchors are `file:symbol` (line numbers drift). Each item lists
**Effort** (S/M/L) and **Risk** (low/med/high).

---

# Architecture / smell

## #18 — Disambiguate the two `conf` packages  ·  Effort S · Risk med

**Problem.** There are two unrelated packages both named `conf`:
- `conf/` (repo root) — `conf/registry.py`, the hydra-zen `store(group=...)`
  handles (`model_store`, `experiment_store`, `sweep_store`).
- `src/ddssm/conf/` — the Hydra YAML library (`config.yaml`, `wandb/`,
  `hydra/sweeper/`).

`import conf.registry` only resolves because `register_experiments()`
(`ddssm._experiment_registry`) injects the repo root onto `sys.path` at runtime.
This causes three concrete problems:
1. **Import-order fragility / test isolation.** Any module doing
   `from conf.registry import ...` (e.g. `experiments/init_centering/*`,
   several tests) fails to import unless `register_experiments()` ran first.
   This is the root cause of two pre-existing test failures that only appear in
   isolation (`test_secondary_metrics.py::test_log_sigma_p2_collapse_runs_on_smoke_model`,
   and the `tests.`-namespace collection error in `test_likelihood/`).
2. **Name collision confusion.** Greps for "conf" hit both; reviewers (and the
   review agents) repeatedly grepped the wrong one. CLAUDE.md's "two parallel
   config worlds" note is a direct symptom.
3. **Packaging smell.** A top-level `conf` package shadows a very common name and
   is not under `src/`, so it isn't part of the installed `ddssm` package.

**Evidence.** `conf/registry.py` vs `src/ddssm/conf/`; the `sys.path` injection in
`ddssm/_experiment_registry.py:_ensure_experiments_on_path`; the isolation
failures confirmed earlier in this review.

**Proposed approach.** Rename the repo-root `conf/` to something unambiguous and
import-safe. Two options:
- **(a) Move it under the package**: `src/ddssm/stores.py` (or
  `src/ddssm/registry.py`), so `from ddssm.stores import experiment_store`. This
  removes the `sys.path` hack entirely *for the store handles* (experiments still
  need the repo root on path so `import experiments` works, but the store
  registry no longer does). Cleanest long-term.
- **(b) Rename in place** to `zen_stores/` (repo root) — lower churn, keeps the
  repo-root location, just kills the `conf` collision.

Recommend (a): it also lets the registry import without `register_experiments()`
having run, which fixes the test-isolation failures. The `experiments/` package
still needs the path injection (it's intentionally outside `src/`), but importing
the *store handles* would no longer depend on it.

**Risk: med** — touches every `from conf.registry import ...` site (≈10 files) and
the `sys.path` bootstrap. Mechanical but wide; must keep `register_experiments()`
working and re-verify `python -m experiments list` + `python -m ddssm.app`.

**Verify.** `uv run pytest tests/test_secondary_metrics.py -q` (the isolation
failure should clear); `python -m experiments list`; full suite.

---

# Logging

## #20 — Log stage boundaries + always-on λ / σ_data² series  ·  Effort S–M · Risk low

**Problem.** Three observability gaps make multi-stage runs hard to read from the
artifacts alone:
1. **No stage marker in the logs.** `StageOrchestrator` prints
   `=== Running {key} ... ===` to stdout (`stages.py:253`) but writes **nothing**
   to `metrics.csv`/TB. A `loss/total` curve spanning stage-1→stage-2 has no
   column saying which stage a row belongs to or where the λ-ramp reset / handoff
   fired, so the curve is uninterpretable without cross-referencing stdout.
2. **λ only logged for `FullELBO`.** `optim/lambda` is emitted only when
   `isinstance(self._active_loss, FullELBO)` (`train.py:404`). Any custom `Loss`
   → no λ column → the triage/health checks that read it go dark.
3. **σ_data² diag is gated by a private flag.** The `diag/sigma_data2/t=*`
   columns require `_report_sigma_data_diag` (`dssd.py:747`); if off, the
   centering-health view silently disappears.

**Evidence.** `stages.py:253` (stdout-only banner); `train.py:404-405`
(`optim/lambda` conditional); `dssd.py:747` (σ_data diag gate). Note #9 already
added an *effective-budget* log line at orchestrator start — this item is the
per-row series complement.

**Proposed approach.**
- Add `stage` and `step_within_stage` columns to every logged row: have
  `StageOrchestrator` set `trainer._current_stage = key` (and the stage start
  step), and include them in `_log_train_step`'s `log_values`. The CSVLogger is
  already column-robust (#19), so adding columns mid-file is safe.
- Make `optim/lambda` unconditional: give the `Loss` ABC a
  `lambda_at(step) -> float | None` method (FullELBO returns `rate_lambda(step)`;
  others return their effective ramp or `None`), and log it whenever it's not
  `None` — independent of the concrete class.
- Default `_report_sigma_data_diag=True` whenever a `sigma_data` buffer exists
  (it already defaults True; just confirm no preset turns it off), or expose it
  as a documented knob.

**Risk: low.** Additive columns + a small `Loss` interface method. The
`test_full_elbo_reproduces_pre_refactor_loss` is unrelated (Gaussian path).

**Verify.** Run a 3+3-step smoke; assert `metrics.csv` has `stage`,
`step_within_stage`, `optim/lambda` for both stages and that the stage column
flips at the boundary. Extend `tests/test_trainer.py` / a stages test.

## #21 — NaN/Inf guard in the logging path  ·  Effort S · Risk low

**Problem.** Nothing guards against non-finite metrics. `MetricStore._tofloat`
(`loggers.py:265`) and `update` (`loggers.py:274`) write raw floats; a NaN/Inf
`loss/total` is persisted as `nan` and training continues for hours. The only NaN
awareness lives downstream in the triage skill's `_safe_float`, not at the
source.

**Evidence.** `loggers.py:_tofloat`/`update` — no `isfinite` check anywhere in
the write path.

**Proposed approach.** In `MetricStore.update` (or `_tofloat`), count non-finite
values per key and expose a `nonfinite/<key>` counter (or a single
`nonfinite/total`). Optionally, add an opt-in early-abort: if `loss/total` is
non-finite for N consecutive logged steps, raise a clear error so a diverged run
fails fast instead of burning compute. Keep it cheap (one `math.isfinite` per
scalar) and off-by-default for the abort behavior to avoid surprising existing
runs.

**Risk: low** — purely additive; the abort path should be opt-in.

**Verify.** Unit test in `tests/test_loggers.py`: feed a NaN, assert the counter
increments and (if abort enabled) the configured exception fires.

## #22 — Per-run `run_summary.json` + `python -m ddssm.report`  ·  Effort M · Risk low

**Problem.** A run dir holds `metrics.csv`, `tb_logs/`, `checkpoints/`,
`resolved_config.yaml` — but nothing that says, at a glance, "healthy / λ done at
step X / σ_data² drift Y / final loss Z / val loss". The only summarizer is the
external `check-training-progress` skill (`.claude/skills/check-training-progress/progress.py`),
which re-derives everything from the CSV each time and isn't shipped with the
framework or unit-tested. `variance/aggregate.py` is the only cross-run roll-up
and it just concatenates JSON into fenced markdown.

**Evidence.** Run-dir contents (`experiment.py:train`); the skill's `progress.py`
reader; `variance/aggregate.py`.

**Proposed approach.**
1. **Emit `run_summary.json`** at `fit`/orchestrator exit:
   `{final_step, loss/total (weighted + unweighted), λ last + warmup-cross step,
   σ_data² mean/drift, val loss, nan_counts, elapsed_s, stages_run}`. This is
   exactly what `progress.py` recomputes — emitting it once at the source makes a
   run self-describing and gives the objective reader a stable place to look.
2. **Promote the reader into the package** as `python -m ddssm.report <run_dir>`:
   move the read-only health logic out of the skill into an in-repo module so it
   ships with the framework and is unit-testable. Accept a `runs/` parent to emit
   one row per run (config hash from `resolved_config.yaml` + final summary) — a
   real cross-run comparison table that replaces the JSON-dump style of
   `variance/aggregate.py`.

**Risk: low** — new artifact + new CLI; doesn't change training. Reuse the
heuristics already encoded in `progress.py` (lambda_state, sigma_data_summary,
head_tail) so behavior matches the skill.

**Verify.** Run a smoke; assert `run_summary.json` exists with the expected keys
and finite values; `python -m ddssm.report <run_dir>` prints a table; point it at
a `runs/` parent and confirm one row per run.

---

# Experiment-authoring workflow

These four reduce the boilerplate/foot-guns in authoring presets, studies, and
sweeps. They're independent and additive (keep the existing APIs).

## #23 — Typed `SweepSpace` with path validation  ·  Effort M · Risk low

**Problem.** Sweep search spaces (`experiments/init_centering/sweeps.py`) are
`dict[str, str]` of dotted Hydra paths → Optuna distribution strings, e.g.
`"experiment.training.stages.n_pretrain": "tag(log, interval(...))"`. Nothing
validates the dotted path against the target config, so a rename in the stages
factory silently turns an axis into a **no-op** (the sweep "runs" but never varies
that knob). The MOO `direction` list length must match the `Objectives` list by
hand (`sweeps.py` ↔ `evals.py`) with no check.

**Evidence.** `sweeps.py` raw param dicts; the `n_pretrain`/`n_stage2` factory
knobs (which only exist because `StagesB = builds(..., populate_full_signature)`);
the hand-matched MOO direction length.

**Proposed approach.** A small `SweepSpace` helper that (a) takes a target config
(e.g. `StagesB`) + field name + distribution, (b) at registration time asserts
the dotted path resolves into the instantiated config (so typos raise at import,
not silently), and (c) for MOO asserts `len(direction) == len(objectives.specs)`.
It emits the *same* Hydra `params` dict, so downstream is byte-identical — purely
a guard + ergonomic layer.

```python
sweep = SweepSpace(target=StagesB, objectives=PilotMOObjective)
sweep.log("n_pretrain", 5, 500)        # validates StagesB exposes n_pretrain
sweep.register("init_ablation_moo")    # asserts direction count == objective count
```

**Risk: low** — additive; rewrite `sweeps.py` to use it (mechanical), emitted
config unchanged.

**Verify.** `tests/`: a typo'd field raises at registration; a valid space emits
the expected `params`; the existing `test_experiment_configs.py::test_sweep_preset_composes`
still passes.

## #24 — Single-call study registration + collision guard  ·  Effort S · Risk low

**Problem.** Registering a study requires **two** decoupled call sites:
`register_study(...)` (so `python -m ddssm.launch` finds it) and
`INIT_CENTERING_STUDY.register(experiment_store)` (so `experiment=NAME` resolves
its points). Doing only one yields a launchable-but-unresolvable study, or vice
versa. Point-name collisions are silent **last-write-wins** (`study.py:Study.register`
just calls `store(config, name=...)`).

**Evidence.** `experiments/init_centering/study.py` (`register_study` call) +
`experiments/init_centering/experiments.py` (the separate
`.register(experiment_store)` call); `src/ddssm/study.py:Study.register`.

**Proposed approach.** Fold point-publishing into `register_study` (or have it
accept the store): `register_study(Study.from_axes(...), into=experiment_store)`
publishes to **both** the launch registry and the experiment group in one call,
and adds a duplicate-name assertion in `Study.from_axes`/`register` (raise on
collision instead of silently overwriting).

**Risk: low** — change two call sites + add a guard. The existing
`tests/test_study.py` pins current behavior; update it for the unified call.

**Verify.** `tests/test_study.py`: one call registers both; a colliding point
name raises.

## #25 — Declarative `Axis.of` to de-triplicate axis mapping  ·  Effort M · Risk med

**Problem.** The axis-value → tags → preset-name mapping is **triplicated** and
must be kept in sync by hand:
- `cells.py:cell_name(...)` (the name function),
- `study.py` `Axis(key=, tags=)` wiring,
- the `name_point=lambda tags: f"{tags['cell']}__{tags['dataset']}"` lambda
  (`init_centering/study.py:173`).

`tests/test_study.py` + `test_init_centering_cells.py` exist mostly to catch
drift between these three — a tell that the coupling is fragile. There's also a
manual `_PARAM_FREE_FORMS` mirror: defined in `model.py` and re-declared in
`cells.py` with a "Mirrors model.py" comment.

**Evidence.** `cells.py:cell_name`, `cells.py:_PARAM_FREE_FORMS` (mirror),
`init_centering/study.py:160-173`, `src/ddssm/study.py:Axis`/`_default_namer`.

**Proposed approach.** Let axis values carry their own identity: a `Named`/`HasTags`
protocol (`.name`, `.tags`) — `Cell` already has `.name`. Add an `Axis.of(values)`
classmethod that defaults `key=lambda v: v.name` and `tags=lambda v: v.tags`,
and default `name_point` to joining the per-axis names. Then the study wiring
collapses to:

```python
Study.from_axes("init_centering",
    axes=[Axis.of(iter_cells()), Axis.of(ABLATION_DATASETS)],
    build=_build, launch=_launch, variants={...})
```

Separately, move `_PARAM_FREE_FORMS` to one shared module imported by both
`model.py` and `cells.py` (kill the manual mirror), and add an assertion in
`_build_init_centering_model` that the same `baseline` object reaches both
transitions (the shared-baseline interlock is currently enforced only by prose).

**Risk: med** — touches the study/axis core (`study.py`) and the family wiring;
needs `AblationDataset` to gain `.name`/`.tags`. Well-covered by existing study
tests, so drift is catchable.

**Verify.** `tests/test_study.py` + `test_init_centering_cells.py` still pass
with the collapsed wiring; add a test that `Axis.of` reproduces the prior point
names exactly (no preset renames).

## #26 — Typed `derive()` for preset variants  ·  Effort S · Risk low

**Problem.** To make a one-knob variant of an existing preset there's no typed
path — only the untyped string DSL `override(exp, "training.stages.n_pretrain=11")`
(`experiments/_make.py:override`), which YAML-parses the RHS and only fails at
call time. Authoring a variant otherwise means re-stating all four slots
(`data/model/hparams/training`) to `experiment(...)`.

**Evidence.** `experiments/_make.py:override` (string-based); `experiment(...)`
keyword-only factory with no "derive from base" path.

**Proposed approach.** Add a typed companion:
`derive(base_exp, training=replace(steps=2000), model=replace(latent_dim=8))`
returning a fresh `ExperimentC` with those slots replaced (via
`dataclasses.replace` on the hydra-zen configs), keeping type-checking and
failing at construction. ~30 lines in `_make.py`, purely additive — keep
`override()` for the string/CLI path.

**Risk: low** — additive helper.

**Verify.** `tests/test_init_centering_factory.py`: `derive` produces a config
whose changed field is set and whose untouched slots are identical to the base.

---

## Suggested order

1. **#18** first — it unblocks the two pre-existing test-isolation failures and is
   the root of the "two config worlds" confusion. Do before more test work.
2. **#20 + #21 + #22** as a logging batch — #20/#21 are small and #22 builds on the
   schema they stabilize (stage column, nan counts feed `run_summary.json`).
3. **#24 + #26** (small workflow wins), then **#23** (sweep validation), then
   **#25** (the larger axis refactor).

All are additive/low-risk except #18 (wide rename) and #25 (study core). None
touch the model math. Branch off `main` per item or per batch; the suite's only
expected failure is the pre-existing `test_full_elbo_reproduces_pre_refactor_loss`
(PyTorch ≥2.9 numerics drift).

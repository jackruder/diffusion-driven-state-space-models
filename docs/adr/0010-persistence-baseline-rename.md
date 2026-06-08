# Rename `IdentityBaseline` / form-string `"identity"` → `PersistenceBaseline` / `"persistence"`

The deterministic baseline μ_p(z_{t-j:t-1}) = z_hist[..., -1] was historically called the
"identity" form because at j=1 it reduces to μ_p(z_{t-1}) = z_{t-1} — the
identity map. With the upcoming higher-order Markov work (EEG-style j>1
encoders) "identity" stops being accurate: the operator is `(B, d, j) → (B, d)`
selecting the most-recent slot, not the identity map on the j-step window. The
correct term is the standard forecasting one — "persistence" (last-value-carried-
forward) — so we renamed before introducing j>1 to avoid permanently embedding
the misleading name in checkpoints, study artifacts, and user-facing preset
names.

## Decision

1. **Class.** `IdentityBaseline` → `PersistenceBaseline` (in
   `src/ddssm/model/centering/baselines.py`). All imports, `__all__` re-exports,
   and the `IdentityBaselineB` hydra-zen builder symbol in
   `src/ddssm/experiment/builders.py` move with it.

2. **Form-string keyword.** The `baseline_form` value `"identity"` → `"persistence"`
   in `experiments/init_centering/cells.py` (`BASELINE_FORMS`,
   `_PARAM_FREE_FORMS`), `experiments/init_centering/model.py` (`Literal` type
   annotation + runtime branch), `experiments/init_centering/report.py`
   (`_BASELINE_FORMS` + the "param-free" cell-omission checks), and every
   parametrize tuple / string in `tests/`. The parameter name `baseline_form`
   itself is unchanged.

3. **Preset names.** The auto-generated study cell name
   `init_identity_pinned_per_t` becomes `init_persistence_pinned_per_t` (via the
   `cell_name(form, mode, tracking)` formatter); `init_identity_pinned_fixed`
   becomes `init_persistence_pinned_fixed`. The single explicit reference in
   `experiments/init_centering/study.py:_B6000_CELLS` is updated.

4. **Docs.** `CONTEXT.md`'s "Baseline form" entry, `README.md`'s preset table,
   `src/ddssm/model/README.md`, `docs/authoring/model.md`, and the
   `select-hyperparameters` SKILL.md are updated; the previous name is preserved
   *only* as a one-line "previously called identity" note pointing here.

## Consequences

- **Preset-name churn.** Any external script, dashboard, or notebook that
  references `init_identity_*` presets will break — replace with
  `init_persistence_*`. The study orchestrator (`python -m ddssm.launch
  init_centering`) regenerates names from the renamed `BASELINE_FORMS` tuple, so
  no separate registration changes are needed.

- **No checkpoint backward compat.** Historical checkpoints and any pickled
  resolved configs that name `IdentityBaseline` or carry `baseline_form:
  identity` will fail to load against the renamed class / fail Hydra's `Literal`
  validation. We considered an import alias (`IdentityBaseline =
  PersistenceBaseline`) but deliberately omitted it: the rename is meant to
  prevent the wrong term from leaking into new j>1 work, and a silent alias
  defeats that. There is no automated migration script.

- **Stale historical artifacts.** Past `metrics.csv`, `resolved_config.yaml`,
  and Optuna study DBs from runs prior to this commit will reference the old
  name (`baseline_form: identity`). Cross-rename comparisons require a manual
  string substitution in the report layer; in practice the historical artifacts
  are read-only and we accept the stale label rather than rewriting them. The
  Phase-E report plots regenerate from `records.jsonl` which is recomputed from
  scratch each run, so this only affects pre-existing JSONLs.

- **`"identity"` is preserved elsewhere.** The diffnets mixer types
  (`time_type="identity"` / `feature_type="identity"`, dispatched to
  `IdentityLayer`) and the encoder aggregator (`IdentityAggregator`, agg_name
  `"identity"`) are *different concepts* (no-op mixers / a different aggregator)
  and are not renamed.

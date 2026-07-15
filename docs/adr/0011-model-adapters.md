# Baseline models enter the Experiment workflow through a ModelAdapter hierarchy

Head-to-head baselines (CSDI now; GluonTS estimators, TimeGrad, … later) must
train, evaluate, and sweep through the **same** machinery as the native DDSSM
model: `python -m ddssm.app experiment=<preset>`, `+sweep=` Optuna searches,
`ObjectiveSpec` reading `metrics.csv`/`metrics.json`, standalone
`python -m ddssm.evaluate/visualize/variance +checkpoint=...`, and an unchanged
run-directory layout. The hard requirements are data parity (identical
splits / context / horizon across families) and metric parity, with no rewrite
of the workflow per family.

Before this change `Experiment.model` was a bare `DDSSM_base` and
`Experiment.train` inlined the whole `DDSSMTrainer` construction + fit. A second
family had nowhere to plug in: the eval/viz/variance runners duck-typed
DDSSM-only surface (`model.log_prob`, `model(x, mask, t, train=False)`, `.j`,
`.sigma_data`, `.transition.sigma_tilde`, …), and the trainer/optimizer knobs
lived half in `DDSSMHyperParamsConf` and half in the curried `TrainerPartial`.

## Decision

`Experiment` stays the single orchestrator and delegates to a **`ModelAdapter`**
— an ABC (`src/ddssm/adapters/base.py`) that integrates one model family behind
a uniform surface: `module` (the raw checkpointable `nn.Module`), `fit`,
`forecast`, `save_checkpoint`, `load_checkpoint`, plus an optional-but-shared
`log_prob`. The underlying model is a plain `nn.Module` **owned** by the adapter.

- **DDSSM is NOT itself an adapter.** `DDSSM_base` stays an untouched
  `nn.Module`; `DDSSMAdapter` wraps it and owns the `DDSSMTrainer` internally
  (the `build_trainer` partial moved off `Experiment` and into the adapter).
- **`CSDIAdapter`** wraps its **own re-vendored** copy of upstream CSDI at
  `src/ddssm/adapters/_csdi_vendor/` — deliberately NOT shared with
  `model/transitions/_csdi_vendor/` (which still serves `CSDITransition`). The
  duplication is accepted now; a unification refactor is future work. The
  adapter copy is kept byte-identical to upstream so the eventual merge is
  mechanical.

Every family defines a **`ModelConfig` subclass** — the uniform config currency
(`src/ddssm/model/config.py`, one field: `batch_size`). All family knobs live
there so sweeps target one place:

- `DDSSMHyperParamsConf(ModelConfig)` — zero field changes; existing presets,
  sweeps, and `tweak()` keep working verbatim. DDSSM architecture stays
  compositional (encoder/decoder/transition composed by family factories like
  `SmokeModel`); it is NOT flattened into the config.
- `CSDIConfig(ModelConfig)` carries **everything** — architecture (`layers`,
  `channels`, `num_steps`, …) AND optimizer (`lr`, `weight_decay`,
  `lr_decay_milestones`, …).

Config precedence mirrors ADR-0004/ADR-0005: the `hparams` argument (from
`Experiment.hparams`, the single source of truth) wins over the adapter's
constructor `config` at `fit`/`load_checkpoint`. For CSDI that governs optimizer
knobs AND lazy module construction; for DDSSM the module is pre-composed, so
`hparams` governs trainer/optimizer knobs only and can never rebuild topology.

Existing presets migrate with **zero edits**: `experiments/_make.experiment(...)`
inspects the model conf's `hydra_zen.get_target`. Adapter-targeting confs get
`config=hparams` curried on; bare DDSSM model confs (which target *functions*
like `_build_init_centering_model`) are wrapped in `DDSSMAdapterC(module=...,
config=hparams, build_trainer=TrainerPartial(...))`. A `wrap: bool = True`
escape hatch on `experiment()` covers the rare opt-out.

### Checkpoint parity

Every adapter round-trips across a fresh process so standalone `evaluate` works
without the training process. Payloads carry a `_format` tag plus the full
adapter-wrapped `model_config_yaml` for drift diffing:

- DDSSM: `ddssm_ckpt_v3` (unchanged; `DDSSMTrainer.save_checkpoint`).
- CSDI: `csdi_ckpt_v1` — `{model_state, optimizer_state, scheduler_state,
  global_step, rng_state, model_config_yaml}`, via the shared `atomic_save` /
  `check_model_config_drift` helpers extracted from `checkpoint.py`.

Cross-format loads raise `ValueError` via an explicit `_format` check in each
adapter — the pre-existing "warn and best-effort partial-load on unknown
format" behaviour would silently mis-load a foreign payload.

### Metric gating

Not every family supports every metric (CSDI has no `log_prob`, no recon-ELBO,
no variance probe). Gating is **method-level**, not a capability taxonomy:

- `MetricNotSupported(NotImplementedError)` is raised at exactly two kinds of
  boundary: the ABC's shared-optional methods (base `log_prob` raises, naming
  the class), and family-internal metrics/plots that begin with
  `model = ctx.require_module(DDSSM_base)` — an `EvalContext`/`PlotContext`
  helper that returns `adapter.module` after an isinstance check or raises.
- The eval, viz, and variance runners catch **`MetricNotSupported`** (the narrow
  subclass, NEVER bare `NotImplementedError`) → log a WARNING skip and omit that
  metric from `metrics.json`. Catching the broad class would silently swallow
  the load-bearing `NotImplementedError`s raised deep inside DDSSM internals
  (`encoder.py`, `transitions.py`, dtype/device op gaps) and mask real bugs.
- This is safe for JSON-source objectives: `ObjectiveSpec._read_json` routes
  missing keys through `_apply_penalty` and `report.py` guards with
  `if k in payload`.

## Considered alternatives

- **Make `DDSSM_base` itself the first `ModelAdapter`.** Rejected: it would
  entangle the research model with orchestration plumbing and force every future
  `nn.Module` edit through the adapter contract. Wrapping keeps `DDSSM_base` a
  pristine `nn.Module` and gives one orchestration path for all families.
- **Share the single `_csdi_vendor/` copy between the transition and the
  baseline adapter.** Rejected for now: the transition copy has repo-specific
  edits and the baseline needs upstream fidelity for its published-number
  sanity check. Duplication now, unification later, byte-identical in the
  meantime.
- **A capability-set / taxonomy on adapters (`supports={"nll", "recon"}`) that
  the runner introspects.** Rejected as premature: method-level gating via
  `require_module` + `try/except MetricNotSupported` is one line at each point
  of need and needs no registration metadata. Revisit only if static
  introspection is ever required.
- **Build the CSDI baseline from `ddssm.nn` blocks** instead of re-vendoring.
  Rejected: fidelity risk — the vendored `forward`/`evaluate` are left untouched
  so the baseline reproduces upstream behaviour exactly.

## Consequences

- `Experiment.model` is now a `ModelAdapter`; `Experiment.hparams` is a
  `ModelConfig`. The `build_trainer` field and the `TrainingScalars.trainable`
  mask (a staged-training remnant — `StageOrchestrator` was already deleted) are
  gone. `prepare_model` became an adapter dispatcher.
- Eval/viz/variance `ctx.model` is now the **adapter** (its `.forecast` /
  `.log_prob` work directly); DDSSM-internal metrics reach the raw module via
  `ctx.require_module(DDSSM_base)`.
- **Old-checkpoint drift warning:** a pre-refactor `ddssm_ckpt_v3` whose saved
  `model_config_yaml` is a *bare* model conf now loads against an
  *adapter-wrapped* expected yaml. `check_model_config_drift` unwraps the
  wrapper's `module:` subtree before diffing, so pre-refactor checkpoints emit a
  single concise WARNING (not an error) and load fine.
- **wandb config-snapshot drift (accepted):** the `resolved_exp` snapshot
  (`app.py`) gains `model.module.*` nesting. Dashboards keyed on the old
  `model.*` paths need a one-time update.
- **Swept-arch standalone eval (accepted, no regression):** evaluating a trial
  checkpoint whose architecture fields were swept requires passing that trial's
  `experiment.hparams.*` overrides on the eval CLI — the same limitation DDSSM
  arch sweeps already have.
- **Untested surfaces, accepted for now (not gated):** wandb sink through the
  CSDI `MetricStore` stack; `python -m experiments sbatch` render on a CSDI
  preset; Optuna pruning (none wired today; adapters don't regress it).
- CSDI fidelity caveats: step-based LR milestones re-derived from the paper's
  epoch schedule; no EMA; `gt_mask` reproduces the upstream forecasting-pattern
  semantics only with `target_strategy="test"`. Sanity-check `csdi_solar` CRPS
  against published numbers before trusting it.

## Landed

- New: `src/ddssm/model/config.py` (`ModelConfig`); `src/ddssm/adapters/`
  (`base.py` ABC + `MetricNotSupported`, `ddssm.py`, `csdi.py`,
  `_csdi_vendor/`); `experiments/csdi/` family (`csdi_smoke` on an in-memory
  windowed data module, `csdi_solar` on GluonTS Solar, `+sweep=csdi_lean`).
- Changed: `DDSSMHyperParamsConf`/`CSDIConfig` subclass `ModelConfig`;
  `Experiment` fields + `train()`/`objective_value()` delegate to the adapter;
  `prepare_model` dispatches to `adapter.load_checkpoint`; `atomic_save` /
  `check_model_config_drift` extracted in `checkpoint.py`; eval/viz/variance
  runners gate on `MetricNotSupported`; `_make.experiment()` wraps/curries;
  data-module ABC renamed `DDSSMDataModule` → `TimeSeriesDataModule`.
- Removed: `TrainingScalars.trainable`, the `Experiment.build_trainer` field,
  and the `StageTrainableConf` builders (`StageOrchestrator` was already gone).
  `DDSSMTrainer._set_trainable` remains as dead code (smaller diff).
- Contract enforced by a parameterized harness (`tests/adapters/`) both families
  instantiate: fit writes `loss/total` train+val rows, cross-process checkpoint
  round-trip, cross-format `ValueError`, forecast shapes in normalized space,
  NullDataModule no-op.

## Follow-up

- Unify the two `_csdi_vendor/` copies once the baseline's fidelity is
  confirmed; keep the adapter copy byte-identical to upstream until then.
- Export `CSDIAdapter` from `ddssm.adapters.__init__` (today imported via the
  submodule path) when the package seam is next revisited.
- Wire GluonTS-estimator / TimeGrad adapters onto the same ABC.

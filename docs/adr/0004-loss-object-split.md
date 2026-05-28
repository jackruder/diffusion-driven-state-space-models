# Split training loss into LossComponents + per-stage Loss object

`DDSSM_base.forward()` currently computes every ELBO term, multiplies
each by its λ, sums them, and returns a single scalar. The trainer
backprops the scalar; the `trainable` mask is the only mechanism for
stage-aware control, and it operates on *which parameters update*, not
on *which terms contribute to the scalar*.

That conflation makes ablations like "drop the recon term entirely"
(CSDI-equivalent: train the diffusion transition only, no encoder /
decoder contribution) awkward — they require either monkey-patching the
forward pass, freezing modules and hoping the recon term becomes a
constant in the loss, or a parallel model class.

## Decision

Refactor `DDSSM_base.forward()` to return a `LossComponents` dataclass
(the four per-term tensors, unweighted). Add a **loss object**
abstraction (registered builder) taking `(components, λ_state) →
scalar`. Each stage in `StageOrchestrator` carries its own loss object
with its own λ schedule (per-stage selection — Depth 3 of the
conversation that produced this ADR).

The **trainable mask** stays. Loss object and trainable mask are
orthogonal mechanisms — see `CONTEXT.md` for the terminology split.

## Considered alternatives

- **Depth 1 — boolean flag inline in `DDSSM_base`.** Rejected: the
  third ablation request would force the refactor anyway, and the
  flags clutter the central hot path.
- **Depth 2 — module-level loss function, single objective per
  experiment.** Rejected: cannot express per-stage term selection,
  which multi-stage runs need (e.g. stage 1 may legitimately want
  only the recon + KL terms, not the transition term).
- **Custom model class per ablation** (e.g. `DDSSMNoSSM` for CSDI
  comparison). Rejected: creates a parallel model lineage; spec
  schema and shared trainer would fork.

## Consequences

- `DDSSM_base.forward()` returns `LossComponents`, not `total_loss`.
  All call sites (trainer, eval runner, variance probe) update.
- Existing presets (`init_centering` × 20, `kdd`, `synthetic`,
  `variance_probe`) migrate to declare their loss object. Default
  builder is `FullELBO` — chosen to reproduce current behaviour
  numerically on a fixed batch (lockdown test).
- λ scheduling moves from `DDSSMTrainer` to the loss object. Per-stage
  λ ramps become explicit in `StagesConf` (today they're a single
  trainer-owned schedule). The loss object holds the schedule *shape*
  as a pure function `λ(step) → float`; the trainer drives `step`,
  passing `step_within_stage` so each stage's λ ramp starts fresh at
  stage boundaries (matching the "each stage is its own training
  phase" semantics that the handoff perturbation already assumes).
- The CSDI-comparison ablation (the experiment that motivated this
  ADR) becomes a `DiffusionOnly` loss object + identity-passthrough
  encoder/decoder builders — no trainer or model surgery required at
  that experiment's site.
- Once λ-schedule fields leave `Hparams`, `model.config.hyperparams`
  has no remaining model-side readers (every surviving reader is the
  trainer). The follow-up cleanup collapses the `Experiment.hparams` /
  `model.config.hyperparams` duplication onto `Experiment.hparams`,
  with `Hparams` shrunk to pure optimisation knobs (lrs, batch_size,
  weight_decay, ema_decay, grad_accum_steps) and the defensive
  re-sync in `Experiment.train` deleted. λ-shape fields live on the
  loss object's config instead.
- The migration sweep also moves the `stages` config from
  `model.config.stages` to `Experiment.training.stages` (each stage's
  loss object is an orchestration concern, not a model concern).
  `StageOrchestrator` then takes `(trainer, training.stages)` and
  stops needing access to `model.config`. Done in the same sweep to
  avoid touching every preset twice.

## Landed in this commit

- `LossComponents` dataclass, `Loss` ABC, `FullELBO`
  (`src/ddssm/losses.py`); `.elbo()`, `.elbo_reg()`, `.total()` helpers.
- `DDSSM_base.forward()` returns `(LossComponents, metrics, stats)`;
  regularizers are unweighted in `LossComponents` and the
  metrics-dict `loss/rate/trans/r_*` keys now report unweighted
  values too.
- `StageSpecConf.loss: Loss | None`; `StageOrchestrator` installs
  `trainer._active_loss` per stage, falling back to a default
  `FullELBO` from the stage's `lambda_ramp` + the model's
  `anchor_lambda` (the latter is the only non-loss source of
  `λ_μp` until `anchor_lambda` migrates).
- Trainer uses `self._active_loss(components, step_within_stage)`;
  per-stage-fresh-step semantics via `_stage_start_step`.
- Orchestrator no longer reads `hparams.lambda_end` /
  `hparams.lambda_sigma_p`; model `forward()` no longer reads
  `hparams.lambda_sigma_p`. Only `_build_default_loss` in the
  trainer still reads λ-shape hparams as a graceful fallback for
  un-migrated single-stage presets.
- `experiments/init_centering/hparams.py` migrated: `SmokeHparams`
  carries no λ fields; stage 1's `FullELBO` is set explicitly with
  `λ_σp = 1e-2`.

## Deferred follow-up

Tracked here so a follow-up session has the punch list:

- **Hparams field removal.** `lambda_schedule`, `lambda_start`,
  `lambda_end`, `lambda_warmup_steps`, `lambda_sigma_p` are still
  defined on `DDSSMHyperParamsConf` for backwards compat with
  un-migrated presets (KDD, synthetic, variance_probe, several test
  fixtures). Removing requires either dropping them from those
  presets (behavior change: no rate-λ ramp for single-fit runs) or
  adding `Experiment.loss` so single-stage presets can declare an
  explicit loss object.
- **`anchor_lambda` migration.** Still lives on the model; the
  init_centering sweep targets `experiment.model.anchor_lambda`.
  Migrating to `experiment.model.stages.stage_2.loss.lambda_mu_p`
  is a sweeper-config breaking change handled separately.

## Landed in the migration sweep

- **`model.config.stages` → `Experiment.training.stages`.** The
  multi-stage spec now lives on `TrainingScalars.stages`;
  `StageOrchestrator` takes it directly (`__init__(trainer, stages)`)
  with no `cfg.stages` indirection. The init_centering preset family
  + the init_ablation sweep keys (`experiment.training.stages.X`)
  migrated.
- **`Experiment.hparams ↔ model.config.hyperparams` collapse.**
  `DDSSM_base` no longer accepts `hyperparams=` and the
  `model.config.hyperparams` field is gone. `DDSSMTrainer` takes
  `hparams=` directly; `experiments._make.experiment(...)` curries it
  onto `TrainerPartial`. The defensive re-sync inside
  `Experiment.train` is deleted.

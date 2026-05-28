# Checkpoint config is authoritative for model architecture; passed `experiment=` is cross-checked

The standalone post-training stages (`ddssm.eval`, `ddssm.visualize`,
`ddssm.variance`) take `experiment=NAME` + `checkpoint=PATH`, build a
fresh model from the re-resolved Hydra config, then patch in the
checkpoint's parameter values via `load_state_dict(strict=True)`. The
checkpoint's `payload["config"]` (written by `save_checkpoint`) is
dead data — never consumed on the load path.

This creates a silent-drift hazard: if the registered preset or its
defaults change between training and post-hoc evaluation, and the
edits happen to preserve parameter shapes, `strict=True` accepts the
load and the user evaluates a different architecture than what
produced the checkpoint. Shape-mismatch cases are caught loudly, but
semantic edits with identical shapes are silent.

## Decision

The model architecture is rebuilt from the checkpoint's saved config,
not from the CLI's `experiment=...`. The passed experiment is used
for data + eval/viz/variance specs, and its `model.config` is
**cross-checked** against the checkpoint's saved config — any
structural difference logs a `WARNING` (with a diff). This catches
silent drift without forcing the user to give up the
`experiment=NAME` shorthand for declaring intent.

Concretely:

- `DDSSMTrainer.save_checkpoint` writes the **resolved YAML** of
  `model.config` (hydra-zen `builds()` shape, not the current
  asdict dump) into `payload["model_config_yaml"]`. The `.pth` is
  self-sufficient for rebuilding the model.
- Standalone CLIs require `experiment=NAME` in the normal case.
  Mismatch between `payload["model_config_yaml"]` and the
  instantiated `experiment.model.config` logs a warning showing the
  diff; the checkpoint wins for model construction; the passed
  experiment supplies data + spec.
- If `experiment=` is omitted, the CLI logs a warning and rebuilds
  the model from the checkpoint alone. Operations needing data /
  eval / viz / variance specs (i.e. all of them) error explicitly —
  the checkpoint deliberately does not carry these (see Considered
  alternatives, option Y).

## Considered alternatives

- **(a) Minimal cleanup — stop writing the dead `config` payload,
  document a "user must pass matching experiment" contract.**
  Rejected: relies on user discipline; silent semantic-drift case
  remains.
- **(b) `resolved_config.yaml` (in the run dir) is authoritative;
  standalone CLIs take `run_dir=...` not `experiment=...`.**
  Rejected for now: real CLI redesign, ties checkpoints to their
  run directories (a `.pth` shared without its run dir can't be
  loaded). Worth revisiting if `.pth`-only handoff is rare.
- **(Y) Save the *full* `Experiment` config (data + eval + viz +
  variance specs) into the checkpoint, so a `.pth` runs end-to-end
  without `experiment=`.** Rejected: eval/viz specs are post-hoc
  choices the user is allowed to vary on the same trained model
  (today's `crps_sum`, tomorrow's `mae`); locking them into the
  checkpoint is the wrong default. Architecture is the only thing
  that *must* match for the parameters to mean anything.

## Consequences

- Checkpoint format gains `payload["model_config_yaml"]` (the
  resolved hydra-zen config). Old checkpoints without this key
  trigger a warning and fall back to "trust the CLI" behaviour
  (current semantics) so the change isn't breaking on read.
- The asdict-dumped `payload["config"]` is preserved for one
  release as a transitional field, then removed.
- Standalone-CLI runners (`eval/runner.py`, `viz/runner.py`,
  `variance/probe.py`) gain a `_load_checkpoint_with_config_check`
  helper that does the rebuild + diff + warn.
- Reproducibility scenario fixed: a preset edit that preserves
  shapes but changes semantics (different encoder builder, same
  output dim) now surfaces as a warning on every post-hoc run
  against pre-edit checkpoints.
- Orthogonal fix triggered by the same audit: variance probe loads
  only `model_state` and ignores `ema_state`, but the diffusion
  sampling path is normally driven from EMA shadows at training
  time. Either load the EMA shadows into `transition` for the probe
  (and other sampling-path stages) or document why live weights are
  the right choice for probing. To be settled when the load helper
  is written.

## Landed

- `DDSSMTrainer.__init__` accepts `model_config_yaml: str | None`; the
  four Hydra entry points (`app.py`, `evaluate.py`, `visualize.py`,
  `variance/cli.py`) snapshot `OmegaConf.to_yaml(cfg.experiment.model,
  resolve=True)` onto `experiment.model_config_yaml` and forward it
  through `build_trainer(...)`.
- `save_checkpoint` writes `payload["model_config_yaml"]`. The legacy
  `payload["config"]` (asdict of the runtime SimpleNamespace) is
  preserved one release as a transitional debugging aid.
- `ddssm.train.load_checkpoint_with_config_check` does the rebuild +
  unified-diff + WARNING. Wired into `eval/runner.py`, `viz/runner.py`,
  and `variance/probe.py`. Shape-mismatching edits still fail loudly
  via `load_state_dict(strict=True)`; semantic-only edits now surface
  as a warning that names the changed lines.
- The "missing `experiment=`" branch (rebuild from checkpoint alone)
  is deferred — every standalone CLI today requires `experiment=` to
  satisfy data/eval/viz specs, so omitting it already errors via Hydra
  resolution.

## Follow-up: checkpoint module (`src/ddssm/checkpoint.py`)

The save/load split (save = trainer method, load = free function)
became a single module. `Checkpoint` (a dataclass) is the payload
schema; `save(trainer, path)`, `load_into_model(...)`, and
`prepare_model(...)` are the entry points. `DDSSMTrainer.save_checkpoint`
/ `restore_from_checkpoint` delegate. The standalone stages load via
`prepare_model`, which folds in the config cross-check so no stage can
forget it (the candidate-1 win lands here).

**EMA-on-load — decided: all inference uses EMA.** The orthogonal EMA
gap is resolved as a `load_ema` parameter on `load_into_model` /
`prepare_model`. The decision: **inference uses the EMA model** — the
transition weights the sampling path used at training time — in every
read-only context.

- `prepare_model` defaults `load_ema=True`, so the standalone stages
  (eval / viz / variance) all load the transition's EMA shadows. Pass
  `load_ema=False` for the rare case wanting raw live weights.
- Validation during training (`DDSSMTrainer._run_validation`) runs
  inside `EMA.swap()`, so val metrics reflect the EMA model and then
  the live weights are restored for the next training step.
- The resume path (`restore_from_checkpoint`) stays on live weights
  (`load_ema=False`) — resuming training continues the live
  trajectory; the EMA shadows go back into the trainer's EMA tracker,
  not the transition.

This is a deliberate **research-results change**: forecast metrics,
the stage-2 ELBO surrogate, the variance-probe numbers, and logged
`val` losses all now reflect the EMA model rather than raw live
weights. Pre-flip baselines are not comparable to post-flip numbers.
The trade-off accepted: EMA fidelity (matches how the model actually
samples) over raw-weight transparency.

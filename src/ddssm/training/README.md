# `ddssm.training`

The training stack for `DDSSM_base`: the core training loop, multi-stage
orchestration, the checkpoint payload schema, and the metric-logging
helpers. It owns everything between an assembled `Experiment` and the
artifacts written to a run directory (`metrics.csv`, `tb_logs/`,
checkpoints). The forward pass always computes the full ELBO; this layer
decides *what trains*, *for how long*, and *what gets recorded*.

## Files

- **`train.py`** — `DDSSMTrainer`, the training harness. Owns per-submodule
  AdamW optimizers (separate LRs for encoder / decoder / z_init / transition),
  optional bf16 AMP, gradient accumulation, EMA tracking (`EMA`), the λ-warmup
  schedule, and step-level logging to CSV / TensorBoard / W&B. `fit(...)` runs
  the loop (step-counted validation, checkpointing, profiling, ELBO-plateau
  early stop) and raises `PreemptError` on a caught preempt signal after saving
  a resume checkpoint. `_set_trainable(...)` toggles `requires_grad` per
  submodule — the single mechanism for stage-aware gradient suppression:
  frozen submodules still run forward but accumulate no gradients.
- **`stages.py`** — `StageOrchestrator` plus its config dataclasses
  (`StagesConf`, `StageSpecConf`, `StageTrainableConf`, `StageLrsConf`,
  `StageSchedulerConf`, `LambdaRampConf`, `EarlyStopSpec`). Runs sequential
  phases (e.g. recon-only → trans-only → joint) in `stages.run` order; for each
  stage it flips `model.stage_selector`, applies the per-stage trainable mask
  and LRs (rebuilding the optimizer, or deferring to a `centering_handoff`
  hook), sets the λ-ramp, and drives `trainer.fit` for the stage's step budget.
  `make_lambda_cosine(...)` builds the cosine λ schedule.
- **`train_utils.py`** — optimizer-construction helpers.
  `param_groups_for_adamw(...)` builds per-component AdamW param groups with
  selective weight decay (norm/bias/embedding/log-var params get zero decay)
  and de-duplicates shared submodules; `make_warmup_cosine(...)` builds a
  linear-warmup + cosine-decay `LambdaLR`.
- **`checkpoint.py`** — the single owner of the `.pth` payload schema. The
  `Checkpoint` dataclass captures model/optimizer/EMA/scaler/scheduler state,
  step counter, and `stage_prefix`; `save`/`load_into_model`/`prepare_model`
  handle atomic writes, format-version tolerance, a model-config cross-check
  (warns on drift), and optional EMA-shadow loading for inference.
- **`loggers.py`** — metric aggregation and sinks. `MetricStore` fans flushed
  rows out to `ConsoleLogger`, `CSVLogger`, `TensorBoardLogger`, and
  `WandbLogger`; `MetricSpec` selects per-key meter kinds
  (`mean`/`sum`/`last`/`ema`) by glob.

## How it fits

An `Experiment` (see `src/ddssm/experiment.py`) supplies a `build_trainer`
partial that constructs the `DDSSMTrainer`; single-stage runs call `fit`
directly while multi-stage presets wrap it in a `StageOrchestrator`. The
per-stage trainable mask — applied through `DDSSMTrainer._set_trainable` — is
the only gradient-suppression mechanism, so stage behavior is fully declarative
in `StagesConf`. Modules use absolute imports (`from ddssm.training... import`).

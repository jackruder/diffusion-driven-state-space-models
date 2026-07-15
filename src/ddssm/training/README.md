# `ddssm.training`

The training stack for `DDSSM_base`: the core training loop, the LR/λ
schedule config, the checkpoint payload schema, and the metric-logging
helpers. It owns everything between a `DDSSMAdapter` (which drives this
stack — see `src/ddssm/adapters/`) and the artifacts written to a run
directory (`metrics.csv`, `tb_logs/`, checkpoints). The forward pass
always computes the full ELBO; this layer decides *for how long* to train
and *what gets recorded*.

## Files

- **`train.py`** — `DDSSMTrainer`, the training harness (owned by
  `DDSSMAdapter`). Owns per-submodule AdamW optimizers (separate LRs for
  encoder / decoder / z_init / transition), optional bf16 AMP, gradient
  accumulation, EMA tracking (`EMA`), the λ-warmup schedule, and step-level
  logging to CSV / TensorBoard / W&B. `fit(...)` runs the loop (step-counted
  validation, checkpointing, profiling, ELBO-plateau early stop) and raises
  `PreemptError` on a caught preempt signal after saving a resume checkpoint.
  (`_set_trainable(...)` — the former per-submodule `requires_grad` mask — is
  now dead code; staged training has been removed.)
- **`stages.py`** — LR-schedule and λ-ramp config dataclasses (`StageLrsConf`,
  `LrScheduleConf`, `LrScheduleGroupConf`, `LambdaRampConf`, `EarlyStopSpec`, …)
  plus their resolvers: `make_lr_lambda(...)` /
  `resolve_lr_schedule_defaults(...)` build the per-component LR lambdas and
  `make_lambda_cosine(...)` the cosine λ schedule — consumed by `train.py` and
  `dssd.py`. The sequential `StageOrchestrator` has been removed; the
  `Stage*Conf` dataclasses that remain are the leftover schedule pieces still
  read by the trainer.
- **`train_utils.py`** — optimizer-construction helpers.
  `param_groups_for_adamw(...)` builds per-component AdamW param groups with
  selective weight decay (norm/bias/embedding/log-var params get zero decay)
  and de-duplicates shared submodules; `make_warmup_cosine(...)` builds a
  linear-warmup + cosine-decay `LambdaLR`.
- **`checkpoint.py`** — owner of the DDSSM (`ddssm_ckpt_v3`) `.pth` payload
  schema. The `Checkpoint` dataclass captures model/optimizer/EMA/scaler/
  scheduler state and step counter; `save`/`load_into_model` handle atomic
  writes (via the shared `atomic_save`), a model-config cross-check (via
  `check_model_config_drift`, which unwraps adapter-wrapper yaml before
  diffing), and optional EMA-shadow loading for inference. `prepare_model` is
  now an adapter dispatcher: it forwards to `adapter.load_checkpoint(...)` so
  each family loads its own format (the CSDI adapter owns `csdi_ckpt_v1`).
- **`loggers.py`** — metric aggregation and sinks. `MetricStore` fans flushed
  rows out to `ConsoleLogger`, `CSVLogger`, `TensorBoardLogger`, and
  `WandbLogger`; `MetricSpec` selects per-key meter kinds
  (`mean`/`sum`/`last`/`ema`) by glob.

## How it fits

`DDSSMAdapter` (see `src/ddssm/adapters/ddssm.py`) constructs the
`DDSSMTrainer` and calls `fit(...)` — `Experiment.train` delegates to the
adapter rather than building a trainer itself. The adapter forwards the
`ModelConfig` (`Experiment.hparams`) so optimizer/LR knobs resolve there. Modules
use absolute imports (`from ddssm.training... import`).

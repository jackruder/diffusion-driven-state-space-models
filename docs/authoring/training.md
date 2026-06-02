# Step 2 — Training & hyperparameters

Training is configured by three pieces passed to
{py:func}`experiments._make.experiment`: `hparams` (optimizer/MC knobs),
`training` (run scalars), and optionally `stages` (multi-stage orchestration).

## Hyperparameters — `Hparams(...)`

`Hparams` wraps `DDSSMHyperParamsConf` (`src/ddssm/model/dssd.py`); the trainer
reads it directly.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `S` | 1 | Monte-Carlo encoder samples per element (memory ↑; variance ↓). |
| `enc_lr` / `dec_lr` / `trans_lr` | 5e-4 | per-submodule learning rates. |
| `batch_size` | 16 | loader batch size (overrides the data module's). |
| `grad_accum_steps` | 4 | optimizer step every N micro-batches (effective batch = `batch_size × grad_accum_steps`). |
| `ema_decay` | 0.999 | EMA of the transition weights (used at val/sampling). |
| `weight_decay` | 1e-2 | AdamW decay; **not** applied to norms/biases/embeddings/decoder logvar (selective grouping in `param_groups_for_adamw`). |
| `clip_grad_norm` | None | global grad-norm clip (None = off). |
| `logvar_min` / `logvar_max` | -7 / 7 | clamp on encoder/decoder log-variances. |

Per-submodule LRs are realized by
{py:func}`ddssm.training.train_utils.param_groups_for_adamw`, which also splits
each group into decay / no-decay sub-groups.

## Run scalars — `Training(...)`

`Training` wraps {py:class}`~ddssm.experiment.experiment.TrainingScalars`:

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `steps` | 1000 | optimizer steps **(single-fit only; ignored under `stages`)**. |
| `log_every` | 50 | metric-flush cadence (ignored under `stages`). |
| `validate_every` | 0 | validation cadence (0 = off; ignored under `stages`). |
| `checkpoint_every` | None | periodic checkpoint cadence. |
| `amp` | True | bf16 autocast. |
| `profile_steps` | 0 | profile the first N steps if > 0. |
| `resume_from` | None | checkpoint path to resume (stage-aware). |
| `trainable` | None | per-module `requires_grad` mask for the single-fit path. |
| `stages` | None | multi-stage spec — see below. |

```{important}
**Single-fit vs. multi-stage.** When `stages` is `None`, one `fit()` loop runs
using `steps`/`log_every`/etc. When `stages` is set, the
{py:class}`~ddssm.training.stages.StageOrchestrator` takes over and those
scalars are **ignored** — each stage carries its own budget. The shipped (and
worked-example) presets are multi-stage.
```

## The trainable mask — the one gradient switch

{py:meth}`DDSSMTrainer._set_trainable <ddssm.training.train.DDSSMTrainer>` flips
`requires_grad` per submodule (`encoder` / `decoder` / `transition` /
`baseline`). The forward pass always computes every ELBO term; frozen submodules
simply don't accumulate gradients. This `StageTrainableConf` mask is the single
mechanism for stage-aware freezing (e.g. pinning the baseline in stage 2).

## Multi-stage — `StagesConf`

`src/ddssm/training/stages.py` defines the orchestration. A `StagesConf` lists
`stage_1` / `stage_2` / `stage_3` (`StageSpecConf`) and a `run` order. Each
`StageSpecConf` owns:

- `steps` — this stage's budget.
- `trainable` (`StageTrainableConf`) — the per-module freeze mask.
- `lrs` (`StageLrsConf`) — per-submodule LRs for the stage.
- `lambda_ramp` (`LambdaRampConf`) — cosine ramp of the rate-λ (`start`, `end`,
  `steps`, `delay`) via `make_lambda_cosine`.
- `loss` — a `FullELBO` carrying the λ schedule + regularizer weights
  (`lambda_sigma_p`, `lambda_mu_p`); per ADR-0004 these live on the loss, not the
  model.
- `centering_handoff` (`CenteringHandoffConf`) — fires before stage 2: rebuilds
  the optimizer and applies the σ_data scaling + encoder perturbation
  (`sigma_pert`).
- `log_every`, `val_every`, `checkpoint_every`, `early_stop` (`EarlyStopSpec`).

The orchestrator iterates `run`, sets `model.stage_selector`, applies the
handoff (unless resumed), sets the trainable mask, installs the loss/λ schedule,
and calls `fit()`. Resuming from a stage-N checkpoint skips earlier stages.

## In the worked example

`synthetic_validation` (in `study.py`'s `_build`) reuses the init-centering
stage builder `StagesB` (itself `builds(_build_init_centering_stages)` in
`experiments/init_centering/hparams.py` — a good reference for hand-rolling a
`StagesConf`) with small budgets, plus a shared `Hparams`/`Training`:

```python
_HPARAMS = Hparams(S=1, batch_size=32, enc_lr=5e-4, dec_lr=5e-4, trans_lr=5e-4)
_TRAINING = Training(steps=400, log_every=25, amp=True)  # steps ignored under stages,
                                                         # but kept > 0 (sanity convention)
exp = experiment(..., hparams=_HPARAMS, training=_TRAINING,
                 stages=StagesB(baseline_mode="pinned", n_pretrain=100, n_stage2=300))
```

Stage 1 (100 steps) trains recon + baseline with a closed-form Gaussian
transition; the handoff freezes the (zero) baseline and perturbs the encoder;
stage 2 (300 steps) trains the diffusion transition. Override budgets from the
CLI for a quick smoke:

```bash
python -m ddssm.app experiment=synthval__harmonic \
    experiment.training.stages.n_pretrain=4 experiment.training.stages.n_stage2=4
```

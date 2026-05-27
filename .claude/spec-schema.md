# Experiment Spec Schema

The shared contract between the three experiment skills:

```
select-hyperparameters  →  spec.yaml  →  experiment-builder  →  preset on disk
                                     ↘
                                       compute-budget  →  launch command
```

All three skills read and write specs in the format below. Changes to this schema are breaking changes to all three skills — update them together.

## Full schema

```yaml
architecture:
  # The model class. All names refer to builders registered in
  # `src/ddssm/builders.py`; `select-hyperparameters` discovers what's
  # available at invocation time and does not enumerate from memory.

  transition: <name>            # required. e.g. "diffusion_v3", "gaussian"
  unet: <name>                  # required if transition is diffusion family; else null
  time_mixer: <name>            # required if unet is csdi residual stack; else null
  feature_mixer: <name>         # required if unet is csdi residual stack; else null
  encoder: <name>               # required
  decoder: <name>               # required
  z_init: <name>                # required

  multi_stage:                  # optional. omit for single-stage training.
    stages:                     # ordered list; runs in declaration order
      - name: <str>             # e.g. "stage_1", "stage_2"
        role: <str>             # informational; e.g. "recon_only", "joint"
        trainable:              # submodules with grad on during this stage
          encoder: <bool>
          decoder: <bool>
          z_init: <bool>
          transition: <bool>
          baseline: <bool>      # only meaningful if centering is in use
        compute_recon: <bool>   # must agree with `trainable` per the interlock
        compute_trans: <bool>   # (see CLAUDE.md § Trainer)
    centering_handoff:          # optional. only valid with multi_stage.
      between: [<stage>, <stage>]
      baseline_form: <name or grid_expression>
      # grid_expression: "<ablation across {form_a, form_b, form_c}>" — signals
      # to experiment-builder that this is a K-axis grid, not a single cell.

hyperparameters:

  fixed:
    # Concrete values that won't vary. Keys are short names matching
    # `Hparams` / `Training` / per-family hparams.py fields. Stay shallow —
    # this is human-edited, not a Hydra tree.
    <name>: <value>             # e.g. batch_size: 16, amp: false

  tuned:
    # Optuna search-space entries. Keys are dotted Hydra paths anchored
    # at `experiment.` — they go verbatim into the sweep dict.
    <experiment.dotted.path>:
      space: <optuna_grammar>   # see table below
      note: <str>               # free-text motivation; preserved as a
                                # comment in the generated sweeps.py

notes:
  # Free-form bullets surfaced into the generated preset's docstring
  # and into the budget skill's context. Use these to capture
  # motivation, related ADR / memory pointers, prior empirical results.
  - <str>
```

## Field semantics

- **Required vs optional.** Required fields must be present and non-`<unset>` before `experiment-builder` will scaffold. Optional fields default to "use the family default" in the generated preset.
- **`<unset>` vs `null`.** Use `<unset>` (literal string) when the user explicitly skipped a question — the downstream builder treats this as "do not materialise this field". Use `null` only when the field is structurally absent (e.g. `unet: null` for a Gaussian transition that has no UNet). The two are distinct: `<unset>` lets the family default apply, `null` declares "this field has no value for this architecture".
- **Stage trainable interlock.** Every `multi_stage.stages[].trainable` flag must agree with `compute_recon` / `compute_trans` per CLAUDE.md's "recon only" rule. The schema lets the user write either; `experiment-builder`'s guards reject inconsistent combinations.
- **Cell grids.** `baseline_form: "<ablation across {zero, identity, mlp}>"` is the canonical way to declare a K-axis ablation in the spec. `experiment-builder` parses this and generates an `iter_cells()`-style loop. Multiple grid axes compose multiplicatively.

## Optuna sweep grammar

The `tuned.<path>.space` field is a string in the project's Optuna grammar. Match the syntax in `experiments/*/sweeps.py` exactly:

| Distribution        | Syntax                                |
|---------------------|---------------------------------------|
| log-uniform float   | `tag(log, interval(LOW, HIGH))`       |
| uniform float       | `interval(LOW, HIGH)`                 |
| log-uniform int     | `tag(log, int(interval(LOW, HIGH)))`  |
| uniform int         | `int(interval(LOW, HIGH))`            |
| categorical         | `choice(A, B, C)`                     |

Numeric literals are floats by default; use the `int(...)` wrapper to force integer sampling. Strings inside `choice(...)` are unquoted symbols, not YAML strings.

## Example — multi-stage diffusion with centering grid

```yaml
architecture:
  transition: diffusion_v3
  unet: csdi_residual_stack
  time_mixer: conv
  feature_mixer: transformer
  encoder: dks_gaussian
  decoder: gaussian
  z_init: gaussian
  multi_stage:
    stages:
      - name: stage_1
        role: recon_only_with_baseline
        trainable: {encoder: true, decoder: true, z_init: false, transition: true, baseline: true}
        compute_recon: true
        compute_trans: false
      - name: stage_2
        role: joint
        trainable: {encoder: true, decoder: true, z_init: false, transition: true, baseline: false}
        compute_recon: true
        compute_trans: true
    centering_handoff:
      between: [stage_1, stage_2]
      baseline_form: "<ablation across {zero, identity, linear, mlp}>"

hyperparameters:
  fixed:
    batch_size: 16
    grad_accum_steps: 1
    amp: false
    lambda_schedule: cosine
    lambda_end: 1.0
  tuned:
    experiment.model.stages.base_lr:
      space: "tag(log, interval(1e-5, 1e-3))"
      note: "encoder LR baseline; decoder + transition LRs derived via multipliers"
    experiment.model.stages.sigma_pert:
      space: "tag(log, interval(1e-3, 5e-2))"
      note: "ADR-0002 forbids 0; lower bound is operationally near-0"
    experiment.model.stages.n_pretrain:
      space: "tag(log, int(interval(50, 500)))"
      note: "parametric μ_p needs convergence; see project-handoff-protocol-invariants memory"

notes:
  - "Round 1 gating: pinned baseline_mode only; learnable held for round 2."
  - "Multi-objective: pair with ddssm_optuna_moo sweeper."
```

## Example — tiny single-trial gaussian

```yaml
architecture:
  transition: gaussian
  unet: null
  time_mixer: null
  feature_mixer: null
  encoder: gaussian
  decoder: gaussian
  z_init: gaussian

hyperparameters:
  fixed:
    batch_size: 32
    lambda_warmup_steps: 200
    enc_lr: 5.0e-4
    dec_lr: 5.0e-4
    zinit_lr: 5.0e-4
    trans_lr: 5.0e-4
  tuned: {}

notes:
  - "Smoke-only; no sweep."
```

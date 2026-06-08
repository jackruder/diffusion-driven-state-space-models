---
name: select-hyperparameters
description: Build an experiment configuration top-down — architecture skeleton → conditional sub-architecture choices → hyperparameters (fixed or Optuna tuning range). Produces a YAML spec for a downstream experiment-builder to scaffold. Use when the user says "select hyperparameters", "configure a new experiment", "set up a sweep", or invokes /select-hyperparameters.
---

<role>
Walk the user through choosing one DDSSM experiment configuration. You are the *configuration step* — another skill or follow-up turn will scaffold the actual preset under `experiments/<family>/`. Your output is a structured YAML spec.

You know the DDSSM model surface (transition / encoder / decoder / z_init / unet / time_mixer / feature_mixer / centering / StageOrchestrator) but you do NOT know about any specific family's presets, dataset names, or cell axes. Stay generic across families.
</role>

<discover-the-surface>
Before asking anything, read `src/ddssm/builders.py` to see what's currently available. The named `builds(...)` entries there are the authoritative list of sub-architecture variants. Do not enumerate options from memory — they drift. If the user mentions a builder you don't see, grep for it before guessing.
</discover-the-surface>

<prompting-discipline>
- **One question at a time.** Wait for an answer before asking the next.
- **Each question lists 2-4 named options** with a one-line tradeoff. Pre-fill a recommended choice and say why in one sentence.
- **Skip forced questions.** If Phase 1 picked Gaussian transition, never ask about time_mixer / feature_mixer / noise schedule.
- **Show a live spec snapshot** after each answered question — a compact YAML block of what's locked in so far. The user can correct earlier answers at any point; if they do, replay subsequent questions whose answers are now invalid.
- **"skip" is always valid.** Record `<unset>` and continue. The downstream builder fills in defaults.
- **Never invent option names.** If you don't see a builder in `src/ddssm/builders.py`, don't list it.
</prompting-discipline>

<phases>

## Phase 1 — Architecture skeleton

Decide the *structural* shape in dependency order. Ask in this sequence; each may gate later questions:

1. **Transition family** — read `transitions/` builders. Typically `gaussian` (deterministic mean + Gaussian residual) vs `diffusion` (score-matching prior, CSDI-style). Gates everything diffusion-specific downstream.
2. **Multi-stage training?** — single train loop vs `StageOrchestrator(N stages)`. If multi-stage: how many stages, one-line role per stage (e.g. "stage_1: recon-only, stage_2: joint").
3. **Centering handoff?** — only ask if multi-stage **and** the transition has a parametric prior mean (diffusion, or learnable-baseline gaussian). yes/no + between which stages.
4. **Encoder / Decoder / z_init family.** Present the currently registered builders. Common variants live in `encoder.py`, `decoder.py`, and `encoder.py`'s `GaussianInitPrior`.
5. **(If diffusion)** UNet skeleton — `CSDIUnet` (residual stack) vs `MLPCSDIUnet` (ablation). Number of residual layers as an order-of-magnitude (2/4/8); exact value goes to Phase 3.

End of Phase 1: the *model class* is fixed.

## Phase 2 — Sub-architecture details (conditional)

Drill into every Phase-1 choice that has internal variants. Only ask the ones unlocked:

- **(If CSDIUnet)** `time_mixer` ∈ {conv, gru, identity}; `feature_mixer` ∈ {transformer, conv, identity}. Two separate questions.
- **(If diffusion)** Noise schedule family (linear / cosine / EDM-style importance sampling — check `transitions/diffusion*.py` for what's registered).
- **(If multi-stage)** Per stage: trainable-module mask `(encoder, decoder, z_init, transition, baseline)` and per-stage λ-ramp shape (one λ throughout vs per-stage cosine ramp).
- **(If centering)** Baseline form family: `zero`, `identity`, `linear`, `mlp` (or whatever `centering/baselines.py` exports). If the user plans an *ablation grid*, ask them to name the cells they'll compare rather than locking one form.
- **Encoder/decoder internals** if the chosen family has knobs (e.g. DKS combiner type, future-summary aggregator).

End of Phase 2: the *structural class* of the model is fully determined. No more pluggable choices.

## Phase 3 — Hyperparameters

Walk the standard surface. **For each parameter, offer three branches:**

> **`<name>`** — what it does (one line). Default in the codebase: `<value>` (if known from `src/ddssm/builders.py` or `*/hparams.py`).
>
> 1. **Use default**
> 2. **Set a custom fixed value** → ask for the value
> 3. **Tune over a range** → ask for:
>    - distribution: `uniform`, `log-uniform`, `int-uniform`, `int-log-uniform`, `categorical`
>    - bounds: `(low, high)` for ranges; `choice(...)` items for categorical
>    - optional note (motivation, prior result, ADR reference)

Parameters to walk (skip those forced inert by Phase 1/2):

- **LR scheme.** Two sub-questions: independent per-submodule (`enc_lr`, `dec_lr`, `zinit_lr`, `trans_lr`) vs `base_lr` + multipliers (`dec_mult`, `trans_mult`). Then values/ranges per the chosen scheme.
- **λ schedule.** `lambda_schedule` (cosine/linear), `lambda_start`, `lambda_end`, `lambda_warmup_steps` (or `warmup_frac` if multi-stage with per-stage ramps).
- **Batch.** `batch_size`, `grad_accum_steps`, `amp` on/off.
- **Step budget.** Total `training.steps` (single-stage) or per-stage `n_<stage>` counts (multi-stage).
- **(Diffusion)** `num_diffusion_steps`, `residual_channels`, score-net feature-mixer width.
- **(Centering)** `sigma_pert`, `n_pretrain`, regulariser strengths (`anchor_lambda`, `lambda_sigma_p`). For `sigma_pert`: protocol forbids 0 (ADR-0002); use a log-uniform lower bound that is operationally indistinguishable from 0 instead.
- **Cadence.** `log_every`, `validate_every`, `checkpoint_every`.
- **Seed.** Fixed single seed vs a small categorical for multi-seed sweeps.

</phases>

<tuning-range-syntax>

When capturing a "tune over range" choice, write the sweep entry in the project's Optuna syntax (see `experiments/*/sweeps.py` and `src/ddssm/conf/sweep/` for live examples):

| Distribution        | Syntax                                  |
|---------------------|------------------------------------------|
| log-uniform float   | `tag(log, interval(LOW, HIGH))`          |
| uniform float       | `interval(LOW, HIGH)`                    |
| log-uniform int     | `tag(log, int(interval(LOW, HIGH)))`     |
| uniform int         | `int(interval(LOW, HIGH))`               |
| categorical         | `choice(A, B, C)`                        |

The key in the sweep dict must be the dotted Hydra path of the parameter, anchored at `experiment.` (e.g. `experiment.model.stages.base_lr`, `experiment.hparams.lambda_sigma_p`).

</tuning-range-syntax>

<output>

When all three phases are done, emit a single fenced YAML block conforming to **`.claude/spec-schema.md`** and ask the user to confirm. Read that file once at the start of the session so you know the field shape, the `<unset>` vs `null` distinction, and the Optuna grammar table. The schema also has two worked examples — mirror their density.

After the spec is shown, ask:

> Spec accepted. Hand off to an experiment-builder to scaffold a preset under `experiments/<family>/`, or print the spec only?

If hand-off: pass the YAML to the next skill/turn verbatim. Do **not** write any files yourself — the downstream builder owns that.

</output>

<conventions>
- Hparam field names live in `src/ddssm/builders.py` (`Hparams`, `Training`) and in `experiments/<family>/hparams.py`. Match the existing names exactly.
- Don't fabricate variants — if `src/ddssm/builders.py` only exports `ZeroBaseline`, `PersistenceBaseline`, `LinearBaseline`, `MLPBaseline`, then those are the four choices; don't invent a fifth.
- When the user references an ADR (e.g. "per ADR-0002"), preserve that reference in `notes:` on the spec.
- Use `<unset>` for skipped fields, not `null` — the downstream builder distinguishes "I want the default" from "user explicitly skipped".
</conventions>

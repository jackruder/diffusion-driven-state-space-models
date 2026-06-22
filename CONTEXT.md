# DDSSM — Project Context

Glossary for DDSSM. Two clusters: project-wide training infrastructure
(used by every experiment) and the init-centering ablation grid
(currently the only mature experiment family).

## Language

### Training infrastructure

**Loss components**:
The per-term tensor outputs of `DDSSM_base.forward()` — `recon`,
`kl_init`, `kl_trans`, `transition`. Computed unconditionally on every
forward pass; selection and λ-weighting are the loss object's job.
See [docs/adr/0004-loss-object-split.md](./docs/adr/0004-loss-object-split.md).
_Avoid_: ELBO terms (ambiguous with the scalar), loss pieces.

**Loss object** (a.k.a. **training objective**):
A registered builder taking `(LossComponents, λ_state) → scalar`. Owns
its own λ schedule. Pickable per-stage in `StageOrchestrator` runs;
default is `FullELBO`. The **single source of truth** for "what scalar
is being backpropped right now."
_Avoid_: Loss function (too generic), objective function (collides with
the Optuna objective below).

**Tuning objective** (current type: `ObjectiveSpec`):
The scalar read from `metrics.csv` / `metrics.json` that Optuna
minimises across trials. Distinct from the **loss object** above —
this one drives *trial ranking*, not training. Currently lives at
`src/ddssm/experiment.py:ObjectiveSpec`.
_Avoid_: Objective (without qualifier — ambiguous with loss object).

**Trainable mask**:
The per-submodule `requires_grad` mask attached to a stage
(`StagesConf.trainable.{encoder, decoder, z_init, transition,
baseline}`). Decides *which parameters update*; orthogonal to which
loss components are summed.
_Avoid_: Frozen modules (describes a state, not the mechanism).

**Standalone stage** (a.k.a. **runner**):
A read-only operation over a trained checkpoint — loads the model,
runs across one data split, never trains. Current members: **eval**,
**viz**, **variance probe** (planned: imputation, counterfactual).
Each has its own Hydra entry point, registry, and context object.
_Avoid_: Pipeline, post-processing step.

**Persistence frame**:
The `GaussianEncoder`'s additive posterior-mean form `μ_t = z_{t-1} + g`,
where `g` is the encoder's free per-step emission (the combiner →
`GaussianDistHead` output) and `z_{t-1}` is the most-recent *realised* latent
(zero at `t = 1`; uses only the most-recent lag regardless of `j`). Selected by
`mu_mode="additive"`, the **default** (`mu_mode="free"` recovers the legacy
`μ_t = g`). The frame anchors the recurrent Jacobian near `I` (stability) and
fits multimodal data better than free emission on the aggregator backbone — see
`docs/encoder-ablation-findings.md`. Independent of the transition's
`baseline_form`, which centers the *prior* mean, not the posterior's.

**Dynamical covariates**:
Time-varying exogenous side info that influences latent dynamics
(holidays, weather, regime flags). Carries domain signal; fed to the encoder
alongside the observations. The live preset uses `covariate_dim = 0`.
_Avoid_: "covariates" without qualifier — ambiguous between dynamical
and static.

**Time embedding**:
Positional encoding of the absolute step index, used for irregular-grid
datasets where adjacent timesteps may be unevenly spaced. Carries no domain
signal — pure position. The live preset runs with `emb_time_dim = 0`.
_Avoid_: "time covariate" — conflates with dynamical covariates.

**Static covariates**:
Series-level metadata invariant within a single sequence (asset
category, geographic region). Projected once and broadcast across `t`.
_Avoid_: "covariates" without qualifier.

### Init-centering ablation

**Cell**:
One point of the 12-cell ablation grid, identified by the triple
`(baseline_form, baseline_mode, tracking_mode)`. Cells are the
**experimental factor of interest**, not Optuna search dimensions.
_Avoid_: Configuration, variant.

**Study**:
The whole ablation as a first-class object
(`experiments/init_centering/study.py:INIT_CENTERING_STUDY`): the family of
experiments to run and compare. It crosses the 12 cells with the 2 datasets
(`datasets.py:ABLATION_DATASETS`) into **24 registered presets** named
`init_<cell>__<dataset>` (e.g. `init_mlp_pinned_per_t__1d`), each baking the
real dataset + dims. The `Study` abstraction is library code
(`src/ddssm/study.py`); it is run by `StudyOrchestrator`
(`src/ddssm/launch.py`) via `python -m ddssm.launch <study> [--select] [--size]`,
and `report.py` consumes its points. _Avoid_: "campaign" (that is the
orchestration/scheduling layer, per the `plan-campaign` skill).

**Canonical cell**:
The triple `(mlp, pinned, per_t)`. The reference point for the two
**control cells** — it is the cell the controls pin and zero a
sweep knob from. **Not** a smoke / pilot cell (see below — smokes
are role-split into two distinct cells).
_Avoid_: Default cell, reference cell, pilot cell.

**Simple-smoke cell**:
The triple `(zero, pinned, fixed)`. Mathematically equivalent to plain
`DiffusionV2` (no centering, no σ_data tracking). Job: numerical V2
anchor + minimum-surface pipeline check. Locked down by the
**V2-reduction test** in `tests/test_init_centering_v2_reduction.py`.
_Avoid_: Simplest cell, baseline cell.

**High-surface-smoke cell**:
The triple `(mlp, learnable, per_t)`. Job: exercise every code path
of the cell machinery (parametric `μ_p`, the `r_mu_p` regulariser
under `learnable`, per-`t` σ_data EMA). If this cell trains end-to-end
without crashing, every cell in the 12-grid plausibly does.
_Avoid_: Pilot cell, max-coverage cell.

**Note — "pilot cell" is deliberately not used.** It was overloaded
(pipeline-correctness validation + canonical-cell preview at once);
the project now uses the two role-specific smoke cells above instead.

**Control cell** *(deprecated)*:
Previously: a single-job preset pinning the canonical cell with one
sweep knob set to 0. **Removed** per
[docs/adr/0002-drop-canonical-controls.md](./docs/adr/0002-drop-canonical-controls.md):
`σ_pert > 0` is mandatory protocol, and `n_pretrain = 0` is
meaningless for parametric `μ_p` cells.
_Avoid_: Don't reintroduce this term for new ablation panels —
they're plain ablation panels, not "controls."

**Handoff**:
The stage-1 → stage-2 transition during multi-stage training:
snapshot μ_p → freeze under Pinned → rebuild optimiser → perturb
encoder → reset σ_data EMA schedule. See
`src/ddssm/centering/handoff.py:perform_centering_handoff`.

**σ_data tracking**:
The `SigmaDataBuffer` records per-`t` target variance for EDM
preconditioning. Three modes: **fixed** (snapshot at stage-2 start),
**global_ema** (single EMA pooled across `t`), **per_t**
(independent per-timestep EMA).

**Baseline form** / **mode**:
- _Form_: which `μ_p` head — `zero`, `persistence`, `linear`, `mlp`.
  (`persistence` was previously called `identity`; renamed because at
  j>1 it's the persistence/last-value baseline `μ_p = z_hist[..., -1]`,
  not the identity-on-the-window — see
  [docs/adr/0010-persistence-baseline-rename.md](./docs/adr/0010-persistence-baseline-rename.md).)
- _Mode_: `pinned` (frozen at handoff) vs `learnable` (updated under
  the `r_mu_p` anchor regulariser, default λ = 1e-2).
Parameter-free forms (`zero`, `persistence`) **auto-degenerate** to
pinned regardless of the requested mode.

**Stage-2 ELBO surrogate**:
The `stage2_elbo_surrogate` headline metric (`src/ddssm/eval/metrics.py:504`).
Defined as `model.forward(train=False)`'s total loss on a held-out
split, averaged across batches.
**Not cell-invariant** — the EDM preconditioning scale, prior
expressivity, and regularizer terms differ per cell. Used as the
Phase-C/D Optuna objective and as the Phase-E ranking column with the
explicit caveat that the ranking is provisional until Phase F
(PF-ODE NLL) lands a genuinely cell-invariant likelihood. See
[docs/adr/0001-stage2-elbo-surrogate-objective.md](./docs/adr/0001-stage2-elbo-surrogate-objective.md).
_Avoid_: ELBO, val loss, headline objective (when precision matters).

**Size axis** *(forward-looking)*:
For each dataset, two architectural sizes are run:
- **tiny**: model `latent_dim` = data's true latent_dim (1 or 4). `channels` and
  `baseline_hidden_dim` = 16 × latent_dim.
- **paper-headline**: 2× over-parametrised. `latent_dim` doubled (2 or 8);
  `channels` and `baseline_hidden_dim` scaled to match.
`j = 1` and `diffusion_layers = 2` are held constant; `diffusion_num_steps = 128`
is the default. The score-net's feature mixer is `conv` (per
[ADR-0003](./docs/adr/0003-score-net-feature-mixer-conv.md)) — there is no
`nheads` knob since attention isn't used at these latent dims. The tiny size
runs the full 12-cell ablation; paper-headline runs only the user-selected
top-N cells per dataset (confirmation study).

**Init-experiment datasets** *(forward-looking)*:
The init-centering ablation runs on two synthetic datasets:
- `nonlinear-bimodal-lift` (D=1): latent `z_t = tanh(z_{t-1}) + δ s_t + σ_z η_t`,
  observation lifted via tanh-MLP. Covers nonlinear dynamics + multimodal jumps.
- `nonlinear-bimodal-lift-mv` *(to be added)*: same family, latent d=4,
  per-dim independent bimodal signs (16 attractors), 4×4 tanh coupling,
  observation lifted to D=8.
Both expose GT latents to unlock `gt_latent_jsd`. LGSSM/Harmonic are NOT
used for the ablation — they are too easy.

**Sweep knobs**:
The two continuous hparams introduced by the handoff protocol:
`N_pretrain` (stage-1 step budget) and `σ_pert` (encoder weight
perturbation at handoff). Sampled by Optuna per trial with
log-uniform priors. Neither reaches 0: per the protocol invariants,
`σ_pert > 0` is mandatory and `n_pretrain = 0` is meaningless for
parametric `μ_p`. The lower bound on `σ_pert` is chosen "small
enough to be effectively off" (specific value TBD).

## Example dialogue

> A: Why does the `init_mlp_*` row dominate the headline table?
>
> B: Because the headline column is the stage-2 ELBO surrogate, and
> the surrogate isn't a cell-fair comparison metric — MLP `μ_p`
> absorbs rate into the prior, so its loss is lower by construction.
> The MLP cells might or might not actually forecast better. We'll
> know once Phase F lands PF-ODE NLL; until then the ranking is
> provisional and the report carries the caveat.

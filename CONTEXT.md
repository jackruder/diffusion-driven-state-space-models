# DDSSM — Init-Centering Experiment Context

Glossary for the init-centering ablation grid (Phases A–E shipped on
branch `claude/update-documents-redo-plan-JzSxw`). This is the ranking
of cells in `model-v2`'s baseline-centering scheme.

## Language

**Cell**:
One point of the 18-cell ablation grid, identified by the triple
`(baseline_form, baseline_mode, tracking_mode)`. Named e.g.
`init_mlp_pinned_per_t`. Cells are the **experimental factor of
interest**, not Optuna search dimensions.
_Avoid_: Configuration, variant.

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
without crashing, every cell in the 18-grid plausibly does.
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
- _Form_: which `μ_p` head — `zero`, `identity`, `linear`, `mlp`.
- _Mode_: `pinned` (frozen at handoff) vs `learnable` (updated under
  the `r_mu_p` anchor regulariser, default λ = 1e-2).
Parameter-free forms (`zero`, `identity`) **auto-degenerate** to
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
is the default. The tiny size runs the full 18-cell ablation; paper-headline
runs only the user-selected top-N cells per dataset (confirmation study).

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

# Drop the `init_canonical_ctrl_*` presets; lock perturbation as mandatory protocol

Phase D shipped two single-job presets — `init_canonical_ctrl_sigma0`
and `init_canonical_ctrl_npretrain0` — that pin the canonical cell
and nail one sweep knob to 0. They existed because Optuna's
log-uniform priors on `sigma_pert` and `n_pretrain` cannot reach 0,
so the 0 case had to be a separate preset.

On grilling, two design invariants from `model-v2.org` make these
controls inappropriate:

1. **`sigma_pert > 0` is mandatory protocol.** The encoder
   perturbation step (§ Stage-1 → stage-2 handoff, step 4) is what
   gives stage 2 an exit ramp from stage-1's sharp local minimum
   into the very different stage-2 loss surface. The sweep range
   covers "small enough to be effectively off"; there is no
   `sigma_pert = 0` mode of the protocol.
2. **`n_pretrain = 0` is not a meaningful control for parametric
   `μ_p`.** Stage 1 is what makes the residual
   `ẑ_t = z̃_t - μ_p(z_{t-1})` close to zero-mean for the diffusion
   target. Skipping stage 1 for `linear` / `mlp` cells means asking
   the diffusion model to denoise uncentered targets — a different
   experiment, not an ablation of the centering scheme.

**Decision:**

- Drop both `init_canonical_ctrl_*` presets, their report rows, and
  their launcher entries.
- Tighten the `sigma_pert` sweep range so the lower bound is
  operationally indistinguishable from 0 (specific value TBD; see
  follow-up task).
- For parameter-free cells (`zero`, `identity`), stage 1 still
  runs (encoder + σ_data buffer initialisation) but its budget may
  be shorter than the parametric cells' budget. (Open question; not
  decided here.)

## Considered alternatives

- **Sweep-level conditional categorical prior (`suggest_categorical([0.0, loguniform(...)])`).**
  Rejected: it makes "no perturbation" a real trial point, which
  contradicts the protocol's correctness story above.
- **Extend the controls to every cell (36 extra runs).** Rejected:
  the controls were always a workaround for the prior's support,
  not a question we wanted answered uniformly across cells.

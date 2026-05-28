# Transitions own their init + transition-KL terms behind a uniform polymorphic interface

After the legacy purge (tag `legacy-transitions`) deleted V1/V2 and the
synthetic/kdd/variance_probe families, the `z_init` / `GaussianInitPrior`
"InitPrior" path in `DDSSM_base` had no remaining users — its only
consumers were the deleted models. The initial-state ELBO term (the
first `j` latents `z_{1:j}`, which have no real observed history) is
otherwise produced by the **VHP-via-diffusion** construction:
`AuxPosterior` infers the auxiliary previous states `z_{-j+1:0}` and the
active transition scores `z_{1:j}` via `transition_kl_init`.

`DDSSM_base` carried three forms of transition-type gating:

- `_accepts_sigma_data(transition)` — `inspect.signature` introspection
  deciding whether to forward the σ_data buffer.
- `hasattr(active, "transition_kl_init")` — guarding the VHP init call.
- A stage-gated encoder-entropy assembly in `_init_kl_loss`: add
  `−H(q_φ)` in stage 1, omit it in stage 2 (the diffusion ESM expansion
  cancels it).

`GaussianTransition` is **not** dead — it is a first-class transition.
Its init term should be computed the *same hierarchical way* as the
diffusion transition (sample `z_aux`, walk the init window with the
transition + masking), differing only in *how a step is scored*
(closed-form Gaussian log-density vs. centered ESM/EDM). The
type-by-type gating is exactly the wrong place for that distinction.

## Decision

1. **`BaseTransition` gains a concrete `transition_kl_init`** that owns
   the shared hierarchical init walk: sample `z_aux` from `q_Φ`, walk
   `t = 1 … j` with the mixed `z_aux → real` history, accumulate the
   per-step score, compute `kl_aux`, and assemble the **complete** init
   decomposition `{loss, entropy, vhp, kl_aux, loss_init}`. Three
   overridable hooks carry the per-transition behaviour:
   - `_score_init_step(...)` (**abstract**) — the per-step
     negative-log-score (Gaussian closed-form / centered Gaussian /
     centered ESM-EDM surrogate).
   - `_init_entropy_term(enc_stats)` — default `−H(q)` over the init
     log-variances; the diffusion path overrides it to `0` (its ESM
     expansion already cancels the entropy).
   - `_update_sigma_data_init(...)` — default no-op; the σ_data-tracking
     transitions override it.

2. **`transition_kl` has a uniform signature** across all transitions —
   `sigma_data` is an accepted kwarg everywhere; a transition that does
   not use it (the plain `GaussianTransition`) simply ignores it.

3. **`DDSSM_base` stops gating on transition type.** `_init_kl_loss`
   becomes a pass-through to `active.transition_kl_init(...)`;
   `_compute_transition_kl` always forwards `sigma_data`. The
   `_accepts_sigma_data` introspection, the `hasattr` check, and the
   stage-gated `−H` assembly are deleted.

4. **The legacy InitPrior path is removed**: `z_init`,
   `GaussianInitPrior`, `BaseInitPrior`, `compute_init_loss`,
   `hierarchical_kl`, the `ZInit` builder, and the
   `z_init`/`aux_posterior` mutual-exclusion. `aux_posterior` becomes a
   mandatory constructor argument on `DDSSM_base`.

## Considered alternatives

- **Keep `_init_kl_loss`'s entropy assembly, drive the cancel decision
  off a `transition.init_cancels_entropy` flag instead of
  `stage_selector`.** Rejected: still a gating site in the model; the
  entropy treatment is genuinely a property of the scoring math, so it
  belongs with the score.
- **Give `GaussianTransition` its own `transition_kl_init`, leave V3 /
  BaselineGaussian untouched (no shared base).** Rejected: triplicates
  the identical aux walk and leaves the "is this hierarchical?" question
  implicit. The shared base makes the contract explicit; the only cost
  is re-validating that V3 / BaselineGaussian numerics are unchanged.
- **Delete `GaussianTransition` too (V3-only model algebra).** Rejected
  by the project owner: the closed-form Gaussian transition is a
  first-class modelling choice; its init KL is closed-form
  (`KL(q(z_{1:j}|x) ‖ p_ψ(z_{1:j}|z_aux))`).

## Consequences

- The init term is **behavior-preserving** for V3 and BaselineGaussian:
  each only ever runs in its own stage, so the old per-stage entropy
  gating maps exactly onto the new per-transition `_init_entropy_term`
  (BaselineGaussian → `−H`, V3 → `0`). A fixed-seed equivalence check
  guards the shared-base refactor.
- A plain `GaussianTransition` model is now **hierarchical**: it requires
  an `aux_posterior`, and its init term always includes `−H(q)` (the
  full closed-form KL — there is no ESM cancellation).
- `DDSSM_base` no longer imports or knows about any specific transition
  class for the init/transition-KL terms; adding a new transition means
  implementing `transition_kl` + the three init hooks, nothing in the
  model.

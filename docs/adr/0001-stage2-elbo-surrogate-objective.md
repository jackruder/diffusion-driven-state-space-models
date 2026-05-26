# Use stage2_elbo_surrogate as the cell-ranking objective, with caveats

The 18-cell ablation grid (Phases C/D) optimises and ranks cells by
`stage2_elbo_surrogate` — the V3 model's total loss on a held-out
split. This metric is **not cell-invariant**: EDM preconditioning
scale depends on `σ_data²(t)` (which differs per `tracking_mode`),
prior expressivity differs per `baseline_form`, and `r_mu_p` is only
active for `learnable` baselines. The ranking therefore conflates
"the cell trained well in its own loss landscape" with "the cell
generalises best."

We accept this for now because the principled alternative —
PF-ODE NLL (`init-experiment.org` § Headline metrics, metric 1) — is
deferred to Phase F. Re-ranking is cheap (it's a post-hoc evaluation
over saved checkpoints, no retraining), so we ship Phases A–E now
with the caveat documented in `CONTEXT.md` and Phase-E report
artefacts, and swap the headline column once Phase F lands.

## Considered alternatives

- **Gate the grid on Phase F.** Rejected: Phase F is open-ended
  (PF-ODE NLL requires an S-sample solve + log-density evaluation
  that isn't implemented), and Phases A–E are independently useful
  for pipeline validation, σ_data drift inspection, and pairwise
  ablation patterns that don't depend on a single ranking number.
- **Rank by `crps_sum_latent` or `gt_latent_jsd`.** Rejected for
  now: `gt_latent_jsd` requires LGSSM data (Harmonic returns
  `available=False`), and `crps_sum_latent` is per-horizon — picking
  a single horizon as the ranking axis is itself an unjustified
  choice.
- **Strip the cell-dependent terms from the eval-time loss.**
  Rejected: Optuna minimised the un-normalised loss during training,
  so a post-hoc-stripped eval metric wouldn't match the inner-loop
  objective and the best-trial selection would diverge from what
  Optuna picked.

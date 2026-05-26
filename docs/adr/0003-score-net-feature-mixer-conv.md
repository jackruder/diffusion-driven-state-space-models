# Score-net feature mixer: conv (not transformer)

The CSDIUnet score net used by ``DiffusionV3Transition`` has two
"mixer" slots per residual block: a **feature mixer** (over the
latent dim ``d``) and a **time mixer** (over the history-window
``L = j + 1``). The factory previously defaulted the feature mixer
to ``TransformerFeatureLayer`` with ``nheads = channels // 8`` (one
attention layer).

At the ablation's latent dims (``d ∈ {1, 2, 4, 8}`` across the size
matrix) attention is doing near-no-op work — self-attention on 1-8
tokens is dominated by parameter count, not expressivity. The
transformer default was a CSDI-paper inheritance assuming hundreds
of multivariate features, which don't exist in this codebase.

**Decision:** swap the feature mixer to ``FeatureMixerConfig(type="conv", n_layers=1)``
uniformly across all cells. ``ConvFeatureLayer``
(``src/ddssm/diffnets.py:185``) treats the latent dim as a sequence
and applies depthwise + pointwise convs; at ``d = 1`` it
short-circuits and returns the input unchanged, so it degenerates
gracefully at the 1D-tiny end of the size matrix. The ``nheads``
plumbing (``channels // 8``) and its divisibility check are removed
from ``_build_init_centering_model``.

The time mixer stays ``conv`` (``ConvTimeLayer``) — even at
``L = 2`` the conv is the mechanism by which the history slot
conditions the target slot's score prediction, and dropping it
would sever that information flow.

## Considered alternatives

- **Keep transformer.** Rejected: parameters spent on attention with
  1-8 tokens isn't worth the compute, and the head_dim=8 scaling
  rule (``nheads = channels // 8``) actually makes the situation
  worse at small ``d`` (1D paper: 4 heads × head_dim 8 = 32-dim
  projection on a 2-token input).
- **``identity`` feature mixer at ``d = 1`` only, conv elsewhere.**
  Rejected: adds per-cell branching to the factory for negligible
  compute savings; ``ConvFeatureLayer`` already short-circuits at
  ``d = 1``.
- **Sweep ``feature_mixer_type`` categorically inside Optuna.**
  Rejected: adds another search dim to a 9-dim sweep that's already
  past the trial-budget rule of thumb. The cell comparison is the
  experimental factor of interest; the mixer is infrastructure that
  shouldn't compete for budget.

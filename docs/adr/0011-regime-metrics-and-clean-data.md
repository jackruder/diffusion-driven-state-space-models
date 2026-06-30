# System-agnostic regime metrics and the `clean_data` batch key

## Context

The Lorenz attractor experiment family needed sample-quality evaluation beyond
training-curve/ELBO metrics. Two evaluation ideas required new design decisions:
(1) scoring metastable lobe-switching behavior, and (2) scoring denoising against
the noise-free ground truth rather than the noisy observations.

## Decisions

### 1. Regime metrics live in `src/ddssm/eval/`, not `experiments/`

The `regime` metric (`src/ddssm/eval/regime.py`) is system-agnostic: the
labelling function (threshold + deadband on one channel) is a parameter, not
hardcoded Lorenz logic. Any mode with metastable switching — Lorenz, a future
double-well SDE, a telegraph-AR process — activates it via `EvalSpec.kwargs`.

Placing it in `src/ddssm/eval/` rather than `experiments/lorenz/` ensures the
registry is importable without the experiments package. This matches the
`bimodal_jsd` precedent (mode-specific metrics in `src/ddssm/eval/metrics.py`).

Lorenz-specific choices (`channel=0`, `deadband=0.3`) live in
`experiments/lorenz/evals.py` as `EvalSpec.kwargs["regime"]` and have no
library footprint.

**Deadband choice for Lorenz:** a deadband=0.1 (matching σ_obs=0.1) still
produces ~10% of interior runs shorter than 3 steps due to within-lobe spirals
that dip x near 0 — the chatter is geometric, not noise-driven. Deadband=0.3 is
the smallest value with zero chatter and run statistics stable up to 0.7 (mean
residence ~20 steps), so it is used.

### 2. `clean_data` is a distinct batch key from `gt_latent`

The two existing ground-truth mechanisms serve different spaces:

| Key | Space | Dim | Consumer |
|-----|-------|-----|----------|
| `gt_latent` | model-latent (d) | d | `crps_sum_latent`, `gt_latent_jsd` |
| `clean_data` | observation (D) | D | `denoise_mse` |

Reusing `gt_latent` would be semantically wrong (Lorenz clean states are D=3,
not d=4/8) and would silently corrupt latent-space metrics if ever enabled
alongside. A first-class `clean_data` key keeps the contract clear.

**Bit-identity invariant:** `clean_data` is stashed from `trajectory` *before*
the `randn_like` noise call, so adding `expose_clean_data=True` makes no extra
RNG calls and leaves `observed_data` bit-identical across all dataset seeds.
This means existing checkpoints remain valid for post-hoc `denoise_mse` eval.

### 3. Eval specs are immutable mid-sweep

The running sweep's `LorenzEval` spec was left unchanged. New metrics were added
to a separate `LorenzForecastEval` spec used by the canonical presets. Once the
sweep completes, `LorenzForecastEval` can be merged into `LorenzEval` for the
next study round. This avoids heterogeneous `metrics.json` files across sweep
trials and prevents long-running later trials from paying the extra forecast cost.

## Alternatives considered

- **Lorenz-specific `lorenz_lobe` metric name**: rejected — the phenomenon
  (metastable residence + switching) is not Lorenz-specific. Parameterization
  is the right abstraction; renaming later would invalidate stored metrics keys.
- **`gt_latent` reuse for clean data**: rejected (space/dim mismatch, semantic
  pollution of existing latent-space metrics).
- **Deadband from obs noise (0.1)**: rejected — produces ~10% short-run chatter
  from within-lobe spiral dynamics, biasing residence-time statistics.

## Consequences

- `regime` metric activated for any future metastable synthetic mode by adding
  per-dataset kwargs to `EvalSpec`; no library changes needed.
- `clean_data` can be exposed for other noisy modes (e.g. a future `double-well`
  mode) with a single `self._all_clean = ...` stash before the noise call.
- MCAR/imputation evaluation (Phase D) would add a third key (`imputation_mse`
  target = `clean_data`), keeping the distinction with `recon_mse` clear.

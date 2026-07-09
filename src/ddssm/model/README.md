# `ddssm.model`

The variational state-space model and its sub-architectures. The model is a
latent SSM trained under an ELBO: an encoder `q_ϕ(z_{1:T} | x_{1:T})` produces
approximate-posterior latent paths, a decoder `p_θ(x_t | z_{t-j+1:t})` is a
diagonal-Gaussian observation model over the length-`j` latent history, and a
**pluggable transition prior** `p_ψ(z_t | z_{t-j:t-1})` (Gaussian or
diffusion-based) scores those paths. The entry class is
[`DDSSM_base`](dssd.py); its `forward` returns the ELBO loss components and
`forecast` autoregressively rolls out and decodes future latents.

## Top-level files

- **`dssd.py`** — `DDSSM_base` (the `nn.Module` owning the ELBO forward pass,
  encoder/decoder/transition dispatch, and the forecast rollout) plus the
  `ProbeBatch` payload (detached latent encodings reused by the variance probe).
  The initial-state term over the first `j` latents is the transition's
  VHP-via-diffusion walk and requires an auxiliary posterior `q_Φ` — there is no
  standalone init-prior module.
- **`encoder.py`** — `BaseEncoder` interface for `q_ϕ`; `sample_paths(...)`
  draws `S` latent paths and their per-step encoder log-densities.
- **`decoder.py`** — `BaseDecoder` interface for `p_θ`; `forward` returns
  `(mu, logvar)` and `log_likelihood` the masked per-step observation likelihood.
- **`losses.py`** — `LossComponents` (the unweighted per-term ELBO bag returned
  by `DDSSM_base.forward()`: `recon`, `init_kl`, `trans_kl`) and the `Loss`
  objects (e.g. `FullELBO`) that weight and sum them, carrying their own λ
  schedule (ADR-0004).

## Subpackages

### `transitions/`
Pluggable transition priors implementing the `BaseTransition` interface
(`transition_kl`, `seq_log_prob`, optional `log_prob`/`sample`/`prior_params`).
- `GaussianTransition` — non-linear diagonal-Gaussian transition (ablation).
- `DiffusionTransition` — CSDI-style diffusion transition with a centered
  ESM target `ẑ_t = z̃_t − μ_p(z_{t-1})`, `σ_data(t)`-driven EDM preconditioning,
  and VHP-via-diffusion at the initial `j` steps.

### `centering/`
Baseline-centering machinery (model-v2). `BaseBaseline` and its parameter-free
forms (`ZeroBaseline`, `PersistenceBaseline`) provide the centering head
`μ_p(z_{t-1})` with a fixed unit prior variance (`σ_p² = 1`); `SigmaDataBuffer`
is the EMA buffer tracking the per-step centered-residual variance `σ_data²(t)`
(modes: `fixed`/`global_ema`/`per_t`). These are pure leaves consumed by reference.

### `likelihood/`
Exact-likelihood evaluation utilities (model-v2). `solve_prob_flow_logdensity`
(probability-flow ODE log-density via the Liouville trace identity),
`iwae_log_likelihood` + `logmeanexp` (IWAE over trajectory samples), and
`vhp_log_prob_init` (importance-sampled initial-state estimator under `q_Φ`).

## How it fits

`DDSSM_base` is composed in Python — by `src/ddssm/builders.py` (for ad-hoc /
notebook assembly) or by the `init_centering` model factory
(`experiments/init_centering/model.py`) — not selected via Hydra config groups.
The concrete building blocks (heads, fusions, score nets, combiners, etc.) come
from `ddssm.nn`. All imports within this package are absolute
(`from ddssm.model...`).

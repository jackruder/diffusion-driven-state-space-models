# `ddssm.nn` — neural building blocks

Reusable, framework-agnostic neural-network components for the DDSSM model. These
are the parts the encoder, decoder, init prior, and transition networks in
`ddssm.model` compose into the variational state-space model (`ddssm.dssd`):
future summaries, history aggregators, fusion/combiner stages, distribution heads,
the CSDI diffusion denoiser, and shared Gaussian/utility helpers. Nothing here owns
training logic or ELBO bookkeeping — these are pure `nn.Module` / function pieces
the higher-level model wires together.

## Files

| File | Provides |
| --- | --- |
| `net_utils.py` | Shared utilities: `time_embedding` (sinusoidal), `get_side_info` (covariate/side-info tensor builder), `hist_abs_time_tokens` (history-window time gather), `Conv1d_with_init`, `get_torch_trans`, `softplus_inv`. |
| `diffnets.py` | Conditional-diffusion blocks: the CSDI U-Net denoiser `CSDIUnet` (+ MLP ablation `MLPCSDIUnet`), `DiffResidualBlock`/`ResidualBlock`, EDM step conditioning `DiffusionEmbedding`, pluggable time/feature mixers (`Conv`/`GRU`/`Transformer`/`Identity` layers + `build_time_layer`/`build_feature_layer` factories and their configs), and the encoder-side `ContextProducer` (+ `MLPContextProducer`). |
| `gaussians.py` | Diagonal-Gaussian helpers: `gaussian_log_prob`, `gaussian_entropy`, `gaussian_kl_divergence`, `logvar_from_raw`/`clamp_logvar`, the `GaussianHead` (mean + stabilised log-var head), and the `GaussianStats` typed dict. |
| `aggregators.py` | History aggregators mapping a `j`-step latent history to one feature vector: `BaseHistoryAggregator` + `Identity`/`GRU`/`MLP`/`Attention`/`ContextProducer` variants. |
| `fusions.py` | Encoder fusions combining the future summary `h_fut` with the aggregated history feature: `BaseEncoderFusion` + `ConcatLinear`/`DKS`/`Gated` variants. |
| `combiners.py` | Encoder combiners that compose an aggregator with a fusion: `BaseEncoderCombiner` + `CompoundCombiner`. |
| `futsum.py` | Future-summary modules `F_ϕ` over the observed sequence (time-reversed mixing): `FutureSummary` base + `GRUFutureSummary` / `TransformerFutureSummary`. |
| `dist_heads.py` | Distribution heads consuming combiner features → `(z, logq, step_params)`: `BaseDistHead` + `GaussianDistHead` (reparameterised sampling, closed-form entropy via `gaussians.py`). |
| `aux_posterior.py` | `AuxPosterior` — diagonal-Gaussian amortised posterior `q_Φ(z_{-j+1:0} | z_{1:j})` for the VHP-via-diffusion construction, with reparameterised `sample` and analytic `kl_against_standard_normal`. |
| `torch_compile.py` | `maybe_compile` — opt-in in-place `torch.compile` wrapper gated by the `DDSSM_TORCH_COMPILE` env var (preserves `state_dict` keys, falls back to eager on failure). |

## How it fits

These blocks are assembled in **Python** when the model is built (see
`experiments/init_centering/model.py` and `src/ddssm/builders.py`) — there are no
CLI config groups for selecting them individually. Import convention is **absolute**:

```python
from ddssm.nn.gaussians import GaussianHead
from ddssm.nn.diffnets import CSDIUnet, build_time_layer
from ddssm.nn.combiners import CompoundCombiner
```

Internal dependencies flow upward: `gaussians` and `net_utils` are the leaf
helpers; `diffnets` builds on `net_utils`; `aggregators`/`futsum` build on
`diffnets`; `combiners` composes `aggregators` + `fusions`; `dist_heads` builds on
`gaussians`.

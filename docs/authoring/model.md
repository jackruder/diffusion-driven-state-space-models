# Step 1 — Deciding & implementing the model

The model is a {py:class}`ddssm.model.dssd.DDSSM_base`: an encoder `q_ϕ(z|x)`, a
decoder `p_θ(x|z)`, and a pluggable transition prior, trained under an ELBO. You
build it by **writing a small factory function** that instantiates the parts and
wires them together. This page covers the menu of choices and how to assemble
them; the worked example is `experiments/synthetic_validation/model.py`.

## Why a factory function

Two constraints decide the shape of the code:

1. **The baseline must be shared.** The μ_p/σ_p *baseline* head is referenced by
   the stage-1 transition, the stage-2 transition, and the model's own
   `baseline` slot. They must be the **same Python object** so its parameters
   are shared across the stage-1 → stage-2 handoff. Nested `builds(...)` configs
   would each instantiate a *separate* baseline — silently breaking the sharing.
   A factory builds it once and passes it by reference.
2. **There is no shape auto-baking.** {py:func}`experiments._make.experiment`
   does not fill `data_dim`/`latent_dim`/`j` into sub-modules — the factory takes
   shapes as arguments and threads them through.

`experiments/init_centering/model.py:SmokeModel` is exactly such a factory
(wrapping `_build_init_centering_model`), but it carries the init-centering
ablation's knobs and scaling heuristics. *
```{tip}
{py:mod}`ddssm.experiment.builders` exposes a `builds(...)` config for every
part (`DDSSM`, `Encoder`, `Decoder`, `DiffTransition`, `ZeroBaselineB`, `Unet`,
…) — handy for overriding a leaf field from a config string. But for the shared
baseline you still need a factory; the builders are the menu, the factory is the
recipe.
```

## The required skeleton

{py:class}`~ddssm.model.dssd.DDSSM_base` requires, at minimum:

- `encoder`, `decoder`, `transition` (the **stage-2** transition),
- shape: `j`, `data_dim`, `latent_dim`, `emb_time_dim`,
- **`aux_posterior`** — mandatory; the initial-state term over `z_{1:j}` is
  computed via the transition's hierarchical VHP walk and needs `q_Φ`,
- and for the centering machinery: `baseline`, `baseline_mode`, `sigma_data`,
  `stage1_transition`.

## The menu of choices

### Shapes
`data_dim` (D, from the dataset), `latent_dim` (d), `j` (history window),
`emb_time_dim` (time-embedding width), `T_max` (≥ data `T`; sizes σ_data²).

### Baseline (μ_p / σ_p head) — `ddssm.model.centering.baselines`
| Class | μ_p | Has params? |
| ----- | --- | ----------- |
| `ZeroBaseline` | ≡ 0 | σ_p MLP only |
| `PersistenceBaseline` | `z_hist[..., -1]` | σ_p MLP only |
| `LinearBaseline` | `A·z_hist + b` | yes |
| `MLPBaseline` | MLP | yes |

With `baseline_mode="pinned"` the baseline is frozen in stage 2; `"learnable"`
trains it under the R_μp anchor regularizer.

### Transitions — `ddssm.model.transitions`
- {py:class}`~ddssm.model.transitions.baseline_gaussian.BaselineGaussianTransition`
  — closed-form Gaussian centered on the baseline; the **stage-1** transition.
- {py:class}`~ddssm.model.transitions.diffusion.DiffusionTransition` — centered
  ESM/EDM diffusion; the **stage-2** transition. Configured by a
  {py:class}`~ddssm.model.transitions.diffusion.DiffusionScheduleConfig`
  (`num_steps`, `S_k`, `k_sampling_mode`, …) and a score network (`unet`).
- {py:class}`~ddssm.model.transitions.transitions.GaussianTransition` — a plain
  Gaussian transition (no centering); rarely used directly.

### Score network (diffusion U-net) — `ddssm.nn.diffnets`
{py:class}`~ddssm.nn.diffnets.CSDIUnet` (`channels`, `n_layers`,
`embedding_dim`) composes a per-channel **time mixer** (`conv`/`gru`/`identity`)
× **feature mixer** (`transformer`/`conv`/`identity`) via `TimeMixerConfig` /
`FeatureMixerConfig` inside a `DiffResidualBlockConfig`. `MLPCSDIUnet` is an MLP
ablation. (At small latent dims, a `conv` feature mixer is the project default.)

### Encoder — `ddssm.model.encoder.GaussianEncoder`
Composed from swappable slots:
- **aggregator** (`ddssm.nn.aggregators`): `Identity` / `GRU` / `MLP` /
  `Attention` / `ContextProducer` (default).
- **fusion** (`ddssm.nn.fusions`): `ConcatLinear` (default) / `DKS` / `Gated`.
- **dist head** (`ddssm.nn.dist_heads.GaussianDistHead`).
- **future summary** (`ddssm.nn.futsum`): `GRUFutureSummary` (default) /
  `TransformerFutureSummary`.

Pass `combiner` / `dist_head` / `fut_summary` to override; the defaults give a
working encoder from just the shapes.

### Decoder — `ddssm.model.decoder.GaussianDecoder`
A `ContextProducer` + `GaussianHead`; configured by shapes + `hidden_dim`.

### Auxiliary posterior & σ_data²
{py:class}`~ddssm.nn.aux_posterior.AuxPosterior` (`q_Φ(z_aux|z_{1:j})`, required)
and {py:class}`~ddssm.model.centering.sigma_data.SigmaDataBuffer`
(`tracking_mode` ∈ `fixed` / `global_ema` / `per_t`).

## Writing the factory (the worked example)

`experiments/synthetic_validation/model.py` is a minimal, fixed-choice instance
of the skeleton. The structure:

```python
def build_synthval_model(*, data_dim=1, latent_dim=1, j=1, emb_time_dim=16,
                         T_max=32, hidden_dim=32, channels=32,
                         diffusion_layers=2, diffusion_num_steps=64) -> DDSSM_base:
    # 1) shared ingredients — built ONCE, passed by reference
    baseline = ZeroBaseline(latent_dim=latent_dim, j=j, hidden_dim=hidden_dim, n_layers=2)
    aux_posterior = AuxPosterior(latent_dim=latent_dim, j=j, hidden_dim=hidden_dim, n_layers=2)
    sigma_data = SigmaDataBuffer(T_max=T_max, tracking_mode="fixed", init_value=1.0)

    # 2) stage-1: closed-form Gaussian centered on the baseline
    stage1 = BaselineGaussianTransition(baseline=baseline, latent_dim=latent_dim,
                                        j=j, emb_time_dim=emb_time_dim)

    # 3) stage-2: centered diffusion with a small conv-mixer score net
    unet = partial(CSDIUnet, channels=channels, n_layers=diffusion_layers,
                   embedding_dim=channels,
                   residual_block=DiffResidualBlockConfig(
                       feature=FeatureMixerConfig(type="conv", n_layers=1)))
    stage2 = DiffusionTransition(baseline=baseline,          # SAME object as stage1
                                 latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
                                 T_max=T_max, unet=unet,
                                 schedule=DiffusionScheduleConfig(num_steps=diffusion_num_steps))

    encoder = GaussianEncoder(data_dim=data_dim, latent_dim=latent_dim, j=j,
                              emb_time_dim=emb_time_dim, use_mask=False, hidden_dim=hidden_dim)
    decoder = GaussianDecoder(data_dim=data_dim, latent_dim=latent_dim, j=j,
                              emb_time_dim=emb_time_dim, hidden_dim=hidden_dim)

    return DDSSM_base(encoder=encoder, decoder=decoder, transition=stage2, j=j,
                      data_dim=data_dim, latent_dim=latent_dim, emb_time_dim=emb_time_dim,
                      use_observation_mask=False,
                      aux_posterior=aux_posterior, baseline=baseline,
                      baseline_mode="pinned", sigma_data=sigma_data,
                      stage1_transition=stage1)

# wrap so it plugs into experiment(model=...)
SynthValModel = builds(build_synthval_model, populate_full_signature=True)
```

Note `baseline` flows into `stage1`, `stage2`, **and** `DDSSM_base(baseline=...)`
as one object — that's the whole reason this is a function. The
`builds(build_synthval_model, populate_full_signature=True)` wrapper turns it
into a config you call as `SynthValModel(data_dim=1, latent_dim=1, j=1)` and pass
to {py:func}`experiments._make.experiment`.

## Swapping choices

To change architecture, edit the factory — e.g. `MLPBaseline` +
`baseline_mode="learnable"` for a learned mean; a `gru` time mixer
(`TimeMixerConfig(type="gru")`); a `TransformerFutureSummary` on the encoder; or
a larger `channels`/`latent_dim`. Verify with
`python -m ddssm.app experiment=<name> --cfg job` and a short smoke run before
committing.

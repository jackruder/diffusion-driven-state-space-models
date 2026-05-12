"""Notebook-first composer for DDSSM experiments.

The single function :func:`make_experiment` takes the shape ints
(``data_dim``, ``latent_dim``, ``j``, ``emb_time_dim``, ``covariate_dim``)
plus a set of already-configured builder calls and returns an
instantiable :class:`Experiment` config.

No Hydra interpolation, no config groups, no store registrations. Every
parameter at every depth is reachable as a keyword argument to a
builder. Read the resulting config back as YAML with :func:`to_yaml` or
write it out to a file with :func:`save_yaml`.

Typical usage in a notebook or org src block::

    from ddssm.builders import (
        Synthetic, Hparams, Training, Eval, Viz, Plot,
        Encoder, Decoder, ZInit, DiffTransition, Unet, Schedule,
    )
    from experiments._make import make_experiment, run, to_yaml

    exp = make_experiment(
        data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
        data=Synthetic(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
        hparams=Hparams(S=1, batch_size=32, enc_lr=5e-4, lambda_warmup_steps=400),
        training=Training(steps=2000, log_every=25, checkpoint_every=500),
        transition=DiffTransition(
            unet=Unet(channels=64, n_layers=4),
            schedule=Schedule(sigma_min=0.01, S_k=20),
        ),
        eval=Eval(metrics=["mae", "crps_sum"], split="val", T_split=32),
        viz=Viz(plots=[Plot("forecast_1d", "forecast.png", kwargs={"n_show": 4})]),
    )
    print(to_yaml(exp))
    run(exp, run_dir="runs/harm_diff")
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import torch
from hydra_zen import instantiate, to_yaml as _zen_to_yaml
from omegaconf import MISSING, OmegaConf

from ddssm.builders import (
    DDSSM,
    Decoder,
    Encoder,
    ExperimentC,
    GaussTransition,
    Hparams,
    Synthetic,
    Training,
    TrainerPartial,
    ZInit,
)

log = logging.getLogger(__name__)


def make_experiment(
    *,
    data_dim: int,
    latent_dim: int,
    j: int = 1,
    emb_time_dim: int = 16,
    covariate_dim: int = 0,
    use_observation_mask: bool = False,
    checkpoint_dir: str = "./checkpoints",
    seed: int | None = 0,
    # Required slots (callers nearly always supply these).
    data: Any = None,
    hparams: Any = None,
    training: Any = None,
    # Optional slots; defaults compose a runnable Gaussian baseline.
    encoder: Any = None,
    decoder: Any = None,
    z_init: Any = None,
    transition: Any = None,
    # Optional eval/viz/variance/objective specs.
    objective: Any = None,
    eval: Any = None,
    viz: Any = None,
    variance: Any = None,
    wandb_config: Any = None,
) -> Any:
    """Assemble an :class:`Experiment` config with shapes baked into every leaf.

    Shape kwargs (``data_dim``, ``latent_dim``, ``j``, ``emb_time_dim``,
    ``covariate_dim``, ``use_observation_mask``) are pushed into the
    encoder/decoder/z_init/transition slots in one place. Callers do
    not pass them again, and the resulting config has concrete integers
    instead of ``${experiment.*}`` interpolation strings.

    Any builder argument may already specify a shape kwarg, in which
    case the caller's value wins (e.g. for variants like ``j=2``).
    """
    if data is None:
        data = Synthetic(D=data_dim)
    if hparams is None:
        hparams = Hparams()
    if training is None:
        training = Training()

    def _fill(b, **shape):
        """Fill MISSING shape fields on a builds() dataclass instance.

        Returns ``b`` unchanged if nothing needs filling; otherwise a
        new instance via :func:`dataclasses.replace`. User-supplied
        values (anything not equal to ``MISSING`` aka ``"???"``) win.
        """
        if b is None:
            return None
        updates = {
            k: v for k, v in shape.items()
            if k in {f.name for f in dataclasses.fields(b)}
            and getattr(b, k) == MISSING
        }
        return dataclasses.replace(b, **updates) if updates else b

    enc_shape = dict(data_dim=data_dim, latent_dim=latent_dim, j=j,
                     emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
                     use_mask=use_observation_mask)
    dec_shape = dict(data_dim=data_dim, latent_dim=latent_dim, j=j,
                     emb_time_dim=emb_time_dim, covariate_dim=covariate_dim)
    zi_shape = dict(latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
                    covariate_dim=covariate_dim)
    tr_shape = dict(latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
                    covariate_dim=covariate_dim)

    encoder = _fill(encoder if encoder is not None else Encoder(), **enc_shape)
    decoder = _fill(decoder if decoder is not None else Decoder(), **dec_shape)
    z_init = _fill(z_init if z_init is not None else ZInit(), **zi_shape)
    transition = _fill(
        transition if transition is not None else GaussTransition(), **tr_shape
    )

    model = DDSSM(
        encoder=encoder, decoder=decoder, z_init=z_init, transition=transition,
        j=j, data_dim=data_dim, latent_dim=latent_dim,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        use_observation_mask=use_observation_mask,
        hyperparams=hparams, checkpoint_dir=checkpoint_dir,
    )

    return ExperimentC(
        data=data,
        model=model,
        build_trainer=TrainerPartial(),
        training=training,
        objective=objective,
        eval=eval,
        viz=viz,
        variance=variance,
        seed=seed,
        wandb_config=wandb_config,
        hparams=hparams,
    )


def to_yaml(exp: Any, *, resolve: bool = True) -> str:
    """Serialize a config to YAML for inspection or saving."""
    return _zen_to_yaml(exp, resolve=resolve)


def tweak(obj: Any, **overrides: Any) -> Any:
    """Apply nested overrides to a builds() dataclass via ``__`` separators.

    Example::

        tweak(exp,
              training__steps=2000,
              training__checkpoint_every=500,
              hparams__lambda_warmup_steps=400,
              model__transition__schedule__sigma_min=0.001)

    Each ``__``-separated path descends one level. Leaf values may be
    scalars or fresh builds() dataclass instances. Returns a new object
    (via :func:`dataclasses.replace`) — the original is untouched.
    """
    nested: dict[str, Any] = {}
    for path, value in overrides.items():
        if "__" not in path:
            nested[path] = value
            continue
        head, rest = path.split("__", 1)
        nested.setdefault(head, {})[rest] = value
    updates: dict[str, Any] = {}
    for k, v in nested.items():
        if isinstance(v, dict):
            cur = getattr(obj, k)
            updates[k] = tweak(cur, **v)
        else:
            updates[k] = v
    return dataclasses.replace(obj, **updates)


def save_yaml(exp: Any, path: str, *, resolve: bool = True) -> None:
    """Persist a config to a YAML file (alongside a run directory)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(to_yaml(exp, resolve=resolve))


def from_yaml(path_or_text: str) -> Any:
    """Load a YAML config previously produced by :func:`save_yaml`.

    Accepts a file path or a YAML string. Returns an OmegaConf
    DictConfig that can be passed to :func:`instantiate` directly.
    """
    if os.path.isfile(path_or_text):
        return OmegaConf.load(path_or_text)
    return OmegaConf.create(path_or_text)


def run(
    exp: Any,
    *,
    device: torch.device | None = None,
    run_dir: str = "./runs/adhoc",
) -> Any:
    """Instantiate ``exp`` and call ``experiment.train(device, run_dir)``.

    Convenience for notebook / org src usage. The returned value is
    whatever ``Experiment.train`` returns — typically the trainer when
    no ``objective`` is set, or a scalar Optuna objective when one is.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(exp, os.path.join(run_dir, "resolved_config.yaml"))

    experiment = instantiate(exp)
    return experiment.train(device=device, run_dir=run_dir)


__all__ = ["make_experiment", "run", "to_yaml", "save_yaml", "from_yaml", "tweak"]

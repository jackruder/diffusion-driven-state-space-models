"""Slim composer + run/override/serialise helpers for DDSSM experiments.

The named models and datasets live under
``experiments/{synthetic,variance_probe,kdd}/`` — each registers
itself to the relevant store in :mod:`conf.registry` on import. To
define a new experiment, import a registered model + dataset and
call :func:`experiment` (it ties them together and keeps
``model.hyperparams`` in sync with the experiment's ``hparams``)::

    from ddssm.builders import Eval, Hparams, Plot, Training, Viz
    from conf.registry import experiment_store
    from experiments._make import experiment, run
    from experiments.synthetic.models import SmallGauss
    from experiments.synthetic.datasets import Harmonic

    exp = experiment(
        data=Harmonic, model=SmallGauss,
        hparams=Hparams(S=1, lambda_warmup_steps=200, batch_size=32,
                        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4),
        training=Training(steps=1000, log_every=25, checkpoint_every=200),
        eval=Eval(metrics=["mae", "crps_sum"], split="val",
                  num_samples=32, T_split=32),
        viz=Viz(plots=[Plot("forecast_1d", "forecast.png",
                            kwargs={"n_show": 4})],
                split="val", num_samples=32, T_split=32),
    )
    experiment_store(exp, name="harmonic_gauss")
    run(exp, run_dir="runs/harmonic_gauss")

For ad-hoc variants in a notebook / sweep, derive from a registered
experiment with :func:`override` (Hydra-CLI-style strings or dicts).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import torch
from hydra_zen import instantiate, to_yaml as _zen_to_yaml
from omegaconf import OmegaConf

from ddssm.builders import ExperimentC, TrainerPartial

log = logging.getLogger(__name__)


def experiment(
    *,
    data: Any,
    model: Any,
    hparams: Any,
    training: Any,
    eval: Any | None = None,
    viz: Any | None = None,
    objective: Any | None = None,
    variance: Any | None = None,
    wandb_config: Any | None = None,
    seed: int | None = 0,
) -> Any:
    """Bind a model + dataset + training into an :class:`ExperimentC` config.

    ``model.hyperparams`` is replaced with ``hparams`` so the two
    Hparams instances inside the resulting tree are identical — the
    trainer reads the experiment-level field and the DDSSM internals
    read ``model.hyperparams``.
    """
    model = dataclasses.replace(model, hyperparams=hparams)
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


def override(obj: Any, *overrides: Any) -> Any:
    """Derive a new experiment by applying Hydra-CLI-style overrides.

    Each positional argument is either:

    * A CLI-style string ``"path.to.field=value"``. The value side is
      YAML-parsed, so ``"training.steps=200"``, ``"data.mode=harmonic"``,
      ``"hparams.S=4"`` and ``"training.amp=true"`` all work.

    * A dict ``{"path.to.field": value}``. Use this form when ``value``
      is a Python object (a builder instance, a callable, etc.) that
      cannot live in a string — e.g.::

          override(exp, {"model.transition.unet": MLPUnet(channels=64)})

    Multiple overrides compose left-to-right. The original ``obj`` is
    never mutated; ``override`` returns a fresh dataclass.

    Examples::

        B = override(A, "training.steps=200",
                        "model.transition.schedule.sigma_min=0.001")

        for mode in ["harmonic", "bimodal", "robot-basis-pursuit"]:
            exp = override(A, f"data.mode={mode}")
            run(exp, run_dir=f"runs/{mode}")
    """
    import yaml

    flat: dict[str, Any] = {}
    for item in overrides:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError(
                    f"CLI override missing '=': {item!r}. "
                    f"Use 'path.to.field=value'."
                )
            key, _, val = item.partition("=")
            flat[key.strip()] = yaml.safe_load(val)
        elif isinstance(item, dict):
            flat.update(item)
        else:
            raise TypeError(
                f"override() expects strings or dicts, got {type(item).__name__}: "
                f"{item!r}"
            )

    return _apply_flat(obj, flat)


def _apply_flat(obj: Any, flat: dict[str, Any]) -> Any:
    """Expand a flat ``{dotted.path: value}`` dict and apply recursively."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cur = nested
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return _apply_nested(obj, nested)


def _apply_nested(obj: Any, d: dict[str, Any]) -> Any:
    updates: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cur = getattr(obj, k)
            if dataclasses.is_dataclass(cur):
                updates[k] = _apply_nested(cur, v)
                continue
        updates[k] = v
    return dataclasses.replace(obj, **updates)


def save_yaml(exp: Any, path: str, *, resolve: bool = True) -> None:
    """Persist a config to a YAML file (alongside a run directory)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(to_yaml(exp, resolve=resolve))


def from_yaml(path_or_text: str) -> Any:
    """Load a YAML config previously produced by :func:`save_yaml`."""
    if os.path.isfile(path_or_text):
        return OmegaConf.load(path_or_text)
    return OmegaConf.create(path_or_text)


def run(
    exp: Any,
    *,
    device: torch.device | None = None,
    run_dir: str = "./runs/adhoc",
) -> Any:
    """Instantiate ``exp`` and call ``experiment.train(device, run_dir)``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(exp, os.path.join(run_dir, "resolved_config.yaml"))
    experiment_obj = instantiate(exp)
    return experiment_obj.train(device=device, run_dir=run_dir)


__all__ = [
    "experiment", "run", "to_yaml", "save_yaml", "from_yaml", "override",
]

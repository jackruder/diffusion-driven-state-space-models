"""Slim composer + run/override/serialise helpers for DDSSM experiments.

Dataset configs are library code in :mod:`ddssm.data.presets`; model
factories live in the experiment families (e.g.
:mod:`experiments.init_centering.model`). To define a new experiment,
import a model factory + a dataset and call :func:`experiment` (it ties
them together and curries ``hparams`` onto the trainer)::

    from ddssm.experiment.builders import (
        Eval,
        Hparams,
        Training,
    )
    from ddssm.experiment.stores import (
        experiment_store,
    )
    from experiments._make import (
        experiment,
        run,
    )
    from experiments.init_centering.model import (
        SmokeModel,
    )
    from ddssm.data.presets import (
        NonlinBimodalLift1D,
    )

    exp = experiment(
        data=NonlinBimodalLift1D,
        model=SmokeModel(
            baseline_form="zero",
            latent_dim=1,
            data_dim=1,
        ),
        hparams=Hparams(
            S=1,
            batch_size=16,
            enc_lr=5e-4,
            dec_lr=5e-4,
            trans_lr=5e-4,
        ),
        training=Training(
            steps=800,
            log_every=25,
            checkpoint_every=200,
        ),
        eval=Eval(
            metrics=[
                "stage2_elbo_surrogate"
            ],
            split="val",
        ),
    )
    experiment_store(
        exp, name="my_cell"
    )
    run(
        exp,
        run_dir="runs/my_cell",
    )

For ad-hoc variants in a notebook / sweep, derive from a registered
experiment with :func:`override` (Hydra-CLI-style strings or dicts).
"""

from __future__ import annotations

import os
from typing import Any
import logging
import dataclasses

import torch
from hydra_zen import to_yaml as _zen_to_yaml, get_target, instantiate
from omegaconf import OmegaConf

from ddssm.adapters import ModelAdapter
from ddssm.experiment.builders import ExperimentC, DDSSMAdapterC, TrainerPartial

log = logging.getLogger(__name__)


def _targets_adapter(model_conf: Any) -> bool:
    """True iff ``model_conf`` already targets a :class:`ModelAdapter` subclass.

    Existing DDSSM presets target *functions* (``_build_init_centering_model`` /
    ``build_gluonts_model``), so ``get_target`` returns a function — a bare
    ``issubclass(t, ModelAdapter)`` would raise ``TypeError`` on it. The
    ``isinstance(t, type)`` guard is therefore mandatory before ``issubclass``.
    """
    t = get_target(model_conf)
    return isinstance(t, type) and issubclass(t, ModelAdapter)


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
    sbatch: Any | None = None,
    seed: int | None = 0,
    wrap: bool = True,
) -> Any:
    """Bind a model + dataset + training into an :class:`ExperimentC` config.

    ``hparams`` is curried onto the model's ``ModelAdapter`` wrapper (its
    ``build_trainer`` ``TrainerPartial`` + ``config``) so the trainer reads it
    directly (per ADR-0004 the model no longer carries a ``hyperparams``
    field). It is also stored on the experiment so callers can introspect or
    ``tweak`` it.

    ``model`` is wrapped in a :class:`DDSSMAdapter` conf (via
    :data:`DDSSMAdapterC`) unless it already targets a
    :class:`~ddssm.adapters.base.ModelAdapter` subclass — in which case
    ``config=hparams`` is curried onto the existing adapter conf instead of
    double-wrapping (future CSDI presets). Pass ``wrap=False`` to skip wrapping
    entirely (escape hatch for callers assembling their own adapter).

    ``sbatch`` is purely metadata at training time; it is read by
    ``python -m experiments sbatch <name>`` when emitting a Slurm
    submit script. Leave ``None`` to inherit the project default in
    :mod:`ddssm.cluster.sbatch`. (Study launches read resources from each
    point's ``PointLaunch.resources`` instead — see ADR-0008.)
    """
    if wrap:
        if _targets_adapter(model):
            # Already an adapter conf (e.g. a future CSDI preset): curry the
            # winning config onto it rather than re-wrapping. ``model`` is a
            # builds() *instance* (a dataclass), so use ``dataclasses.replace``
            # rather than calling it.
            model = dataclasses.replace(model, config=hparams)
        else:
            # Bare DDSSM model conf: wrap in a DDSSMAdapter so Experiment.model
            # is a ModelAdapter. The TrainerPartial now lives INSIDE the wrapper.
            model = DDSSMAdapterC(
                module=model,
                config=hparams,
                build_trainer=TrainerPartial(hparams=hparams),
            )
    return ExperimentC(
        data=data,
        model=model,
        build_trainer=TrainerPartial(hparams=hparams),
        training=training,
        objective=objective,
        eval=eval,
        viz=viz,
        variance=variance,
        seed=seed,
        wandb_config=wandb_config,
        hparams=hparams,
        sbatch=sbatch,
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

          override(
              exp,
              {
                  "model.module.transition.unet": MLPUnet(
                      channels=64
                  )
              },
          )

    Multiple overrides compose left-to-right. The original ``obj`` is
    never mutated; ``override`` returns a fresh dataclass.

    Examples::

        B = override(
            A,
            "training.steps=200",
            "model.module.transition.schedule.sigma_min=0.001",
        )

        for mode in [
            "harmonic",
            "bimodal",
            "robot-basis-pursuit",
        ]:
            exp = override(
                A, f"data.mode={mode}"
            )
            run(
                exp,
                run_dir=f"runs/{mode}",
            )
    """
    import yaml

    flat: dict[str, Any] = {}
    for item in overrides:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError(
                    f"CLI override missing '=': {item!r}. Use 'path.to.field=value'."
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
    """Instantiate ``exp``, train, then resolve the objective value (if any).

    Mirrors :func:`ddssm.app.main` so notebook / script callers get the
    same return shape as a Hydra run (``None`` when no objective is set,
    otherwise the scalar / list returned by :meth:`Experiment.objective_value`).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(exp, os.path.join(run_dir, "resolved_config.yaml"))
    experiment_obj = instantiate(exp)
    experiment_obj.train(device=device, run_dir=run_dir)
    return experiment_obj.objective_value(device=device, run_dir=run_dir)


__all__ = [
    "experiment",
    "from_yaml",
    "override",
    "run",
    "save_yaml",
    "to_yaml",
]

"""Hydra entry point for DDSSM training.

Usage::

    # Default experiment (init_smoke_simple).
    python -m ddssm.app

    # Pick a different registered preset (see ``python -m experiments list``).
    python -m ddssm.app experiment=init_smoke_high_surface

    # Override any field at any depth via dot-notation. The shipped presets
    # are multi-stage, so the step budget lives under ``training.stages.*``
    # (here the factory args n_pretrain / n_stage2) — ``training.steps`` is
    # only read by the single-fit (no-stages) path.
    python -m ddssm.app experiment=init_smoke_simple \\
        experiment.training.stages.n_pretrain=200 \\
        experiment.training.stages.n_stage2=400

    # Swap the dataset baked into a preset (data store is packaged at
    # experiment.data; use +data= to append).
    python -m ddssm.app experiment=init_smoke_simple +data=harmonic

    # Optuna sweep using a pre-defined search space.
    python -m ddssm.app --multirun \\
        experiment=init_smoke_high_surface \\
        +sweep=init_ablation \\
        hydra.sweeper.n_trials=20

Experiments are discovered from ``experiments/*.py`` in the repo root;
see :mod:`ddssm._experiment_registry`.

Under ``DDSSM_PREEMPTIVE=1`` (set by the preempt-aware sbatch preamble —
see ADR-0009), :func:`main` performs an app-level trial-resume hand-off:
it loads the Optuna study, looks up the current RUNNING trial by
param-match against the cfg's sampled hparams, injects any pending
``resume_from`` saved by a previous preempt cycle into
``cfg.experiment.training.resume_from``, and on a
:class:`ddssm.train.PreemptError` enqueues a retry trial via
``study.add_trial(...)`` carrying the saved checkpoint path. The retry
machinery is deliberately app-level (no monkey-patch, no
``RetryFailedTrialCallback``) because Optuna's callback path does not
fire from the sweeper's exception-catch branch — see ADR-0009
"Considered alternatives" and the gate test under Phase 1 of the
implementation plan.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import hydra
import optuna
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna.trial import FrozenTrial, TrialState

from ._experiment_registry import register_experiments
from .train import PreemptError

register_experiments()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preempt-aware helpers (ADR-0009 §5).
#
# These are module-level so the unit tests in ``tests/test_app_preempt.py``
# can exercise the lookup / enqueue logic without driving a full Hydra
# multirun. The high-level orchestration lives in :func:`apply_preempt_hooks`
# and the ``try/except PreemptError`` block inside :func:`main`.
# ---------------------------------------------------------------------------


_PARAM_FLOAT_RTOL = 1e-6
_PARAM_FLOAT_ATOL = 1e-12


def _params_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two Optuna params dicts with float tolerance.

    Optuna stores sampled floats with full IEEE-754 precision; the cfg
    values arrive from Hydra via the override CLI as parsed floats too,
    but downstream substitutions / interpolations can lose a ULP or two.
    Compare with a small relative tolerance to be robust.
    """
    if a.keys() != b.keys():
        return False
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, float) or isinstance(vb, float):
            try:
                if not math.isclose(
                    float(va), float(vb),
                    rel_tol=_PARAM_FLOAT_RTOL, abs_tol=_PARAM_FLOAT_ATOL,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if va != vb:
                return False
    return True


def _find_current_running_trial_by_params(
    study: optuna.Study, hparams: dict[str, Any],
) -> FrozenTrial | None:
    """Find the unique RUNNING trial whose params match ``hparams``.

    Per Phase 0 finding I1 (see the implementation plan), the
    Hydra-Optuna sweeper does NOT expose the current Optuna trial to the
    task function — neither via env var nor via cfg injection. The only
    handle we have on the running trial in ``app.py`` is the cfg's
    sampled hparam values; we recover the trial by matching them against
    the ``RUNNING`` rows in the study.

    Returns the unique matching FrozenTrial, or None if zero or more than
    one trial matches (ambiguous — log a warning and skip resume rather
    than crash, per ADR-0009 §5).
    """
    if not hparams:
        log.warning(
            "Preempt: hparams dict is empty; cannot match a running trial. "
            "Skipping resume.",
        )
        return None

    running = study.get_trials(states=[TrialState.RUNNING], deepcopy=False)
    matches = [t for t in running if _params_equal(t.params, hparams)]

    if len(matches) == 0:
        log.warning(
            "Preempt: no RUNNING trial matches the cfg's sampled hparams "
            "(searched %d running trials). Skipping resume.",
            len(running),
        )
        return None
    if len(matches) > 1:
        log.warning(
            "Preempt: %d RUNNING trials match the cfg's sampled hparams "
            "(expected 1). Skipping resume to avoid corrupting state.",
            len(matches),
        )
        return None
    return matches[0]


def _get_resume_from_user_attrs(trial: FrozenTrial) -> str | None:
    """Return ``trial.user_attrs["resume_from"]`` or None if absent.

    Set by :func:`_enqueue_preempt_retry` on the retry trial; on the
    retry-side, ``apply_preempt_hooks`` reads it back and injects it
    into ``cfg.experiment.training.resume_from``.
    """
    val = trial.user_attrs.get("resume_from")
    if val is None:
        return None
    return str(val)


def _enqueue_preempt_retry(
    study: optuna.Study,
    current_trial: FrozenTrial,
    resume_from: str,
) -> None:
    """Enqueue a WAITING retry trial copying ``current_trial``'s params.

    The retry carries ``user_attrs["resume_from"]`` (so the next pickup
    can mid-trial resume from the saved ckpt) and
    ``user_attrs["retried_from"]`` (provenance / debugging). The retry's
    state is WAITING so the sampler picks it up on the next
    ``study.ask()`` (this invocation or a subsequent requeue).

    See ADR-0009 §5 — the explicit-enqueue path that replaces
    ``RetryFailedTrialCallback`` (which empirically does not fire on
    ``study.tell(trial, FAILED)`` from the sweeper's catch-block).
    """
    retry = optuna.trial.create_trial(
        params=dict(current_trial.params),
        distributions=dict(current_trial.distributions),
        state=TrialState.WAITING,
        user_attrs={
            "resume_from": resume_from,
            "retried_from": current_trial.number,
        },
    )
    study.add_trial(retry)
    log.info(
        "Preempt: enqueued retry trial (retried_from=%d) with resume_from=%s",
        current_trial.number, resume_from,
    )


def _collect_sampled_hparams_from_cfg(
    cfg: DictConfig, param_keys: list[str],
) -> dict[str, Any]:
    """Extract the values of ``param_keys`` from ``cfg`` via dot-paths.

    Hydra-Optuna applies sampled params as CLI overrides like
    ``experiment.training.n_pretrain=123``; the resolved cfg therefore
    carries those values at the same dot-path. Missing keys are silently
    skipped (caller's match logic will fail-soft on missing keys).
    """
    hparams: dict[str, Any] = {}
    for key in param_keys:
        val = OmegaConf.select(cfg, key, default=None)
        if val is not None:
            hparams[key] = val
    return hparams


def _resolve_sweeper_field(cfg: DictConfig, name: str) -> Any:
    """Return ``cfg.hydra.sweeper.<name>`` or None if absent / unresolvable."""
    try:
        return OmegaConf.select(cfg, f"hydra.sweeper.{name}", default=None)
    except Exception:  # pragma: no cover — defensive
        return None


def apply_preempt_hooks(
    cfg: DictConfig,
) -> tuple[FrozenTrial | None, optuna.Study | None]:
    """Wire up preempt-aware trial-resume hand-off on the cfg.

    No-op when ``DDSSM_PREEMPTIVE`` is unset (non-preemptive runs are
    completely untouched). When set:

    1. Load the Optuna study from
       ``cfg.hydra.sweeper.{study_name,storage}``.
    2. Find the current RUNNING trial by param-match against the cfg.
       The set of relevant param keys is taken from each running trial's
       own ``.params`` (Optuna already records the canonical key names
       there).
    3. If a matching trial is found and it carries
       ``user_attrs["resume_from"]`` (set by a previous preempt cycle),
       inject the path into ``cfg.experiment.training.resume_from``.

    Returns ``(current_trial, study)`` so the caller can enqueue a retry
    via :func:`_enqueue_preempt_retry` on a :class:`PreemptError`. Either
    field may be None if the study could not be loaded or the trial
    could not be matched — the caller should fall through to plain
    (non-preempt-aware) training in that case.
    """
    if os.environ.get("DDSSM_PREEMPTIVE") != "1":
        return None, None

    study_name = _resolve_sweeper_field(cfg, "study_name")
    storage = _resolve_sweeper_field(cfg, "storage")
    if not study_name or not storage:
        log.warning(
            "DDSSM_PREEMPTIVE=1 but cfg.hydra.sweeper.{study_name,storage} "
            "is not set — preempt hand-off requires Optuna sweep mode. "
            "Falling back to plain training (no resume, no retry enqueue).",
        )
        return None, None

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except (KeyError, Exception) as e:  # noqa: BLE001 — defensive
        log.warning(
            "Preempt: failed to load study %r at %r: %s. "
            "Falling back to plain training.",
            study_name, storage, e,
        )
        return None, None

    # Collect the union of param names across all RUNNING trials and try to
    # match each. We iterate per-trial because in principle different trials
    # could carry different param sets (e.g. mid-search with conditional
    # search spaces).
    running = study.get_trials(states=[TrialState.RUNNING], deepcopy=False)
    matches: list[FrozenTrial] = []
    for t in running:
        cfg_hparams = _collect_sampled_hparams_from_cfg(cfg, list(t.params.keys()))
        if cfg_hparams and _params_equal(cfg_hparams, t.params):
            matches.append(t)

    if len(matches) == 0:
        log.warning(
            "Preempt: no RUNNING trial matches the cfg's sampled hparams "
            "(searched %d running trials). Skipping resume.",
            len(running),
        )
        return None, study
    if len(matches) > 1:
        log.warning(
            "Preempt: %d RUNNING trials match the cfg's sampled hparams "
            "(expected 1). Skipping resume to avoid corrupting state.",
            len(matches),
        )
        return None, study

    current_trial = matches[0]
    resume_from = _get_resume_from_user_attrs(current_trial)
    if resume_from is not None:
        OmegaConf.update(
            cfg, "experiment.training.resume_from", resume_from, merge=False,
        )
        log.info(
            "Preempt: resuming trial %d from ckpt %s",
            current_trial.number, resume_from,
        )

    return current_trial, study


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Phase 0: preempt hand-off (no-op unless DDSSM_PREEMPTIVE=1).
    current_trial, study = apply_preempt_hooks(cfg)

    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    log.info("Device=%s run_dir=%s", device, run_dir)

    try:
        with open(f"{run_dir}/resolved_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    except OSError as e:
        log.warning("Could not persist resolved_config.yaml: %s", e)

    experiment = instantiate(cfg.experiment)
    # ADR-0005: snapshot the resolved model YAML so the trainer can
    # persist it into checkpoints (and post-training stages can diff
    # against it on load).
    experiment.model_config_yaml = OmegaConf.to_yaml(
        cfg.experiment.model, resolve=True,
    )
    try:
        experiment.train(device=device, run_dir=run_dir)
    except PreemptError as e:
        # Preempt path: enqueue a retry trial carrying the saved ckpt
        # path, then re-raise so the sweeper marks the current trial
        # FAILED via its normal exception path. See ADR-0009 §5.
        if current_trial is not None and study is not None:
            try:
                _enqueue_preempt_retry(study, current_trial, e.resume_from)
            except Exception as enqueue_err:  # noqa: BLE001 — best-effort
                log.error(
                    "Preempt: failed to enqueue retry trial: %s. "
                    "The trial will be marked FAILED but no retry is queued.",
                    enqueue_err,
                )
        else:
            log.warning(
                "PreemptError raised but no current_trial/study handle "
                "is available; cannot enqueue retry. The trial will be "
                "marked FAILED with no resume path.",
            )
        raise
    return experiment.objective_value(device=device, run_dir=run_dir)


if __name__ == "__main__":
    main()

"""Named variance-probe experiments + sweep preset.

DiffusionV2 transition + Probe spec over short (300-step) runs on a
handful of synthetic datasets. The ``variance_probe`` *sweep* preset
at the bottom is a config preset (not an Optuna search space) — it
enables W&B logging and tweaks the probe runner's
``R``/``n_batches``/``seeds`` so a probe run is reproducible across
the synthetic modes::

    # train
    python -m ddssm.app \\
        experiment=variance_probe_lgssm +sweep=variance_probe

    # run probe from the checkpoint
    python -m ddssm.variance \\
        experiment=variance_probe_lgssm \\
        checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth' \\
        +sweep=variance_probe
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import experiment_store, sweep_store

from experiments._make import experiment
from experiments.variance_probe.data import (
    NonlinearBimodalLift, ProbeBimodal, ProbeBimodalNoisy, ProbeLGSSM,
)
from experiments.variance_probe.evals import LossTail, VarianceProbe
from experiments.variance_probe.hparams import Probe as ProbeHparams, Probe300
from experiments.variance_probe.model import ProbeMediumModel, ProbeSmall


# ---------------------------------------------------------------------------
# Compose helper: every probe run shares the same objective + variance spec.
# ---------------------------------------------------------------------------


def _probe(*, data, model):
    return experiment(
        data=data, model=model,
        hparams=ProbeHparams,
        training=Probe300,
        objective=LossTail, variance=VarianceProbe,
    )


variance_probe_lgssm = _probe(data=ProbeLGSSM, model=ProbeSmall.model)
experiment_store(variance_probe_lgssm, name="variance_probe_lgssm")

variance_probe_bimodal_clean = _probe(data=ProbeBimodal, model=ProbeSmall.model)
experiment_store(variance_probe_bimodal_clean, name="variance_probe_bimodal_clean")

variance_probe_bimodal_noisy = _probe(data=ProbeBimodalNoisy, model=ProbeSmall.model)
experiment_store(variance_probe_bimodal_noisy, name="variance_probe_bimodal_noisy")

variance_probe_nonlinear_bimodal_lift = _probe(
    data=NonlinearBimodalLift, model=ProbeMediumModel.model,
)
experiment_store(
    variance_probe_nonlinear_bimodal_lift,
    name="variance_probe_nonlinear_bimodal_lift",
)


# ---------------------------------------------------------------------------
# Sweep preset (config preset, not an Optuna search space).
# ---------------------------------------------------------------------------

VarianceProbeSweep = make_config(
    hydra_defaults=["_self_", {"override /wandb": "enabled"}],
    experiment=dict(
        wandb_config=dict(
            project="ddssm-variance-probe",
            tags=["variance-probe"],
        ),
        variance=dict(
            R=32,
            n_batches=1,
            seeds=[0],
        ),
    ),
)
sweep_store(VarianceProbeSweep, name="variance_probe")

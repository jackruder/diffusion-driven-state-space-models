"""Named variance-probe experiments (DiffusionV2 + Probe spec)."""

from __future__ import annotations

from ddssm.builders import Objective, Probe

from conf.registry import experiment_store

from experiments._make import experiment
from experiments.variance_probe.datasets import (
    NonlinearBimodalLift, ProbeBimodal, ProbeBimodalNoisy, ProbeLGSSM,
)
from experiments.variance_probe.hparams import Probe as ProbeHparams
from experiments.variance_probe.models import ProbeMedium, ProbeSmall
from experiments.variance_probe.training import Probe300


_OBJECTIVE = Objective(metric="loss/total", split="train", tail_frac=0.1)
_VARIANCE = Probe()


def _probe(*, data, model):
    return experiment(
        data=data, model=model,
        hparams=ProbeHparams,
        training=Probe300,
        objective=_OBJECTIVE, variance=_VARIANCE,
    )


variance_probe_lgssm = _probe(data=ProbeLGSSM, model=ProbeSmall)
experiment_store(variance_probe_lgssm, name="variance_probe_lgssm")

variance_probe_bimodal_clean = _probe(data=ProbeBimodal, model=ProbeSmall)
experiment_store(variance_probe_bimodal_clean, name="variance_probe_bimodal_clean")

variance_probe_bimodal_noisy = _probe(data=ProbeBimodalNoisy, model=ProbeSmall)
experiment_store(variance_probe_bimodal_noisy, name="variance_probe_bimodal_noisy")

variance_probe_nonlinear_bimodal_lift = _probe(
    data=NonlinearBimodalLift, model=ProbeMedium,
)
experiment_store(
    variance_probe_nonlinear_bimodal_lift,
    name="variance_probe_nonlinear_bimodal_lift",
)

"""Named KDD Cup 2018 PM2.5 experiments + Optuna sweep preset.

Phase-1 sweep mirrors ``scripts/experiments/kdd/optuna-kdd-p1.py`` so
the Hydra-native path produces equivalent search behaviour::

    python -m ddssm.app --multirun \\
        experiment=kdd_gauss \\
        +sweep=kdd_phase1 \\
        hydra.sweeper.n_trials=50 \\
        hydra.sweeper.study_name=ddssm_kdd_recon
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import experiment_store, sweep_store

from experiments._make import experiment
from experiments.kdd.data import KDDData
from experiments.kdd.evals import KDDEval, KDDViz
from experiments.kdd.hparams import Diff8k, Gauss5k, KDDHparams
from experiments.kdd.model import KDD


kdd_gauss = experiment(
    data=KDDData, model=KDD.gauss_model,
    hparams=KDDHparams,
    training=Gauss5k,
    eval=KDDEval, viz=KDDViz,
)
experiment_store(kdd_gauss, name="kdd_gauss")

kdd_diffusion = experiment(
    data=KDDData, model=KDD.diff_model,
    hparams=KDDHparams,
    training=Diff8k,
    eval=KDDEval, viz=KDDViz,
)
experiment_store(kdd_diffusion, name="kdd_diffusion")


# ---------------------------------------------------------------------------
# Optuna sweep preset (mirrors scripts/experiments/kdd/optuna-kdd-p1.py).
# ---------------------------------------------------------------------------

KDDPhase1 = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params={
                "experiment.hyperparams.enc_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.dec_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.zinit_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.trans_lr": "interval(1e-5, 3e-4)",
                "experiment.hyperparams.lambda_warmup_steps":
                    "range(50, 4000, step=50)",
                "experiment.hyperparams.lambda_end": "interval(0.7, 2.0)",
                "experiment.hyperparams.weight_decay": "interval(1e-5, 1e-2)",
                "experiment.hyperparams.batch_size": "choice(32, 64, 128)",
            },
        ),
    ),
)
sweep_store(KDDPhase1, name="kdd_phase1")

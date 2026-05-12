"""Synthetic verification experiment base presets.

Three family bases cover the full set of synthetic verification
experiments documented in ``verifications.org``.  Each base uses the
Gaussian transition by default (``transition=gaussian``); switch to the
diffusion transition and adjust the training budget with CLI overrides:

.. code-block:: sh

    # harmonic + diffusion
    python -m ddssm.app experiment=harmonic \\
        transition=diffusion \\
        'experiment.training.steps=2000' \\
        'experiment.training.checkpoint_every=500' \\
        'experiment.hyperparams.lambda_warmup_steps=400'

    # harmonic with 4× observation noise
    python -m ddssm.app experiment=harmonic \\
        'experiment.data.mode=harmonic-noisy'

    # harmonic with second-order AR latent (j=2)
    python -m ddssm.app experiment=harmonic \\
        'experiment.j=2'

    # bimodal + diffusion
    python -m ddssm.app experiment=bimodal \\
        transition=diffusion \\
        'experiment.training.steps=2000' \\
        'experiment.training.checkpoint_every=500' \\
        'experiment.hyperparams.lambda_warmup_steps=400'

    # robot_2d + diffusion
    python -m ddssm.app experiment=robot_2d \\
        transition=diffusion \\
        'experiment.training.steps=4000' \\
        'experiment.hyperparams.lambda_warmup_steps=800'

Registered experiment names
----------------------------
- ``harmonic``  — D=1, j=1, harmonic mode;  Gaussian by default
- ``bimodal``   — D=1, j=1, bimodal mode, S=4; Gaussian by default
- ``robot_2d``  — D=2, j=2, robot-basis-pursuit; Gaussian by default
"""

from __future__ import annotations

from .._infra import (
    build_experiment_conf,
    DDSSMHyperParamsConf,
    ObjectiveSpecConf,
    SyntheticDataModuleConf,
    TrainingScalarsConf,
    store,
)
from .._eval_viz import (
    BimodalEvalConf,
    BimodalVizConf,
    HarmonicEvalConf,
    HarmonicVizConf,
    Robot2DEvalConf,
    Robot2DVizConf,
)


# ---------------------------------------------------------------------------
# Harmonic: clean sine-wave signal.
# Noisy variant: experiment.data.mode=harmonic-noisy
# j=2 variant:   experiment.j=2
# Diffusion:      transition=diffusion experiment.training.steps=2000
#                 experiment.training.checkpoint_every=500
#                 experiment.hyperparams.lambda_warmup_steps=400
# ---------------------------------------------------------------------------

HarmonicExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=200,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(HarmonicExperimentConf, group="experiment", name="harmonic")


# ---------------------------------------------------------------------------
# Bimodal: multimodality benchmark — headline metric: energy score.
# Diffusion:  transition=diffusion experiment.training.steps=2000
#             experiment.training.checkpoint_every=500
#             experiment.hyperparams.lambda_warmup_steps=400
# ---------------------------------------------------------------------------

BimodalExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="bimodal", T=64, N_per_split=1024, batch_size=32),
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=200,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=4,
    ),
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=BimodalEvalConf,
    viz_conf=BimodalVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(BimodalExperimentConf, group="experiment", name="bimodal")


# ---------------------------------------------------------------------------
# Robot navigation 2D: spatial trajectory, D=2, j=2.
# Diffusion:  transition=diffusion experiment.training.steps=4000
#             experiment.hyperparams.lambda_warmup_steps=800
# ---------------------------------------------------------------------------

Robot2DExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="robot-basis-pursuit", T=64, D=2,
                                      N_per_split=1024, batch_size=32),
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=400,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=2000, log_every=50, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=Robot2DEvalConf,
    viz_conf=Robot2DVizConf,
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(Robot2DExperimentConf, group="experiment", name="robot_2d")


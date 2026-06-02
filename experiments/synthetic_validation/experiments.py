"""Register the ``synthetic_validation`` presets — one per synthetic dataset.

The worked example for ``docs/authoring/``: a single simple model
(:func:`experiments.synthetic_validation.model.build_synthval_model`) trained on
several 1-D synthetic datasets, to check it recovers known dynamics. Each
dataset becomes its own preset ``synthval__<tag>``::

    python -m experiments list | grep synthval
    python -m ddssm.app experiment=synthval__harmonic --cfg job   # inspect
    python -m ddssm.app experiment=synthval__harmonic              # train

All datasets here are ``D=1, T=32``, so one model shape (``data_dim=1,
latent_dim=1, j=1``) fits all of them. The two-stage schedule is reused from the
init-centering family's stage builder (a small recon/baseline stage followed by
the diffusion stage); see ``docs/authoring/training.md``.
"""

from __future__ import annotations

from ddssm.data.presets import LGSSM, Bimodal, Harmonic
from ddssm.experiment.builders import Eval, Hparams, Objective, Training
from ddssm.experiment.stores import experiment_store
from experiments._make import experiment
from experiments.init_centering.hparams import StagesB
from experiments.synthetic_validation.model import SynthValModel

# Dataset axis: tag -> library dataset preset (all D=1, T=32).
DATASETS = {
    "harmonic": Harmonic,   # noisy sine waves
    "lgssm": LGSSM,         # linear-Gaussian state space
    "bimodal": Bimodal,     # bimodal random walk
}

# One model shape / hparams / training spec, shared across datasets.
_hparams = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    enc_lr=5e-4, dec_lr=5e-4, trans_lr=5e-4, ema_decay=0.997,
)
# `steps` is ignored under `stages` (the budget comes from the stage specs);
# kept positive as the single-fit fallback / sanity value (= n_pretrain + n_stage2).
_training = Training(steps=400, log_every=25, amp=True)

for _tag, _data in DATASETS.items():
    _exp = experiment(
        data=_data,
        model=SynthValModel(data_dim=1, latent_dim=1, j=1),
        hparams=_hparams,
        training=_training,
        # Small two-stage budget so a cell finishes quickly.
        stages=StagesB(baseline_mode="pinned", n_pretrain=100, n_stage2=300),
        eval=Eval(metrics=["mae", "crps_sum", "stage2_elbo_surrogate"], split="val"),
        objective=Objective(metric="loss/total", split="train", source="csv"),
    )
    experiment_store(_exp, name=f"synthval__{_tag}")

"""Compose-and-instantiate tests for the Hydra ``experiment`` config group.

Each registered experiment must:

* Compose at config time (``hydra.compose``) without raising.
* Yield an :class:`Experiment` instance whose ``data``, ``model``,
  ``training``, and ``objective`` fields are populated.
* Produce a non-empty model parameter count.

These tests don't run training -- they only check config plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate

import ddssm.conf  # noqa: F401  -- registers ConfigStore entries
from ddssm.data.datamodule import DDSSMDataModule
from ddssm.experiment import Experiment, ObjectiveSpec, TrainingScalars

CONF_DIR = (Path(__file__).resolve().parent.parent / "conf").as_posix()

EXPERIMENTS = [
    "synthetic_gauss",
    "synthetic_diffusion",
    "kdd_gauss",
    "kdd_diffusion",
    "harmonic",
    "bimodal",
    "robot_2d",
]

SWEEPS = ["synthetic_lr", "kdd_phase1"]


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.training.steps > 0
    assert cfg.experiment.model._target_.endswith("DDSSM_base")
    assert cfg.experiment.data._target_  # data module target is set


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_instantiates(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    assert isinstance(expt.data, DDSSMDataModule)
    assert isinstance(expt.training, TrainingScalars)
    assert isinstance(expt.objective, ObjectiveSpec)
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


def test_default_experiment_is_synthetic_gauss() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert cfg.experiment.model.transition._target_.endswith("GaussianTransition")
    assert cfg.experiment.model.encoder._target_.endswith("GaussianEncoder")
    assert cfg.experiment.model.decoder._target_.endswith("GaussianDecoder")
    assert cfg.experiment.model.z_init._target_.endswith("GaussianInitPrior")


# ---------------------------------------------------------------------------
# Encoder / Decoder / InitPrior plug-and-play overrides.
#
# Mirrors the existing ``transition=…`` override coverage: each module
# slot is a Hydra config group, so ``encoder=NAME``, ``decoder=NAME``, and
# ``z_init=NAME`` overrides must compose and instantiate cleanly on every
# registered experiment.
# ---------------------------------------------------------------------------

MODULE_GROUP_OVERRIDES = [
    "encoder=gaussian",
    "decoder=gaussian",
    "z_init=gaussian",
]

MLP_MODULE_GROUP_OVERRIDES = [
    "encoder=gaussian_mlp",
    "decoder=gaussian_mlp",
    "z_init=gaussian_mlp",
]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_module_group_overrides_compose(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={name}"] + MODULE_GROUP_OVERRIDES,
        )
    assert cfg.experiment.model.encoder._target_.endswith("GaussianEncoder")
    assert cfg.experiment.model.decoder._target_.endswith("GaussianDecoder")
    assert cfg.experiment.model.z_init._target_.endswith("GaussianInitPrior")


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_module_group_overrides_instantiate(name: str) -> None:
    """Model still builds with non-empty parameter count when each module
    slot is selected via its config group instead of the hard-coded path."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={name}"] + MODULE_GROUP_OVERRIDES,
        )
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


def test_mlp_architecture_ablation_overrides_compose() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=harmonic",
                "transition=diffusion_mlp",
            ]
            + MLP_MODULE_GROUP_OVERRIDES,
        )

    assert cfg.experiment.model.transition._target_.endswith("DiffusionTransition")
    assert cfg.experiment.model.transition.unet._target_.endswith("MLPCSDIUnet")
    assert cfg.experiment.model.encoder.context._target_.endswith("MLPContextProducer")
    assert cfg.experiment.model.decoder.context._target_.endswith("MLPContextProducer")
    assert cfg.experiment.model.z_init.context._target_.endswith("MLPContextProducer")
    assert cfg.experiment.model.z_init.aux_context._target_.endswith("MLPContextProducer")


def test_mlp_architecture_ablation_overrides_instantiate() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=harmonic",
                "transition=diffusion_mlp",
            ]
            + MLP_MODULE_GROUP_OVERRIDES,
        )
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


def test_objective_returns_inf_on_missing_csv(tmp_path) -> None:
    obj = ObjectiveSpec(metric="loss/total", split="train", tail_frac=0.1)
    assert obj.read(str(tmp_path / "missing.csv")) == float("inf")


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"+sweep={name}"],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    assert "optuna" in sweeper._target_.lower()
    assert sweeper.direction == "minimize"
    assert len(sweeper.params) > 0


def test_experiment_and_sweep_combine() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=synthetic_gauss", "+sweep=synthetic_lr"],
            return_hydra_config=True,
        )
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert "optuna" in cfg.hydra.sweeper._target_.lower()



# ---------------------------------------------------------------------------
# Synthetic verification: base presets + transition/override combos.
#
# Each preset uses transition=${transition} (top-level group, default:
# gaussian).  Override with transition=diffusion plus the extra training
# scalars shown in verifications.org.
# ---------------------------------------------------------------------------

SYNTH_BASE_PRESETS = ["harmonic", "bimodal", "robot_2d"]

SYNTH_DIFFUSION_OVERRIDES = [
    "transition=diffusion",
    "experiment.training.steps=2000",
    "experiment.training.checkpoint_every=500",
    "experiment.hyperparams.lambda_warmup_steps=400",
]


@pytest.mark.parametrize("name", SYNTH_BASE_PRESETS)
def test_synth_base_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.training.steps > 0
    assert cfg.experiment.model._target_.endswith("DDSSM_base")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")


@pytest.mark.parametrize("name", SYNTH_BASE_PRESETS)
def test_synth_base_preset_instantiates(name: str) -> None:
    """Model builds and eval/viz specs are present for every base preset."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    assert isinstance(expt.data, DDSSMDataModule)
    assert isinstance(expt.training, TrainingScalars)
    assert isinstance(expt.objective, ObjectiveSpec)
    assert expt.eval is not None, f"{name}: eval spec is None"
    assert expt.viz is not None, f"{name}: viz spec is None"
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


@pytest.mark.parametrize("name", SYNTH_BASE_PRESETS)
def test_synth_diffusion_override_composes(name: str) -> None:
    """transition=diffusion override must resolve cleanly on every base preset."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={name}"] + SYNTH_DIFFUSION_OVERRIDES,
        )
    assert cfg.experiment.transition._target_.endswith("DiffusionTransition")
    assert cfg.experiment.training.steps == 2000


@pytest.mark.parametrize("expected_data_dim,expected_j,overrides", [
    (1, 1, ["experiment=harmonic"]),
    (1, 2, ["experiment=harmonic", "experiment.j=2"]),
    (1, 1, ["experiment=bimodal"]),
    (2, 2, ["experiment=robot_2d"]),
])
def test_synth_shape_fields(
    expected_data_dim: int, expected_j: int, overrides: list
) -> None:
    """Shape fields in the composed config must match what the model will be built with."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    assert cfg.experiment.data_dim == expected_data_dim
    assert cfg.experiment.j == expected_j


@pytest.mark.parametrize("overrides,expected_metrics", [
    (["experiment=harmonic"],  ["mae", "crps_sum"]),
    (["experiment=harmonic", "experiment.data.mode=harmonic-noisy"], ["mae", "crps_sum"]),
    (["experiment=bimodal"],   ["energy_score", "crps_sum"]),
    (["experiment=robot_2d"],  ["energy_score", "crps_sum"]),
])
def test_synth_eval_metrics(overrides: list, expected_metrics: list) -> None:
    """Eval metric list must match the family spec (harmonic→mae, bimodal/robot→energy_score)."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    assert list(cfg.experiment.eval.metrics) == expected_metrics


@pytest.mark.parametrize("overrides,expected_first_plot", [
    (["experiment=harmonic"],  "forecast_1d"),
    (["experiment=bimodal"],   "forecast_1d"),
    (["experiment=robot_2d"],  "forecast_2d_spatial"),
])
def test_synth_viz_first_plot(overrides: list, expected_first_plot: str) -> None:
    """Robot preset must use the 2D spatial plot; all others use forecast_1d."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    assert cfg.experiment.viz.plots[0].name == expected_first_plot


@pytest.mark.parametrize("name", SYNTH_BASE_PRESETS)
def test_synth_eval_conf_has_explicit_t_split(name: str) -> None:
    """Synthetic eval confs with forecasting metrics MUST carry an explicit T_split.

    SyntheticDataModule.metadata.forecast_split is always None, so without an
    explicit T_split the eval runner would silently pass None to _iter_forecast_batches
    and raise at metric-compute time.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    t_split = cfg.experiment.eval.T_split
    assert t_split is not None and int(t_split) > 0, (
        f"{name}: eval.T_split={t_split!r} — forecast metrics (mae/energy_score/crps_sum) "
        "require a non-None T_split on synthetic datasets"
    )

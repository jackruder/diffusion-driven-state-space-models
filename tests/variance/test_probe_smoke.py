from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra_zen import instantiate

import ddssm.conf  # noqa: F401
from ddssm.variance.runner import ProbeCell, ProbePlotSpec, ProbeSpec, variance

CONF_DIR = (Path(__file__).resolve().parents[2] / "conf").as_posix()


def _make_experiment(name: str):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    return instantiate(cfg.experiment)


def test_encode_for_probe_matches_transition_term() -> None:
    expt = _make_experiment("synthetic_diffusion")
    model = expt.model
    loader = expt.data.train_loader()
    batch = expt.data.batch_transform(next(iter(loader)), torch.device("cpu"))
    with torch.no_grad():
        probe_batch = model.encode_for_probe(batch)
        bs = probe_batch.zs.shape[0] * probe_batch.zs.shape[1]
        d = probe_batch.zs.shape[2]
        sk = int(model.transition.S_k)
        mc_override = {
            "k_idx": torch.zeros((bs, sk), dtype=torch.long),
            "eps": torch.zeros((bs, d, sk), dtype=probe_batch.zs.dtype),
        }
        # probe path (what the variance runner does)
        trans_probe = model.transition.transition_kl(
            **probe_batch.as_kwargs(), mc_override=mc_override
        )
        # internal model path (what forward calls)
        trans_internal = model._compute_transition_kl(
            **probe_batch.as_kwargs(), mc_override=mc_override
        )
    assert torch.allclose(trans_probe["kl"], trans_internal["kl"], atol=1e-5, rtol=1e-5)


def test_variance_runner_smoke(tmp_path: Path) -> None:
    expt = _make_experiment("variance_probe_lgssm")
    spec = ProbeSpec(
        cells=[ProbeCell("esm", "uniform"), ProbeCell("dsm", "uniform")],
        R=2,
        n_batches=1,
        seeds=[0],
        force_per_k=False,
        plots=[ProbePlotSpec(name="summary_table", save_filename="summary_table.png")],
    )
    out = variance(expt, spec, device=torch.device("cpu"), run_dir=str(tmp_path))
    assert "summary" in out and "metrics" in out
    assert (tmp_path / "variance_raw.csv").is_file()
    assert (tmp_path / "variance_summary.json").is_file()
    assert (tmp_path / "summary_table.png").is_file()

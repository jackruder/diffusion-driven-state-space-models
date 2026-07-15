"""Guards for the DDSSM_COMPILE_STEP=1 two-region compiled train step.

The compiled path is a structural alternative to the eager microstep
loop — same metric key set, optimizer moves, EMA moves. Bit-for-bit
equality with eager is NOT expected (bf16 autocast + inline loss
composition vs FullELBO.__call__).
"""

from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

from ddssm.model.dssd import DDSSM_base
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.model.transitions.transitions import GaussianTransition
from ddssm.nn.aggregators import IdentityAggregator
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.diffnets import (
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.futsum import GRUFutureSummary
from ddssm.nn.gaussians import GaussianHead
from ddssm.training.train import DDSSMTrainer, _make_split_step

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2

_CTX = partial(
    ContextProducer,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_FS = partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1)


def _make_tiny_model():
    combiner = partial(
        CompoundCombiner,
        aggregator=partial(IdentityAggregator),
        fusion=partial(ConcatLinearFusion),
    )
    enc = GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=combiner,
        dist_head=partial(GaussianDistHead),
        fut_summary=_FS,
    )
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=GaussianHead,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=GaussianHead,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=CHANNELS, n_layers=1)
    return DDSSM_base(
        encoder=enc,
        decoder=dec,
        transition=trans,
        aux_posterior=aux,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
    )


class _SyntheticBatch(Dataset):
    def __init__(self, B: int = 2, T: int = 4):
        self.B = B
        self.T = T

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        return {
            "observed_data": torch.randn(DATA_DIM, self.T),
            "observation_mask": torch.ones(DATA_DIM, self.T),
            "timepoints": torch.arange(self.T, dtype=torch.float32),
        }


def test_make_split_step_returns_two_callables():
    """``_make_split_step`` returns a (fwd_bwd, opt_ema) pair per docs."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params_flat = [p for g in opt.param_groups for p in g["params"]]
    fwd_bwd, opt_ema = _make_split_step(
        model, active_loss=None, optimizer=opt,
        params_flat=params_flat, max_norm=1.0, ema=None,
    )
    assert callable(fwd_bwd)
    assert callable(opt_ema)


def test_fit_compiled_step_runs_and_updates_params(tmp_path, monkeypatch):
    """DDSSM_COMPILE_STEP=1 fit runs and moves the model."""
    monkeypatch.setenv("DDSSM_COMPILE_STEP", "1")
    # Sub-module compiles off — CPU nn.GRU trips a dynamo polyfill.
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "0")

    torch.manual_seed(42)
    model = _make_tiny_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "metrics.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    loader = DataLoader(_SyntheticBatch(B=2, T=4), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=3,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    after = model.state_dict()
    moved = any(
        not torch.equal(before[k], after[k])
        for k, v in after.items() if torch.is_floating_point(v)
    )
    assert moved, "compiled fit did not update any float parameter"


def test_fit_compiled_step_metric_keys_match_eager(tmp_path, monkeypatch):
    """Compiled and eager paths log the same metric keys (regression guard
    for the ``_build_metrics_from_flat`` contract)."""
    import csv as _csv

    def _run(compile_step: str) -> set[str]:
        monkeypatch.setenv("DDSSM_COMPILE_STEP", compile_step)
        monkeypatch.setenv("DDSSM_TORCH_COMPILE", "0")
        torch.manual_seed(0)
        model = _make_tiny_model()
        csv_path = tmp_path / f"metrics_{compile_step}.csv"
        trainer = DDSSMTrainer(
            model=model,
            device=torch.device("cpu"),
            csv_log_path=str(csv_path),
            tensorboard_dir=str(tmp_path / f"tb_{compile_step}"),
            quiet=True,
        )
        loader = DataLoader(_SyntheticBatch(B=2, T=4), batch_size=2)
        trainer.fit(
            train_loader=loader,
            val_loader=None,
            total_steps=2,
            validate_every=0,
            log_every=1,
            checkpoint_every=None,
            amp=False,
        )
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            return set(reader.fieldnames or [])

    eager_cols = _run("0")
    compiled_cols = _run("1")
    model_keys_eager = {c for c in eager_cols if c.startswith(("loss/", "calib/", "diag/"))}
    model_keys_compiled = {c for c in compiled_cols if c.startswith(("loss/", "calib/", "diag/"))}
    assert model_keys_eager == model_keys_compiled, (
        f"metric key mismatch\n"
        f"  eager-only:    {sorted(model_keys_eager - model_keys_compiled)}\n"
        f"  compiled-only: {sorted(model_keys_compiled - model_keys_eager)}"
    )


def test_ema_shadow_moves_under_compiled_step(tmp_path, monkeypatch):
    """EMA shadow tensors update in-graph under the compiled path.

    Guards the ``_float_live``/``_int_pairs`` caching in :class:`EMA`:
    if the cached refs got stale (e.g. a swap replacing tensor
    identity), the shadows would stop moving.
    """
    monkeypatch.setenv("DDSSM_COMPILE_STEP", "1")
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "0")
    torch.manual_seed(7)
    model = _make_tiny_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "m.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    keys = list(trainer.ema._float_keys)[:4]
    shadow_before = [
        trainer.ema.shadow[k].detach().clone() for k in keys
    ]
    loader = DataLoader(_SyntheticBatch(B=2, T=4), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=3,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    shadow_after = [
        trainer.ema.shadow[k].detach().clone() for k in keys
    ]
    moved = any(
        not torch.equal(a, b) for a, b in zip(shadow_before, shadow_after)
    )
    assert moved, "EMA shadow did not move under compiled fit path"

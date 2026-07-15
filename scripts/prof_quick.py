"""Fast steady-state train-step wall-time check (~1 min per run when cache is warm).

Runs the current trainer path (no fused-step, no outer compile) for the same
h2h preset used across the arflow-encoder perf tuning. Prints ``step=Xms`` and
projected 20K-step training time, plus dynamo/inductor cache stats and a
CPU-time profiler ranking of the top dispatch hotspots.

Env / prerequisites:
    * Run under the venv: ``.venv/bin/python scripts/prof_quick.py``.
    * ``DDSSM_TORCH_COMPILE`` defaults to strict; set ``=0`` for eager.
    * Requires a working CUDA device (uses ``torch.cuda.synchronize()``).

Usage:
    .venv/bin/python scripts/prof_quick.py
"""
import os
import sys
import time

os.environ.setdefault("DDSSM_TORCH_COMPILE", "strict")
import torch  # noqa: E402  (import after env setup)

# Make the repo importable without ``pip install -e .``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_module  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from ddssm.experiment.registry import register_experiments  # noqa: E402
from ddssm.model.losses import FullELBO  # noqa: E402
from ddssm.training.train import _compile_optimizer_step  # noqa: E402

N_WARMUP = 15
N_TIMED = 10
BATCH_SIZE = 64
PRESET = "h2h__gaussian_csdilike_ais_big_wideenc_conv_20k_gjsd_lrsched_split__nlblmv__j4"


def build():
    register_experiments()
    with initialize_config_module("ddssm.conf", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment={PRESET}",
                f"++experiment.hparams.batch_size={BATCH_SIZE}",
                "++experiment.hparams.use_split_loss=false",
                "experiment.training.steps=1",
                "experiment.training.log_every=100000",
                "experiment.training.validate_every=0",
                "experiment.training.checkpoint_every=100000",
            ],
        )
    exp = instantiate(cfg.experiment)
    device = torch.device("cuda")
    exp.model.to(device)
    trainer = exp.build_trainer(
        model=exp.model, device=device,
        csv_log_path="/tmp/prof_ignore.csv",
        tensorboard_dir=None, checkpoint_dir=None, wandb_config=None,
        hparams=exp.hparams,
    )
    trainer._active_loss = trainer._build_default_loss(total_steps=1)
    loss_wants_split = isinstance(trainer._active_loss, FullELBO) and getattr(
        trainer._active_loss, "use_split_loss", False
    )
    if loss_wants_split:
        trainer._install_split_topology()
    else:
        from types import SimpleNamespace
        trainer._rebuild_optimizer(SimpleNamespace(
            enc_lr=exp.hparams.enc_lr, dec_lr=exp.hparams.dec_lr,
            trans_lr=exp.hparams.trans_lr,
        ))
    trainer._scaler = torch.amp.GradScaler("cuda", enabled=False)
    # Mirror what fit() does — compile optimizer.step so the profile
    # measures the same code path as production.
    _compile_optimizer_step(trainer._optimizers)
    return exp, trainer, device


def one_step(trainer, batch, amp):
    for opt in trainer._optimizers:
        opt.zero_grad(set_to_none=True)
    loss, _, _ = trainer._compute_loss_and_metrics(batch, amp=amp)
    trainer._backward_loss(loss, trainer._scaler, amp)
    trainer._optimizer_step(trainer._scaler, amp)


def main():
    exp, trainer, device = build()
    loader = exp.data.train_loader()
    xform = exp.data.batch_transform
    it = iter(loader)

    def nb():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return xform(b, device) if xform is not None else {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in b.items()
        }

    for _ in range(N_WARMUP):
        one_step(trainer, nb(), True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        one_step(trainer, nb(), True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"step={1000*(t1-t0)/N_TIMED:.1f}ms  proj20K={20000*(t1-t0)/N_TIMED/60:.1f}min")

    # Quick torch.profiler over 10 steps for top CPU dispatch.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            one_step(trainer, nb(), True)
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))

    import torch._dynamo as _dyn
    dyn = dict(_dyn.utils.counters.get("stats", {}))
    if "unique_graphs" in _dyn.utils.counters:
        dyn.update(_dyn.utils.counters["unique_graphs"])
    hits = sum(
        v for k, v in _dyn.utils.counters.get("inductor", {}).items()
        if "hit" in k.lower()
    )
    misses = sum(
        v for k, v in _dyn.utils.counters.get("inductor", {}).items()
        if "miss" in k.lower()
    )
    print(f"dyn stats={dyn}  inductor hits={hits} miss={misses}")


if __name__ == "__main__":
    main()

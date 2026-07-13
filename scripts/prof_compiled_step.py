"""Compare baseline / fused-eager / fused-compiled train-step wall times.

Reuses one built trainer across all three variants (warm cache carries over)
and prints per-variant ``step=Xms  proj20K=Ymin  peakVRAM=ZMB``. Used to
measure the payoff of :func:`ddssm.training.train._make_fused_step` and its
outer-``torch.compile`` variant relative to the current ``fit()`` codepath.

Env vars:
    * ``DDSSM_TORCH_COMPILE`` — defaults to ``"strict"``; sub-module compiles
      inside the trainer are governed by this.
    * ``DDSSM_TORCH_COMPILE_MODE`` — forwarded to ``torch.compile(..., mode=)``
      on the outer fused variant. Set to ``"reduce-overhead"`` to test CUDA
      graphs on the fused step.

The preset forces ``use_split_loss=False`` so we're in the single-loss
regime the fused step assumes.

Usage:
    .venv/bin/python scripts/prof_compiled_step.py
"""
import os
import sys
import time

os.environ.setdefault("DDSSM_TORCH_COMPILE", "strict")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import compose, initialize_config_module  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from ddssm.experiment.registry import register_experiments  # noqa: E402
from ddssm.training.train import _compile_optimizer_step, _make_fused_step  # noqa: E402

N_WARMUP = 8
N_TIMED = 5
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
                # Fused step assumes single-loss; force off explicitly.
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
    from types import SimpleNamespace
    trainer._rebuild_optimizer(SimpleNamespace(
        enc_lr=exp.hparams.enc_lr, dec_lr=exp.hparams.dec_lr,
        trans_lr=exp.hparams.trans_lr,
    ))
    trainer._scaler = torch.amp.GradScaler("cuda", enabled=False)
    _compile_optimizer_step(trainer._optimizers)
    return exp, trainer, device


def _current_trainer_step(trainer, batch, amp):
    for opt in trainer._optimizers:
        opt.zero_grad(set_to_none=True)
    loss, _, _ = trainer._compute_loss_and_metrics(batch, amp=amp)
    trainer._backward_loss(loss, trainer._scaler, amp)
    trainer._optimizer_step(trainer._scaler, amp)


def _time_it(step_fn, nb, label):
    for _ in range(N_WARMUP):
        step_fn(nb())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        step_fn(nb())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = 1000 * (t1 - t0) / N_TIMED
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(
        f"  {label:16s} step={ms:6.1f}ms  proj20K={20000*ms/1000/60:5.1f}min  "
        f"peakVRAM={peak_mb:6.0f}MB"
    )


def main():
    print(f"preset: {PRESET}")
    print(f"batch={BATCH_SIZE}, warmup={N_WARMUP}, timed={N_TIMED}")
    compile_mode = os.environ.get("DDSSM_TORCH_COMPILE_MODE", "default")
    print(f"DDSSM_TORCH_COMPILE_MODE={compile_mode!r}")
    print()

    # Build once; reuse the trainer / model / optimizer state across all
    # three variants. Warm caches carry over. Only the step function differs.
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

    baseline = lambda batch: _current_trainer_step(trainer, batch, True)

    opt = trainer._optimizers[0]
    params_flat = [p for g in opt.param_groups for p in g["params"]]
    max_norm = (
        float(trainer.clip_grad_norm)
        if trainer.clip_grad_norm is not None else float("inf")
    )
    fused_eager = _make_fused_step(
        trainer.model, trainer._active_loss, opt, params_flat, max_norm,
    )
    _compile_mode = os.environ.get("DDSSM_TORCH_COMPILE_MODE", "").strip() or None
    fused_compiled = torch.compile(
        _make_fused_step(
            trainer.model, trainer._active_loss, opt, params_flat, max_norm,
        ),
        **({"mode": _compile_mode} if _compile_mode else {}),
    )
    lam = torch.tensor(
        float(trainer._active_loss.rate_lambda(1)), device=device
    )
    fused_eager_step = lambda batch, _f=fused_eager, _l=lam: _f(batch, _l)
    fused_compiled_step = lambda batch, _f=fused_compiled, _l=lam: _f(batch, _l)

    _time_it(baseline, nb, "baseline")
    _time_it(fused_eager_step, nb, "fused_eager")
    _time_it(fused_compiled_step, nb, "fused_compiled")


if __name__ == "__main__":
    main()

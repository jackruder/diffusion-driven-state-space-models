"""Bench the two-region compiled train step against the eager trainer path.

Prints per-variant ``step=Xms  proj20K=Ymin  peakVRAM=ZMB`` for:
  * ``baseline`` — current ``DDSSMTrainer._compute_loss_and_metrics`` +
    ``_backward_loss`` + ``_optimizer_step`` (the pre-Phase-C path).
  * ``split_eager`` — the two-region split step as plain Python
    (measures the Python-overhead cost of the fused-step shape).
  * ``split_compiled`` — the docs-blessed pattern: fwd+bwd compiled via
    the compiled_autograd hook, opt.step compiled standalone
    (fullgraph=False).

Env vars:
  * ``DDSSM_TORCH_COMPILE`` — defaults to ``"strict"``; sub-module compiles
    inside the trainer are governed by this.
  * ``DDSSM_TORCH_COMPILE_MODE`` — forwarded to ``torch.compile(..., mode=)``
    on the fwd+bwd region. On this preset ``reduce-overhead`` / ``max-autotune``
    regress wall time (denoiser shape variance blocks stable capture).

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
from ddssm.training.train import _make_split_step  # noqa: E402

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
                # Fused step assumes single-loss.
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
    return exp, trainer, device


def _baseline_step(trainer, batch, amp):
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

    baseline = lambda batch: _baseline_step(trainer, batch, True)

    opt = trainer._optimizers[0]
    params_flat = [p for g in opt.param_groups for p in g["params"]]
    max_norm = (
        float(trainer.clip_grad_norm)
        if trainer.clip_grad_norm is not None else float("inf")
    )
    split_fwd_bwd, split_opt_ema = _make_split_step(
        trainer.model, trainer._active_loss, opt, params_flat, max_norm,
        ema=trainer.ema,
    )
    _mode = os.environ.get("DDSSM_TORCH_COMPILE_MODE", "").strip() or None
    _fwd_kwargs = {"mode": _mode} if _mode else {}
    if _mode in {"reduce-overhead", "max-autotune"}:
        _fwd_kwargs["dynamic"] = False
    compiled_fwd_bwd = torch.compile(split_fwd_bwd, **_fwd_kwargs)
    # opt.step compiled standalone (torch's "Compiling the optimizer" recipe).
    # Wrapping opt_ema in a compile would nest and thrash on per-param-group
    # ``maybe_fallback`` guards.
    if not getattr(opt.step, "_ddssm_compiled", False):
        opt.step = torch.compile(opt.step, fullgraph=False)
        opt.step._ddssm_compiled = True

    lam = torch.tensor(
        float(trainer._active_loss.rate_lambda(1)), device=device
    )

    def split_eager_step(batch):
        split_fwd_bwd(batch, lam)
        split_opt_ema()

    def split_compiled_step(batch):
        compiled_fwd_bwd(batch, lam)
        split_opt_ema()

    _time_it(baseline, nb, "baseline")
    _time_it(split_eager_step, nb, "split_eager")
    _time_it(split_compiled_step, nb, "split_compiled")


if __name__ == "__main__":
    main()

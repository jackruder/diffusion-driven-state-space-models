"""Benchmark torch.compile vs eager on a real DDSSM training step.

Runs the actual trainer fit loop for a preset, timing each optimizer step
(CUDA-synced) so the one-time compile spike is separated from steady state.
Spawns one eager worker (``DDSSM_TORCH_COMPILE=0``) and one compiled worker
(``DDSSM_TORCH_COMPILE=1``) in separate processes, then prints a comparison.

Usage:
    python scripts/bench_compile.py <preset> [--steps N] [-o KEY=VAL ...]

Examples:
    python scripts/bench_compile.py init_smoke_simple --steps 80
    python scripts/bench_compile.py init_smoke_simple --steps 50 -o experiment.hparams.batch_size=512

Note (NixOS): inductor needs ``TRITON_LIBCUDA_PATH`` and ``TRITON_PTXAS_PATH``
or it silently falls back to eager (DDSSM's maybe_compile sets
``suppress_errors=True``). This script auto-detects them if unset. Verify a real
compile happened: the compiled run's first step shows a multi-second spike.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
from pathlib import Path
import argparse
import statistics as st
import subprocess

REPO = Path(__file__).resolve().parent.parent


def _worker(preset: str, steps: int, tag: str, extra: list[str]) -> None:
    """One run: fit `steps` steps, print a BENCHJSON line of per-step ms."""
    import warnings

    warnings.filterwarnings("ignore")
    import torch

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    from hydra import compose, initialize_config_dir
    from ddssm.train import DDSSMTrainer
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from ddssm._experiment_registry import register_experiments

    register_experiments()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "src/ddssm/conf")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment={preset}",
                f"experiment.training.steps={steps}",
                "experiment.training.log_every=100000",
                "experiment.training.validate_every=0",
                "experiment.training.checkpoint_every=100000",
            ]
            + extra,
        )
    exp = instantiate(cfg.experiment)
    exp.model.config.checkpoint_dir = "/tmp/bench_ckpts"

    ts: list[float] = []
    _orig = DDSSMTrainer._optimizer_step

    def _timed(self, scaler, amp):
        r = _orig(self, scaler, amp)
        torch.cuda.synchronize()
        ts.append(time.perf_counter())
        return r

    DDSSMTrainer._optimizer_step = _timed

    tk = dict(
        model=exp.model,
        device=device,
        csv_log_path="/tmp/bench_metrics.csv",
        tensorboard_dir="/tmp/bench_tb",
        wandb_config=None,
    )
    if exp.hparams is not None:
        tk["hparams"] = exp.hparams
    if exp.model_config_yaml is not None:
        tk["model_config_yaml"] = exp.model_config_yaml
    trainer = exp.build_trainer(**tk)
    if exp.training.trainable is not None:
        trainer._set_trainable(exp.training.trainable)

    train_loader = exp.data.train_loader()
    nparams = sum(p.numel() for p in exp.model.parameters())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        batch_transform=exp.data.batch_transform,
        **exp.training.fit_kwargs(),
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    intervals = [ts[0] - t0] + [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    print(
        "BENCHJSON "
        + json.dumps({
            "tag": tag,
            "preset": preset,
            "nparams": nparams,
            "total_fit_s": total,
            "intervals_ms": [round(x * 1000, 3) for x in intervals],
        })
    )


def _run_worker(
    preset: str, steps: int, tag: str, compile_on: bool, extra: list[str]
) -> dict:
    """Spawn a worker subprocess with the right env; parse its BENCHJSON."""
    env = dict(os.environ)
    env["DDSSM_TORCH_COMPILE"] = "1" if compile_on else "0"
    # NixOS: triton hardcodes /sbin/ldconfig and ships a non-runnable ptxas.
    if compile_on:
        env.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")
        ptxas = shutil.which("ptxas")
        if ptxas and "TRITON_PTXAS_PATH" not in env:
            env["TRITON_PTXAS_PATH"] = ptxas
    cmd = [sys.executable, __file__, preset, "--steps", str(steps), "--worker", tag]
    for o in extra:
        cmd += ["-o", o]
    out = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("BENCHJSON "):
            return json.loads(line[len("BENCHJSON ") :])
    raise RuntimeError(
        f"{tag} worker produced no result.\nSTDERR tail:\n"
        + "\n".join(out.stderr.splitlines()[-15:])
    )


def _steady(x: list[float]) -> float:
    return st.median(x[len(x) // 2 :])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="extra hydra override, e.g. -o experiment.hparams.batch_size=512",
    )
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker is not None:
        _worker(args.preset, args.steps, args.worker, args.override)
        return

    print(f"benchmarking '{args.preset}' ({args.steps} steps each) ...")
    eager = _run_worker(args.preset, args.steps, "eager", False, args.override)
    print("  eager done; compiling (first step will be slow) ...")
    comp = _run_worker(args.preset, args.steps, "compiled", True, args.override)

    ei, ci = eager["intervals_ms"], comp["intervals_ms"]
    es, cs = _steady(ei), _steady(ci)
    overhead = (
        sum(max(0.0, x - cs) for x in ci) - sum(max(0.0, x - es) for x in ei)
    ) / 1000
    speedup = es / cs
    be = overhead / ((es - cs) / 1000) if es > cs else None

    print(
        f"\npreset={args.preset}  params={eager['nparams']:,}  steps={len(ei)}  "
        f"overrides={args.override or '(none)'}"
    )
    print(f"  eager    steady = {es:8.2f} ms/step   (step0 {ei[0]:.0f} ms)")
    print(
        f"  compiled steady = {cs:8.2f} ms/step   (step0 {ci[0]:.0f} ms  <- compile spike)"
    )
    print(f"  speedup         = {speedup:.2f}x")
    print(f"  compile overhead= ~{overhead:.1f} s (one-time)")
    print(
        "  break-even      = "
        + (f"~{be:.0f} steps" if be else "never (compiled slower on this config)")
    )


if __name__ == "__main__":
    main()

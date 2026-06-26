"""Speed benchmark + profile: arflow vs gaussian encoder (the throughput claim).

The whole point of ``ARFlowEncoder`` is killing ``GaussianEncoder``'s sequential
``for t`` loop. This times (a) the encoder ``sample_paths`` (fwd+bwd) across T and
(b) a full DDSSM train step, gaussian vs arflow, then runs a torch profile of one
step at large T. Eager by default (the structural parallelism win shows without
compile); set ``DDSSM_TORCH_COMPILE=1`` + the NixOS triton env to add compile.

Run::

    .venv/bin/python experiments/arflow_headtohead/bench.py
"""

import torch  # preload first so numpy's C-extensions resolve on NixOS
import os
import sys
import time
from functools import partial

# Make ``import experiments`` resolve when run as a bare script (mirrors registry.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.profiler import profile, ProfilerActivity

from ddssm.model.encoder import GaussianEncoder, ARFlowEncoder
from ddssm.nn.futsum import TransformerFutureSummary
from experiments.gluonts_forecast.model import build_gluonts_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Head-to-head model dims (nonlin_bimodal_lift_mv): D=8, latent=8, width=16, C=32.
B, D, LATENT, WIDTH, CH, NHEADS = 32, 8, 8, 16, 32, 4


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _make_encoder(kind: str, j: int = 1):
    fut = partial(
        TransformerFutureSummary, summary_dim=WIDTH, nheads=NHEADS, transformer_layers=1
    )
    common = dict(
        data_dim=D, latent_dim=LATENT, j=j, emb_time_dim=0, use_mask=False,
        hidden_dim=WIDTH, fut_summary=fut,
    )
    if kind == "gaussian":
        return GaussianEncoder(**common).to(DEVICE)
    return ARFlowEncoder(
        **common, channels=CH, nheads=NHEADS, backbone="transformer"
    ).to(DEVICE)


def _time_ms(fn, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    return (time.perf_counter() - t0) / reps * 1e3


def bench_encoder() -> None:
    print(f"\n# Encoder sample_paths (fwd+bwd), {DEVICE} -- ms/call")
    print(f"{'T':>5} {'gaussian':>10} {'arflow':>10} {'speedup':>8}")
    for T in (32, 64, 128, 192):
        obs = torch.randn(B, D, T, device=DEVICE)
        te = torch.zeros(B, T, 0, device=DEVICE)

        def step(enc):
            def _f():
                enc.zero_grad(set_to_none=True)
                zs, _lq, _st = enc.sample_paths(obs, te, S=1)
                zs.sum().backward()
            return _f

        g, a = _make_encoder("gaussian"), _make_encoder("arflow")
        tg = _time_ms(step(g), reps=20, warmup=5)
        ta = _time_ms(step(a), reps=20, warmup=5)
        print(f"{T:>5} {tg:>10.2f} {ta:>10.2f} {tg / ta:>7.2f}x")


def _full_model(encoder_type: str, T: int):
    return build_gluonts_model(
        data_dim=D, latent_dim=LATENT, j=1, T_max=T, channels=CH, nheads=NHEADS,
        summary_layers=1, diffusion_layers=2, num_steps=64, grad_checkpoint=False,
        encoder_type=encoder_type,
    ).to(DEVICE).train()


def _full_batch(T: int):
    x = torch.randn(B, D, T, device=DEVICE)
    mask = torch.ones(B, D, T, device=DEVICE)
    tp = torch.arange(T, device=DEVICE).unsqueeze(0).expand(B, -1)
    return x, mask, tp


def bench_full_step() -> None:
    print(f"\n# Full DDSSM train step (fwd+bwd), {DEVICE} -- ms/step")
    print(f"{'T':>5} {'gaussian':>10} {'arflow':>10} {'speedup':>8}")
    for T in (32, 192):
        x, mask, tp = _full_batch(T)

        def step(m):
            def _f():
                m.zero_grad(set_to_none=True)
                c, _m, _s = m(observed_data=x, observation_mask=mask, timepoints=tp)
                c.total().backward()
            return _f

        g, a = _full_model("gaussian", T), _full_model("arflow", T)
        tg = _time_ms(step(g), reps=10, warmup=3)
        ta = _time_ms(step(a), reps=10, warmup=3)
        print(f"{T:>5} {tg:>10.2f} {ta:>10.2f} {tg / ta:>7.2f}x")


def profile_step(encoder_type: str, T: int = 192) -> None:
    print(f"\n# Profile: full step, encoder={encoder_type}, T={T}")
    m = _full_model(encoder_type, T)
    x, mask, tp = _full_batch(T)

    def _f():
        m.zero_grad(set_to_none=True)
        c, _m, _s = m(observed_data=x, observation_mask=mask, timepoints=tp)
        c.total().backward()

    for _ in range(3):
        _f()
    _sync()
    acts = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        for _ in range(5):
            _f()
        _sync()
    sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
    out = f"runs/bench_profile_{encoder_type}_T{T}.json"
    prof.export_chrome_trace(out)
    print(f"  chrome trace -> {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    bench_encoder()
    bench_full_step()
    profile_step("gaussian")
    profile_step("arflow")

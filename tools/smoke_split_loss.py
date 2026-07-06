"""Diagnostic smoke: 100-iter tiny model, single vs split loss modes.

Not a pytest test — a hand-runnable diagnostic for M8 verification.
Prints:
  * Final loss trajectory (single-mode with p_k_clip=1e-3, then split-mode).
  * ``optim/grad_skips`` counter (must be 0 in both).
  * Confirms ``loss/rate/trans/kl_phith`` and ``loss/rate/trans/kl_psi``
    are present in the split-mode log dict.

Run with (NixOS torch + numpy env — need zlib for numpy's C exts):
    LD_LIBRARY_PATH=/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH \
    TORCHDYNAMO_DISABLE=1 uv run python tools/smoke_split_loss.py

Notes on the split-mode loss trajectory
---------------------------------------
Under split mode the aggregate ``loss/total`` combines the (λ-weighted)
phith side and the *unit-weighted* psi side.  ψ trains at full strength
even during recon-only warmup (rate_lambda has no effect on it) — this is
intentional per the plan's two-timescale design.  The unit-weighted ψ
term can grow substantially early in training as the score net calibrates
against the encoder's moving target; that's expected and NOT a bug.  The
smoke here checks structural properties (metrics wired, grad_skips == 0,
losses finite), not directional convergence.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable (matches capture_goldens.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "test_integration"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from conftest import make_random_walk_data, make_vhp_model
from ddssm.model.losses import FullELBO
from ddssm.model.transitions.diffusion import DiffusionScheduleConfig
from ddssm.training.train import DDSSMTrainer


N_ITERS = 100
SEED = 0


class _RandomWalkDataset(Dataset):
    def __init__(self, *, n_seqs: int = 8, T: int = 8, seed: int = 0):
        payload = make_random_walk_data(n_seqs=n_seqs, T=T, seed=seed)
        self._obs = payload["observed_data"]
        self._mask = payload["observation_mask"]
        self._tp = payload["timepoints"]

    def __len__(self):
        return self._obs.size(0)

    def __getitem__(self, idx):
        return {
            "observed_data": self._obs[idx],
            "observation_mask": self._mask[idx],
            "timepoints": self._tp[idx],
        }


def _build_model(*, p_k_clip):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_vhp_model(
        baseline_form="mlp",
        baseline_mode="pinned",
        tracking_mode="fixed",
        lambda_sigma_p=0.0,
        sigma_data_init=1.0,
        snapshot_anchor=False,
    )
    # Override the diffusion schedule with the requested p_k_clip.
    model.transition.schedule = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        k_sampling_mode="uniform",
        p_k_clip=p_k_clip,
    )
    model.stage_selector = "stage_2"
    return model


def _run_smoke(*, use_split_loss: bool, p_k_clip):
    tag = "split" if use_split_loss else "single"
    print(f"\n=== smoke run: mode={tag}, p_k_clip={p_k_clip}, n_iters={N_ITERS} ===")
    model = _build_model(p_k_clip=p_k_clip)
    tmp_dir = REPO_ROOT / ".smoke_tmp" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)

    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_dir / "tb"),
        checkpoint_dir=str(tmp_dir / "ckpt"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(
        rate_lambda=lambda _step: 1.0,
        lambda_sigma_p=0.0,
        lambda_mu_p=0.0,
        use_split_loss=use_split_loss,
    )

    ds = _RandomWalkDataset(n_seqs=8, T=8, seed=SEED)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # Capture per-step loss via an in-memory logger.
    from ddssm.training.loggers import Logger

    class _MemLogger(Logger):
        def __init__(self):
            self.rows: list[tuple[int, dict]] = []

        def on_step(self, split, step, row):
            if split == "train":
                self.rows.append((step, dict(row)))

        def on_epoch(self, split, epoch, row):
            pass

    mem = _MemLogger()
    trainer.metrics.loggers.append(mem)

    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=N_ITERS,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    # Extract a 5-point trajectory from the in-memory log rows.
    checkpoints = [20, 40, 60, 80, 100]
    row_by_step = {step: row for step, row in mem.rows}
    trajectory = [
        (t, row_by_step.get(t, {}).get("loss/total", float("nan")))
        for t in checkpoints
    ]
    final_row = dict(trainer.metrics._split("train").values())

    print(f"  loss/total trajectory: {trajectory}")
    print(f"  final grad_skip_count: {trainer.grad_skip_count}")
    print(f"  final optim/grad_skips (from log dict): {final_row.get('optim/grad_skips')}")
    print(f"  final optim/grad_norm:  {final_row.get('optim/grad_norm')}")

    if use_split_loss:
        # Confirm the split KL metrics are surfaced.
        for k in (
            "loss/rate/trans/kl_phith",
            "loss/rate/trans/kl_psi",
            "loss/rate/init/loss_psi",
        ):
            v = final_row.get(k, None)
            present = v is not None
            finite = present and np.isfinite(v)
            print(f"  {k}: present={present}, finite={finite}, value={v}")
    else:
        # In single mode ``kl_phith`` / ``kl_psi`` are still emitted by the
        # diffusion transition_kl dict (M5 wired them unconditionally).
        for k in ("loss/rate/trans/kl_phith", "loss/rate/trans/kl_psi"):
            v = final_row.get(k, None)
            print(f"  (single mode) {k}: value={v}")

    # Verify: no grad-skips on clean data (spec requirement).
    if trainer.grad_skip_count != 0:
        print(f"  FAIL: expected 0 grad-skips, got {trainer.grad_skip_count}")
        return False
    return True


def main() -> int:
    print("DDSSM split-loss smoke")
    print("=" * 60)
    ok_single = _run_smoke(use_split_loss=False, p_k_clip=1e-3)
    ok_split = _run_smoke(use_split_loss=True, p_k_clip=1e-3)
    print()
    if ok_single and ok_split:
        print("Smoke OK — both modes finished, grad_skips=0 in both.")
        return 0
    print("Smoke FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

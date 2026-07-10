"""Smoke test for ``use_time_embedding=False`` (regular-timestep regime).

Builds the init-centering factory with the absolute-time path off and runs
a forward + backward + forecast to confirm every consumer's
``self.emb_time_dim > 0`` guard collapses the time-conditioning ops without
breaking shapes. Parameterised over ``DDSSM_TORCH_COMPILE`` so we explicitly
exercise both the eager and the Inductor graphs against ``emb_time_dim=0``.
"""

from __future__ import annotations

import torch
import pytest

from experiments.init_centering.model import _build_init_centering_model


def _make_batch(B: int, T: int, D: int, device: torch.device) -> dict:
    return {
        "observed_data": torch.randn(B, D, T, device=device),
        "observation_mask": torch.ones(B, D, T, device=device),
        "timepoints": torch
        .arange(T, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(B, -1)
        .contiguous(),
    }


@pytest.mark.parametrize("compile_flag", ["0", "1"])
def test_no_time_embedding_forward_and_forecast(
    monkeypatch: pytest.MonkeyPatch, compile_flag: str
) -> None:
    """Forward, backward, and forecast all run with ``emb_time_dim=0``."""
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", compile_flag)

    model = _build_init_centering_model(use_time_embedding=False, data_dim=1)
    assert model.emb_time_dim == 0
    assert model.transition.emb_time_dim == 0

    device = torch.device("cpu")
    model.to(device)

    batch = _make_batch(B=2, T=8, D=1, device=device)
    components, metrics, _ = model(**batch)
    loss = components.recon + components.init_kl + components.trans_kl
    assert torch.isfinite(loss)
    loss.backward()

    # ---- Forecast: autoregressive rollout. ----
    H, L2 = 6, 4
    with torch.no_grad():
        out = model.forecast(
            x_hist=torch.randn(2, 1, H, device=device),
            x_mask=torch.ones(2, 1, H, device=device),
            past_time=torch
            .arange(H, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(2, -1)
            .contiguous(),
            future_time=torch
            .arange(H, H + L2, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(2, -1)
            .contiguous(),
            num_samples=2,
        )
    assert out["pred_mean"].shape == (2, 1, L2)
    assert torch.isfinite(out["pred_mean"]).all()


def test_use_time_embedding_true_round_trip() -> None:
    """``use_time_embedding=True`` restores the original ``emb_time_dim``."""
    model = _build_init_centering_model(
        use_time_embedding=True, emb_time_dim=16, data_dim=1
    )
    assert model.emb_time_dim == 16
    assert model.transition.emb_time_dim == 16

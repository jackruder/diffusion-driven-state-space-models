from __future__ import annotations

import torch

from ddssm.data.synthetic import SyntheticDataset


def test_nonlinear_bimodal_lift_shapes_and_seed_determinism() -> None:
    d1 = SyntheticDataset(
        mode="nonlinear-bimodal-lift",
        split="train",
        N_per_split=8,
        T=16,
        D=4,
        dataset_seed=123,
    )
    d2 = SyntheticDataset(
        mode="nonlinear-bimodal-lift",
        split="train",
        N_per_split=8,
        T=16,
        D=4,
        dataset_seed=123,
    )
    assert d1.data.shape == (8, 4, 16)
    assert torch.allclose(d1.data, d2.data)

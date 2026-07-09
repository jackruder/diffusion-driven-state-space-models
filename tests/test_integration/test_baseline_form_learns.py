"""The remaining baseline forms (Zero / Persistence) are parameter-free.

Post-refactor there is no Linear / MLP baseline and no σ_p head. Both
surviving forms have literally zero trainable parameters.
"""

from __future__ import annotations

import pytest

from .conftest import make_vhp_model

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("baseline_form", ["zero", "persistence"])
def test_parameter_free_baseline_has_no_parameters(baseline_form: str) -> None:
    """Zero / Persistence expose zero trainable parameters."""
    model = make_vhp_model(baseline_form=baseline_form)
    total = sum(p.numel() for p in model.baseline.parameters())
    assert total == 0, (
        f"{baseline_form}: baseline should be parameter-free; got {total} params"
    )

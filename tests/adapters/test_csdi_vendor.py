"""Guards the re-vendored CSDI model copy under ``ddssm.adapters._csdi_vendor``.

The adapters package deliberately keeps its own byte-identical copy of the
upstream CSDI code (separate from ``ddssm.model.transitions._csdi_vendor``) so a
future unification refactor stays mechanical.
"""

from pathlib import Path

import ddssm


def test_import_smoke() -> None:
    """Re-vendored CSDI symbols import under the adapters namespace package."""
    from ddssm.adapters._csdi_vendor.main_model import CSDI_Forecasting
    from ddssm.adapters._csdi_vendor.diff_models import diff_CSDI

    assert CSDI_Forecasting is not None
    assert diff_CSDI is not None


def test_byte_identical_to_source() -> None:
    """The adapters copy stays byte-for-byte identical to the source vendor."""
    pkg_root = Path(ddssm.__file__).parent
    source_dir = pkg_root / "model" / "transitions" / "_csdi_vendor"
    vendor_dir = pkg_root / "adapters" / "_csdi_vendor"

    for name in ("main_model.py", "diff_models.py"):
        source_bytes = (source_dir / name).read_bytes()
        vendor_bytes = (vendor_dir / name).read_bytes()
        assert vendor_bytes == source_bytes, f"{name} differs from source vendor copy"

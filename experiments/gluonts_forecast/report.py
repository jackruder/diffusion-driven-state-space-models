"""Reporting for the gluonts_forecast benchmark — comparison table.

Renders a methods × datasets CRPS-sum table merging the DDSSM finalist numbers
with a baked dict of published baselines. The disk-scan of finalist
``metrics.json`` is wired after the campaign produces them; for now this module
owns the ``PUBLISHED`` reference numbers and the table formatter.
"""

from __future__ import annotations

from collections.abc import Mapping

# Published CRPS-sum (normalized) from the source papers. Fill in / verify
# against the exact tables when reporting (CSDI Table 1; TimeGrad; GP-Copula).
PUBLISHED: dict[str, dict[str, float]] = {
    "CSDI": {
        "solar": 0.338,
        "electricity": 0.041,
        "traffic": 0.073,
        "taxi": 0.355,
        "wiki": 0.207,
    },
    "TimeGrad": {
        "solar": 0.287,
        "electricity": 0.021,
        "traffic": 0.044,
        "taxi": 0.114,
        "wiki": 0.0485,
    },
    "GP-Copula": {
        "solar": 0.337,
        "electricity": 0.024,
        "traffic": 0.078,
        "taxi": 0.208,
        "wiki": 0.086,
    },
}

DATASET_ORDER = ["solar", "electricity", "traffic", "taxi", "wiki"]


def comparison_table(
    ddssm_crps_sum: Mapping[str, float],
    *,
    published: Mapping[str, Mapping[str, float]] = PUBLISHED,
) -> str:
    """Render a markdown methods × datasets CRPS-sum table.

    Args:
        ddssm_crps_sum: ``{dataset: crps_sum}`` for our best finalist per dataset.
        published: baseline method → ``{dataset: crps_sum}``.
    """
    header = "| Method | " + " | ".join(DATASET_ORDER) + " |"
    sep = "|" + "---|" * (len(DATASET_ORDER) + 1)
    rows = [header, sep]
    for method, by_ds in published.items():
        cells = [f"{by_ds.get(d, float('nan')):.4g}" for d in DATASET_ORDER]
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    ours = [f"{ddssm_crps_sum.get(d, float('nan')):.4g}" for d in DATASET_ORDER]
    rows.append("| **DDSSM (ours)** | " + " | ".join(ours) + " |")
    return "\n".join(rows)


__all__ = ["DATASET_ORDER", "PUBLISHED", "comparison_table"]

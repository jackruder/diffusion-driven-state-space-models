"""Retired legacy Optuna driver.

Use the Hydra-native sweeps documented in ``verifications.org`` instead, e.g.
``python -m ddssm.app --multirun experiment=bimodal transition=diffusion
hydra/sweeper=ddssm_optuna``.
"""

from __future__ import annotations


def main() -> None:
    """Exit with guidance for the supported sweep workflow."""
    raise SystemExit(
        "This legacy driver has been retired. Use Hydra multirun sweeps with "
        "`python -m ddssm.app --multirun ...`."
    )


if __name__ == "__main__":
    main()

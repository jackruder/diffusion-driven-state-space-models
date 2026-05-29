"""Deprecated — superseded by ``launch_study --mode tiny``.

Kept as a thin forwarding shim so existing commands keep working. New usage::

    python -m experiments.init_centering.launch_study --mode tiny ...
"""

from __future__ import annotations

import sys

from experiments.init_centering import launch_study


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    print(
        "[deprecated] launch_ablation_tiny -> 'launch_study --mode tiny'; forwarding.",
        file=sys.stderr,
    )
    return launch_study.main(["--mode", "tiny", *argv])


if __name__ == "__main__":
    sys.exit(main())

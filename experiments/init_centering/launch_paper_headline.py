"""Deprecated — superseded by ``launch_study --mode paper``.

Kept as a thin forwarding shim so existing commands keep working. New usage::

    python -m experiments.init_centering.launch_study --mode paper --top-cells ... ...
"""

from __future__ import annotations

import sys

from experiments.init_centering import launch_study


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    print(
        "[deprecated] launch_paper_headline -> 'launch_study --mode paper'; forwarding.",
        file=sys.stderr,
    )
    return launch_study.main(["--mode", "paper", *argv])


if __name__ == "__main__":
    sys.exit(main())

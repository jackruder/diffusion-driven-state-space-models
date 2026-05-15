"""``python -m experiments`` — dispatch to the experiments CLI."""

from __future__ import annotations

import sys

from experiments._cli import main


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Sphinx configuration for the DDSSM documentation.

Build with the ``[docs]`` extra installed::

    pip install -e .[docs]
    sphinx-build -b html docs docs/_build/html
"""

import sys
from pathlib import Path

# Make the package importable for autodoc (src/ layout) and let the top-level
# ``experiments`` package resolve too (it lives at the repo root, not under src/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

project = "DDSSM"
author = "Jack Ruder"
copyright = "2026, Jack Ruder"  # noqa: A001 - Sphinx-mandated config name
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Generate stub pages for everything referenced by autosummary directives.
# Member documentation is requested per-module in the recursive autosummary
# template (leaf modules only), so package __init__ pages don't re-document
# their re-exports — which would otherwise raise "duplicate object" warnings.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Heavy / optional third-party deps that need not be importable to build docs.
autodoc_mock_imports = ["mamba_ssm", "wandb"]

# Google-style docstrings (matches the repo's ruff pydocstyle convention).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render docstring ``Attributes:`` sections as :ivar: fields rather than separate
# attribute directives, so they don't collide with autodoc-documented dataclass
# fields ("duplicate object description").
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "adr", "README.md"]

html_theme = "furo"
html_title = "DDSSM"

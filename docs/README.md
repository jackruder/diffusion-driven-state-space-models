# Building the docs

The documentation is built with [Sphinx](https://www.sphinx-doc.org) using
`autodoc` + `autosummary` (API reference generated from the source docstrings)
and `myst-parser` (the Markdown guides).

```bash
# Install the docs toolchain (and the package, for autodoc imports)
pip install -e .[docs]          # or: uv sync --extra docs

# Build HTML
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html

# Strict build (treat warnings as errors — what you want in CI)
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Layout

- `index.md` — landing page + table of contents.
- `architecture.md` — implementation/architecture guide.
- `api.rst` — recursive `autosummary` root; stub pages are generated under
  `_autosummary/` at build time (git-ignored).
- `_templates/autosummary/module.rst` — recursive module template.
- `conf.py` — Sphinx config (adds `src/` and the repo root to `sys.path` for
  autodoc; mocks heavy optional imports).
- `adr/` — Architecture Decision Records (excluded from the Sphinx build).

Each `src/ddssm/<pkg>/README.md` documents one subsystem; those are the best
entry point when working inside a package.

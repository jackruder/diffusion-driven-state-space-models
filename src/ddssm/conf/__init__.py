"""Marker file so Hydra can resolve ``config_path="../conf"`` from
sibling-package entry points (``ddssm.variance``, ``ddssm.evaluate``,
``ddssm.visualize``) as the importable module ``ddssm.conf``.

The directory still holds plain YAML — no Python — but Hydra's path
resolution needs this empty package marker for entries that escape
out of one package and back into another.
"""

"""Model-family adapters: one class per family behind a uniform surface.

Each adapter integrates a model family (DDSSM, re-vendored CSDI, …) with the
single :class:`~ddssm.experiment.experiment.Experiment` orchestrator. This
package exposes only the abstract seam; concrete adapters land in later modules.
"""

from ddssm.adapters.base import ModelAdapter, MetricNotSupported

__all__ = ["MetricNotSupported", "ModelAdapter"]

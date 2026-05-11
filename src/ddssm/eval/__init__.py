"""Independent evaluation stage for trained DDSSM models.

Three layers, each replaceable:

* ``metrics``: stateless metric functions, each taking an
  :class:`EvalContext` and returning a JSON-serialisable result.
  Registered in ``METRIC_REGISTRY`` so an ``EvalSpec`` can list them
  by name. Add new metrics by registering them here.
* ``runner``: glue that loads a checkpoint, builds the data module's
  test/val loader, walks the metrics specified by an ``EvalSpec``,
  and writes a single ``metrics.json`` to the run dir.
* CLI (``ddssm.evaluate``): a Hydra entry point that resolves an
  experiment + checkpoint and invokes the runner.

Train, evaluate, and visualize are independent stages; nothing in
this module is called from training.
"""

from .runner import EvalSpec, evaluate
from .metrics import METRIC_REGISTRY, EvalContext, register_metric

__all__ = [
    "METRIC_REGISTRY",
    "EvalContext",
    "EvalSpec",
    "evaluate",
    "register_metric",
]

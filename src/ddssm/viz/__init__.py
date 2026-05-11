"""Independent visualization stage for trained DDSSM models.

Same three-layer shape as :mod:`ddssm.eval`:

* ``plots``: stateless plot functions, each taking a :class:`PlotContext`
  and a ``save_path``. Registered in ``PLOT_REGISTRY`` so a ``VizSpec``
  can list them by name. Add new plot kinds by registering them here.
* ``runner``: glue that loads a checkpoint, builds the data loader,
  walks the plots specified by a :class:`VizSpec`, and saves PNGs to
  the run dir.
* CLI (``ddssm.visualize``): a Hydra entry point that resolves an
  experiment + checkpoint and invokes the runner.

Train, evaluate, and visualize are independent stages; nothing here
runs during training.
"""

from .plots import PLOT_REGISTRY, PlotContext, register_plot
from .runner import VizSpec, PlotSpec, visualize

__all__ = [
    "PLOT_REGISTRY",
    "PlotContext",
    "PlotSpec",
    "VizSpec",
    "register_plot",
    "visualize",
]

"""Training scalars for variance-probe experiments (300-step runs).

``checkpoint_every`` is set tight (25 steps → 12 checkpoints) so the
``ddssm.variance ... +per_step=true`` sweep can probe the entire
training trajectory and animate how the variance landscape evolves.
"""

from __future__ import annotations

from ddssm.builders import Training

from conf.registry import training_store


Probe300 = Training(steps=300, log_every=20, checkpoint_every=25, amp=False)
training_store(Probe300, name="probe_300")

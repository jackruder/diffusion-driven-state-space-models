"""Training scalars for variance-probe experiments (300-step runs)."""

from __future__ import annotations

from ddssm.builders import Training

from conf.registry import training_store


Probe300 = Training(steps=300, log_every=20, checkpoint_every=100, amp=False)
training_store(Probe300, name="probe_300")

"""KDD training scalars (5k Gaussian / 8k diffusion, AMP on)."""

from __future__ import annotations

from ddssm.builders import Training

from conf.registry import training_store


Gauss5k = Training(steps=5000, log_every=50, checkpoint_every=500, amp=True)
Diff8k = Training(steps=8000, log_every=50, checkpoint_every=500, amp=True)

training_store(Gauss5k, name="kdd_gauss_5k")
training_store(Diff8k, name="kdd_diff_8k")

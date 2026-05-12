"""Training-scalar variants for the synthetic-data experiments."""

from __future__ import annotations

from ddssm.builders import Training

from conf.registry import training_store


# 500-step smoke (LGSSM).
Smoke500 = Training(steps=500, log_every=25, amp=False)

# 1000-step Gaussian harmonic / bimodal default.
Gauss1k = Training(steps=1000, log_every=25, checkpoint_every=200, amp=False)

# 1000-step diffusion smoke (LGSSM diffusion).
Diff1k = Training(steps=1000, log_every=25, amp=False)

# 2000-step diffusion harmonic / bimodal / robot2d_gauss default.
Diff2k = Training(steps=2000, log_every=25, checkpoint_every=500, amp=False)

# 2000-step robot 2D Gaussian (log_every=50).
RobotGauss = Training(steps=2000, log_every=50, checkpoint_every=500, amp=False)

# 4000-step robot 2D diffusion.
RobotDiff = Training(steps=4000, log_every=50, checkpoint_every=500, amp=False)

training_store(Smoke500, name="smoke_500")
training_store(Gauss1k, name="gauss_1k")
training_store(Diff1k, name="diff_1k")
training_store(Diff2k, name="diff_2k")
training_store(RobotGauss, name="robot_gauss_2k")
training_store(RobotDiff, name="robot_diff_4k")

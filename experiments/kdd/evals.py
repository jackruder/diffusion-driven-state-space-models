"""KDD eval spec (test split, mae + crps_sum)."""

from __future__ import annotations

from ddssm.builders import Eval

from conf.registry import eval_store


KDD = Eval(metrics=["mae", "crps_sum"], split="test", num_samples=32)
eval_store(KDD, name="kdd")

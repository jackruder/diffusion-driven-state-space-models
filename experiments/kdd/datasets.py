"""KDD Cup 2018 PM2.5 data module."""

from __future__ import annotations

from ddssm.builders import KDD

from conf.registry import data_store


KDDData = KDD(batch_size=128, eval_step_size=24)
data_store(KDDData, name="kdd")

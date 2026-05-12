"""Noise schedules for the synthetic-data diffusion transitions."""

from __future__ import annotations

from ddssm.builders import Schedule

from conf.registry import schedule_store


Default = Schedule()
schedule_store(Default, name="default")

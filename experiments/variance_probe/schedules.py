"""Noise schedules for variance-probe experiments (DiffusionV2)."""

from __future__ import annotations

from ddssm.builders import ScheduleV2

from conf.registry import schedule_store


V2 = ScheduleV2()
schedule_store(V2, name="v2")

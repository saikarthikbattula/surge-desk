"""Deterministic event-time replay with injectable wall clock (Agent 1)."""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Iterator

from . import config
from .contracts import Report
from .textutil import parse_iso


def busiest_window_start(reports: list[Report], hours: float = 1.0) -> datetime:
    if not reports or not math.isfinite(hours) or hours <= 0:
        raise ValueError("A nonempty report stream and positive finite hours are required")
    times = [parse_iso(r.timestamp) for r in reports]
    width = timedelta(hours=hours)
    end = best_count = best_start = 0
    for start, timestamp in enumerate(times):
        while end < len(times) and times[end] < timestamp + width:
            end += 1
        if end - start > best_count:
            best_count, best_start = end - start, start
    return times[best_start]


def iter_batches(reports: list[Report], speed: float | str, start: datetime | None = None,
                 duration_s: float | None = None, batch_seconds: float = config.BATCH_SECONDS,
                 clock=time.monotonic, sleep=time.sleep) -> Iterator[tuple[float, list[Report]]]:
    max_mode = speed == "max"
    numeric_speed = 20.0 if max_mode else float(speed)
    if not math.isfinite(numeric_speed) or numeric_speed <= 0:
        raise ValueError("speed must be positive and finite, or 'max'")
    if not math.isfinite(batch_seconds) or batch_seconds <= 0:
        raise ValueError("batch_seconds must be positive and finite")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s < 0):
        raise ValueError("duration_s must be nonnegative and finite")
    if not reports or duration_s == 0:
        return
    start = start if start is not None else parse_iso(reports[0].timestamp)
    origin = clock()
    factor = config.REPLAY_COMPRESSION * numeric_speed
    slot = None
    batch = []

    def deliver(index, members):
        offset = (index + 1) * batch_seconds
        if duration_s is not None:
            offset = min(offset, duration_s)
        if max_mode:
            return clock(), members
        scheduled = origin + offset
        remaining = scheduled - clock()
        if remaining > 0:
            sleep(remaining)
        return scheduled, members

    for report in reports:
        offset = (parse_iso(report.timestamp) - start).total_seconds() / factor
        if offset < 0:
            continue
        if duration_s is not None and offset >= duration_s:
            break
        index = math.floor(offset / batch_seconds)
        if batch and index != slot:
            yield deliver(slot, batch)
            batch = []
            if not max_mode and duration_s is not None and clock() >= origin + duration_s:
                return
        slot = index
        batch.append(report)
    if batch and (max_mode or duration_s is None or clock() < origin + duration_s):
        yield deliver(slot, batch)

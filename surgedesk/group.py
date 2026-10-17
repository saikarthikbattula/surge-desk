"""Online incident grouping with the fixture reference's location and tie rules."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, fileio
from .contracts import Report, Tagged
from .stubs import reference_incident_location
from .textutil import parse_iso

SVILookup = Callable[[str | None], float | None]


@dataclass
class IncidentState:
    incident_id: str
    event: str
    reports: list[Report]
    tags: list[Tagged]
    sum_vec: np.ndarray
    last_seen: datetime
    _times: list[datetime] = field(init=False, repr=False)

    def __post_init__(self):
        self.sum_vec = np.asarray(self.sum_vec, dtype=np.float64)
        self._times = [parse_iso(r.timestamp) for r in self.reports]

    @property
    def first_seen(self) -> datetime:
        return self._times[0]

    @property
    def centroid(self) -> np.ndarray:
        norm = np.linalg.norm(self.sum_vec)
        return (self.sum_vec / norm if norm else self.sum_vec).astype(np.float32)

    def members_until(self, now: datetime) -> tuple[list[Report], list[Tagged]]:
        end = bisect_right(self._times, now)
        return self.reports[:end], self.tags[:end]

    def location(self, now: datetime | None = None) -> tuple[str | None, str | None]:
        tags = self.tags if now is None else self.members_until(now)[1]
        return reference_incident_location(tags)


class GroupState:
    def __init__(self, event: str):
        self.event = event
        self.incidents: dict[str, IncidentState] = {}
        self.assignments: dict[str, str] = {}

    def add(self, report: Report, tag: Tagged, emb: np.ndarray) -> str | None:
        """Assign an actionable report using time, location, and cosine similarity."""
        assert report.event == self.event, "Report belongs to another event"
        assert report.report_id == tag.report_id, "Report/tag IDs differ"
        if not tag.is_actionable:
            return None
        assert report.report_id not in self.assignments, "Report already assigned"
        emb = np.asarray(emb, dtype=np.float32)
        assert emb.shape == (config.EMBED_DIM,) and np.isfinite(emb).all()
        now = parse_iso(report.timestamp)
        window = timedelta(hours=config.WINDOW_HOURS)
        candidates = sorted(
            (inc for inc in self.incidents.values() if now - inc.last_seen <= window),
            key=lambda inc: inc.last_seen, reverse=True,
        )[:config.MAX_CANDIDATES]
        eligible, thresholds, centroids = [], [], []
        for inc in candidates:
            place, county = inc.location()
            match = bool(
                (tag.county_fips and county and tag.county_fips == county)
                or (tag.places and place and tag.places[0].lower() == place.lower())
            )
            if (tag.places or tag.county_fips) and (place or county) and not match:
                continue
            eligible.append(inc)
            thresholds.append(config.SIM_PLACE if match else config.SIM_NOPLACE)
            norm = np.linalg.norm(inc.sum_vec)
            centroids.append(inc.sum_vec / norm if norm else inc.sum_vec)
        best = None
        best_sim = -np.inf
        if eligible:
            similarities = np.stack(centroids) @ emb
            for inc, sim, threshold in zip(eligible, similarities, thresholds):
                if sim >= threshold and sim > best_sim:
                    best, best_sim = inc, sim
        if best is None:
            iid = f"{self.event}-inc-{len(self.incidents) + 1}"
            best = IncidentState(iid, self.event, [], [],
                                 np.zeros(config.EMBED_DIM, dtype=np.float64), now)
            self.incidents[iid] = best
        best.reports.append(report)
        best.tags.append(tag)
        best._times.append(now)
        best.sum_vec += emb
        best.last_seen = now
        self.assignments[report.report_id] = best.incident_id
        return best.incident_id

    def open_incidents(self, now: datetime) -> list[IncidentState]:
        """Return incidents with a member at or before now within the time window."""
        out = []
        for inc in self.incidents.values():
            end = bisect_right(inc._times, now)
            if end and now - inc._times[end - 1] <= timedelta(hours=config.WINDOW_HOURS):
                out.append(inc)
        return out


def load_stage1(event: str, src_dir: Path | None = None) -> tuple[list[Report], list[Tagged], np.ndarray]:
    src = Path(src_dir) if src_dir is not None else config.event_dir(event)
    reports = fileio.read_jsonl(src / config.REPORTS_FILE, Report)
    tags = fileio.read_jsonl(src / config.TAGGED_FILE, Tagged)
    emb = fileio.load_embeddings(src / config.EMBEDDINGS_FILE)
    assert len(reports) == len(tags) == len(emb), "Stage 1 row counts differ"
    assert emb.shape == (len(reports), config.EMBED_DIM), "Embedding shape differs"
    assert all(r.report_id == t.report_id and t.emb_row == i
               for i, (r, t) in enumerate(zip(reports, tags))), "Stage 1 row order differs"
    assert all(r.event == event for r in reports), "Stage 1 event differs"
    assert len({r.report_id for r in reports}) == len(reports), "Duplicate report IDs"
    assert [(parse_iso(r.timestamp), r.report_id) for r in reports] == sorted(
        (parse_iso(r.timestamp), r.report_id) for r in reports), "Reports must be time ordered"
    return reports, tags, emb


def group_event(event: str, src_dir: Path | None = None, out_dir: Path | None = None) -> GroupState:
    reports, tags, emb = load_stage1(event, src_dir)
    state = GroupState(event)
    for r, t, vec in zip(reports, tags, emb):
        state.add(r, t, vec)
    out = Path(out_dir) if out_dir is not None else config.event_dir(event)
    sizes = [len(inc.reports) for inc in state.incidents.values()]
    stats = dict(n_reports=len(reports), n_actionable=sum(t.is_actionable for t in tags),
                 n_incidents=len(sizes), singleton_share=sizes.count(1) / len(sizes) if sizes else 0.0,
                 mean_size=sum(sizes) / len(sizes) if sizes else 0.0,
                 max_size=max(sizes, default=0), size_histogram=dict(sorted(Counter(sizes).items())))
    fileio.write_jsonl(out / "groups.jsonl", [
        dict(incident_id=i.incident_id, report_ids=[r.report_id for r in i.reports])
        for i in state.incidents.values()
    ])
    fileio.write_json(out / "group_stats.json", stats)
    print(stats)
    return state

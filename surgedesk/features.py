"""Snapshot rows built exclusively from the shared feature definitions."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .contracts import FEATURE_KEYS, Report
from .group import GroupState, IncidentState, SVILookup
from .stubs import reference_features, reference_incident_location, reference_urgent_proxy, snapshot_times
from .textutil import parse_iso, to_iso

assert "gold" not in " ".join(FEATURE_KEYS)

SNAPSHOT_COLUMNS = ["incident_id", "snapshot_time", "place", "county_fips", "svi",
                    "first_seen", "last_seen", "report_ids",
                    *[k for k in FEATURE_KEYS if k != "svi"], "urgent_proxy"]
# Person 4 resolved the duplicate SVI column: keep one numeric feature column,
# using -1 for unknown and restoring None in Incident metadata.


def snapshot_features(inc: IncidentState, now: datetime, svi_lookup: SVILookup) -> dict[str, float] | None:
    reports, tags = inc.members_until(now)
    if not reports:
        return None
    _, county = reference_incident_location(tags)
    return reference_features([replace(r, gold_label=None) for r in reports], tags, now, svi_lookup(county))


def _snapshot_row(inc: IncidentState, now: datetime, svi_lookup: SVILookup) -> list:
    reports, tags = inc.members_until(now)
    place, county = reference_incident_location(tags)
    svi = svi_lookup(county)
    feats = reference_features([replace(r, gold_label=None) for r in reports], tags, now, svi)
    return [inc.incident_id, to_iso(now), place, county, feats["svi"], reports[0].timestamp,
            reports[-1].timestamp, json.dumps([r.report_id for r in reports]),
            *[feats[k] for k in FEATURE_KEYS if k != "svi"], reference_urgent_proxy(reports)]


def build_snapshots(event: str, state: GroupState, reports: list[Report], svi_lookup: SVILookup,
                    out_dir: Path | None = None) -> pd.DataFrame:
    assert state.event == event
    rows = []
    if reports:
        for now in snapshot_times(parse_iso(reports[0].timestamp), parse_iso(reports[-1].timestamp)):
            rows.extend(_snapshot_row(inc, now, svi_lookup) for inc in state.open_incidents(now))
    frame = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    out = Path(out_dir) if out_dir is not None else config.event_dir(event)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / config.SNAPSHOTS_FILE, index=False)
    return frame

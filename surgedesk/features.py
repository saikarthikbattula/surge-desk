"""Snapshot rows built exclusively from the shared feature definitions."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config
from .contracts import FEATURE_KEYS, RANK_FEATURE_KEYS, Report
from .group import GroupState, IncidentState, SVILookup
from .stubs import reference_features, reference_incident_location, reference_urgent_proxy, snapshot_times
from .textutil import mask_privacy, parse_iso, to_iso

assert "gold" not in " ".join(FEATURE_KEYS)


def feature_keys(event: str) -> list[str]:
    return RANK_FEATURE_KEYS if config.ranking_objective(event) == "assistance_v1" else FEATURE_KEYS


def assistance_signals(report: Report) -> dict:
    """Conservative text cues, never gold labels or Granite output.

    A cue indicates a reason for review, not a verified current emergency.
    Check clauses independently so old context does not suppress a new request.
    """
    current, historical, casualty, evidence = _signal_values(mask_privacy(report.text), parse_iso(report.timestamp).year)
    return {"current_assistance": current, "historical_or_speculative": historical,
            "casualty_only": casualty, "evidence": list(evidence)}


@lru_cache(maxsize=config.ASSISTANCE_CACHE_SIZE)
def _signal_values(text: str, year: int):
    """Bounded cache of masked-text cues, reused across snapshot rows."""
    current, historical, casualty = False, False, False
    evidence = []
    for clause in re.split(r"[.!?;\n]+|\bbut\b", text, flags=re.I):
        low = clause.lower().replace("’", "'")
        years = re.findall(r"\b(?:in|of|during|from)\s+((?:18|19|20)\d{2})\b|"
                           r"\b((?:18|19|20)\d{2})(?:'s)?\s+(?:death|toll|storm|hurricane|flood|earthquake)\b", low)
        old = bool(re.search(config.ASSISTANCE_HISTORY, low) or
                   re.search(config.ASSISTANCE_SPECULATION, low) or
                   any(int(y) < year for pair in years for y in pair if y))
        historical |= old
        casualty |= bool(re.search(config.CASUALTY_PATTERN, low))
        if old or re.search(config.ASSISTANCE_NEGATION, low):
            continue
        if any(re.search(pattern, low) for pattern in config.ASSISTANCE_PATTERNS):
            current = True
            evidence.append(clause.strip())
    return current, historical, casualty and not current, tuple(evidence)


def _features(reports, tags, now, svi):
    values = reference_features([replace(r, gold_label=None) for r in reports], tags, now, svi)
    if config.ranking_objective(reports[0].event) == "assistance_v1":
        signals = [assistance_signals(r) for r in reports]
        for name in ("current_assistance", "historical_or_speculative", "casualty_only"):
            active = [s[name] and (name != "current_assistance" or
                       now - parse_iso(r.timestamp) <= timedelta(hours=config.WINDOW_HOURS))
                      for r, s in zip(reports, signals)]
            values[f"share_{name}"] = round(sum(active) / len(signals), 6)
    return values

# Keep one numeric SVI column, using -1 for unknown and restoring None in metadata.
SNAPSHOT_COLUMNS = ["incident_id", "snapshot_time", "place", "county_fips", "svi",
                    "first_seen", "last_seen", "report_ids",
                    *[k for k in FEATURE_KEYS if k != "svi"], "urgent_proxy"]


def snapshot_columns(event):
    return ["incident_id", "snapshot_time", "place", "county_fips", "svi",
            "first_seen", "last_seen", "report_ids",
            *[k for k in feature_keys(event) if k != "svi"], "urgent_proxy"]


def snapshot_features(inc: IncidentState, now: datetime, svi_lookup: SVILookup) -> dict[str, float] | None:
    reports, tags = inc.members_until(now)
    if not reports:
        return None
    _, county = reference_incident_location(tags)
    return _features(reports, tags, now, svi_lookup(county))


def _snapshot_row(inc: IncidentState, now: datetime, svi_lookup: SVILookup) -> list:
    reports, tags = inc.members_until(now)
    place, county = reference_incident_location(tags)
    svi = svi_lookup(county)
    feats = _features(reports, tags, now, svi)
    return [inc.incident_id, to_iso(now), place, county, feats["svi"], reports[0].timestamp,
            reports[-1].timestamp, json.dumps([r.report_id for r in reports]),
            *[feats[k] for k in feature_keys(inc.event) if k != "svi"], reference_urgent_proxy(reports)]


def build_snapshots(event: str, state: GroupState, reports: list[Report], svi_lookup: SVILookup,
                    out_dir: Path | None = None) -> pd.DataFrame:
    assert state.event == event
    rows = []
    if reports:
        for now in snapshot_times(parse_iso(reports[0].timestamp), parse_iso(reports[-1].timestamp)):
            rows.extend(_snapshot_row(inc, now, svi_lookup) for inc in state.open_incidents(now))
    frame = pd.DataFrame(rows, columns=snapshot_columns(event))
    out = Path(out_dir) if out_dir is not None else config.event_dir(event)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / config.SNAPSHOTS_FILE, index=False)
    return frame

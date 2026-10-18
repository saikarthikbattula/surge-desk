"""Snapshot rows built exclusively from the shared feature definitions."""
from __future__ import annotations

import json
import hashlib
import html
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


def feature_definition(event: str) -> str | None:
    """Bind assistance models to executable features and shared cue policy.

    A feature order alone cannot detect changed semantics. Legacy archives and
    frozen fixtures deliberately retain their original loading behavior.
    """
    if config.ranking_objective(event) != "assistance_v1":
        return None
    policy = {name: getattr(config, name) for name in (
        "ASSISTANCE_PATTERNS", "ASSISTANCE_NEGATION", "ASSISTANCE_SPECULATION",
        "ASSISTANCE_HISTORY", "CASUALTY_PATTERN", "WINDOW_HOURS")}
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(json.dumps(policy, sort_keys=True).encode())
    return digest.hexdigest()


def assistance_signals(report: Report) -> dict:
    """Conservative text cues, never gold labels or Granite output.

    A cue indicates a reason for review, not a verified current emergency.
    Check clauses independently so old context does not suppress a new request.
    """
    current, historical, casualty, evidence = _signal_values(
        mask_privacy(report.text), parse_iso(report.timestamp).year,
        tuple(config.ASSISTANCE_PATTERNS), config.ASSISTANCE_NEGATION,
        config.ASSISTANCE_HISTORY, config.ASSISTANCE_SPECULATION, config.CASUALTY_PATTERN)
    return {"current_assistance": current, "historical_or_speculative": historical,
            "casualty_only": casualty, "evidence": list(evidence)}


def _clauses(text):
    """Analyze decoded clauses, but quote exact spans of the masked source.

    Entity semicolons are not sentence boundaries. Retain an offset map so
    evidence remains a source substring even when decoding changes its length.
    Split coordinated predicates, not resource lists such as food and water.
    """
    decoded, starts, ends = [], [], []
    cursor = 0
    for entity in re.finditer(r"&(?:#[xX][0-9a-fA-F]+|#\d+|[a-zA-Z][a-zA-Z0-9]+);?", text):
        for index in range(cursor, entity.start()):
            decoded.append(text[index])
            starts.append(index)
            ends.append(index + 1)
        value = html.unescape(entity.group())
        decoded.extend(value)
        starts.extend([entity.start()] * len(value))
        ends.extend([entity.end()] * len(value))
        cursor = entity.end()
    for index in range(cursor, len(text)):
        decoded.append(text[index])
        starts.append(index)
        ends.append(index + 1)
    decoded = "".join(decoded)
    boundary = (r"[.!?;\n]+|\bbut\b|"
                r"\b(?:and|while|yet)\s+(?=(?:i|we|they|he|she|my|our|the|a|an|"
                r"need|needs|require|requires|request|please|still|not|don['’]t|doesn['’]t|do|does)\b)|"
                r",\s*(?=(?:i|we|they|he|she|my|our)\b)")
    cursor = 0
    for match in re.finditer(boundary, decoded, re.I):
        if cursor < match.start():
            yield decoded[cursor:match.start()], text[starts[cursor]:ends[match.start() - 1]].strip()
        cursor = match.end()
    if cursor < len(decoded):
        yield decoded[cursor:], text[starts[cursor]:ends[-1]].strip()


@lru_cache(maxsize=config.ASSISTANCE_CACHE_SIZE)
def _signal_values(text: str, year: int, patterns, negation, history, speculation, casualty_pattern):
    """Bounded cache of masked-text cues, reused across snapshot rows."""
    current, historical, casualty = False, False, False
    evidence = []
    for clause, source in _clauses(text):
        low = clause.lower().replace("’", "'")
        years = re.findall(r"\b(?:in|of|during|from)\s+((?:18|19|20)\d{2})\b|"
                           r"\b((?:18|19|20)\d{2})(?:'s)?\s+(?:death|toll|storm|hurricane|flood|earthquake)\b", low)
        old = bool(re.search(history, low) or
                   re.search(speculation, low) or
                   any(int(y) < year for pair in years for y in pair if y))
        historical |= old
        casualty |= bool(re.search(casualty_pattern, low))
        if old or re.search(negation, low):
            continue
        matches = [m for pattern in patterns for m in re.finditer(pattern, low)]
        # A context marker scopes forward within its clause. A request before
        # trailing conditional advice remains observed ("we need water if you
        # can deliver"). Existing unnamed shared patterns remain compatible.
        contexts = [m for m in matches if m.lastgroup == "context"]
        if any(m.lastgroup != "context" and not any(c.start() <= m.start() for c in contexts)
               for m in matches):
            current = True
            evidence.append(source)
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

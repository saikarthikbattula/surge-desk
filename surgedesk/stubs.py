"""Reference and stub implementations.

Two jobs:
1. `reference_features` and `reference_incident_location` are the EXACT definitions
   of the ranker features and incident place. Agent 2's features.py must return
   identical values (tests/test_agent2_*.py compares against fixtures built from these).
2. `stub_*` functions are deliberately dumb versions of every stage with the real
   interfaces. `python -m surgedesk.pipeline --stub` and the self-test use them so the
   whole flow (and the UI) works before the real stages exist. Stub output goes to
   outputs/_stub/ and is never used for results.

Owner: foundation (frozen).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from . import config
from .contracts import (
    FEATURE_KEYS,
    SHARE_CATEGORIES,
    URGENT_LABELS,
    VULN_TERMS,
    Briefing,
    Incident,
    Report,
    Tagged,
    empty_fields,
    is_actionable,
    zero_timing,
)
from .textutil import dedupe_key, minutes_between, parse_iso, to_iso

# =========================================================================== reference definitions


def find_terms(text: str, terms: list[str]) -> list[str]:
    """Whole-word/phrase, case-insensitive matches of `terms`, deduplicated, in order of first appearance."""
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for term in terms:
        m = re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low)
        if m:
            hits.append((m.start(), term))
    hits.sort()
    seen, out = set(), []
    for _, t in hits:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def reference_incident_location(tags: list[Tagged]) -> tuple[str | None, str | None]:
    """Incident place = most common non-null places[0] among members; county = most common
    non-null county_fips. Ties go to the value seen earliest (members are in time order)."""

    def most_common(values: list[str | None]) -> str | None:
        vals = [v for v in values if v]
        if not vals:
            return None
        counts = Counter(vals)
        best = max(counts.values())
        for v in vals:  # earliest among the tied
            if counts[v] == best:
                return v
        return None

    place = most_common([t.places[0] if t.places else None for t in tags])
    county = most_common([t.county_fips for t in tags])
    return place, county


def reference_features(
    reports: list[Report], tags: list[Tagged], now: datetime, svi: float | None
) -> dict[str, float]:
    """Ranker features for one incident as of `now`.

    `reports` and `tags` are the incident's members (same order, time-ordered) with
    timestamp <= now. Values are rounded to 6 decimals. Keys/order = FEATURE_KEYS.
    """
    assert reports and len(reports) == len(tags), "need >= 1 member and matching tags"
    ts = [parse_iso(r.timestamp) for r in reports]
    assert all(t <= now for t in ts), "members must have timestamp <= now"
    n = len(reports)
    last60 = sum(1 for t in ts if now - timedelta(minutes=60) < t <= now)
    prev60 = sum(1 for t in ts if now - timedelta(minutes=120) < t <= now - timedelta(minutes=60))
    place, county = reference_incident_location(tags)
    cats = [t.pred_category for t in tags]
    f = {
        "n_reports": n,
        "n_unique_texts": len({dedupe_key(r.text) for r in reports}),
        "reports_last_60m": last60,
        "growth_ratio": last60 / max(1, prev60),
        "age_min": minutes_between(min(ts), now),
        "minutes_since_last": minutes_between(max(ts), now),
        "has_place": 1.0 if (place or county) else 0.0,
        "svi": svi if svi is not None else -1.0,
        "vuln_mentions": sum(len(t.vuln_terms) for t in tags),
        "vuln_share": sum(1 for t in tags if t.vuln_terms) / n,
        "mean_pred_conf": sum(t.pred_conf for t in tags) / n,
    }
    for c in SHARE_CATEGORIES:
        f[f"share_{c}"] = cats.count(c) / n
    out = {k: round(float(f[k]), 6) for k in FEATURE_KEYS}
    return out


def reference_urgent_proxy(reports: list[Report]) -> int | None:
    """1 if any member's gold label is in URGENT_LABELS, 0 otherwise, None if no gold labels at all."""
    golds = [r.gold_label for r in reports]
    if all(g is None for g in golds):
        return None
    return int(any(g in URGENT_LABELS for g in golds))


def snapshot_times(start: datetime, end: datetime, every_min: int = config.SNAPSHOT_MIN) -> list[datetime]:
    """Snapshot grid: first tick is `start` floored to the interval, then every `every_min`
    minutes, last tick is the first one >= `end`."""
    step = timedelta(minutes=every_min)
    floor = start.replace(second=0, microsecond=0)
    floor -= timedelta(minutes=floor.minute % every_min)
    out = [floor]
    while out[-1] < end:
        out.append(out[-1] + step)
    return out


# =========================================================================== stubs (dumb on purpose)
_STUB_RULES = [
    ("requests_or_urgent_needs", r"\b(help|need|trapped|stranded|rescue|boat)\b"),
    ("injured_or_dead_people", r"\b(injured|dead|killed|hurt)\b"),
    ("infrastructure_and_utility_damage", r"\b(power|damage|roof|collapsed|water supply|pumps)\b"),
    ("displaced_people_and_evacuations", r"\b(evacuat\w*|shelter)\b"),
    ("caution_and_advice", r"\b(avoid|warning|turn around)\b"),
    ("sympathy_and_support", r"\b(pray\w*|thoughts|stay strong)\b"),
]


def _stub_vector(text: str) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(config.EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def stub_tag_batch(reports: list[Report], start_row: int = 0) -> tuple[list[Tagged], np.ndarray]:
    """Keyword rules instead of a classifier; no places. Same signature shape as Tagger.tag_batch."""
    tagged, vecs = [], []
    for i, r in enumerate(reports):
        cat = "other_relevant_information"
        for c, pat in _STUB_RULES:
            if re.search(pat, r.text, re.IGNORECASE):
                cat = c
                break
        tagged.append(
            Tagged(
                report_id=r.report_id, emb_row=start_row + i, pred_category=cat, pred_conf=0.5,
                places=[], county_fips=None, vuln_terms=find_terms(r.text, VULN_TERMS),
                is_actionable=is_actionable(cat),
            )
        )
        vecs.append(_stub_vector(r.text))
    emb = np.vstack(vecs) if vecs else np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    return tagged, emb


class StubGroupState:
    """Groups actionable reports by predicted category only. Same interface as group.GroupState."""

    def __init__(self, event: str):
        self.event = event
        self.incidents: dict[str, dict] = {}
        self._by_cat: dict[str, str] = {}

    def add(self, report: Report, tag: Tagged, emb: np.ndarray) -> str | None:
        if not tag.is_actionable:
            return None
        iid = self._by_cat.get(tag.pred_category)
        if iid is None:
            iid = f"{self.event}-inc-{len(self.incidents) + 1}"
            self._by_cat[tag.pred_category] = iid
            self.incidents[iid] = {"reports": [], "tags": []}
        self.incidents[iid]["reports"].append(report)
        self.incidents[iid]["tags"].append(tag)
        return iid


def stub_rank_now(state: StubGroupState, now: datetime, lam: float = 0.0) -> list[Incident]:
    """FIFO: earliest incident ranks first. Scores are fake (1/rank)."""
    rows = []
    for iid, inc in state.incidents.items():
        pairs = [(r, t) for r, t in zip(inc["reports"], inc["tags"]) if parse_iso(r.timestamp) <= now]
        if not pairs:
            continue
        reps, tags = [p[0] for p in pairs], [p[1] for p in pairs]
        place, county = reference_incident_location(tags)
        rows.append((parse_iso(reps[0].timestamp), iid, reps, tags, place, county))
    rows.sort(key=lambda x: (x[0], x[1]))
    out = []
    for rank, (_, iid, reps, tags, place, county) in enumerate(rows, 1):
        score = round(1.0 / rank, 6)
        out.append(
            Incident(
                incident_id=iid, event=state.event, snapshot_time=to_iso(now),
                report_ids=[r.report_id for r in reps], first_seen=reps[0].timestamp,
                last_seen=reps[-1].timestamp, place=place, county_fips=county, svi=None,
                features=reference_features(reps, tags, now, None), score=score,
                score_adj=score, lam=lam, rank=rank, rank_adj=rank, shap_top=[],
                urgent_proxy=reference_urgent_proxy(reps),
            )
        )
    return out


def template_briefing_text(incident: Incident, fields: dict) -> str:
    """The official template wording (Agent 3's template_briefing must produce exactly this)."""
    n = incident.features["n_reports"]
    where = fields["location"]["value"] or incident.place or "an unknown location"
    need = fields["need"]["value"] or "not stated"
    vuln = fields["vulnerable"]["value"] or "none mentioned"
    reason = "no explanation available"
    if incident.shap_top:
        reason = ", ".join(
            f"{_plain(k)}" for k, v in incident.shap_top[:2] if v > 0
        ) or reason
    return (
        f"{int(n)} report{'s' if n != 1 else ''} near {where}. Need: {need}. "
        f"Vulnerable people: {vuln}. Ranked high because of: {reason}."
    )


def _plain(key: str) -> str:
    from .contracts import FEATURE_PLAIN

    return FEATURE_PLAIN.get(key, key)


def stub_brief(incident: Incident) -> Briefing:
    fields = empty_fields()
    return Briefing(
        incident_id=incident.incident_id, event=incident.event,
        snapshot_time=incident.snapshot_time,
        cache_key=hashlib.sha1(("stub|" + incident.incident_id).encode()).hexdigest(),
        rep_report_ids=incident.report_ids[:1], fields=fields, sentences=[],
        briefing=template_briefing_text(incident, fields), used_fallback=True,
        model_file="stub", timing=zero_timing(),
    )


def run_stub_pipeline(event: str, reports: list[Report]) -> dict:
    """Run every stage with stubs on `reports`; write all handoff files to outputs/_stub/<event>/."""
    from .fileio import save_embeddings, write_jsonl

    reports = sorted(reports, key=lambda r: (r.timestamp, r.report_id))
    tagged, emb = stub_tag_batch(reports)
    state = StubGroupState(event)
    for r, t in zip(reports, tagged):
        state.add(r, t, emb[t.emb_row])
    now = parse_iso(reports[-1].timestamp)
    incidents = stub_rank_now(state, now)
    briefings = [stub_brief(i) for i in incidents[: config.TOP_N_BRIEF]]
    d = config.event_dir(event, stub=True)
    write_jsonl(d / config.REPORTS_FILE, reports)
    write_jsonl(d / config.TAGGED_FILE, tagged)
    save_embeddings(d / config.EMBEDDINGS_FILE, emb)
    write_jsonl(d / config.INCIDENTS_FILE, incidents)
    write_jsonl(d / config.BRIEFINGS_FILE, briefings)
    return {"reports": len(reports), "incidents": len(incidents), "briefings": len(briefings), "dir": str(d)}

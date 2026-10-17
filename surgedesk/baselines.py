"""Deterministic snapshot orderings used by evaluation."""
from __future__ import annotations

from .contracts import Incident, URGENT_KEYWORDS
from .stubs import find_terms
from .textutil import parse_iso


def order_fifo(incidents: list[Incident]) -> list[str]:
    return [i.incident_id for i in sorted(incidents, key=lambda i: (parse_iso(i.first_seen), i.incident_id))]


def order_volume(incidents: list[Incident]) -> list[str]:
    return [i.incident_id for i in sorted(incidents, key=lambda i: (
        -i.features["n_reports"], parse_iso(i.first_seen), i.incident_id))]


def order_keyword(incidents: list[Incident], texts_by_id: dict[str, str]) -> list[str]:
    return [i.incident_id for i in sorted(incidents, key=lambda i: (
        -sum(len(find_terms(texts_by_id[rid], URGENT_KEYWORDS)) for rid in i.report_ids),
        parse_iso(i.first_seen), i.incident_id))]


def order_surgedesk(incidents: list[Incident]) -> list[str]:
    return [i.incident_id for i in sorted(incidents, key=lambda i: i.rank_adj)]

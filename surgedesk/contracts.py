"""Data contracts shared by every stage. FROZEN.

Every handoff file is JSON Lines; each line is one of the dataclasses below,
serialized with .to_dict(). Read with surgedesk.fileio.read_jsonl(path, cls) and
write with surgedesk.fileio.write_jsonl(path, rows) so validation always runs.

Owner: foundation. Do not change field names, types, or constants without the
whole team agreeing; three other agents depend on them.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

from .textutil import parse_iso


class ContractError(ValueError):
    """Raised when a record does not match its contract."""


# =========================================================================== constants
CATEGORIES: list[str] = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "dont_know_cant_judge",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
]

# Reports in these predicted categories never join an incident.
NON_ACTIONABLE: frozenset[str] = frozenset(
    {"not_humanitarian", "sympathy_and_support", "dont_know_cant_judge"}
)

# Gold labels that make an incident snapshot urgent_proxy = 1 (evaluation/training target only).
URGENT_LABELS: frozenset[str] = frozenset(
    {"requests_or_urgent_needs", "injured_or_dead_people", "missing_or_found_people"}
)

# Predicted-category shares used as ranker features (order matters for FEATURE_KEYS).
SHARE_CATEGORIES: list[str] = [
    "requests_or_urgent_needs",
    "injured_or_dead_people",
    "missing_or_found_people",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "caution_and_advice",
    "other_relevant_information",
]

# Exact ranker feature keys, in this order. Definitions: handoffs/agent2_group_rank.md section 5.
FEATURE_KEYS: list[str] = [
    "n_reports",
    "n_unique_texts",
    "reports_last_60m",
    "growth_ratio",
    "age_min",
    "minutes_since_last",
    "has_place",
    "svi",
    "vuln_mentions",
    "vuln_share",
    "mean_pred_conf",
] + [f"share_{c}" for c in SHARE_CATEGORIES]

# Plain-English names used in the UI and in Granite briefings.
FEATURE_PLAIN: dict[str, str] = {
    "n_reports": "number of reports",
    "n_unique_texts": "number of distinct reports",
    "reports_last_60m": "reports in the last hour",
    "growth_ratio": "reports growing fast",
    "age_min": "how long ago it started",
    "minutes_since_last": "time since the latest report",
    "has_place": "a known location",
    "svi": "area social vulnerability",
    "vuln_mentions": "mentions of vulnerable people",
    "vuln_share": "share of reports mentioning vulnerable people",
    "mean_pred_conf": "classifier confidence",
    "share_requests_or_urgent_needs": "urgent requests for help",
    "share_injured_or_dead_people": "reports of injuries or deaths",
    "share_missing_or_found_people": "missing people reports",
    "share_displaced_people_and_evacuations": "evacuation and displacement",
    "share_infrastructure_and_utility_damage": "infrastructure damage",
    "share_rescue_volunteering_or_donation_effort": "rescue and donation activity",
    "share_caution_and_advice": "warnings and advice",
    "share_other_relevant_information": "other relevant information",
}

# Vulnerable-person lexicon (case-insensitive, whole word or phrase).
VULN_TERMS: list[str] = [
    "elderly", "senior", "seniors", "grandma", "grandpa", "grandmother", "grandfather",
    "disabled", "disability", "wheelchair", "oxygen", "dialysis", "insulin", "pregnant",
    "baby", "babies", "infant", "newborn", "toddler", "kids", "children", "bedridden",
    "hospice", "nursing home", "special needs", "trapped", "stranded",
]

# Keyword baseline vocabulary (case-insensitive, whole word or phrase).
URGENT_KEYWORDS: list[str] = [
    "help", "trapped", "rescue", "stranded", "injured", "dead", "missing",
    "sos", "roof", "water rising", "need",
]

BRIEF_FIELDS: list[str] = ["hazard", "location", "people_affected", "vulnerable", "need"]
SENTENCE_TAGS: list[str] = BRIEF_FIELDS + ["reason"]
ACTIONS: list[str] = ["accept", "override", "dismiss"]
SOURCES: list[str] = ["humaid", "synthetic"]

HAND_LABEL_COLUMNS: list[str] = [
    "report_id", "text", "hazard", "location", "people_affected", "vulnerable", "need",
    "labeler", "notes",
]
PURITY_COLUMNS: list[str] = ["incident_id", "report_id", "text", "same_incident", "labeler"]

# JSON schema Granite must fill for one incident (or one report). Shared with evaluation.
_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "evidence": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["value", "evidence", "confidence"],
}
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {f: _FIELD_SCHEMA for f in BRIEF_FIELDS},
    "required": list(BRIEF_FIELDS),
}

_LABEL_ALIASES = {
    "dont_know_or_cant_judge": "dont_know_cant_judge",
    "don_t_know_cant_judge": "dont_know_cant_judge",
    "rescue_volunteering_and_donation_effort": "rescue_volunteering_or_donation_effort",
    "injured_and_dead_people": "injured_or_dead_people",
    "missing_and_found_people": "missing_or_found_people",
    "displaced_people_and_evacuation": "displaced_people_and_evacuations",
    "requests_or_urgent_need": "requests_or_urgent_needs",
}


def normalize_label(raw: str) -> str:
    """Map any HumAID label spelling ('Requests or urgent needs', 'requests_or_urgent_needs') to CATEGORIES."""
    s = raw.strip().lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = _LABEL_ALIASES.get(s, s)
    if s not in CATEGORIES:
        raise ContractError(f"Unknown HumAID label {raw!r} (normalized {s!r}). Known: {CATEGORIES}")
    return s


def is_actionable(category: str) -> bool:
    return category not in NON_ACTIONABLE


# =========================================================================== helpers
def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _is_iso(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    try:
        parse_iso(s)
        return True
    except ValueError:
        return False


def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


def _is_fips(x: Any) -> bool:
    return x is None or (isinstance(x, str) and len(x) == 5 and x.isdigit())


class _Record:
    """Base: strict from_dict (unknown or missing keys raise), to_dict, validate."""

    KIND: ClassVar[str] = "record"

    @classmethod
    def from_dict(cls, d: dict) -> "_Record":
        names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
        unknown = set(d) - set(names)
        missing = [n for n in names if n not in d]
        _req(not unknown, f"{cls.KIND}: unknown keys {sorted(unknown)}")
        _req(not missing, f"{cls.KIND}: missing keys {missing}")
        obj = cls(**{n: d[n] for n in names})  # type: ignore[call-arg]
        obj.validate()
        return obj

    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    def validate(self) -> None:  # overridden
        raise NotImplementedError


# =========================================================================== records
@dataclass
class Report(_Record):
    """One incoming report. outputs/<event>/reports.jsonl, sorted by timestamp then report_id."""

    KIND: ClassVar[str] = "Report"
    report_id: str
    event: str
    text: str
    timestamp: str                 # ISO 8601 UTC, e.g. 2017-08-27T10:00:00Z
    source: str                    # "humaid" or "synthetic"
    gold_label: str | None         # one of CATEGORIES; NEVER a model input at inference
    is_synthetic: bool

    def validate(self) -> None:
        _req(isinstance(self.report_id, str) and self.report_id != "", "Report.report_id must be a non-empty str")
        _req(isinstance(self.event, str) and self.event != "", "Report.event must be a non-empty str")
        _req(isinstance(self.text, str), "Report.text must be a str")
        _req(_is_iso(self.timestamp), f"Report.timestamp not ISO: {self.timestamp!r}")
        _req(self.source in SOURCES, f"Report.source must be one of {SOURCES}")
        _req(self.gold_label is None or self.gold_label in CATEGORIES, f"Report.gold_label invalid: {self.gold_label!r}")
        _req(isinstance(self.is_synthetic, bool), "Report.is_synthetic must be bool")
        _req(self.is_synthetic == (self.source == "synthetic"), "Report.is_synthetic must match source")


@dataclass
class Tagged(_Record):
    """Stage 1 output per report. outputs/<event>/tagged.jsonl, same order as reports.jsonl."""

    KIND: ClassVar[str] = "Tagged"
    report_id: str
    emb_row: int                   # row in embeddings.npy (float32 [n, 384], L2-normalized)
    pred_category: str             # one of CATEGORIES
    pred_conf: float               # max class probability
    places: list[str]              # matched place names, best match first
    county_fips: str | None        # 5-digit county FIPS of best match
    vuln_terms: list[str]          # matched VULN_TERMS, in order of appearance, deduplicated
    is_actionable: bool            # pred_category not in NON_ACTIONABLE

    def validate(self) -> None:
        _req(isinstance(self.report_id, str) and self.report_id != "", "Tagged.report_id must be non-empty")
        _req(isinstance(self.emb_row, int) and not isinstance(self.emb_row, bool) and self.emb_row >= 0, "Tagged.emb_row must be int >= 0")
        _req(self.pred_category in CATEGORIES, f"Tagged.pred_category invalid: {self.pred_category!r}")
        _req(_is_prob(self.pred_conf), "Tagged.pred_conf must be in [0, 1]")
        _req(isinstance(self.places, list) and all(isinstance(p, str) for p in self.places), "Tagged.places must be list[str]")
        _req(_is_fips(self.county_fips), f"Tagged.county_fips must be 5-digit str or null: {self.county_fips!r}")
        _req(isinstance(self.vuln_terms, list) and all(t in VULN_TERMS for t in self.vuln_terms), "Tagged.vuln_terms must be from VULN_TERMS")
        _req(self.is_actionable == is_actionable(self.pred_category), "Tagged.is_actionable inconsistent with pred_category")


@dataclass
class Incident(_Record):
    """Stage 3 output: one incident at one snapshot. outputs/<event>/incidents.jsonl."""

    KIND: ClassVar[str] = "Incident"
    incident_id: str               # "<event>-inc-<n>", n starting at 1, stable across snapshots
    event: str
    snapshot_time: str             # ranking is "as of" this time
    report_ids: list[str]          # members with timestamp <= snapshot_time, in time order
    first_seen: str
    last_seen: str
    place: str | None
    county_fips: str | None
    svi: float | None              # CDC SVI RPL_THEMES in [0, 1]; None if unknown
    features: dict[str, float]     # exactly FEATURE_KEYS
    score: float                   # ranker probability
    score_adj: float               # after fairness adjustment at lam
    lam: float
    rank: int                      # 1 = highest score within this snapshot
    rank_adj: int                  # 1 = highest score_adj within this snapshot
    shap_top: list[list]           # up to 5 [feature_key, contribution], by |contribution| desc
    urgent_proxy: int | None       # evaluation label; None when gold labels are unavailable

    def validate(self) -> None:
        _req(isinstance(self.incident_id, str) and "-inc-" in self.incident_id, f"Incident.incident_id malformed: {self.incident_id!r}")
        _req(_is_iso(self.snapshot_time), "Incident.snapshot_time not ISO")
        _req(isinstance(self.report_ids, list) and len(self.report_ids) >= 1, "Incident.report_ids must be non-empty")
        _req(_is_iso(self.first_seen) and _is_iso(self.last_seen), "Incident.first_seen/last_seen not ISO")
        _req(parse_iso(self.first_seen) <= parse_iso(self.last_seen) <= parse_iso(self.snapshot_time), "Incident times out of order")
        _req(self.place is None or isinstance(self.place, str), "Incident.place must be str or null")
        _req(_is_fips(self.county_fips), "Incident.county_fips must be 5-digit str or null")
        _req(self.svi is None or _is_prob(self.svi), "Incident.svi must be in [0, 1] or null")
        _req(isinstance(self.features, dict) and list(self.features) == FEATURE_KEYS, "Incident.features keys must equal FEATURE_KEYS in order")
        _req(all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.features.values()), "Incident.features values must be finite numbers")
        _req(_is_prob(self.score) and _is_prob(self.score_adj), "Incident.score/score_adj must be in [0, 1]")
        _req(isinstance(self.lam, (int, float)) and self.lam >= 0, "Incident.lam must be >= 0")
        _req(isinstance(self.rank, int) and self.rank >= 1 and isinstance(self.rank_adj, int) and self.rank_adj >= 1, "Incident.rank/rank_adj must be ints >= 1")
        _req(isinstance(self.shap_top, list) and len(self.shap_top) <= 5, "Incident.shap_top must be a list of <= 5 pairs")
        for pair in self.shap_top:
            _req(isinstance(pair, list) and len(pair) == 2 and pair[0] in FEATURE_KEYS and isinstance(pair[1], (int, float)), f"Incident.shap_top bad pair {pair!r}")
        _req(self.urgent_proxy in (None, 0, 1), "Incident.urgent_proxy must be 0, 1, or null")


@dataclass
class Briefing(_Record):
    """Stage 4 output for one incident at one snapshot. outputs/<event>/briefings.jsonl."""

    KIND: ClassVar[str] = "Briefing"
    incident_id: str
    event: str
    snapshot_time: str
    cache_key: str                 # sha1(incident_id | sorted report_ids | model file name)
    rep_report_ids: list[str]      # 1 or 2 reports Granite read
    fields: dict[str, dict]        # BRIEF_FIELDS -> {value, evidence, confidence, grounded}
    sentences: list[dict]          # {text, tags, accepted, reject_reason}
    briefing: str                  # accepted sentences joined with spaces, or the template
    used_fallback: bool
    model_file: str                # file name only (or "template" / "stub")
    timing: dict[str, float]       # extract_s, brief_s, prompt_tokens, completion_tokens

    def validate(self) -> None:
        _req(isinstance(self.incident_id, str) and "-inc-" in self.incident_id, "Briefing.incident_id malformed")
        _req(_is_iso(self.snapshot_time), "Briefing.snapshot_time not ISO")
        _req(isinstance(self.cache_key, str) and self.cache_key != "", "Briefing.cache_key empty")
        _req(isinstance(self.rep_report_ids, list) and 1 <= len(self.rep_report_ids) <= 2, "Briefing.rep_report_ids must have 1 or 2 ids")
        _req(isinstance(self.fields, dict) and list(self.fields) == BRIEF_FIELDS, "Briefing.fields keys must equal BRIEF_FIELDS in order")
        for name, fv in self.fields.items():
            _req(isinstance(fv, dict) and set(fv) == {"value", "evidence", "confidence", "grounded"}, f"Briefing.fields[{name}] must have value/evidence/confidence/grounded")
            _req(fv["value"] is None or isinstance(fv["value"], str), f"Briefing.fields[{name}].value must be str or null")
            _req(fv["evidence"] is None or isinstance(fv["evidence"], str), f"Briefing.fields[{name}].evidence must be str or null")
            _req(_is_prob(fv["confidence"]), f"Briefing.fields[{name}].confidence must be in [0, 1]")
            _req(isinstance(fv["grounded"], bool), f"Briefing.fields[{name}].grounded must be bool")
            _req(not (fv["value"] is not None and not fv["grounded"]), f"Briefing.fields[{name}]: ungrounded fields must have value null")
        _req(isinstance(self.sentences, list), "Briefing.sentences must be a list")
        for s in self.sentences:
            _req(isinstance(s, dict) and set(s) == {"text", "tags", "accepted", "reject_reason"}, f"Briefing sentence malformed: {s!r}")
            _req(isinstance(s["tags"], list) and all(isinstance(t, str) for t in s["tags"]), "Briefing sentence tags must be list[str]")
            _req(isinstance(s["accepted"], bool), "Briefing sentence accepted must be bool")
            _req((s["reject_reason"] is None) == s["accepted"], "reject_reason must be null exactly when accepted")
        _req(isinstance(self.briefing, str) and self.briefing.strip() != "", "Briefing.briefing must be non-empty")
        _req(isinstance(self.used_fallback, bool), "Briefing.used_fallback must be bool")
        _req(isinstance(self.model_file, str) and self.model_file != "", "Briefing.model_file empty")
        _req(set(self.timing) == {"extract_s", "brief_s", "prompt_tokens", "completion_tokens"}, "Briefing.timing keys wrong")


@dataclass
class Decision(_Record):
    """One coordinator action from the review UI. outputs/<event>/decisions.jsonl (append-only)."""

    KIND: ClassVar[str] = "Decision"
    incident_id: str
    event: str
    snapshot_time: str
    action: str                    # accept | override | dismiss
    new_rank: int | None           # required for override, else null
    note: str
    reviewer: str
    timestamp: str                 # when the click happened (UTC)
    lam: float

    def validate(self) -> None:
        _req(self.action in ACTIONS, f"Decision.action must be one of {ACTIONS}")
        _req(_is_iso(self.snapshot_time) and _is_iso(self.timestamp), "Decision times not ISO")
        if self.action == "override":
            _req(isinstance(self.new_rank, int) and self.new_rank >= 1, "Decision.new_rank required (int >= 1) for override")
        else:
            _req(self.new_rank is None, "Decision.new_rank must be null unless action is override")
        _req(isinstance(self.note, str) and isinstance(self.reviewer, str), "Decision.note/reviewer must be str")
        _req(isinstance(self.lam, (int, float)) and self.lam >= 0, "Decision.lam must be >= 0")


RECORD_TYPES = {
    "reports.jsonl": Report,
    "tagged.jsonl": Tagged,
    "incidents.jsonl": Incident,
    "briefings.jsonl": Briefing,
    "decisions.jsonl": Decision,
}


def empty_fields() -> dict[str, dict]:
    """All BRIEF_FIELDS null and ungrounded (used by templates and stubs)."""
    return {f: {"value": None, "evidence": None, "confidence": 0.0, "grounded": False} for f in BRIEF_FIELDS}


def zero_timing() -> dict[str, float]:
    return {"extract_s": 0.0, "brief_s": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}


__all__ = [n for n in dir() if not n.startswith("_")]

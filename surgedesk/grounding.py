"""Literal evidence and sentence checks for Stage 4 briefings."""
from __future__ import annotations

import math
import re

from . import config
from .contracts import BRIEF_FIELDS, SENTENCE_TAGS, Incident
from .textutil import content_words, normalize_for_match


def evidence_ok(evidence: str | None, sources: list[str]) -> bool:
    """Check that a quote occurs in one of the masked reports Granite read."""
    if not isinstance(evidence, str) or sum(not char.isspace() for char in evidence) < 2:
        return False
    quote = normalize_for_match(evidence)
    return any(quote in normalize_for_match(source) for source in sources)


def _geographic_use(evidence: str, sources: list[str]) -> bool:
    """Reject quotes used only as storm names or comparison targets.

    This is a conservative context guard, not a geographic entity validator.
    A separate geographic occurrence of the same name can still support it.
    """
    quote = normalize_for_match(evidence)
    event_storm = re.fullmatch(r"(?:hurricane|cyclone|typhoon|tropical_storm)_(.+)_\d{4}", config.DEMO_EVENT)
    storm_names = {event_storm[1].replace("_", " ")} if event_storm else set()
    for source in sources:
        storm_names.update(name.lower() for name in re.findall(
            r"\b(?:hurricane|cyclone|typhoon|tropical\s+storm)\s*#?([a-z]+)", source, re.I))
    for source in sources:
        text = normalize_for_match(source)
        for match in re.finditer(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", text):
            prefix = text[:match.start()]
            if re.search(r"\b(?:hurricane|cyclone|typhoon|tropical\s+storm)\s*#?$", prefix):
                continue
            if re.search(r"\b(?:than|compared\s+to|unlike)\s*$", prefix):
                continue
            if quote in storm_names:
                geographic_prefix = re.search(r"\b(?:in|near|at|outside|city\s+of|town\s+of)\s*$", prefix)
                # State abbreviations are explicit place qualifiers.
                original_suffix = re.sub(r"\s+", " ", source).strip()[match.end():]
                geographic_suffix = re.match(r"(?:\s*,\s*|\s+)[A-Z]{2}\b", original_suffix)
                if not geographic_prefix and not geographic_suffix:
                    continue
            return True
    return False


def ground_fields(raw: dict | None, sources: list[str]) -> dict[str, dict]:
    """Keep only fields with literal quotes from the masked source texts."""
    raw = raw if isinstance(raw, dict) else {}
    fields = {}
    for name in BRIEF_FIELDS:
        candidate = raw.get(name)
        candidate = candidate if isinstance(candidate, dict) else {}
        evidence = candidate.get("evidence")
        if evidence_ok(evidence, sources) and (name != "location" or _geographic_use(evidence, sources)):
            confidence = candidate.get("confidence")
            if isinstance(confidence, bool):
                confidence = 0.0
            try:
                confidence = float(confidence)
            except (TypeError, ValueError, OverflowError):
                confidence = 0.0
            if math.isnan(confidence):
                confidence = 0.0
            confidence = min(1.0, max(0.0, confidence))
            quote = evidence.strip()
            fields[name] = {
                "value": quote,
                "evidence": quote,
                "confidence": confidence,
                "grounded": True,
            }
        else:
            fields[name] = {
                "value": None,
                "evidence": None,
                "confidence": 0.0,
                "grounded": False,
            }
    return fields


def split_tagged_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Split model output into sentences and their bracketed source tags."""
    matches = list(re.finditer(r"(.+?)\s*\[([^\[\]]*)\]\s*\.?", text, re.S))
    out = []
    for match in matches:
        sentence = match.group(1).strip()
        if not sentence.endswith("."):
            sentence += "."
        tags = [tag.strip().lower() for tag in match.group(2).split(",") if tag.strip()]
        out.append((sentence, tags))
    remainder = text[matches[-1].end():].strip() if matches else text.strip()
    if len(remainder.split()) >= 3:
        if not remainder.endswith("."):
            remainder += "."
        out.append((remainder, []))
    return out


def numbers_allowed(incident: Incident, fields: dict) -> set[str]:
    """Numbers supplied by the ranker or grounded report quotes."""
    numbers = {
        str(int(incident.features["n_reports"])),
        str(int(incident.features["reports_last_60m"])),
    }
    for name in BRIEF_FIELDS:
        field = fields.get(name, {})
        if field.get("grounded") and isinstance(field.get("value"), str):
            numbers.update(re.findall(r"\d+", field["value"]))
    return numbers


def check_sentence(sentence: str, tags: list[str], fields: dict, reasons: list[str],
                   numbers: set[str]) -> tuple[bool, str | None]:
    """Check tagged claims, ranking reasons, and every number in a sentence."""
    if not tags:
        return False, "no tags"
    for tag in tags:
        if tag not in SENTENCE_TAGS:
            return False, f"unknown tag '{tag}'"

    words = set(content_words(sentence))
    for tag in tags:
        if tag == "reason":
            continue
        value = fields.get(tag, {}).get("value")
        if value is None:
            return False, f"tagged field '{tag}' is unknown"
        field_words = content_words(value)
        if field_words and sum(word in words for word in field_words) / len(field_words) < 0.5:
            return False, f"tagged field '{tag}' not supported by the sentence"

    if "reason" in tags:
        phrase_supported = any(
            (reason_words := content_words(reason))
            and sum(word in words for word in reason_words) / len(reason_words) >= 0.5
            for reason in reasons
        )
        number_supported = any(number in numbers for number in re.findall(r"\d+", sentence))
        if not phrase_supported and not number_supported:
            return False, "reason not supported"

    for number in re.findall(r"\d+", sentence):
        if number not in numbers:
            return False, f"unsupported number '{number}'"
    # A ranking feature about rescue/donation activity cannot establish an
    # actual unmet need. Check the claim even if the model omits its field tag.
    if re.search(r"\b(?:needs?|needed|requires?|required|requiring)\b", sentence, re.I):
        need = fields.get("need", {})
        if not need.get("grounded") or not need.get("value"):
            return False, "need claim without grounded need"
        if "need" not in tags:
            return False, "need claim missing need tag"
    return True, None

"""Literal evidence and sentence checks for Stage 4 briefings."""
from __future__ import annotations

import math
import re

from . import config
from .contracts import BRIEF_FIELDS, SENTENCE_TAGS, Incident, feature_reason
from .textutil import content_words, normalize_for_match


def evidence_ok(evidence: str | None, sources: list[str]) -> bool:
    """Check that a quote occurs in one of the masked reports Granite read."""
    if not isinstance(evidence, str) or sum(not char.isspace() for char in evidence) < 2:
        return False
    quote = normalize_for_match(evidence)
    return any(quote in normalize_for_match(source) for source in sources)


def _current_assertion(evidence: str, sources: list[str], name: str) -> bool:
    """Require a current assertion, rather than a historical or forecast quote.

    These are conservative exclusion rules, not a claim of semantic validation.
    Check each occurrence separately so another current report can support it.
    """
    quote = normalize_for_match(evidence)
    event_year = re.search(r"_(\d{4})$", config.DEMO_EVENT)
    for source in sources:
        text = normalize_for_match(source)
        for match in re.finditer(re.escape(quote), text):
            start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;\n") + 1
            ends = [pos for mark in ".!?;\n" if (pos := text.find(mark, match.end())) >= 0]
            clause = text[start:min(ends) if ends else len(text)]
            if re.search(r"\b(?:historical|historically|previous|past|in comparison|back in)\b", clause):
                continue
            # Date context distinguishes the year 2000 from '2000 people'.
            years = re.findall(
                r"\b(?:in|of|during|from)\s+((?:18|19|20)\d{2})\b|"
                r"\b((?:18|19|20)\d{2})(?:'s)?\s+(?:death|toll|storm|hurricane|flood|earthquake)\b",
                clause)
            years = [year for pair in years for year in pair if year]
            if event_year and any(int(year) < int(event_year[1]) for year in years):
                continue
            # Requests such as 'could you send a boat' are still requests.
            if name != "need" and re.search(
                r"\b(?:could|might|would|forecast\w*|predict\w*|expected|projected)\b|"
                r"\bmay\s+(?:be|have|cause|affect|displace|kill|injure|reach|rise|approach)\b", clause
            ):
                continue
            if name == "people_affected":
                # 'Death toll is rising' supplies neither who nor how many.
                person = re.search(
                    r"\b(?:people|persons?|residents?|famil\w*|children|child|kids?|bab\w*|"
                    r"newborns?|adults?|patients?|victims?|responders?|workers?|men|man|women|woman|"
                    r"elderly|seniors?|grand\w*|parents?|mothers?|fathers?|mom|dad|neighbors?)\b", quote)
                if not person and not re.search(r"\d", quote):
                    continue
                # Bare counts need people/casualty context in the same sentence.
                if not person and not re.search(
                    r"\b(?:people|persons?|residents?|famil\w*|dead|died|death|killed|injured|"
                    r"evacuated|displaced|missing|trapped|rescued|casualties)\b", clause):
                    continue
            return True
    return False


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
        if (evidence_ok(evidence, sources) and _current_assertion(evidence, sources, name)
                and (name != "location" or _geographic_use(evidence, sources))):
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
        sentence = _clean_sentence(match.group(1))
        if not sentence.endswith("."):
            sentence += "."
        tags = [tag.strip().lower() for tag in match.group(2).split(",") if tag.strip()]
        out.append((sentence, tags))
    remainder = _clean_sentence(text[matches[-1].end():] if matches else text)
    if len(remainder.split()) >= 3:
        if not remainder.endswith("."):
            remainder += "."
        out.append((remainder, []))
    return out


def _clean_sentence(text: str) -> str:
    """Remove list labels and dangling punctuation, not factual numbers."""
    text = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", "", text.strip())
    return re.sub(r"[,;:]\s*\.?$", ".", text).strip()


def _numbers(text: str) -> list[str]:
    # Treat 12,000 as one number; '12' and '000' cannot support a claim of 12.
    return [value.replace(",", "") for value in re.findall(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?", text)]


def numbers_allowed(incident: Incident, fields: dict) -> set[str]:
    """Numbers supplied by the ranker or grounded report quotes."""
    numbers = {
        str(int(incident.features["n_reports"])),
        str(int(incident.features["reports_last_60m"])),
    }
    for name in BRIEF_FIELDS:
        field = fields.get(name, {})
        if field.get("grounded") and isinstance(field.get("value"), str):
            numbers.update(_numbers(field["value"]))
    if "share_current_assistance" in incident.features:
        for key, contribution in incident.shap_top:
            if contribution > 0:
                numbers.update(_numbers(feature_reason(key, incident.features)))
    return numbers


def check_sentence(sentence: str, tags: list[str], fields: dict, reasons: list[str],
                   numbers: set[str]) -> tuple[bool, str | None]:
    """Check tagged claims, ranking reasons, and every number in a sentence."""
    sentence = _clean_sentence(sentence)
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
        number_supported = any(number in numbers for number in _numbers(sentence))
        if not phrase_supported and not number_supported:
            return False, "reason not supported"

    for number in _numbers(sentence):
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
    if tags == ["reason"]:
        # A matching reason word must not license an unrelated factual claim.
        # Keep ranking explanations in the requested, visibly separate form.
        if not re.match(r"ranked high because of\b", sentence, re.I):
            return False, "reason not supported"
        supported = set(content_words(" ".join(reasons))) | {
            "ranked", "high", "because", "reports", "report",
        } | numbers
        if {word for word in words if not word.isdigit()} - supported:
            return False, "reason not supported"
        if any(": " in reason for reason in reasons):
            remainder = re.sub(r"^ranked high because of\s*", "", sentence, flags=re.I).lower()
            for reason in sorted(reasons, key=len, reverse=True):
                remainder = remainder.replace(reason.lower(), "")
            remainder = re.sub(r"\band\b|[\s,;.]+", "", remainder)
            if remainder:
                return False, "reason not supported"
    return True, None

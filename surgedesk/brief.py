"""Stage 4: representative reports, grounded extraction, and briefings."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy

from . import config, stubs
from .contracts import (
    BRIEF_FIELDS,
    EXTRACTION_SCHEMA,
    FEATURE_PLAIN,
    Briefing,
    Incident,
    Report,
    Tagged,
    empty_fields,
    zero_timing,
)
from .fileio import append_jsonl, load_embeddings, read_jsonl, write_jsonl
from .grounding import check_sentence, ground_fields, numbers_allowed, split_tagged_sentences
from .llm import GraniteLLM, get_llm
from .textutil import mask_privacy


EXTRACT_SYSTEM = (
    "You extract facts from disaster reports for an emergency coordinator. Use ONLY the report text. "
    "For each field, copy the shortest exact phrase from a report as evidence. If a report does not state "
    "a field, set value and evidence to null and confidence to 0. Never guess. "
    "The quote must express that field, not merely appear in the report. Storm names and historical "
    "comparisons are not locations. A donation offer is not an unmet need. Return JSON only: one object "
    "with the five requested fields, without report-number wrappers, markdown, or explanations."
)
EXTRACT_USER = (
    "Fields:\n"
    "- hazard: the physical threat (flooding, fire, wind damage, collapse, outage)\n"
    "- location: the most specific place named\n"
    "- people_affected: who or how many people are affected\n"
    "- vulnerable: any elderly, disabled, medical, pregnant, or child mention\n"
    "- need: what is being asked for\n\n"
    "Reports:\n{reports}\n\n"
    "Return JSON with keys hazard, location, people_affected, vulnerable, need. "
    "Each is an object with value, evidence, confidence (0 to 1)."
)

BRIEF_SYSTEM = (
    "You write short briefings for an emergency coordinator. Use ONLY the facts given. Write 2 or 3 sentences. "
    "End every sentence with tags in square brackets naming the facts it uses, chosen from: "
    "hazard, location, people_affected, vulnerable, need, reason. Do not add any fact that is not given. "
    "Use one short factual claim per sentence and put its tags immediately after that sentence. "
    "Omit unknown fields. Tags contain field names only, never values. "
    "Do not infer emergencies, injuries, urgency, vulnerability, or rescue needs from ranking features. "
    "For a ranking explanation, write 'Ranked high because of' followed by the given reason phrases "
    "and tag it [reason]. Rescue/donation activity describes a ranking feature, not a request for aid."
)
BRIEF_USER = (
    "Facts:\nhazard: {hazard}\nlocation: {location}\npeople_affected: {people_affected}\n"
    "vulnerable: {vulnerable}\nneed: {need}\nnumber of reports: {n_reports}\n"
    "reasons it ranked high: {reasons}\n\nWrite the briefing."
)


def select_representatives(incident: Incident, reports_by_id: dict[str, Report],
                           tags_by_id: dict[str, Tagged], embeddings: numpy.ndarray) -> list[Report]:
    """Choose the centroid report and the report with most vulnerable mentions."""
    report_ids = incident.report_ids
    rows = [tags_by_id[report_id].emb_row for report_id in report_ids]
    centroid = embeddings[rows].sum(axis=0)
    length = float(numpy.linalg.norm(centroid))
    if length:
        centroid = centroid / length
    similarities = [float(numpy.dot(embeddings[row], centroid)) for row in rows]
    first = report_ids[int(numpy.argmax(similarities))]
    second = max(
        enumerate(report_ids),
        key=lambda pair: (
            len(tags_by_id[pair[1]].vuln_terms),
            len(reports_by_id[pair[1]].text),
            -pair[0],
        ),
    )[1]
    selected = [first] if first == second else [first, second]
    return [reports_by_id[report_id] for report_id in selected]


def extract_fields(llm: GraniteLLM, reports: list[Report]) -> tuple[dict, dict]:
    """Extract fields from masked reports, then verify each evidence quote."""
    sources = [mask_privacy(report.text) for report in reports]
    numbered = "\n".join(f"[{index}] {source}" for index, source in enumerate(sources, 1))
    user = EXTRACT_USER.format(reports=numbered)
    parsed, timing = llm.chat_json(EXTRACT_SYSTEM, user, EXTRACTION_SCHEMA, config.EXTRACT_MAX_TOKENS)
    return ground_fields(parsed, sources), timing


def _positive_reasons(incident: Incident) -> list[str]:
    return [FEATURE_PLAIN[key] for key, value in incident.shap_top if value > 0][:3]


def write_briefing(llm: GraniteLLM, incident: Incident, fields: dict) -> tuple[list[dict], dict]:
    """Generate and verify each tagged sentence before it can appear in a briefing."""
    reasons = _positive_reasons(incident)
    field_values = {
        name: mask_privacy(fields.get(name, {}).get("value") or "unknown")
        for name in BRIEF_FIELDS
    }
    user = BRIEF_USER.format(
        **field_values,
        n_reports=int(incident.features["n_reports"]),
        reasons="; ".join(reasons),
    )
    answer, timing = llm.chat_text(BRIEF_SYSTEM, user, config.BRIEF_MAX_TOKENS)
    allowed = numbers_allowed(incident, fields)
    sentences = []
    for sentence, tags in split_tagged_sentences(answer):
        accepted, reject_reason = check_sentence(sentence, tags, fields, reasons, allowed)
        sentences.append({
            "text": mask_privacy(sentence),
            "tags": tags,
            "accepted": accepted,
            "reject_reason": reject_reason,
        })
    return sentences, timing


def template_briefing(incident: Incident, fields: dict) -> str:
    return stubs.template_briefing_text(incident, fields)


def cache_key(incident: Incident, model_file: str) -> str:
    raw = "|".join([incident.incident_id] + sorted(incident.report_ids) + [model_file])
    return hashlib.sha1(raw.encode()).hexdigest()


def _source_data(src_dir: Path) -> tuple[dict[str, Report], dict[str, Tagged], numpy.ndarray]:
    reports = read_jsonl(src_dir / config.REPORTS_FILE, Report)
    tagged = read_jsonl(src_dir / config.TAGGED_FILE, Tagged)
    embeddings = load_embeddings(src_dir / config.EMBEDDINGS_FILE)
    return (
        {report.report_id: report for report in reports},
        {tag.report_id: tag for tag in tagged},
        embeddings,
    )


def _timing(extract: dict, brief: dict) -> dict[str, float]:
    return {
        "extract_s": float(extract.get("seconds", 0.0)),
        "brief_s": float(brief.get("seconds", 0.0)),
        "prompt_tokens": float(extract.get("prompt_tokens", 0.0)) + float(brief.get("prompt_tokens", 0.0)),
        "completion_tokens": float(extract.get("completion_tokens", 0.0)) + float(brief.get("completion_tokens", 0.0)),
    }


def _log_if_needed(llm: GraniteLLM, path: Path, kind: str, timing: dict) -> None:
    """Injected fakes need the same call log that GraniteLLM writes for itself."""
    log_path = getattr(llm, "log_path", None)
    if log_path is None or Path(log_path) != path:
        append_jsonl(path, {"kind": kind, "timing": timing, "error": None})


def _recheck_cached(row: Briefing, incident: Incident, reports_by_id: dict[str, Report]) -> Briefing:
    """Apply current rejection checks to cached text without another model call."""
    sources = [mask_privacy(reports_by_id[rid].text) for rid in row.rep_report_ids]
    fields = ground_fields(row.fields, sources)
    reasons = _positive_reasons(incident)
    allowed = numbers_allowed(incident, fields)
    sentences = []
    for old in row.sentences:
        accepted, reason = check_sentence(old["text"], old["tags"], fields, reasons, allowed)
        sentences.append({**old, "accepted": accepted, "reject_reason": reason})
    accepted = [sentence["text"] for sentence in sentences if sentence["accepted"]]
    return replace(row, fields=fields, sentences=sentences, used_fallback=not accepted,
                   briefing=" ".join(accepted) if accepted else template_briefing(incident, fields))


def brief_incidents(event: str, incidents: list[Incident], top_n: int = config.TOP_N_BRIEF,
                    force: bool = False, llm: GraniteLLM | None = None,
                    src_dir: Path | None = None, out_dir: Path | None = None) -> list[Briefing]:
    """Brief one fairness-adjusted snapshot, persisting each completed incident."""
    src_dir = Path(src_dir) if src_dir is not None else config.event_dir(event)
    out_dir = Path(out_dir) if out_dir is not None else config.event_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / config.BRIEFINGS_FILE
    log_path = out_dir / config.LLM_LOG_FILE
    existing = read_jsonl(output, Briefing) if output.exists() else []
    chosen = sorted(incidents, key=lambda incident: incident.rank_adj)[:max(0, top_n)]
    if not chosen:
        return []

    model_path = Path(config.GRANITE_PATH)
    if not model_path.is_absolute():
        model_path = config.ROOT / model_path
    template_only = llm is None and not model_path.is_file()
    model_file = (
        "template" if template_only else
        getattr(llm, "model_file", "fake") if llm is not None else model_path.name
    )
    if template_only:
        print(
            f"WARNING: Granite model file is missing: {model_path}. "
            "Using template briefings. See handoffs/person3_granite.md."
        )

    result = []
    source_data = None
    for incident in chosen:
        key = cache_key(incident, model_file)
        cached = next((row for row in existing if (
            row.cache_key == key and row.snapshot_time == incident.snapshot_time
        )), None)
        if cached is not None and not force:
            if source_data is None:
                source_data = _source_data(src_dir)
            checked = _recheck_cached(cached, incident, source_data[0])
            if checked != cached:
                existing = [checked if old is cached else old for old in existing]
                write_jsonl(output, existing)
            result.append(checked)
            continue

        if source_data is None:
            source_data = _source_data(src_dir)
        reports_by_id, tags_by_id, embeddings = source_data
        representatives = select_representatives(incident, reports_by_id, tags_by_id, embeddings)
        if template_only:
            fields = empty_fields()
            sentences = []
            briefing_text = template_briefing(incident, fields)
            timing = zero_timing()
            fallback = True
        else:
            if llm is None:
                llm = get_llm(model_path=model_path, log_path=log_path)
            fields, extract_timing = extract_fields(llm, representatives)
            _log_if_needed(llm, log_path, "json", extract_timing)
            if any(field["grounded"] for field in fields.values()):
                sentences, brief_timing = write_briefing(llm, incident, fields)
                _log_if_needed(llm, log_path, "text", brief_timing)
            else:
                # With no verified facts, another generation can only explain
                # ranking features. Use the honest template without that call.
                sentences = []
                brief_timing = {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            accepted = [sentence["text"] for sentence in sentences if sentence["accepted"]]
            fallback = not accepted
            briefing_text = " ".join(accepted) if accepted else template_briefing(incident, fields)
            timing = _timing(extract_timing, brief_timing)

        row = Briefing(
            incident_id=incident.incident_id,
            event=incident.event,
            snapshot_time=incident.snapshot_time,
            cache_key=key,
            rep_report_ids=[report.report_id for report in representatives],
            fields=fields,
            sentences=sentences,
            briefing=briefing_text,
            used_fallback=fallback,
            model_file=model_file,
            timing=timing,
        )
        row.validate()
        existing = [old for old in existing if not (
            old.incident_id == row.incident_id and old.snapshot_time == row.snapshot_time
        )]
        existing.append(row)
        write_jsonl(output, existing)
        result.append(row)
    return result


def extract_for_eval(event: str, report_ids: list[str], force: bool = False,
                     llm: GraniteLLM | None = None, src_dir: Path | None = None,
                     out_dir: Path | None = None) -> Path:
    """Extract one report at a time, saving predictions after each call."""
    src_dir = Path(src_dir) if src_dir is not None else config.event_dir(event)
    out_dir = Path(out_dir) if out_dir is not None else config.event_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "extraction_preds.jsonl"
    log_path = out_dir / config.LLM_LOG_FILE
    existing = read_jsonl(output) if output.exists() else []
    completed = {row["report_id"] for row in existing}
    pending = [report_id for report_id in dict.fromkeys(report_ids)
               if force or report_id not in completed]
    if not pending:
        if not output.exists():
            write_jsonl(output, existing)
        return output
    if llm is None:
        llm = get_llm(log_path=log_path)
    reports = read_jsonl(src_dir / config.REPORTS_FILE, Report)
    by_id = {report.report_id: report for report in reports}
    for report_id in pending:
        fields, timing = extract_fields(llm, [by_id[report_id]])
        _log_if_needed(llm, log_path, "json", timing)
        existing = [row for row in existing if row["report_id"] != report_id]
        existing.append({"report_id": report_id, "fields": fields, "timing": timing})
        write_jsonl(output, existing)
        completed.add(report_id)
    return output

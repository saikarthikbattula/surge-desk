"""Fast, offline acceptance checks for Stage 4; Granite is opt-in only."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from surgedesk import brief, config, llm, stubs, textutil
from surgedesk.contracts import BRIEF_FIELDS, FEATURE_PLAIN, Briefing, Incident, Report, Tagged, empty_fields, zero_timing
from surgedesk.fileio import load_embeddings, read_jsonl, write_jsonl
from surgedesk.grounding import (
    check_sentence,
    evidence_ok,
    ground_fields,
    numbers_allowed,
    split_tagged_sentences,
)
from surgedesk.testing import fixture_dir, skip


def _fixture():
    root = fixture_dir()
    reports = read_jsonl(root / config.REPORTS_FILE, Report)
    tagged = read_jsonl(root / config.TAGGED_FILE, Tagged)
    incidents = read_jsonl(root / config.INCIDENTS_FILE, Incident)
    briefings = read_jsonl(root / config.BRIEFINGS_FILE, Briefing)
    embeddings = load_embeddings(root / config.EMBEDDINGS_FILE)
    return reports, tagged, incidents, briefings, embeddings


def _last_snapshot(incidents):
    snapshot = max(incident.snapshot_time for incident in incidents)
    return [incident for incident in incidents if incident.snapshot_time == snapshot]


class FakeLLM:
    model_file = "fake"

    def __init__(self):
        self.json_calls = 0
        self.text_calls = 0

    def chat_json(self, system, user, schema, max_tokens):
        assert system == brief.EXTRACT_SYSTEM
        assert schema == brief.EXTRACTION_SCHEMA
        assert max_tokens == config.EXTRACT_MAX_TOKENS
        assert "Reports:\n[1] " in user
        self.json_calls += 1
        phrase = "Water rising fast" if "Water rising fast" in user else None
        raw = {
            "hazard": {"value": "flood", "evidence": phrase, "confidence": 0.9},
            "location": {"value": "invented", "evidence": "Atlantis", "confidence": 0.9},
        }
        return raw, {"seconds": 0.01, "prompt_tokens": 12, "completion_tokens": 8,
                     "ok": True, "attempts": 1}

    def chat_text(self, system, user, max_tokens):
        assert system == brief.BRIEF_SYSTEM
        assert max_tokens == config.BRIEF_MAX_TOKENS
        self.text_calls += 1
        return "Water rising fast. [hazard] Twelve people need rescue. [people_affected]", {
            "seconds": 0.02, "prompt_tokens": 10, "completion_tokens": 9,
            "ok": True, "attempts": 1,
        }


def test_chat_json_plain_without_grammar_and_quote_guard():
    class PlainModel:
        def __init__(self):
            self.calls = []

        def create_chat_completion(self, **kwargs):
            assert set(kwargs) == {"messages", "max_tokens", "temperature"}
            self.calls.append(kwargs)
            content = (
                "not JSON" if len(self.calls) == 1 else
                json.dumps({"Report 1": {
                    "hazard": "invented tsunami", "location": "Meyerland Houston",
                }})
            )
            return {"choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    adapter = llm.GraniteLLM.__new__(llm.GraniteLLM)
    adapter.model = PlainModel()
    adapter.log_path = None
    source = "Water rising fast near Meyerland Houston. @privatehandle needs help."
    parsed, timing = adapter.chat_json("Use only the report", source,
                                       brief.EXTRACTION_SCHEMA, 128)
    assert timing["ok"] and timing["attempts"] == 2
    assert timing["prompt_tokens"] == 20 and timing["completion_tokens"] == 10
    assert "@privatehandle" not in adapter.model.calls[0]["messages"][1]["content"]
    assert "@user" in adapter.model.calls[0]["messages"][1]["content"]
    assert adapter.model.calls[1]["messages"][1]["content"].endswith("Return only valid JSON.")
    assert parsed["location"] == {"value": "Meyerland Houston",
                                  "evidence": "Meyerland Houston", "confidence": 0.0}
    fields = ground_fields(parsed, [textutil.mask_privacy(source)])
    assert fields["location"]["grounded"]
    assert not fields["hazard"]["grounded"]


def test_chat_json_retries_unrelated_object():
    class PlainModel:
        def __init__(self):
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            content = ({"message": "sorry"} if self.calls == 1 else
                       {"hazard": {"value": None, "evidence": None, "confidence": 0}})
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

    adapter = llm.GraniteLLM.__new__(llm.GraniteLLM)
    adapter.model = PlainModel()
    adapter.log_path = None
    parsed, timing = adapter.chat_json("system", "report", brief.EXTRACTION_SCHEMA, 128)
    assert timing["ok"] and timing["attempts"] == 2
    assert parsed["hazard"]["value"] is None


def test_representatives_match_fixture():
    reports, tagged, incidents, briefings, embeddings = _fixture()
    reports_by_id = {report.report_id: report for report in reports}
    tags_by_id = {tag.report_id: tag for tag in tagged}
    incidents_by_key = {(incident.incident_id, incident.snapshot_time): incident for incident in incidents}
    for fixture_briefing in briefings:
        incident = incidents_by_key[(fixture_briefing.incident_id, fixture_briefing.snapshot_time)]
        actual = brief.select_representatives(incident, reports_by_id, tags_by_id, embeddings)
        assert [report.report_id for report in actual] == fixture_briefing.rep_report_ids


def test_evidence_ok():
    sources = ["Water  rising fast near Meyerland Houston; @user requested a boat."]
    assert evidence_ok("WATER rising   FAST", sources)
    assert not evidence_ok("floodwaters are rising", sources)
    assert not evidence_ok("", sources)
    assert not evidence_ok(None, sources)
    assert not evidence_ok("W", sources)


def test_ground_fields_rejects_fabrication():
    reports, _, incidents, _, _ = _fixture()
    source = reports[0].text
    fields = ground_fields({
        "hazard": {"value": "severe flood", "evidence": "Water rising fast", "confidence": 1.7},
        "location": {"value": "Atlantis", "evidence": "Atlantis", "confidence": 0.9},
    }, [source])
    assert list(fields) == BRIEF_FIELDS
    assert fields["hazard"] == {"value": "Water rising fast", "evidence": "Water rising fast",
                                "confidence": 1.0, "grounded": True}
    assert fields["location"] == {"value": None, "evidence": None,
                                  "confidence": 0.0, "grounded": False}
    assert all(fields[name]["value"] is None for name in BRIEF_FIELDS[2:])
    incident = incidents[0]
    row = Briefing(
        incident_id=incident.incident_id, event=incident.event,
        snapshot_time=incident.snapshot_time, cache_key=brief.cache_key(incident, "fake"),
        rep_report_ids=incident.report_ids[:1], fields=fields, sentences=[],
        briefing=brief.template_briefing(incident, fields), used_fallback=True,
        model_file="fake", timing={"extract_s": 0.0, "brief_s": 0.0,
                                   "prompt_tokens": 0.0, "completion_tokens": 0.0},
    )
    row.validate()


def test_split_tagged_sentences():
    assert split_tagged_sentences("Sentence one. [location, need] Sentence two [reason].") == [
        ("Sentence one.", ["location", "need"]), ("Sentence two.", ["reason"]),
    ]
    assert split_tagged_sentences("Sentence one [HaZaRd]. This has no tags here.") == [
        ("Sentence one.", ["hazard"]), ("This has no tags here.", []),
    ]
    assert split_tagged_sentences("This sentence has no tags.") == [
        ("This sentence has no tags.", []),
    ]


def test_location_quotes_need_geographic_context():
    def location(quote, sources):
        return ground_fields({"location": {"value": quote, "evidence": quote, "confidence": 1}}, sources)["location"]

    for quote, source in (("Harvey", "Hurricane Harvey flooded Houston."),
                          ("Harvey", "#HurricaneHarvey flooded Houston."),
                          ("Katrina", "Harvey is worse than Katrina.")):
        assert evidence_ok(quote, [source])  # Literal occurrence alone is insufficient.
        assert not location(quote, [source])["grounded"]
    assert location("Houston", ["Hurricane Harvey flooded Houston."])["grounded"]
    assert location("Galveston County", ["Flooding in Galveston County."])["grounded"]
    assert location("Harvey", ["Hurricane Harvey flooded Harvey, LA."])["grounded"]
    assert location("Harvey", ["  Hurricane   Harvey flooded   Harvey, LA."])["grounded"]
    assert location("Harvey", ["Hurricane Harvey arrived.", "Flooding in Harvey."])["grounded"]
    with patch.object(config, "DEMO_EVENT", "hurricane_harvey_2017"):
        assert not location("Harvey", ["#Harvey flooding."])["grounded"]
        assert location("Harvey", ["Harvey LA has flooding."])["grounded"]
        assert location("Harvey", ["Harvey,LA has flooding."])["grounded"]
        assert location("Houston", ["#Houston flooding #Harvey."])["grounded"]


def test_ranking_reason_does_not_establish_an_unmet_need():
    sentence = "The urgency is underscored by the single report and the need for immediate rescue and donation efforts."
    reasons = ["reports of injuries or deaths", "classifier confidence", "rescue and donation activity"]
    fields = empty_fields()
    assert check_sentence(sentence, ["reason"], fields, reasons, {"1"}) == (
        False, "need claim without grounded need")
    assert check_sentence("Ranked high because of rescue and donation activity.", ["reason"], fields, reasons, {"1"}) == (True, None)
    assert check_sentence("Ranked high because of urgent requests for help.", ["reason"], fields,
                          ["urgent requests for help"], set()) == (True, None)
    fields["need"] = {"value": "need rescue", "evidence": "need rescue", "grounded": True, "confidence": 1.0}
    assert check_sentence("People need rescue.", ["need"], fields, [], set()) == (True, None)
    assert check_sentence("People need rescue due to rescue and donation activity.", ["reason"], fields, reasons, set()) == (
        False, "need claim missing need tag")


def test_check_sentence_fixture():
    _, _, incidents, briefings, _ = _fixture()
    fixture_briefing = next(row for row in briefings if row.incident_id == "fixture_event-inc-1")
    incident = next(row for row in incidents if (
        row.incident_id == fixture_briefing.incident_id
        and row.snapshot_time == fixture_briefing.snapshot_time
    ))
    reasons = [FEATURE_PLAIN[key] for key, value in incident.shap_top if value > 0][:3]
    allowed = numbers_allowed(incident, fixture_briefing.fields)
    for sentence in fixture_briefing.sentences:
        assert check_sentence(sentence["text"], sentence["tags"], fixture_briefing.fields,
                              reasons, allowed) == (sentence["accepted"], sentence["reject_reason"])


def test_check_sentence_rules():
    fields = empty_fields()
    assert check_sentence("News here.", ["imaginary"], fields, [], set()) == (
        False, "unknown tag 'imaginary'")
    fields["people_affected"] = {"value": "people", "evidence": "people",
                                  "confidence": 1.0, "grounded": True}
    assert check_sentence("12 people need help.", ["people_affected"], fields, [], {"4"}) == (
        False, "unsupported number '12'")
    assert check_sentence("Ranked high because of 4 reports.", ["reason"], fields, [], {"4"}) == (
        True, None)


def test_template_matches_stub():
    _, _, incidents, _, _ = _fixture()
    for incident in _last_snapshot(incidents):
        fields = empty_fields()
        assert brief.template_briefing(incident, fields) == stubs.template_briefing_text(incident, fields)


def test_brief_incidents_without_model():
    _, _, incidents, _, _ = _fixture()
    original_path = config.GRANITE_PATH
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            config.GRANITE_PATH = Path(temp_dir) / "missing.gguf"
            rows = brief.brief_incidents(
                "fixture_event", _last_snapshot(incidents), top_n=3,
                src_dir=fixture_dir(), out_dir=Path(temp_dir),
            )
        finally:
            config.GRANITE_PATH = original_path
        assert len(rows) == 3
        saved = read_jsonl(Path(temp_dir) / config.BRIEFINGS_FILE, Briefing)
        assert len(saved) == 3
        assert [row.rank_adj for row in sorted(_last_snapshot(incidents), key=lambda i: i.rank_adj)[:3]] == [1, 2, 3]
        for row in rows:
            row.validate()
            assert row.model_file == "template"
            assert row.used_fallback
            assert row.fields == empty_fields()
            assert row.sentences == []
            incident = next(i for i in incidents if i.incident_id == row.incident_id and i.snapshot_time == row.snapshot_time)
            assert row.briefing == stubs.template_briefing_text(incident, empty_fields())


def test_fake_llm_pipeline_and_resume():
    _, _, incidents, _, _ = _fixture()
    target = [next(i for i in _last_snapshot(incidents) if i.incident_id == "fixture_event-inc-1")]
    fake = FakeLLM()
    with tempfile.TemporaryDirectory() as temp_dir:
        destination = Path(temp_dir)
        rows = brief.brief_incidents("fixture_event", target, top_n=1, llm=fake,
                                     src_dir=fixture_dir(), out_dir=destination)
        assert fake.json_calls == fake.text_calls == 1
        assert rows[0].fields["hazard"]["grounded"]
        assert rows[0].fields["location"]["value"] is None
        assert rows[0].sentences[0]["accepted"]
        assert not rows[0].sentences[1]["accepted"]
        assert not rows[0].used_fallback
        assert rows[0].timing == {"extract_s": 0.01, "brief_s": 0.02,
                                  "prompt_tokens": 22.0, "completion_tokens": 17.0}
        assert len(read_jsonl(destination / config.LLM_LOG_FILE)) == 2
        assert brief.brief_incidents("fixture_event", target, top_n=1, llm=fake,
                                     src_dir=fixture_dir(), out_dir=destination) == rows
        assert fake.json_calls == fake.text_calls == 1
        report_id = target[0].report_ids[0]
        predictions = brief.extract_for_eval("fixture_event", [report_id], llm=fake,
                                              src_dir=fixture_dir(), out_dir=destination)
        assert len(read_jsonl(predictions)) == 1
        brief.extract_for_eval("fixture_event", [report_id], llm=fake,
                               src_dir=fixture_dir(), out_dir=destination)
        assert fake.json_calls == 2


def test_cache_keeps_snapshot_metadata():
    _, _, incidents, _, _ = _fixture()
    same_members = [incident for incident in incidents if incident.incident_id == "fixture_event-inc-3"
                    and len(incident.report_ids) == 1][:2]
    assert len(same_members) == 2
    fake = FakeLLM()
    with tempfile.TemporaryDirectory() as temp_dir:
        for incident in same_members:
            rows = brief.brief_incidents("fixture_event", [incident], top_n=1, llm=fake,
                                         src_dir=fixture_dir(), out_dir=Path(temp_dir))
            assert rows[0].snapshot_time == incident.snapshot_time
        saved = read_jsonl(Path(temp_dir) / config.BRIEFINGS_FILE, Briefing)
        assert len(saved) == 2
        assert saved[0].cache_key == saved[1].cache_key


def test_cached_unsupported_need_is_rejected_without_model_calls():
    _, _, incidents, _, _ = _fixture()
    incident = replace(_last_snapshot(incidents)[0],
                       shap_top=[["share_rescue_volunteering_or_donation_effort", 1.0]])
    fake = FakeLLM()
    sentence = "The urgency is underscored by the single report and the need for immediate rescue and donation efforts."
    old = Briefing(incident_id=incident.incident_id, event=incident.event, snapshot_time=incident.snapshot_time,
                   cache_key=brief.cache_key(incident, fake.model_file), rep_report_ids=incident.report_ids[:1],
                   fields=empty_fields(), sentences=[dict(text=sentence, tags=["reason"], accepted=True, reject_reason=None)],
                   briefing=sentence, used_fallback=False, model_file=fake.model_file, timing=zero_timing())
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        write_jsonl(destination / config.BRIEFINGS_FILE, [old])
        result = brief.brief_incidents(incident.event, [incident], llm=fake,
                                      src_dir=fixture_dir(), out_dir=destination)[0]
        assert fake.json_calls == fake.text_calls == 0
        assert result.used_fallback and not result.sentences[0]["accepted"]
        assert result.sentences[0]["reject_reason"] == "need claim without grounded need"
        assert result.briefing == brief.template_briefing(incident, result.fields)
        assert read_jsonl(destination / config.BRIEFINGS_FILE, Briefing) == [result]


def test_extract_real_granite_granite():
    model_path = os.environ.get("SURGEDESK_GRANITE")
    if not model_path or not Path(model_path).is_file():
        skip("Set SURGEDESK_GRANITE to the local Granite GGUF file")
    reports, tagged, incidents, briefings, embeddings = _fixture()
    first = briefings[0]
    incident = next(i for i in incidents if i.incident_id == first.incident_id
                    and i.snapshot_time == first.snapshot_time)
    representatives = brief.select_representatives(
        incident, {r.report_id: r for r in reports}, {t.report_id: t for t in tagged}, embeddings,
    )
    model = llm.get_llm(model_path=Path(model_path))
    try:
        fields, timing = brief.extract_fields(model, representatives)
        assert list(fields) == BRIEF_FIELDS
        assert any(field["grounded"] for field in fields.values())
        record = Briefing(
            incident_id=incident.incident_id, event=incident.event,
            snapshot_time=incident.snapshot_time,
            cache_key=brief.cache_key(incident, model.model_file),
            rep_report_ids=[report.report_id for report in representatives],
            fields=fields, sentences=[], briefing=brief.template_briefing(incident, fields),
            used_fallback=True, model_file=model.model_file, timing=zero_timing(),
        )
        record.validate()
        print(f"Granite extract: {timing['seconds']:.3f} seconds")
    finally:
        llm.unload()

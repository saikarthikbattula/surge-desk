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


def test_phone_number_is_masked_before_extraction_and_briefing_fake_calls():
    phone = "713-555-0100"
    source = f"Please send a ventilator to Pine Clinic. Call {phone} for supplies."
    report = replace(_fixture()[0][0], text=source)
    incident = _last_snapshot(_fixture()[2])[0]

    class PrivateFake:
        def chat_json(self, system, user, schema, max_tokens):
            assert phone not in system + user
            assert "<phone>" in user
            return {"need": {"value": "Please send a ventilator",
                             "evidence": "Please send a ventilator", "confidence": 1}}, {}

        def chat_text(self, system, user, max_tokens):
            assert phone not in system + user
            assert "<phone>" in user
            return "Reports state: Please send a ventilator. [need]", {}

    fake = PrivateFake()
    fields, _ = brief.extract_fields(fake, [report])
    assert fields["need"]["grounded"]
    # Check the briefing boundary too, including a manually supplied old quote.
    fields["need"] = {"value": f"Please send a ventilator; call {phone}",
                      "evidence": f"Please send a ventilator; call {phone}",
                      "confidence": 1.0, "grounded": True}
    brief.write_briefing(fake, incident, fields)


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


def test_current_assertions_exclude_historical_and_speculative_counts():
    def fields(quote, sources, name="people_affected"):
        return ground_fields({name: {"evidence": quote, "confidence": 1}}, sources)[name]

    historical = "Harbor City in 1900 had a death toll of 8,000 or more. Today's storm could approach that number."
    with patch.object(config, "DEMO_EVENT", "hurricane_harvey_2017"):
        assert not fields("8,000 or more", [historical])["grounded"]
        assert not fields("8,000 or more", ["Harbor City's 1900 Death toll reached 8,000 or more."])["grounded"]
        assert not fields("Harbor City", [historical], "location")["grounded"]
        for qualifier in ("could", "might", "may", "would"):
            assert not fields("800 people", [f"The storm {qualifier} displace 800 people."])["grounded"]
        assert not fields("death toll is rising", ["The death toll is rising."])["grounded"]
        assert fields("First Responder", ["First Responder found dead after the storm."])["grounded"]
        assert fields("8,000 or more", [historical, "8,000 or more people evacuated today."])["grounded"]
        assert fields("2000 people", ["2000 people evacuated today."])["grounded"]
        assert fields("2000 people", ["On May 5, 2000 people evacuated."])["grounded"]
        assert fields("my grandma", ["My grandma is trapped upstairs."])["grounded"]
        assert fields("send a boat", ["Could you send a boat?"], "need")["grounded"]


def test_sentence_list_labels_and_whole_numbers():
    fields = empty_fields()
    fields["people_affected"] = {"value": "12,000 people", "evidence": "12,000 people",
                                  "confidence": 1, "grounded": True}
    allowed = numbers_allowed(_last_snapshot(_fixture()[2])[0], fields)
    assert "12000" in allowed and "12" not in allowed and "000" not in allowed
    parsed = split_tagged_sentences("2. Reports state 12,000 people,. [people_affected]")
    assert parsed == [("Reports state 12,000 people.", ["people_affected"])]
    assert check_sentence(*parsed[0], fields, [], allowed) == (True, None)
    assert check_sentence("12 people.", ["people_affected"], fields, [], allowed)[0] is False
    # Do not remove the factual count when there is no list punctuation.
    assert split_tagged_sentences("2 people evacuated. [people_affected]")[0][0].startswith("2 people")
    assert check_sentence("Ranked high because of classifier confidence and potential mass casualties.",
                          ["reason"], fields, ["classifier confidence"], allowed)[0] is False


def test_cached_historical_fields_are_removed_without_generation():
    reports, _, incidents, _, _ = _fixture()
    incident = _last_snapshot(incidents)[0]
    representative = replace(reports[0], report_id=incident.report_ids[0],
                             text="Harbor City in 1900 reported 8,000 dead. Today's storm could cause more deaths.")
    fields = empty_fields()
    fields["people_affected"] = {"value": "8,000", "evidence": "8,000", "confidence": 1, "grounded": True}
    text = "1. Reports state 8,000 people,."
    row = Briefing(incident_id=incident.incident_id, event=incident.event, snapshot_time=incident.snapshot_time,
                   cache_key=brief.cache_key(incident, "fake"), rep_report_ids=[representative.report_id],
                   fields=fields, sentences=[dict(text=text, tags=["people_affected"], accepted=True, reject_reason=None)],
                   briefing=text, used_fallback=False, model_file="fake", timing=zero_timing())
    with patch.object(config, "DEMO_EVENT", "hurricane_harvey_2017"), \
         patch.object(brief, "get_llm", side_effect=AssertionError("Review must not load Granite")):
        reviewed = brief.review_cached_briefings([row], [incident], [representative])
        assert reviewed[0].used_fallback
        assert not reviewed[0].fields["people_affected"]["grounded"]
        assert reviewed[0].sentences[0]["text"] == "Reports state 8,000 people."
        assert brief.review_cached_briefings([row], [replace(incident, report_ids=incident.report_ids[1:])],
                                             [representative]) == []


def test_cache_recheck_masks_old_phone_sentence_and_invalidates_old_evidence():
    import shutil

    reports, _, incidents, _, _ = _fixture()
    incident = _last_snapshot(incidents)[0]
    representative = incident.report_ids[0]
    phone = "713-555-0100"
    quote = f"Call {phone} for supplies"
    old_fields = empty_fields()
    old_fields["need"] = {"value": quote, "evidence": quote,
                          "confidence": 1.0, "grounded": True}
    sentence = f"Reports state: {quote}."
    old = Briefing(incident_id=incident.incident_id, event=incident.event,
                   snapshot_time=incident.snapshot_time,
                   cache_key=brief.cache_key(incident, "fake"),
                   rep_report_ids=[representative], fields=old_fields,
                   sentences=[{"text": sentence, "tags": ["need"],
                               "accepted": True, "reject_reason": None}],
                   briefing=sentence, used_fallback=False,
                   model_file="fake", timing=zero_timing())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_dir = root / "source"
        output_dir = root / "output"
        shutil.copytree(fixture_dir(), source_dir)
        changed = [replace(report, text=quote + ".") if report.report_id == representative else report
                   for report in reports]
        write_jsonl(source_dir / config.REPORTS_FILE, changed)
        output_dir.mkdir()
        write_jsonl(output_dir / config.BRIEFINGS_FILE, [old])
        fake = FakeLLM()
        result = brief.brief_incidents(incident.event, [incident], top_n=1, llm=fake,
                                      src_dir=source_dir, out_dir=output_dir)[0]
        assert fake.json_calls == fake.text_calls == 0
        assert not result.fields["need"]["grounded"]
        assert result.fields["need"]["evidence"] is None
        assert result.used_fallback
        assert result.sentences[0]["text"] == "Reports state: Call <phone> for supplies."
        assert not result.sentences[0]["accepted"]
        saved_text = (output_dir / config.BRIEFINGS_FILE).read_text()
        assert phone not in saved_text


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


def test_brief_prompt_omits_unknown_fields_and_checks_reason_values():
    from surgedesk.contracts import feature_reason
    incident = _last_snapshot(_fixture()[2])[0]
    values = dict(incident.features)
    values.update(share_current_assistance=1.0, share_historical_or_speculative=0.0, share_casualty_only=0.0)
    values["share_rescue_volunteering_or_donation_effort"] = 0.0
    incident = replace(incident, features=values, shap_top=[["share_rescue_volunteering_or_donation_effort", 1.0]])
    fields = empty_fields()
    fields["people_affected"] = dict(value="First Responder", evidence="First Responder", confidence=1.0, grounded=True)
    class QuotesOnly:
        def chat_text(self, system, user, max_tokens):
            assert "people_affected: First Responder" in user
            assert "hazard:" not in user and "location:" not in user and "need:" not in user
            return "Reports state: First Responder. [people_affected]", dict(seconds=0, prompt_tokens=0, completion_tokens=0)
    sentences, _ = brief.write_briefing(QuotesOnly(), incident, fields)
    assert sentences[0]["accepted"]
    reason = feature_reason("share_rescue_volunteering_or_donation_effort", values)
    assert "0%" in reason
    allowed = numbers_allowed(incident, fields)
    assert check_sentence(f"Ranked high because of {reason}.", ["reason"], fields, [reason], allowed)[0]
    assert not check_sentence(f"Ranked high because of {reason.replace('0%', '100%')}.",
                              ["reason"], fields, [reason], allowed | {"100"})[0]


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


def test_no_grounded_facts_skip_sentence_generation():
    class NoFacts(FakeLLM):
        def chat_json(self, *args):
            self.json_calls += 1
            return None, dict(seconds=0.01, prompt_tokens=10, completion_tokens=0, ok=False, attempts=1)

        def chat_text(self, *args):
            raise AssertionError("No verified facts: do not spend another generation call")

    fake = NoFacts()
    incident = _last_snapshot(_fixture()[2])[0]
    with tempfile.TemporaryDirectory() as directory:
        result = brief.brief_incidents(incident.event, [incident], llm=fake,
                                      src_dir=fixture_dir(), out_dir=Path(directory))[0]
        assert fake.json_calls == 1 and result.used_fallback
        assert result.sentences == [] and result.timing["brief_s"] == 0.0
        assert result.briefing == brief.template_briefing(incident, empty_fields())


def test_notebook_fast_review_budget_and_cached_only_loading():
    import shutil
    from contextlib import redirect_stdout
    from io import StringIO

    notebook = json.loads((config.ROOT / "notebooks/03_brief.ipynb").read_text())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / config.FIXTURE_EVENT
        shutil.copytree(fixture_dir(), source)
        labels = root / "labels"
        labels.mkdir()
        # A present label file must not turn Run All into an extraction batch.
        (labels / "hand_labels.csv").write_text("report_id\n" + _fixture()[0][0].report_id + "\n")
        model_path = root / "fake"  # Matches OfflineFakeLLM's model_file for the cache-only rerun.
        with patch.object(config, "OUTPUTS", root), patch.object(config, "LABELS_DIR", labels), \
             patch.object(config, "DEMO_EVENT", config.FIXTURE_EVENT), patch.object(config, "TOP_N_BRIEF", 10), \
             patch.object(config, "GRANITE_PATH", model_path), \
             patch.dict(os.environ, {"SURGEDESK_TOP_N": "10", "SURGEDESK_GRANITE": str(model_path)}), \
             patch.object(brief, "extract_for_eval", side_effect=AssertionError("No automatic label batch")), \
             patch.object(llm, "get_llm", side_effect=AssertionError("No model load for fake/cached review")):
            for cached in (False, True):
                if cached:
                    model_path.write_bytes(b"placeholder; must never be loaded")
                scope = {}
                with patch.object(brief, "extract_fields", wraps=brief.extract_fields) as extraction, redirect_stdout(StringIO()):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), "03_brief.ipynb", "exec"), scope)
                assert scope["RUN_SPEED_PROBE"] is False and scope["bench_rows"] == []
                assert scope["RUN_EXTRACTION_EVAL"] is False
                assert len(scope["briefings"]) == 3
                assert extraction.call_count == (0 if cached else 3)
                assert len(scope["new_calls"]) == (0 if cached else 3)
                assert {row["status"] for row in scope["quality_rows"]} == {
                    "cache rechecked" if cached else "generated this run"}
                assert sum(row["field_total"] for row in scope["quality_rows"]) == 15


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


def test_synthetic_extraction_examples_and_short_grounded_briefings():
    # Invented reports only. Fake success checks plumbing, not Granite quality.
    cases = [
        ("Please send a ventilator to Pine Clinic.",
         {"location": "Pine Clinic", "need": "Please send a ventilator"}),
        ("We need blankets at Cedar Hall.",
         {"location": "Cedar Hall", "need": "need blankets"}),
        ("Donate to our storm relief fund.", {}),
        ("Officials report 6 people died.", {"people_affected": "6 people died"}),
        ("In 1900, 8,000 people died; this storm could be worse.", {}),
        ("More updates soon.", {}),
    ]
    base_report = _fixture()[0][0]
    incident = _last_snapshot(_fixture()[2])[0]
    for source, expected in cases:
        class LiteralFake:
            def chat_json(self, system, user, schema, max_tokens):
                assert user.split("Reports:\n", 1)[1].startswith("[1] " + source)
                return {name: dict(value=quote, evidence=quote, confidence=1)
                        for name, quote in expected.items()}, dict(seconds=0, attempts=1)

            def chat_text(self, system, user, max_tokens):
                assert "number of reports:" not in user
                assert "explanation phrases:" not in user
                assert "reason" not in system
                assert "rank" not in user.lower()
                for name in BRIEF_FIELDS:
                    assert (name + ":" in user) == (name in expected)
                name = next(iter(expected))
                return f'Reports state: {expected[name]}. [{name}]', dict(seconds=0, attempts=1)

        model = LiteralFake()
        fields, _ = brief.extract_fields(model, [replace(base_report, text=source)])
        assert {name: item["value"] for name, item in fields.items() if item["grounded"]} == expected
        sentences, timing = brief.write_briefing(model, incident, fields)
        if expected:
            assert len(sentences) == 1 and sentences[0]["accepted"]
        else:
            assert sentences == [] and timing["attempts"] == 0


def test_prompt_examples_cannot_supply_extraction_evidence():
    class ContaminatedFake:
        def chat_json(self, *args):
            # Both quotes occur in instructions/examples, neither in Reports.
            return {"need": dict(evidence="need blankets"),
                    "location": dict(evidence="Pine Clinic")}, {}

    fields, _ = brief.extract_fields(ContaminatedFake(), [replace(_fixture()[0][0], text="More updates soon.")])
    assert fields == empty_fields()


def test_chat_json_malformed_output_stops_after_two_attempts():
    class MalformedModel:
        def __init__(self):
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            assert set(kwargs) == {"messages", "max_tokens", "temperature"}
            assert kwargs["max_tokens"] == config.EXTRACT_MAX_TOKENS
            self.calls += 1
            return {"choices": [{"message": {"content": "{broken"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    model = llm.GraniteLLM.__new__(llm.GraniteLLM)
    model.model, model.log_path = MalformedModel(), None
    raw, timing = model.chat_json("system", "report", brief.EXTRACTION_SCHEMA, config.EXTRACT_MAX_TOKENS)
    assert raw is None and not timing["ok"]
    assert timing["attempts"] == model.model.calls == 2
    assert timing["prompt_tokens"] == 14 and timing["completion_tokens"] == 6


def test_adapter_error_logs_do_not_echo_backend_text():
    class FailingModel:
        def create_chat_completion(self, **kwargs):
            raise ValueError("PRIVATE_SYNTHETIC_CONTACT should never reach a log")

    model = llm.GraniteLLM.__new__(llm.GraniteLLM)
    model.model = FailingModel()
    with tempfile.TemporaryDirectory() as directory:
        model.log_path = Path(directory) / "calls.jsonl"
        model.chat_json("system", "report", brief.EXTRACTION_SCHEMA, config.EXTRACT_MAX_TOKENS)
        model.chat_text("system", "report", config.BRIEF_MAX_TOKENS)
        records = read_jsonl(model.log_path)
        assert len(records) == 2
        assert all(row["error"] == "ValueError" for row in records)
        assert "PRIVATE_SYNTHETIC_CONTACT" not in model.log_path.read_text()


def test_extra_hazard_urgency_vulnerability_and_need_claims_are_rejected():
    fields = ground_fields({"location": dict(evidence="Pine Clinic"),
                            "need": dict(evidence="Please send a ventilator")},
                           ["Please send a ventilator to Pine Clinic."])
    cases = [
        ("Pine Clinic has a fire.", ["location"], "unsupported hazard claim 'fire'"),
        ("Pine Clinic is in immediate danger.", ["location"], "unsupported urgency claim 'immediate'"),
        ("Pine Clinic has elderly patients.", ["location"], "unsupported vulnerability claim 'elderly'"),
        ("Please send a ventilator and blankets.", ["need"], "unsupported need claim 'blankets'"),
    ]
    for sentence, tags, reason in cases:
        assert check_sentence(sentence, tags, fields, ["urgent requests for help"], set()) == (False, reason)
    assert check_sentence("Reports state: Please send a ventilator.", ["need"], fields, [], set()) == (True, None)
    fields = empty_fields()
    assert not check_sentence("Ranked high because of classifier confidence and fire.",
                              ["reason"], fields, ["classifier confidence"], set())[0]


def test_direct_briefing_skips_no_grounded_fields():
    class NoText:
        def chat_text(self, *args):
            raise AssertionError("Empty fields must not call the model")

    sentences, timing = brief.write_briefing(NoText(), _last_snapshot(_fixture()[2])[0], empty_fields())
    assert sentences == [] and timing["seconds"] == timing["attempts"] == 0


def test_pipeline_cache_rejects_foreign_or_missing_representatives():
    reports, _, incidents, _, _ = _fixture()
    incident = _last_snapshot(incidents)[0]
    foreign = next(report.report_id for report in reports if report.report_id not in incident.report_ids)
    for representative in [foreign, "missing-report"]:
        fake = FakeLLM()
        old = Briefing(incident_id=incident.incident_id, event=incident.event,
                       snapshot_time=incident.snapshot_time, cache_key=brief.cache_key(incident, "fake"),
                       rep_report_ids=[representative], fields=empty_fields(),
                       sentences=[], briefing="Old template", used_fallback=True,
                       model_file="fake", timing=zero_timing())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_jsonl(destination / config.BRIEFINGS_FILE, [old])
            result = brief.brief_incidents(incident.event, [incident], llm=fake,
                                          src_dir=fixture_dir(), out_dir=destination)[0]
            assert fake.json_calls == 1
            assert result.rep_report_ids and set(result.rep_report_ids) <= set(incident.report_ids)


def test_cache_recheck_retains_original_recorded_timings():
    incident = _last_snapshot(_fixture()[2])[0]
    timing = dict(extract_s=12.5, brief_s=3.25, prompt_tokens=123., completion_tokens=45.)
    old = Briefing(incident_id=incident.incident_id, event=incident.event,
                   snapshot_time=incident.snapshot_time, cache_key=brief.cache_key(incident, "fake"),
                   rep_report_ids=incident.report_ids[:1], fields=empty_fields(), sentences=[],
                   briefing="Old template", used_fallback=True, model_file="fake", timing=timing)
    fake = FakeLLM()
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        write_jsonl(destination / config.BRIEFINGS_FILE, [old])
        result = brief.brief_incidents(incident.event, [incident], llm=fake,
                                      src_dir=fixture_dir(), out_dir=destination)[0]
        assert result.timing == timing
        assert fake.json_calls == fake.text_calls == 0

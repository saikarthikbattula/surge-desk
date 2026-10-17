"""Foundation tests (owner: foundation). Run with: python -m surgedesk.selftest -k foundation"""
from __future__ import annotations

from datetime import datetime, timezone

from surgedesk import config
from surgedesk.contracts import (
    CATEGORIES, FEATURE_KEYS, Briefing, ContractError, Incident, Report, Tagged, normalize_label,
)
from surgedesk.stubs import find_terms, reference_features, snapshot_times
from surgedesk.testing import assert_close, load_fixture
from surgedesk.textutil import (
    datetime_to_tweet_id, mask_privacy, parse_iso, to_iso, tweet_id_to_datetime,
)


def test_tweet_id_roundtrip():
    dt = datetime(2017, 8, 27, 10, 0, tzinfo=timezone.utc)
    assert tweet_id_to_datetime(datetime_to_tweet_id(dt, 5)) == dt


def test_iso_roundtrip():
    s = "2017-08-27T10:00:00Z"
    assert to_iso(parse_iso(s)) == s


def test_mask_privacy():
    assert mask_privacy("hi @bob see http://t.co/x1") == "hi @user see <url>"


def test_normalize_label_variants():
    assert normalize_label("Requests or urgent needs") == "requests_or_urgent_needs"
    assert normalize_label("dont_know_cant_judge") == "dont_know_cant_judge"
    assert normalize_label("Don't know or can't judge") == "dont_know_cant_judge"
    for c in CATEGORIES:
        assert normalize_label(c) == c
    try:
        normalize_label("weather")
        raise AssertionError("expected ContractError")
    except ContractError:
        pass


def test_find_terms_whole_word():
    assert find_terms("Grandma is TRAPPED with kids", ["kids", "trapped", "grandma"]) == ["grandma", "trapped", "kids"]
    assert find_terms("kidsfest", ["kids"]) == []


def test_contract_rejects_unknown_keys():
    r = load_fixture("reports.jsonl", Report)[0].to_dict()
    r["extra"] = 1
    try:
        Report.from_dict(r)
        raise AssertionError("expected ContractError")
    except ContractError:
        pass


def test_fixture_features_match_reference():
    reports = {r.report_id: r for r in load_fixture("reports.jsonl", Report)}
    tags = {t.report_id: t for t in load_fixture("tagged.jsonl", Tagged)}
    for inc in load_fixture("incidents.jsonl", Incident):
        f = reference_features([reports[i] for i in inc.report_ids], [tags[i] for i in inc.report_ids],
                               parse_iso(inc.snapshot_time), inc.svi)
        assert list(f) == FEATURE_KEYS
        for k in FEATURE_KEYS:
            assert_close(f[k], inc.features[k], msg=f"{inc.incident_id} {k}")


def test_snapshot_grid():
    s = parse_iso("2017-08-27T10:07:00Z")
    e = parse_iso("2017-08-27T11:01:00Z")
    assert [to_iso(t) for t in snapshot_times(s, e, 30)] == [
        "2017-08-27T10:00:00Z", "2017-08-27T10:30:00Z", "2017-08-27T11:00:00Z", "2017-08-27T11:30:00Z"]


def test_briefing_fixture_evidence_is_substring():
    reports = {r.report_id: r.text.lower() for r in load_fixture("reports.jsonl", Report)}
    for b in load_fixture("briefings.jsonl", Briefing):
        sources = [reports[r] for r in b.rep_report_ids]
        for name, fv in b.fields.items():
            if fv["evidence"]:
                assert any(fv["evidence"].lower() in s for s in sources), (b.incident_id, name)


def test_states_for_event():
    assert config.states_for_event("hurricane_harvey_2017") == ["TX", "LA"]
    assert config.states_for_event("kerala_floods_2018") == []
    assert config.DEMO_EVENT not in config.train_events([config.DEMO_EVENT, "hurricane_irma_2017"])

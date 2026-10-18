"""Offline acceptance checks for incident grouping, ranking, and adjustment."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from surgedesk import baselines, config, fairness, fileio, rank
from surgedesk.contracts import FEATURE_KEYS, RANK_FEATURE_KEYS, Incident, Report, Tagged
from surgedesk.features import SNAPSHOT_COLUMNS, assistance_signals, build_snapshots, feature_definition, snapshot_features
from surgedesk.group import GroupState, group_event, load_stage1
from surgedesk.testing import skip, load_fixture, fixture_dir
from surgedesk.textutil import parse_iso, to_iso


def _lookup():
    values = {i.county_fips: i.svi for i in load_fixture(config.INCIDENTS_FILE, Incident)}
    return values.get


def _last():
    incidents = load_fixture(config.INCIDENTS_FILE, Incident)
    last = max(i.snapshot_time for i in incidents)
    return [i for i in incidents if i.snapshot_time == last]


def _state(event=config.FIXTURE_EVENT):
    reports, tags, emb = load_stage1(event, fixture_dir(event))
    state = GroupState(event)
    for r, t, vec in zip(reports, tags, emb):
        state.add(r, t, vec)
    return state, reports, tags, emb


def _train(out):
    try:
        import xgboost  # noqa: F401
    except ImportError:
        skip("xgboost not installed")
    path = out / config.RANKER_PATH.name
    meta = rank.train_ranker([config.FIXTURE_TRAIN_EVENT], svi_lookup=lambda f: None,
                             outputs_root=config.FIXTURES / "outputs", ranker_path=path,
                             allow_fixture=True)
    return meta, rank.Ranker(path)


def test_grouping_matches_fixture():
    with TemporaryDirectory() as tmp:
        for event in (config.FIXTURE_EVENT, config.FIXTURE_TRAIN_EVENT):
            out = Path(tmp) / event
            state = group_event(event, src_dir=fixture_dir(event), out_dir=out)
            expected = load_fixture("groups_expected.jsonl", event=event)
            assert fileio.read_jsonl(out / "groups.jsonl") == expected
            assert list(state.assignments) == [r.report_id for r, t in zip(*load_stage1(event, fixture_dir(event))[:2]) if t.is_actionable]
            stats = fileio.read_json(out / "group_stats.json")
            assert stats["n_incidents"] == len(expected)
            assert sum(int(size) * count for size, count in stats["size_histogram"].items()) == stats["n_actionable"]


def test_snapshots_match_fixture():
    state, reports, _, _ = _state()
    with TemporaryDirectory() as tmp:
        rows = build_snapshots(config.FIXTURE_EVENT, state, reports, _lookup(), Path(tmp))
        assert list(rows.columns) == SNAPSHOT_COLUMNS and rows.columns.is_unique
        expected = {(i.incident_id, i.snapshot_time): i for i in load_fixture(config.INCIDENTS_FILE, Incident)}
        assert set(zip(rows.incident_id, rows.snapshot_time)) == set(expected)
        for row in rows.to_dict("records"):
            inc = expected[(row["incident_id"], row["snapshot_time"])]
            for key in FEATURE_KEYS:
                assert abs(row[key] - inc.features[key]) <= 1e-6, key
            assert (None if pd.isna(row["place"]) else row["place"]) == inc.place
            assert (None if pd.isna(row["county_fips"]) else row["county_fips"]) == inc.county_fips
            assert row["urgent_proxy"] == inc.urgent_proxy


def test_open_rule():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    first = parse_iso(reports[0].timestamp)
    later = first + timedelta(hours=config.WINDOW_HOURS * 2)
    state = GroupState(config.FIXTURE_EVENT)
    state.add(reports[0], tags[0], emb[0])
    second = replace(reports[0], report_id=reports[1].report_id, timestamp=to_iso(later))
    state.add(second, replace(tags[0], report_id=second.report_id), emb[0])
    assert len(state.incidents) == 2
    assert state.open_incidents(first - timedelta(microseconds=1)) == []
    assert len(state.open_incidents(first + timedelta(hours=config.WINDOW_HOURS))) == 1
    assert state.open_incidents(first + timedelta(hours=config.WINDOW_HOURS, microseconds=1)) == []
    assert [i.incident_id for i in state.open_incidents(later)] == [list(state.incidents)[-1]]


def test_different_places_never_merge():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    state = GroupState(config.FIXTURE_EVENT)
    state.add(reports[0], tags[0], emb[0])
    state.add(reports[1], tags[1], emb[0])
    assert tags[0].county_fips != tags[1].county_fips
    assert len(state.incidents) == 2


def test_grouping_similarity_tie_uses_recency():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    state = GroupState(config.FIXTURE_EVENT)
    first = state.add(reports[0], tags[0], emb[0])
    recent = state.add(reports[1], tags[1], emb[0])
    assert first != recent
    unlocated = replace(tags[2], places=[], county_fips=None)
    assert state.add(reports[2], unlocated, emb[0]) == recent
    assert next(iter(state.incidents.values())).sum_vec.dtype == np.float64
    assert next(iter(state.incidents.values())).centroid.dtype == np.float32


def test_stage1_alignment_rejected():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        fileio.write_jsonl(out / config.REPORTS_FILE, reports)
        fileio.write_jsonl(out / config.TAGGED_FILE, [replace(tags[0], emb_row=1), *tags[1:]])
        fileio.save_embeddings(out / config.EMBEDDINGS_FILE, emb)
        try:
            load_stage1(config.FIXTURE_EVENT, out)
        except AssertionError:
            pass
        else:
            raise AssertionError("Misaligned embeddings must be rejected")


def test_fairness_adjust_and_apply():
    assert fairness.adjust(0.5, 0.8, 1.0) == 0.9
    assert fairness.adjust(0.9, 1.0, 1.0) == 1.0
    assert fairness.adjust(0.5, None, 1.0) == 0.5
    incidents = _last()
    before = [i.to_dict() for i in incidents]
    adjusted = fairness.apply(incidents, 1.0)
    assert [i.rank_adj for i in adjusted] == list(range(1, len(incidents) + 1))
    assert [i.to_dict() for i in incidents] == before
    assert all(a is not b for a in adjusted for b in incidents)
    assert all(a.score_adj >= b.score_adj for a, b in zip(adjusted, adjusted[1:]))
    table = fairness.fairness_table(incidents, k=len(incidents) + 1)
    assert (table.jaccard_vs_lam0 == 1.0).all()
    assert pd.isna(fairness.fairness_table([replace(i, urgent_proxy=None) for i in incidents]).precision_at_k).all()


def test_baselines_fixture():
    incidents = _last()
    texts = {r.report_id: r.text for r in load_fixture(config.REPORTS_FILE, Report)}
    orders = [baselines.order_fifo(incidents), baselines.order_volume(incidents),
              baselines.order_keyword(incidents, texts), baselines.order_surgedesk(incidents)]
    assert orders[0][0] == f"{config.FIXTURE_EVENT}-inc-1"
    for order in orders:
        assert sorted(order) == sorted(i.incident_id for i in incidents)


def test_train_ranker_fixture():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        meta, model = _train(out)
        state, reports, _, _ = _state()
        rows = build_snapshots(config.FIXTURE_EVENT, state, reports, _lookup(), out)
        scores, explanations = model.score(rows), model.explain(rows)
        assert len(scores) == len(rows) and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        assert len(explanations) == len(rows)
        for pairs in explanations:
            assert len(pairs) <= 5
            assert all(len(pair) == 2 and pair[0] in FEATURE_KEYS and np.isfinite(pair[1]) for pair in pairs)
            assert [abs(p[1]) for p in pairs] == sorted([abs(p[1]) for p in pairs], reverse=True)
        assert meta["feature_keys"] == FEATURE_KEYS
        assert meta["train_events"] == [config.FIXTURE_TRAIN_EVENT]
        assert meta["params"]["n_jobs"] == config.N_THREADS
        assert len(model.score(rows.iloc[:0])) == 0 and model.explain(rows.iloc[:0]) == []


def test_rank_event_fixture():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        _, model = _train(out)
        incidents = rank.rank_event(config.FIXTURE_EVENT, force=True, svi_lookup=_lookup(),
                                    src_dir=fixture_dir(), out_dir=out, ranker=model)
        assert fileio.validate_file(out / config.INCIDENTS_FILE) == len(incidents)
        for snap in {i.snapshot_time for i in incidents}:
            subset = [i for i in incidents if i.snapshot_time == snap]
            assert [i.rank for i in subset] == list(range(1, len(subset) + 1))
            assert [i.rank_adj for i in subset] == [i.rank for i in subset]
        assert rank.rank_event(config.FIXTURE_EVENT, out_dir=out) == incidents
        state, _, _, _ = _state()
        now = parse_iso(rank.pick_demo_snapshot(incidents))
        live = rank.rank_now(state, now, model, _lookup())
        assert live == [i for i in incidents if i.snapshot_time == to_iso(now)]
        assert rank.rank_now(state, now - timedelta(days=config.WINDOW_HOURS), model, _lookup()) == []


def test_no_gold_in_features():
    state, reports, tags, emb = _state()
    unlabeled = [replace(r, gold_label=None) for r in reports]
    other = GroupState(config.FIXTURE_EVENT)
    for r, t, vec in zip(unlabeled, tags, emb):
        other.add(r, t, vec)
    with TemporaryDirectory() as tmp:
        a = build_snapshots(config.FIXTURE_EVENT, state, reports, _lookup(), Path(tmp))
        b = build_snapshots(config.FIXTURE_EVENT, other, unlabeled, _lookup(), Path(tmp))
        pd.testing.assert_frame_equal(a[FEATURE_KEYS], b[FEATURE_KEYS])
        assert b.urgent_proxy.isna().all()
        inc = next(iter(state.incidents.values()))
        assert snapshot_features(inc, inc.first_seen - timedelta(microseconds=1), _lookup()) is None
        assert snapshot_features(inc, inc.last_seen, _lookup()) == snapshot_features(other.incidents[inc.incident_id], inc.last_seen, _lookup())


def test_demo_training_guard_and_snapshot_override():
    try:
        rank.train_ranker([config.DEMO_EVENT])
    except AssertionError:
        pass
    else:
        raise AssertionError("Demo training must be rejected")
    with patch.dict("os.environ", {"SURGEDESK_SNAPSHOT": _last()[0].snapshot_time}):
        assert rank.pick_demo_snapshot([]) == _last()[0].snapshot_time


def test_assistance_cues_reject_history_forecasts_offers_and_resolved_needs():
    source = load_fixture(config.REPORTS_FILE, Report)[0]
    positive = ["Please send a boat, a family is trapped upstairs.", "We need drinking water.",
                "Two residents are stranded on a roof.", "Police searching for a missing child.",
                "We are not trapped but need food.", "Could someone help my neighbor?", "Volunteers needed at the shelter.",
                "SOS!", "Help!", "We need assistance.", "We need an ambulance.", "Anyone have a boat near the school?",
                "My neighbors need to be rescued.", "We are not rescued and need help.", "We are not rescued yet."]
    negative = ["A responder found dead after the storm.", "In 1900 the storm left families stranded.",
                "The storm could cause people to become trapped.", "We don't need water.",
                "Family rescued and safe now.", "We offer water and food to volunteers.",
                "There is no rescue needed.", "The death toll is rising.", "Missing child found safe."]
    for text in positive:
        report = replace(source, text=text, gold_label=None)
        assert assistance_signals(report)["current_assistance"], text
        assert assistance_signals(replace(report, gold_label="injured_or_dead_people")) == assistance_signals(report)
    for text in negative:
        assert not assistance_signals(replace(source, text=text))["current_assistance"], text


def test_assistance_features_have_no_future_or_gold_leakage():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    event = "operational_test_2017"
    earlier = replace(reports[0], event=event, text="Officials report six deaths.", gold_label="injured_or_dead_people")
    later = replace(reports[1], event=event, text="Please send rescue boats.", gold_label="requests_or_urgent_needs")
    state = GroupState(event)
    tag = replace(tags[0], report_id=earlier.report_id, places=[], county_fips=None)
    state.add(earlier, tag, emb[0])
    from types import SimpleNamespace
    with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
        before_later = snapshot_features(next(iter(state.incidents.values())), parse_iso(earlier.timestamp), lambda f: None)
        model = SimpleNamespace(meta={"objective": "assistance_v1", "train_events": ["training_event"],
                                      "feature_definition": feature_definition(event)},
                                score=lambda rows: rows["share_current_assistance"].to_numpy(),
                                explain=lambda rows: [[] for _ in range(len(rows))])
        earlier_ranked = rank.rank_now(state, parse_iso(earlier.timestamp), model, lambda f: None)
    state.add(later, replace(tag, report_id=later.report_id), emb[0])
    incident = next(iter(state.incidents.values()))
    with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
        before = snapshot_features(incident, parse_iso(earlier.timestamp), lambda f: None)
        after = snapshot_features(incident, parse_iso(later.timestamp), lambda f: None)
        assert list(before) == RANK_FEATURE_KEYS
        assert before["share_current_assistance"] == 0 and after["share_current_assistance"] == 0.5
        assert before["share_casualty_only"] == 1
        # Adding a later member cannot retroactively change an earlier snapshot.
        assert before == before_later
        assert rank.rank_now(state, parse_iso(earlier.timestamp), model, lambda f: None) == earlier_ranked
        incident.reports = [replace(r, gold_label=None) for r in incident.reports]
        assert snapshot_features(incident, parse_iso(later.timestamp), lambda f: None) == after
        from surgedesk.stubs import reference_urgent_proxy
        assert reference_urgent_proxy([earlier]) == 0
        assert reference_urgent_proxy([later]) == 1
        assert reference_urgent_proxy([replace(later, gold_label=None)]) is None


def _require_coordinated_cue_policy():
    if not any("(?P<resource>" in p for p in config.ASSISTANCE_PATTERNS):
        skip("Specific-need policy awaits Person 1/Agent 4 shared config patch; do not deploy the owned-files PR alone")


def test_specific_assistance_pairs_with_coordinated_policy():
    _require_coordinated_cue_policy()
    source = load_fixture(config.REPORTS_FILE, Report)[0]
    positive = [
        # Concrete resources do not imply a confirmed life-threatening rescue.
        "Urgently requesting personal-care goods, clothes, bedding, food and water.",
        "Please donate blankets and diapers to the neighborhood collection team.",
        "An elderly resident needs a nebulizer.", "My mother needs oxygen.",
        "My friend urgently needs help distributing fuel in Lakeside.",
        "The rescue team reportedly needs food and gas; where can we send them?",
        "The shelter reports it is running out of food.",
        "We need shelter tonight.", "WE NEED SHELTER TONIGHT!",
        "We need shelter if you can help.", "We need water and don't need food.",
        "We need water and don’t need food.",
        "A family needs help.", "My neighbor needs assistance.",
        "We need medical attention.", "One injured person requires aid.",
        "We need drinking water.", "Water is urgently needed at the depot.",
        "Anyone have a boat near the school?", "Volunteers needed at the shelter.",
        "My neighbors need to be rescued.", "We are not rescued yet.",
        "Two residents are stranded upstairs.", "A child is missing near the park.",
        "Police searching for a missing child.", "SOS!", "Help us!",
        "We don't need food but need water.",
        "We don't need food and we need insulin.",
        "A family was rescued and we need an ambulance for another resident.",
        "The family was rescued and needs medication.",
        "CNN interviewed a rescued resident; my father needs oxygen.",
        "Please donate if you can; we need insulin.",
        "If you need shelter, go to the center; we need a boat for our neighbor.",
        "We need food &amp; water.", "We need blankets &#38; diapers!",
        "We need food &AMP; water.",
        "We don&apos;t need water; my child needs insulin.",
        "Last year families were stranded; we need rescue boats now.",
        "They predict floods tomorrow; a family is trapped now.",
        "A rescued woman needs insulin for tonight.",
    ]
    negative = [
        "Please donate and help people in need.",
        "Find out how to help families after the storm.",
        "Thousands of students were affected and need help.",
        "People need help, not more interviews of a rescued woman.",
        "We are not an intake center; if you need shelter, go to the civic center.",
        "Donate today because people need help.",
        "Join our general medical fundraiser to help people.",
        "Fundraising for families who need medical care.",
        "If you need shelter tonight, visit the school.",
        "If we need shelter tomorrow, we will call.",
        "Whenever anyone needs water, call the center.",
        "Here are food and gas donation publicity photos.",
        "A rescued woman described her interview and the food she received.",
        "Family rescued and safe now.", "They were rescued and no longer need help.",
        "We don't need water.", "We do not need a nebulizer.",
        "We don&apos;t need food &amp; water.",
        "The missing child was found safe.", "Nobody is trapped here; no rescue needed.",
        "They might need medicine tomorrow.", "The storm could leave families stranded.",
        "Last year we needed water.", "In 1912 families were stranded.",
        "We offer water and food to volunteers.", "We need your help.",
        "People need help.", "A large population needs assistance.",
        "Medicine, boats, water, shelter.", "Please help the hurricane campaign.",
        "The community's need for help is huge.",
    ]
    for expected, examples in ((True, positive), (False, negative)):
        for text in examples:
            report = replace(source, text=text, gold_label=None)
            signals = assistance_signals(report)
            assert signals["current_assistance"] == expected, (expected, text, signals)
            assert assistance_signals(replace(report, gold_label="requests_or_urgent_needs")) == signals
            assert bool(signals["evidence"]) == expected


def test_html_analysis_preserves_source_evidence_and_stored_reports():
    source = load_fixture(config.REPORTS_FILE, Report)[0]
    from surgedesk.textutil import mask_privacy
    for text, evidence in (
        ("We need food &amp; water.", "We need food &amp; water"),
        ("We need water&#33; No further update.", "We need water"),
        ("@sample We need food &amp; water; https://example.test", "@user We need food &amp; water"),
        ("We don&apos;t need food; we need water.", "we need water"),
    ):
        report = replace(source, text=text)
        signals = assistance_signals(report)
        assert signals["evidence"] == [evidence], signals
        assert evidence in mask_privacy(text)
        assert report.text == text


def test_assistance_feature_policy_invalidates_models_and_memory_cache():
    from types import SimpleNamespace
    event = "operational_test_2017"
    source = replace(load_fixture(config.REPORTS_FILE, Report)[0], event=event, text="We need water.")
    with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
        definition = feature_definition(event)
        model = SimpleNamespace(meta={"objective": "assistance_v1", "train_events": ["training_event"],
                                      "feature_definition": definition})
        rank._check_model(model, event)
        assert assistance_signals(source)["current_assistance"]
        with patch.object(config, "ASSISTANCE_PATTERNS", ()):
            assert not assistance_signals(source)["current_assistance"]
            assert feature_definition(event) != definition
            try:
                rank._check_model(model, event)
            except ValueError as exc:
                assert "Retrain" in str(exc)
            else:
                raise AssertionError("Same-schema old feature semantics must require retraining")
        assert assistance_signals(source)["current_assistance"]
        del model.meta["feature_definition"]
        try:
            rank._check_model(model, event)
        except ValueError as exc:
            assert "Retrain" in str(exc)
        else:
            raise AssertionError("Pre-fingerprint assistance model must require retraining")
        assert feature_definition(config.FIXTURE_EVENT) is None


def test_old_assistance_cues_expire_while_new_context_keeps_incident_open():
    reports, tags, emb = load_stage1(config.FIXTURE_EVENT, fixture_dir())
    event, start = "operational_test_2017", parse_iso(reports[0].timestamp)
    state = GroupState(event)
    for idx, (hours, text) in enumerate(((0, "Please send a boat."), (1, "A casualty update."), (7, "Another casualty update."))):
        report = replace(reports[idx], event=event, text=text, timestamp=to_iso(start + timedelta(hours=hours)))
        tagged = replace(tags[0], report_id=report.report_id, places=[], county_fips=None)
        state.add(report, tagged, emb[0])
    assert len(state.incidents) == 1
    with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
        inc = next(iter(state.incidents.values()))
        assert snapshot_features(inc, start, lambda f: None)["share_current_assistance"] == 1
        assert snapshot_features(inc, start + timedelta(hours=7), lambda f: None)["share_current_assistance"] == 0


def test_current_assistance_priority_survives_fairness_even_with_lower_model_score():
    base = _last()[0]
    def item(idx, current, score):
        values = dict(base.features)
        values.update(share_current_assistance=float(current), share_historical_or_speculative=0.0,
                      share_casualty_only=float(not current))
        return replace(base, event="operational_test_2017", incident_id=f"operational_test_2017-inc-{idx}",
                       features=values, score=score, score_adj=score, rank=idx, rank_adj=idx,
                       svi=None if current else 1.0)
    frame = pd.DataFrame({"share_current_assistance": [1.0, 0.0]})
    scores = rank._priority_scores(frame, [0.0, 1.0], "assistance_v1")
    request, news = item(1, True, scores[0]), item(2, False, scores[1])
    request.validate()
    news.validate()
    for lam in config.LAMBDAS + [10.0]:
        ordered = fairness.apply([news, request], lam)
        assert ordered[0].incident_id == request.incident_id
    assert np.array_equal(rank._priority_scores(frame, [0.0, 1.0], "legacy"), [0.0, 1.0])


def test_old_model_and_ranked_schema_require_explicit_rebuild():
    from types import SimpleNamespace
    event = "operational_test_2017"
    with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
        for action in (lambda: rank._check_model(SimpleNamespace(meta={"train_events": ["training_event"]}), event),
                       lambda: rank.validate_ranked_outputs(event, _last())):
            try:
                action()
            except ValueError as exc:
                assert "ranker" in str(exc)
            else:
                raise AssertionError("Old production rankings/models must not silently serve the new objective")


def test_assistance_training_backup_rerank_live_parity_and_stale_cache():
    import shutil
    try:
        import xgboost
    except ImportError:
        skip("xgboost not installed")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        training, demo = "assistance_train_2017", "assistance_demo_2017"
        for event, fixture in ((training, config.FIXTURE_TRAIN_EVENT), (demo, config.FIXTURE_EVENT)):
            d = root / event
            shutil.copytree(fixture_dir(fixture), d)
            fileio.write_jsonl(d / config.REPORTS_FILE,
                              [replace(r, event=event) for r in load_fixture(config.REPORTS_FILE, Report, fixture)])
        model_path = root / "models" / config.RANKER_PATH.name
        with patch.object(config, "DEMO_EVENT", demo), patch.object(config, "LEGACY_RANKER_DIR", root / "legacy"), \
             patch.object(config, "RANKING_OBJECTIVE", "legacy"):
            rank.train_ranker([training], svi_lookup=lambda f: None, outputs_root=root, ranker_path=model_path)
            original = model_path.read_bytes()
        with patch.object(config, "DEMO_EVENT", demo), patch.object(config, "LEGACY_RANKER_DIR", root / "legacy"), \
             patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
            meta = rank.train_ranker([training], svi_lookup=lambda f: None, outputs_root=root, ranker_path=model_path)
            assert meta["objective"] == "assistance_v1" and meta["feature_keys"] == RANK_FEATURE_KEYS
            assert meta["train_events"] == [training] and demo not in meta["train_events"]
            assert (config.LEGACY_RANKER_DIR / model_path.name).read_bytes() == original
            model = rank.Ranker(model_path)
            out = root / "ranked"
            rows = rank.rank_event(demo, force=True, src_dir=root / demo, out_dir=out, ranker=model,
                                   svi_lookup=lambda f: None)
            assert all(list(i.features) == RANK_FEATURE_KEYS for i in rows)
            assert rank.rank_event(demo, src_dir=root / demo, out_dir=out, ranker=model) == rows
            assert fileio.read_json(out / config.RANK_PROVENANCE_FILE)["legacy_model_comparison"]
            score_rows = fileio.read_jsonl(out / config.RANK_SCORES_FILE)
            assert all(r["legacy_model_score"] is not None for r in score_rows)
            state = group_event(demo, src_dir=root / demo, out_dir=root / "state")
            now = parse_iso(rank.pick_demo_snapshot(rows))
            live = rank.rank_now(state, now, model, lambda f: None)
            assert live == [i for i in rows if i.snapshot_time == to_iso(now)]
            for inc in state.incidents.values():
                inc.reports = [replace(r, gold_label=None) for r in inc.reports]
            without_gold = rank.rank_now(state, now, model, lambda f: None)
            assert [replace(i, urgent_proxy=None) for i in live] == without_gold
            # Same-schema assistance models from before this feature revision fail
            # at load time, including cached/offline and live entry points.
            meta_path = model_path.parent / config.RANKER_META_PATH.name
            old_meta = dict(meta)
            old_meta.pop("feature_definition")
            fileio.write_json(meta_path, old_meta)
            try:
                rank.Ranker(model_path)
            except ValueError as exc:
                assert "Retrain" in str(exc)
            else:
                raise AssertionError("Old assistance model must not load")
            fileio.write_json(meta_path, meta)
            # The production schema crosses evaluation, UI, and brief contracts
            # without loading MiniLM or Granite or modifying frozen fixtures.
            from surgedesk import brief, evaluate, ui
            from types import SimpleNamespace
            for name in (config.REPORTS_FILE, config.TAGGED_FILE, config.EMBEDDINGS_FILE):
                shutil.copy2(root / demo / name, out / name)
            with patch.object(config, "LABELS_DIR", root / "labels"), patch.object(config, "DOCS", root / "docs"), \
                 patch.object(config, "RANKER_META_PATH", model_path.parent / config.RANKER_META_PATH.name):
                summary = evaluate.run_all(demo, src_dir=out)
                assert summary["n_reports"] == len(load_fixture(config.REPORTS_FILE, Report))
                assert (out / "ranking_objective_comparison.csv").exists()
                html = ui.render_html(demo, src_dir=out)
                assert "Priority score" in html and "not a calibrated probability" in html
                fake = SimpleNamespace(model_file="offline-test", chat_json=lambda *args: (
                    {}, dict(seconds=0.0, prompt_tokens=0, completion_tokens=0, ok=True)))
                chosen = [i for i in rows if i.snapshot_time == to_iso(now)]
                checked = brief.brief_incidents(demo, chosen, top_n=3, llm=fake, src_dir=out, out_dir=out)
                assert len(checked) == 3 and all(b.used_fallback for b in checked)
            # Cached rankings cannot silently survive a changed source report.
            reports = fileio.read_jsonl(root / demo / config.REPORTS_FILE, Report)
            fileio.write_jsonl(root / demo / config.REPORTS_FILE,
                              [replace(reports[0], text=reports[0].text + " Please help."), *reports[1:]])
            try:
                rank.rank_event(demo, src_dir=root / demo, out_dir=out, ranker=model)
            except ValueError as exc:
                assert "stale" in str(exc)
            else:
                raise AssertionError("Changed reports must invalidate ranking cache")

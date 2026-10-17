"""Offline acceptance checks for incident grouping, ranking, and adjustment."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from surgedesk import baselines, config, fairness, fileio, rank
from surgedesk.contracts import FEATURE_KEYS, Incident, Report
from surgedesk.features import SNAPSHOT_COLUMNS, build_snapshots, snapshot_features
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

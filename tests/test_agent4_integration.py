"""Fast, offline Agent 4 acceptance tests; upstream APIs are faked only in tests."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from surgedesk import benchmark, config, evaluate, fileio, ui
from surgedesk.contracts import BRIEF_FIELDS, Briefing, Decision, Incident, Report, empty_fields
from surgedesk.pipeline import StageUnavailable
from surgedesk.testing import fixture_dir, load_fixture


def _copy(root):
    destination = Path(root) / config.FIXTURE_EVENT
    shutil.copytree(fixture_dir(), destination)
    return destination


def _fixture_groups(d):
    members = {}
    for incident in fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident):
        members.setdefault(incident.incident_id, set()).update(incident.report_ids)
    groups = [dict(incident_id=iid, report_ids=sorted(ids)) for iid, ids in members.items()]
    fileio.write_jsonl(d / "groups.jsonl", groups)
    return groups


def _row(idx, urgent, first="2017-08-27T10:00:00Z", snap="2017-08-27T10:30:00Z", rank=None):
    return replace(load_fixture(config.INCIDENTS_FILE, Incident)[0],
                   incident_id=f"fixture_event-inc-{idx}", urgent_proxy=urgent,
                   snapshot_time=snap, first_seen=first, last_seen=first,
                   rank=rank or idx, rank_adj=rank or idx)


def test_stub_pipeline_cli():
    result = subprocess.run([sys.executable, "-m", "surgedesk.pipeline", "--stub"],
                            cwd=config.PACKAGE_DIR.parent, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "[stub] start" in result.stdout and "[stub] done in" in result.stdout
    assert fileio.validate_file(config.event_file(config.FIXTURE_EVENT, config.INCIDENTS_FILE, stub=True)) > 0


def test_render_html_stub():
    html = ui.render_html(stub=True)
    incidents = fileio.read_jsonl(config.event_file(config.FIXTURE_EVENT, config.INCIDENTS_FILE, stub=True), Incident)
    assert ui.BANNER in html
    assert html.count('class="incident-card"') == min(config.TOP_K_EVAL, len(incidents))
    assert "template" in html


def test_render_html_fixture_briefing():
    with tempfile.TemporaryDirectory() as root:
        d = _copy(root)
        html = ui.render_html(config.FIXTURE_EVENT, src_dir=d)
        assert 'data-incident-id="fixture_event-inc-1"' in html
        assert "Meyerland Houston" in html
        assert "1 sentences rejected" in html
        assert "2017-" in html


def test_ranking_metrics_hand_example():
    rows = [_row(1, 1), _row(2, 1, first="2017-08-27T10:10:00Z"),
            _row(3, 0, first="2017-08-27T10:05:00Z"), _row(4, 0, first="2017-08-27T10:15:00Z")]
    surge = evaluate.ranking_metrics(rows, [i.incident_id for i in rows], k=2)
    fifo = evaluate.ranking_metrics(rows, [i.incident_id for i in sorted(rows, key=lambda i: i.first_seen)], k=2)
    assert surge["p_at_10"] == 1.0 and fifo["p_at_10"] == 0.5
    assert abs(surge["ndcg_at_10"] - 1.0) < 1e-9
    assert abs(fifo["ndcg_at_10"] - 1 / (1 + 1 / math.log2(3))) < 1e-9
    assert surge["r_at_10"] == 1.0


def test_time_to_surface_hand_example():
    start = "2017-08-27T10:05:00Z"
    early, later = "2017-08-27T10:15:00Z", "2017-08-27T10:30:00Z"
    rows = [_row(1, 1, first=start, snap=s) for s in (early, later)]
    rows += [_row(2, 1, first=start, snap=s) for s in (early, later)]
    others = [f"fixture_event-inc-{n}" for n in range(3, 13)]
    orders = {early: others + [rows[0].incident_id, "fixture_event-inc-2"],
              later: [rows[0].incident_id] + others + ["fixture_event-inc-2"]}
    result = evaluate.time_to_surface(rows, orders)
    assert result[0]["minutes"] == 25.0 and not result[0]["censored"]
    assert result[1]["minutes"] is None and result[1]["censored"]


def test_operational_ranking_compares_both_targets_and_preserved_model():
    with tempfile.TemporaryDirectory() as root:
        d = Path(root)
        event = "operational_eval_2017"
        reports = load_fixture(config.REPORTS_FILE, Report)[:2]
        reports = [replace(reports[0], event=event, gold_label="requests_or_urgent_needs", text="Please send a boat."),
                   replace(reports[1], event=event, gold_label="injured_or_dead_people", text="Officials report six deaths.")]
        rows = []
        for idx, report in enumerate(reports, 1):
            item = _row(idx, 1 if idx == 1 else 0)
            values = dict(item.features)
            values.update(share_current_assistance=1.0 if idx == 1 else 0.0,
                          share_historical_or_speculative=0.0, share_casualty_only=0.0 if idx == 1 else 1.0)
            rows.append(replace(item, event=event, report_ids=[report.report_id], features=values))
        fileio.write_jsonl(d / config.REPORTS_FILE, reports)
        fileio.write_jsonl(d / config.INCIDENTS_FILE, rows)
        fileio.write_jsonl(d / config.RANK_SCORES_FILE,
                          [dict(incident_id=i.incident_id, snapshot_time=i.snapshot_time,
                                legacy_model_score=0.1 if i.rank == 1 else 0.99) for i in rows])
        with patch.object(config, "RANKING_OBJECTIVE", "assistance_v1"):
            result = evaluate.eval_ranking(event, d)
        comparisons = {(r["target"], r["method"]): r for r in result["target_comparison"]}
        new = comparisons["assistance_category_proxy", "SurgeDesk"]
        old = comparisons["assistance_category_proxy", "LegacySurgeDesk"]
        assert new["ndcg_at_10"] == 1.0 and old["ndcg_at_10"] < new["ndcg_at_10"]
        assert comparisons["legacy_broad_proxy", "SurgeDesk"]["p_at_10"] == 1.0
        assert result["objective"] == "assistance_v1"
        assert (d / "ranking_objective_comparison.csv").exists()


def test_extraction_match_rule():
    assert evaluate.extraction_match("Meyerland Houston", "near Meyerland Houston")
    assert not evaluate.extraction_match("Meyerland Houston", "Rockport")
    assert evaluate.extraction_match(None, None)
    assert evaluate.extraction_match("", None)
    assert not evaluate.extraction_match("Meyerland", None)


def test_decision_logging():
    with tempfile.TemporaryDirectory() as root:
        d = Path(root)
        incident = _row(1, 1)
        decision = ui._log_decision(d, incident, "accept", note="Contact @name https://example.com", reviewer="P1")
        assert decision.action == "accept"
        assert fileio.validate_file(d / config.DECISIONS_FILE) == 1
        assert "@name" not in (d / config.DECISIONS_FILE).read_text()
        try:
            ui._log_decision(d, incident, "override")
        except ValueError as exc:
            assert "new_rank" in str(exc)
        else:
            raise AssertionError("override without new_rank accepted")
        assert len(fileio.read_jsonl(d / config.DECISIONS_FILE, Decision)) == 1
        ui._log_decision(d, incident, "override", new_rank=2)
        assert fileio.validate_file(d / config.DECISIONS_FILE) == 2


def test_summary_keys_on_fixture():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.object(config, "DOCS", root / "docs"), patch.object(config, "RANKER_META_PATH", root / "absent.json"):
            result = evaluate.run_all(config.FIXTURE_EVENT, src_dir=d)
        assert set(result) == {"event", "measured_on", "n_reports", "n_actionable", "n_incidents", "singleton_share",
                               "classifier_macro_f1", "ranker_dev_auc", "ranking", "time_to_surface",
                               "fairness_demo_snapshot", "extraction", "briefings", "purity"}
        assert set(result["measured_on"]) == {"arch", "cores", "ram_gb", "nnpa_flag"}
        assert result == fileio.read_json(d / config.SUMMARY_FILE)
        assert result["n_reports"] == len(load_fixture(config.REPORTS_FILE, Report))
        assert (root / "docs" / "RESULTS.md").exists()
        assert "NaN" not in (d / config.SUMMARY_FILE).read_text()


def test_incident_review_preserves_ranks_masks_sources_and_marks_unbriefed():
    with tempfile.TemporaryDirectory() as root:
        d = _copy(root)
        reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
        reports = [replace(r, text=r.text + " Contact @private_reviewer https://private.example/path") for r in reports]
        tags = fileio.read_jsonl(d / config.TAGGED_FILE, evaluate.Tagged)
        incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident)
        briefs = fileio.read_jsonl(d / config.BRIEFINGS_FILE, Briefing)
        before = [i.to_dict() for i in incidents]
        with patch("surgedesk.brief.get_llm", side_effect=AssertionError("Review export must not load Granite")):
            result = evaluate._write_incident_review(d, config.FIXTURE_EVENT, incidents, reports, tags, briefs)
        frame = pd.read_csv(d / "incident_review.csv", dtype=str, keep_default_na=False)
        assert len(frame) == result["n_exported_member_reports"]
        assert frame.incident_id.nunique() == result["n_exported_incidents"]
        assert set(frame.briefing_status) == {"generated_checked", "template_fallback", "not_briefed"}
        assert not frame.loc[frame.briefing_status == "not_briefed", "representative"].eq("True").any()
        assert "gold_label" not in frame
        assert "@private_reviewer" not in (d / "incident_review.csv").read_text()
        assert "https://private.example" not in (d / "incident_review.csv").read_text()
        assert frame.text.str.contains("@user", regex=False).all()
        assert [i.to_dict() for i in incidents] == before
        expected = {(i.incident_id, rid) for i in incidents if i.snapshot_time == result["snapshot_time"]
                    and i.rank <= config.TOP_K_EVAL for rid in i.report_ids}
        assert set(zip(frame.incident_id, frame.report_id)) == expected
        # Missing inputs replace old review rows with an empty, labeled export.
        skipped = evaluate._write_incident_review(d, config.FIXTURE_EVENT, [], reports, tags, briefs)
        assert skipped["status"] == "skipped"
        assert pd.read_csv(d / "incident_review.csv").empty


def test_label_sheet_sampling_and_scientific_ids():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        with patch.object(config, "LABELS_DIR", root / "labels"):
            path = evaluate.make_label_sheet(config.FIXTURE_EVENT, n=8, n_actionable=5, src_dir=d)
            labels = pd.read_csv(path, dtype=str, keep_default_na=False)
            assert list(labels) == list(evaluate.HAND_LABEL_COLUMNS)
            assert labels.report_id.is_unique and set(labels.labeler) <= {"P1", "P2", "P3", "P4"}
            assert all(not labels[f].str.len().any() for f in BRIEF_FIELDS)
            previous = path.read_bytes()
            evaluate.make_label_sheet(config.FIXTURE_EVENT, n=1, n_actionable=1, src_dir=d)
            assert path.read_bytes() == previous
            path.write_text("report_id\n9.01746E+17\n")
            try:
                evaluate._read_labels(path)
            except ValueError as exc:
                assert "plain text" in str(exc)
            else:
                raise AssertionError("Scientific notation must fail")


def test_extraction_metrics_with_labels():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        rid = load_fixture(config.REPORTS_FILE, Report)[0].report_id
        row = dict.fromkeys(evaluate.HAND_LABEL_COLUMNS, "")
        row.update(report_id=rid, labeler="P1", location="Meyerland Houston")
        fields = empty_fields()
        fields["location"] = dict(value="near Meyerland Houston", evidence="Meyerland Houston", confidence=1.0, grounded=True)
        with patch.object(config, "LABELS_DIR", root / "labels"):
            config.LABELS_DIR.mkdir()
            pd.DataFrame([row]).to_csv(config.LABELS_DIR / "hand_labels.csv", index=False)
            fileio.write_jsonl(d / "extraction_preds.jsonl", [dict(report_id=rid, fields=fields, timing=dict(ok=True, seconds=2.0))])
            result = evaluate.eval_extraction(config.FIXTURE_EVENT, d)
        assert result["location"]["match_rate"] == 1.0
        assert result["location"]["null_agreement"] is None
        assert result["hazard"]["null_agreement"] == 1.0
        assert result["json_valid_rate"] == 1.0 and result["mean_seconds_per_report"] == 2.0


def test_extraction_evaluation_rechecks_saved_grounding_and_preserves_original_timing():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
        rid = reports[0].report_id
        reports[0] = replace(reports[0], text="Harbor City's 1900 Death toll reached 8,000 people.")
        fileio.write_jsonl(d / config.REPORTS_FILE, reports)
        fields = empty_fields()
        fields["people_affected"] = dict(value="8,000 people", evidence="8,000 people", confidence=1, grounded=True)
        predictions = d / "extraction_preds.jsonl"
        fileio.write_jsonl(predictions, [dict(report_id=rid, fields=fields, timing=dict(ok=True, seconds=42.0))])
        saved = predictions.read_bytes()
        row = dict.fromkeys(evaluate.HAND_LABEL_COLUMNS, "")
        row.update(report_id=rid, labeler="AI")
        with patch.object(config, "LABELS_DIR", root / "labels"), \
             patch.object(config, "DEMO_EVENT", "hurricane_harvey_2017"), \
             patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai"}), \
             patch("surgedesk.brief.get_llm", side_effect=AssertionError("No model calls in evaluation")):
            config.LABELS_DIR.mkdir()
            pd.DataFrame([row]).to_csv(config.LABELS_DIR / "ai_extraction_labels_DRAFT.csv", index=False)
            result = evaluate.eval_extraction(config.FIXTURE_EVENT, d)
        assert result["people_affected"]["match_rate"] == 1.0
        assert result["people_affected"]["predicted_non_null_rate"] == 0.0
        assert result["mean_seconds_per_report"] == 42.0
        assert predictions.read_bytes() == saved


def test_evaluation_freshness_detects_changed_labels_briefings_and_code():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.object(config, "DOCS", root / "docs"), \
             patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai", "SURGEDESK_GENERATE_EXTRACTIONS": "0"}):
            assert evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["status"] == "missing"
            fileio.write_json(d / config.SUMMARY_FILE, {})
            assert evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["status"] == "unverified"
            config.LABELS_DIR.mkdir()
            label = config.LABELS_DIR / "ai_extraction_labels_DRAFT.csv"
            row = dict.fromkeys(evaluate.HAND_LABEL_COLUMNS, "")
            row.update(report_id=load_fixture(config.REPORTS_FILE, Report)[0].report_id, labeler="AI")
            pd.DataFrame([row]).to_csv(label, index=False)
            original_labels = label.read_bytes()
            evaluate.run_all(config.FIXTURE_EVENT, d)
            # Notebook 05 uses the saved AI mode even in a new default-human kernel.
            with patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "human"}):
                assert evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["status"] == "current"
            row["location"] = "Houston"
            pd.DataFrame([row]).to_csv(label, index=False)
            assert "labels/extraction" in evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["changed_inputs"]
            label.write_bytes(original_labels)
            briefs = fileio.read_jsonl(d / config.BRIEFINGS_FILE, Briefing)
            original_briefs = (d / config.BRIEFINGS_FILE).read_bytes()
            briefs[0] = replace(briefs[0], used_fallback=True)
            fileio.write_jsonl(d / config.BRIEFINGS_FILE, briefs)
            assert config.BRIEFINGS_FILE in evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["changed_inputs"]
            (d / config.BRIEFINGS_FILE).write_bytes(original_briefs)
            altered = root / "altered_package"
            altered.mkdir()
            # Simulate a later code update without touching the checkout.
            with patch.object(config, "PACKAGE_DIR", altered):
                assert "code/grounding.py" in evaluate.evaluation_freshness(config.FIXTURE_EVENT, d)["changed_inputs"]


def test_ai_purity_excludes_uncertain_and_unreviewed_rows():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        group = max(_fixture_groups(d), key=lambda g: len(g["report_ids"]))
        incident = group["incident_id"]
        values = [("yes", "AI"), ("no", "AI"), ("uncertain", "AI"), ("", "AI"), ("yes", "")]
        rows = [dict(incident_id=incident, report_id=group["report_ids"][idx % len(group["report_ids"])], text="fixture", same_incident=v, labeler=who)
                for idx, (v, who) in enumerate(values)]
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai"}):
            config.LABELS_DIR.mkdir()
            path = config.LABELS_DIR / "ai_purity_labels_DRAFT.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = evaluate.eval_grouping(config.FIXTURE_EVENT, d)
            assert result["purity"] == 0.5
            assert result["purity_rows_total"] == 5
            assert result["purity_rows_evaluated"] == 2 and result["purity_rows_excluded"] == 3
            assert result["label_provenance"]["source"] == "ai"
            pd.DataFrame(rows[2:4]).to_csv(path, index=False)
            assert evaluate.eval_grouping(config.FIXTURE_EVENT, d)["purity"] is None
            rows[0]["same_incident"] = "yees"
            pd.DataFrame(rows).to_csv(path, index=False)
            try:
                evaluate.eval_grouping(config.FIXTURE_EVENT, d)
            except ValueError as exc:
                assert "invalid same_incident" in str(exc)
            else:
                raise AssertionError("Misspelled decisions must not silently become negatives")


def test_regrouping_invalidates_purity_and_mismatched_ui_briefings():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        groups = _fixture_groups(d)
        group = groups[0]
        labels = [dict(incident_id=group["incident_id"], report_id=rid, text="fixture",
                       same_incident="yes", labeler="AI") for rid in group["report_ids"]]
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai"}):
            config.LABELS_DIR.mkdir()
            pd.DataFrame(labels).to_csv(config.LABELS_DIR / "ai_purity_labels_DRAFT.csv", index=False)
            assert evaluate.eval_grouping(config.FIXTURE_EVENT, d)["purity"] == 1.0
            all_ids = [r.report_id for r in load_fixture(config.REPORTS_FILE, Report)]
            added = next(rid for rid in all_ids if rid not in group["report_ids"])
            group["report_ids"].append(added)
            fileio.write_jsonl(d / "groups.jsonl", groups)
            result = evaluate.eval_grouping(config.FIXTURE_EVENT, d)
            assert result["purity"] is None and result["purity_rows_stale"] == len(labels)
        data = ui.load_demo(config.FIXTURE_EVENT, src_dir=d)
        key, old = next(iter(data.briefings.items()))
        incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident)
        current = next(i for i in incidents if (i.incident_id, i.snapshot_time) == key)
        added = next(rid for rid in all_ids if rid not in current.report_ids)
        incidents = [replace(i, report_ids=i.report_ids + [added]) if (i.incident_id, i.snapshot_time) == key else i
                     for i in incidents]
        fileio.write_jsonl(d / config.INCIDENTS_FILE, incidents)
        assert key not in ui.load_demo(config.FIXTURE_EVENT, src_dir=d).briefings
        with patch.object(config, "LABELS_DIR", root / "absent_labels"), patch.object(config, "DOCS", root / "docs"):
            assert evaluate.run_all(config.FIXTURE_EVENT, d)["briefings"]["n"] == 2


def test_ui_and_evaluation_recheck_historical_cached_claims_without_loading():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        rows = fileio.read_jsonl(d / config.BRIEFINGS_FILE, Briefing)
        old = rows[0]
        rep_id = old.rep_report_ids[0]
        reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
        reports = [replace(r, text="In 1900, Harbor City reported 8,000 dead.") if r.report_id == rep_id else r
                   for r in reports]
        fileio.write_jsonl(d / config.REPORTS_FILE, reports)
        fields = empty_fields()
        fields["people_affected"] = dict(value="8,000", evidence="8,000", confidence=1, grounded=True)
        claim = dict(text="1. Reports state 8,000 people,.", tags=["people_affected"], accepted=True, reject_reason=None)
        old = replace(old, fields=fields, sentences=[claim], rep_report_ids=[rep_id],
                      briefing=claim["text"], used_fallback=False)
        fileio.write_jsonl(d / config.BRIEFINGS_FILE, [old])
        from surgedesk import brief
        with patch.object(config, "DEMO_EVENT", "hurricane_harvey_2017"), \
             patch.object(config, "LABELS_DIR", root / "labels"), patch.object(config, "DOCS", root / "docs"), \
             patch.object(brief, "get_llm", side_effect=AssertionError("Cached review must not load Granite")):
            displayed = ui.load_demo(config.FIXTURE_EVENT, src_dir=d).briefings[(old.incident_id, old.snapshot_time)]
            assert displayed.used_fallback and not displayed.fields["people_affected"]["grounded"]
            summary = evaluate.run_all(config.FIXTURE_EVENT, d)
            assert summary["briefings"] == dict(n=1, fields_grounded_share=0.0,
                                                 sentences_accepted_share=0.0, fallback_share=1.0)


def test_ai_reference_mode_keeps_reviewed_nulls_and_stored_predictions():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        rid = load_fixture(config.REPORTS_FILE, Report)[0].report_id
        row = dict.fromkeys(evaluate.HAND_LABEL_COLUMNS, "")
        row.update(report_id=rid, labeler="AI")
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai"}):
            config.LABELS_DIR.mkdir()
            pd.DataFrame([row]).to_csv(config.LABELS_DIR / "ai_extraction_labels_DRAFT.csv", index=False)
            assert len(evaluate._hand_labels()) == 1
            fileio.write_jsonl(d / "extraction_preds.jsonl", [dict(report_id=rid, fields=empty_fields(), timing=dict(ok=True, seconds=1.0))])
            result = evaluate.eval_extraction(config.FIXTURE_EVENT, d)
            assert result["n_labeled"] == 1 and result["hazard"]["match_rate"] == 1.0
            assert result["label_provenance"]["provisional"]
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "human"}):
            assert evaluate._hand_labels() is None  # Never silently reuse AI drafts in human mode.


def test_ai_run_all_can_disable_granite_and_reports_provenance():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d = _copy(root)
        rid = load_fixture(config.REPORTS_FILE, Report)[0].report_id
        row = dict.fromkeys(evaluate.HAND_LABEL_COLUMNS, "")
        row.update(report_id=rid, location="Meyerland Houston", labeler="AI")
        with patch.object(config, "LABELS_DIR", root / "labels"), patch.object(config, "DOCS", root / "docs"), \
             patch.dict(os.environ, {"SURGEDESK_LABEL_SOURCE": "ai", "SURGEDESK_GENERATE_EXTRACTIONS": "0"}), \
             patch("surgedesk.brief.extract_for_eval", side_effect=AssertionError("Granite must not be called")):
            config.LABELS_DIR.mkdir()
            pd.DataFrame([row]).to_csv(config.LABELS_DIR / "ai_extraction_labels_DRAFT.csv", index=False)
            result = evaluate.run_all(config.FIXTURE_EVENT, d)
            assert result["extraction"]["status"] == "skipped"
            provenance = fileio.read_json(d / "label_provenance.json")
            assert provenance["source"] == "ai" and not provenance["generate_missing_extractions"]
            assert "PROVISIONAL AI-BASED EVALUATION" in (config.DOCS / "RESULTS.md").read_text()


def test_upstream_missing_has_owner():
    with patch.dict(sys.modules, {"surgedesk.fairness": None}):
        try:
            ui._adjust([_row(1, 1)], 1.0)
        except StageUnavailable as exc:
            assert "stage fairness not implemented yet (owner: Agent 2)" in str(exc)
        else:
            raise AssertionError("Missing fairness implementation must be explicit")
        assert ui._adjust([_row(1, 1)], 0.0)[0].lam == 0


def test_benchmark_schedule_latency_and_live_embeddings():
    reports = load_fixture(config.REPORTS_FILE, Report)[:2]
    calls = []
    state = types.SimpleNamespace(add=lambda r, t, e: calls.append((r.gold_label, r.report_id)))
    tags = load_fixture(config.TAGGED_FILE, evaluate.Tagged)[:2]
    fake_tagger = types.SimpleNamespace(tag_batch=lambda batch: (tags, np.zeros((len(batch), config.EMBED_DIM))))
    modules = {"replay": types.SimpleNamespace(busiest_window_start=lambda r: parse_time(r[0].timestamp),
                       iter_batches=lambda *args: iter([(100.0, reports)])),
               "tag": types.SimpleNamespace(Tagger=lambda event: fake_tagger),
               "group": types.SimpleNamespace(GroupState=lambda event: state),
               "rank": types.SimpleNamespace(Ranker=lambda: object(), default_svi_lookup=lambda: (lambda f: None),
                       rank_now=lambda *args: [_row(1, 1)])}
    with patch.object(benchmark, "require_module", side_effect=lambda name, stage: modules[name]), patch.object(benchmark.fileio, "read_jsonl", return_value=reports), patch.object(benchmark.time, "monotonic", side_effect=[90.0, 102.0, 105.0]):
        [(row, _)] = list(benchmark.iter_stream(config.FIXTURE_EVENT, "max", 1.0))
    assert row["process_s"] == 3.0 and row["latency_s"] == 5.0
    assert len(calls) == 2 and all(gold is None for gold, _ in calls)
    assert benchmark.latency_stats([1.0, 3.0])["p50_latency_s"] == 2.0


def parse_time(value):
    return evaluate.parse_iso(value)


def test_ranking_evaluation_writes_all_methods_and_censoring():
    with tempfile.TemporaryDirectory() as root:
        d = _copy(root)
        rows = [_row(1, 1), _row(2, 1, first="2017-08-27T10:10:00Z"),
                _row(3, 0, first="2017-08-27T10:05:00Z"), _row(4, 0, first="2017-08-27T10:15:00Z")]
        fileio.write_jsonl(d / config.INCIDENTS_FILE, rows)
        fake = types.SimpleNamespace(
            order_surgedesk=lambda rows: [r.incident_id for r in sorted(rows, key=lambda i: i.rank_adj)],
            order_fifo=lambda rows: [r.incident_id for r in sorted(rows, key=lambda i: i.first_seen)],
            order_volume=lambda rows: [r.incident_id for r in rows],
            order_keyword=lambda rows, texts: [r.incident_id for r in rows])
        with patch.object(evaluate, "require_module", return_value=fake):
            result = evaluate.eval_ranking(config.FIXTURE_EVENT, d)
        assert set(result["ranking"]) == {"SurgeDesk", "FIFO", "Volume", "Keyword"}
        assert result["ranking"]["SurgeDesk"]["ndcg_at_10"] == 1.0
        assert abs(result["ranking"]["FIFO"]["ndcg_at_10"] - (1 + 0.5) / (1 + 1 / math.log2(3))) < 1e-9
        assert len(pd.read_csv(d / "ranking_metrics.csv")) == 4
        assert len(pd.read_csv(d / "time_to_surface.csv")) == 8
        assert (d / "plots/time_to_surface.png").exists()


def test_fairness_evaluation_uses_each_snapshot():
    with tempfile.TemporaryDirectory() as root:
        d = _copy(root)
        seen = []

        def table(rows):
            seen.append(rows[0].snapshot_time)
            return pd.DataFrame([dict(lam=lam, n_incidents=len(rows), precision_at_k=0.5,
                                      high_svi_share_top_k=0.25, unknown_svi_share_top_k=0.0,
                                      jaccard_vs_lam0=1.0) for lam in config.LAMBDAS])

        with patch.object(evaluate, "require_module", return_value=types.SimpleNamespace(fairness_table=table)):
            result = evaluate.eval_fairness(config.FIXTURE_EVENT, d)
        assert len(seen) == len({i.snapshot_time for i in load_fixture(config.INCIDENTS_FILE, Incident)})
        assert result["demo_snapshot"] == max(seen)
        assert len(result["demo"]) == len(config.LAMBDAS)
        assert (d / "fairness.csv").exists() and (d / "plots/fairness.png").exists()


def test_granite_benchmark_bypasses_briefing_cache():
    with tempfile.TemporaryDirectory() as root:
        d = _copy(root)
        # The committed .npy fixture is little-endian. This isolated benchmark
        # test uses fake representatives, so create its input in native byte
        # order through the same writer used by the real embedding stage.
        n_reports = len(fileio.read_jsonl(d / config.REPORTS_FILE, Report))
        vectors = np.zeros((n_reports, config.EMBED_DIM),
                           dtype=np.dtype(np.float32).newbyteorder("S"))
        vectors[:, 0] = 1.0
        fileio.save_embeddings(d / config.EMBEDDINGS_FILE, vectors)
        calls = []
        fake_llm = types.SimpleNamespace(load_s=1.25, model_file="test-fixed.gguf")

        def representatives(incident, reports, tags, embeddings):
            assert embeddings.dtype == np.dtype(np.float32)
            assert embeddings.shape == (n_reports, config.EMBED_DIM)
            return [reports[incident.report_ids[0]]]

        fake = types.SimpleNamespace(
            select_representatives=representatives,
            extract_fields=lambda llm, reports: (calls.append("extract") or empty_fields(), dict(seconds=2.0, completion_tokens=10)),
            write_briefing=lambda llm, incident, fields: (calls.append("brief") or [], dict(seconds=3.0, completion_tokens=20)))
        modules = {"brief": fake, "llm": types.SimpleNamespace(get_llm=lambda **kwargs: fake_llm)}
        with patch.object(benchmark, "require_module", side_effect=lambda name, stage: modules[name]), patch.object(config, "event_dir", return_value=d):
            frame = benchmark.bench_granite(config.FIXTURE_EVENT, n=2)
        assert calls == ["extract", "brief", "extract", "brief"]
        assert list(frame.tokens_per_s) == [6.0, 6.0]
        saved = fileio.read_json(d / "benchmark_granite_summary.json")
        assert saved["load_s"] == 1.25 and saved["peak_rss_bytes"] > 0
        assert saved["model_file"] == "test-fixed.gguf"
        assert saved["configured_threads"] == config.N_THREADS
        assert "including prompt processing" in saved["timing_note"]


def test_pipeline_cli_owner_diagnosis():
    result = subprocess.run([sys.executable, "-m", "surgedesk.pipeline", "--stage", "data"],
                            cwd=config.PACKAGE_DIR.parent, text=True, capture_output=True)
    if "not implemented yet" in result.stderr:
        assert result.returncode == 1 and "owner: Agent 1" in result.stderr
        assert "Traceback" not in result.stderr

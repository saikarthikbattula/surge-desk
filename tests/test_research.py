"""Offline tests for the solo research workflow; no external model required."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from surgedesk import config, fileio, research, research_data, embed
from surgedesk.contracts import CATEGORIES, Report
from surgedesk.testing import load_fixture


def _manifest():
    return research_data.freeze(research_data.make_manifest(
        config.FIXTURES / "humaid", {"train": [config.FIXTURE_TRAIN_EVENT], "dev": [config.FIXTURE_EVENT]},
        "Synthetic compatibility check; no held-out real-disaster claim", synthetic=True))


def _reject(fn):
    try:
        fn()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("Expected rejection")


def test_manifest_is_split_preserving_and_tamper_evident():
    manifest = _manifest()
    assert {r["original_split"] for r in manifest["rows"]} == {"train", "dev", "test"}
    train = research_data.selected_reports(manifest, "train")
    dev = research_data.selected_reports(manifest, "dev")
    assert train and dev
    assert not ({r.report_id for _, r in train} & {r.report_id for _, r in dev})
    assert not ({r["text_sha256"] for r, _ in train} & {r["text_sha256"] for r, _ in dev})
    altered = deepcopy(manifest)
    altered["rows"][0]["partition"] = "train"
    _reject(lambda: research_data.validate_manifest(altered))
    _reject(lambda: research_data.selected_reports(manifest, "dev", [config.FIXTURE_TRAIN_EVENT]))
    _reject(lambda: research_data.make_manifest(config.FIXTURES / "humaid",
                                               {"train": [config.DEMO_EVENT], "dev": [config.FIXTURE_EVENT]}, "test"))


def test_public_text_free_specification_reconstructs_identical_workloads():
    from surgedesk.provenance import report_identity
    manifest = _manifest()
    spec = {"synthetic": True, "partitions": manifest["partitions"], "rationale": manifest["rationale"],
            "tsv_checksums": manifest["sources"], "excluded_keys": [x["key"] for x in manifest["exclusions"]],
            "workload_fingerprints": {role: report_identity([r for _, r in research_data.selected_reports(manifest, role)]) for role in config.RESEARCH_PARTITIONS}}
    rebuilt = research_data.reconstruct_manifest(spec, config.FIXTURES / "humaid")
    for role in config.RESEARCH_PARTITIONS:
        assert [(row["key"], r.gold_label) for row, r in research_data.selected_reports(rebuilt, role)] == [(row["key"], r.gold_label) for row, r in research_data.selected_reports(manifest, role)]


def test_near_duplicate_review_and_source_changes():
    rows = [{"key": "a", "partition": "train"}, {"key": "b", "partition": "dev"}]
    pairs = research_data.near_pairs(rows, ["one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten eleven"])
    assert len(pairs) == 1 and pairs[0]["decision"] == "pending"
    manifest = _manifest()
    manifest["near_duplicates"] = pairs
    manifest["state"] = "draft"
    _reject(lambda: research_data.freeze(manifest))
    with tempfile.TemporaryDirectory() as temp:
        import shutil
        root = Path(temp) / "humaid"
        shutil.copytree(config.FIXTURES / "humaid", root)
        copied = research_data.freeze(research_data.make_manifest(root,
             {"train": [config.FIXTURE_TRAIN_EVENT], "dev": [config.FIXTURE_EVENT]}, "test", True))
        source = root / next(iter(copied["sources"]))
        source.write_text(source.read_text() + "\n", encoding="utf-8")
        _reject(lambda: research_data.selected_reports(copied, "train"))


def test_near_duplicate_prefix_index_matches_exhaustive_jaccard():
    rng = np.random.default_rng(config.SEED)
    texts, rows = [], []
    for i in range(30):
        tokens = ["token" + str(n) for n in rng.choice(30, 8, replace=False)]
        for suffix in ([], ["extra" + str(i)]):
            rows.append({"key": str(len(rows)), "partition": "train" if not suffix else "dev"})
            texts.append(" ".join(tokens + suffix))
    actual = {(p["left"], p["right"]) for p in research_data.near_pairs(rows, texts)}
    expected = set()
    for i, text in enumerate(texts):
        for j in range(i):
            if rows[j]["partition"] == rows[i]["partition"]:
                continue
            a, b = set(text.split()), set(texts[j].split())
            if len(a & b) / len(a | b) >= config.RESEARCH_NEAR_DUPLICATE and text != texts[j]:
                expected.add((str(j), str(i)))
    assert actual == expected


def test_results_view_marks_incomplete_artifacts_unavailable():
    with tempfile.TemporaryDirectory() as temp, patch.object(config, "RESEARCH_DIR", Path(temp)):
        fileio.write_json(Path(temp) / "partial" / config.RESEARCH_RESULT_FILE, {"incomplete": True})
        values = research.saved_results()
        assert len(values) == 1 and values[0]["status"] == "unavailable"
        assert "mean_event_macro_f1" not in values[0]


def test_metrics_have_fixed_taxonomy_and_missing_support():
    a, b = CATEGORIES[:2]
    values = research.metrics([a, a, b], [a, b, b])
    assert values["accuracy"] == 2 / 3
    assert values["per_class"][a] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert values["per_class"][b]["precision"] == 0.5
    assert values["per_class"][CATEGORIES[-1]]["recall"] is None
    assert abs(values["macro_f1"] - (4 / 3) / len(CATEGORIES)) < 1e-12
    assert research.metrics([], [])["macro_f1"] is None
    _reject(lambda: research.metrics([a], []))


def test_research_tfidf_end_to_end_is_reproducible_and_train_only():
    import joblib
    manifest = _manifest()
    with tempfile.TemporaryDirectory() as temp, patch.object(config, "RESEARCH_DIR", Path(temp)):
        model = research.train(manifest, "model")
        artifact = joblib.load(model / config.RESEARCH_MODEL_FILE)
        train_text = " ".join(r.text.lower() for _, r in research_data.selected_reports(manifest, "train"))
        assert all(word in train_text for word in artifact["vectorizer"].vocabulary_ if " " not in word)
        a = research.evaluate(manifest, model, "first")
        b = research.evaluate(manifest, model, "second")
        result = research.compare(a, b)
        assert result["identical_predictions"] and result["identical_quality_metrics"]
        assert b"SYNTHETIC FIXTURE" in (a / config.RESEARCH_CONFUSION_FILE).read_bytes()
        _reject(lambda: research.evaluate(manifest, model, "test", "test"))
        _reject(lambda: research.train(manifest, "model"))
        selection_path = Path(temp) / "selection.json"
        research.select(manifest, [a, b], model, selection_path)
        assert fileio.read_json(selection_path)["selected_model_sha256"]
        review_path = Path(temp) / "review.json"
        research.review(manifest, a, review_path)
        assert "classification_errors" in fileio.read_json(review_path)
        altered = fileio.read_json(a / config.RESEARCH_RESULT_FILE)
        altered["partition"] = "test"
        fileio.write_json(a / config.RESEARCH_RESULT_FILE, altered)
        _reject(lambda: research.compare(a, b))


def test_embedding_fingerprint_rejects_same_shape_text_reorder_model_and_corruption():
    reports = load_fixture(config.REPORTS_FILE, Report)
    class Encoder:
        threads, device = config.N_THREADS, "cpu"
        calls = 0
        def encode(self, texts):
            self.calls += 1
            return np.ones((len(texts), config.EMBED_DIM), dtype=np.float32)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with patch.object(config, "OUTPUTS", root), patch.object(config, "MINILM_DIR", root / "model"):
            encoder = Encoder()
            path = config.event_file(config.FIXTURE_EVENT, config.REPORTS_FILE)
            fileio.write_jsonl(path, reports)
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            assert encoder.calls == 1
            changed = deepcopy(reports)
            changed[0].text = "Changed text with identical report IDs"
            fileio.write_jsonl(path, changed)
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            fileio.write_jsonl(path, list(reversed(changed)))
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            encoder.model_dir = root / "model"
            encoder.model_dir.mkdir()
            (encoder.model_dir / "weights").write_bytes(b"different model")
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            target = config.event_file(config.FIXTURE_EVENT, config.EMBEDDINGS_FILE)
            target.write_bytes(b"incomplete")
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            assert encoder.calls == 5
            config.event_file(config.FIXTURE_EVENT, config.EMBED_MANIFEST_FILE).unlink()
            embed.embed_event(config.FIXTURE_EVENT, embedder=encoder)
            assert encoder.calls == 6
            assert np.load(target).dtype.str == ">f4"


def test_missing_memory_is_unavailable():
    import sys
    from surgedesk.provenance import Measurement
    with patch.dict(sys.modules, {"psutil": None}):
        with Measurement() as m:
            pass
    assert m.as_dict()["peak_rss_bytes"] is None
    assert m.as_dict()["memory_unavailable_reason"]


def test_reserved_evaluation_requires_selection_and_locks_further_selection():
    import csv
    import shutil
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data = root / "humaid"
        shutil.copytree(config.FIXTURES / "humaid", data)
        source_reports = load_fixture(config.REPORTS_FILE, Report)
        with (data / "reserved_event_test.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["tweet_id", "tweet_text", "class_label"])
            for i in range(2):
                writer.writerow([str(int(source_reports[i].report_id) + 2000000000),
                                 "entirely reserved vocabulary " + ("alpha" if i else "omega"), CATEGORIES[i]])
        manifest = research_data.freeze(research_data.make_manifest(data,
             {"train": [config.FIXTURE_TRAIN_EVENT], "dev": [config.FIXTURE_EVENT], "test": ["reserved_event"]},
             "Synthetic final-evaluation gating test", True))
        with patch.object(config, "RESEARCH_DIR", root / "runs"):
            model = research.train(manifest, "model")
            dev = research.evaluate(manifest, model, "dev")
            _reject(lambda: research.evaluate(manifest, model, "unauthorized-test", "test"))
            selection = root / "selected.json"
            research.select(manifest, [dev], model, selection)
            final = research.evaluate(manifest, model, "final", "test", selection=selection)
            assert fileio.read_json(final / config.RESEARCH_RESULT_FILE)["pooled"]["n"] == 2
            _reject(lambda: research.select(manifest, [dev], model, root / "reselected.json"))
            value = fileio.read_json(selection)
            value["criterion"] = "changed after seeing test results"
            fileio.write_json(selection, value)
            _reject(lambda: research.evaluate(manifest, model, "changed-selection", "test", selection=selection))


def test_location_evidence_keeps_conflicting_counties_uncertain():
    from surgedesk.geo import Gazetteer, PlaceMatcher
    matcher = PlaceMatcher(config.FIXTURE_EVENT, Gazetteer(config.FIXTURES / "gazetteer"))
    conflict = matcher.match_with_evidence("Houston and Mission TX need water")
    assert conflict["places"] == ["Houston", "Mission"]
    assert conflict["legacy_county_fips"] == "48201"
    assert conflict["county_fips"] is None and conflict["uncertainty"] == "ambiguous"
    approximate = matcher.match_with_evidence("Houston TX needs water")
    assert approximate["county_fips"] == "48201" and approximate["uncertainty"] == "approximate"
    exact = matcher.match_with_evidence("Harris County needs water")
    assert exact["uncertainty"] == "county_name"
    assert matcher.match_with_evidence("Houston MS needs water")["county_fips"] is None
    gaz = Gazetteer(config.FIXTURES / "gazetteer")
    houston = next(p for p in gaz.places if p["NAME"] == "Houston city" and p["USPS"] == "TX")
    mission = next(p for p in gaz.places if p["NAME"] == "Mission city")
    gaz.places.append(dict(houston, ALAND="1", INTPTLAT=mission["INTPTLAT"], INTPTLONG=mission["INTPTLONG"]))
    same_state = PlaceMatcher(config.FIXTURE_EVENT, gaz).match_with_evidence("Houston TX")
    assert same_state["legacy_county_fips"] == "48201"
    assert same_state["county_fips"] is None


def test_conservative_locations_reject_development_audit_false_positives():
    from surgedesk.geo import Gazetteer, PlaceMatcher
    gaz = Gazetteer(config.FIXTURES / "gazetteer")
    prototype = next(p for p in gaz.places if p["USPS"] == "TX")
    for name in ("North", "Advance", "Cash", "Speed", "Kelly", "Simpson"):
        gaz.places.append(dict(prototype, NAME=name + " city"))
    matcher = PlaceMatcher(config.FIXTURE_EVENT, gaz)
    with patch.object(config, "LOCATION_POLICY", "conservative"):
        for text in ("moving north", "in advance of landfall", "withdraw cash", "wind speed", "spoke with Kelly Exampleperson", "Saffir-Simpson scale"):
            assert matcher.match(text) == ([], None), text
        assert matcher.match("Kelly TX is flooded")[0] == ["Kelly"]
        assert matcher.match("Simpson TX is flooded")[0] == ["Simpson"]


def test_pipeline_recovery_skips_completed_stage_and_rebuilds_on_change():
    from surgedesk.pipeline_state import run
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        event = config.FIXTURE_EVENT
        output = root / event / config.REPORTS_FILE
        calls = []
        def execute(stage, name, force):
            calls.append(stage)
            assert force
            fileio.write_jsonl(output, load_fixture(config.REPORTS_FILE, Report))
            fileio.write_json(root / "events.json", [event])
            return {"reports": len(load_fixture(config.REPORTS_FILE, Report))}
        with patch.object(config, "OUTPUTS", root), patch.object(config, "PIPELINE_RECEIPTS", root / "receipts"):
            first = run("data", event, False, execute)
            assert run("data", event, False, execute) == first and len(calls) == 1
            output.write_text("partial write", encoding="utf-8")
            run("data", event, False, execute)
            assert len(calls) == 2
            # A failed later stage never removes the valid data receipt.
            def interrupted(*args):
                raise RuntimeError("injected interruption")
            try:
                run("embed", event, False, interrupted)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected injected failure")
            run("data", event, False, execute)
            assert len(calls) == 2


def test_research_classifier_is_compatible_with_stage1_tagger():
    import joblib
    from surgedesk.tag import Tagger
    from surgedesk.geo import Gazetteer, PlaceMatcher
    manifest = _manifest()
    with tempfile.TemporaryDirectory() as temp, patch.object(config, "RESEARCH_DIR", Path(temp)):
        model = research.train(manifest, "model")
        artifact = joblib.load(model / config.RESEARCH_MODEL_FILE)
        reports = load_fixture(config.REPORTS_FILE, Report)
        embeddings = load_fixture(config.EMBEDDINGS_FILE)
        tagger = Tagger(config.FIXTURE_EVENT, model / config.RESEARCH_MODEL_FILE,
                        matcher=PlaceMatcher(config.FIXTURE_EVENT, Gazetteer(config.FIXTURES / "gazetteer")))
        tagged, returned = tagger.tag_batch(reports, embeddings)
        from surgedesk.textutil import mask_privacy
        expected = artifact["model"].predict(artifact["vectorizer"].transform([mask_privacy(r.text) for r in reports]))
        assert [t.pred_category for t in tagged] == expected.tolist()
        assert np.array_equal(returned, embeddings)


def test_synthetic_recording_contains_evidence_and_validated_traceable_decisions():
    from surgedesk.research_walkthrough import run
    from surgedesk.contracts import Decision
    with tempfile.TemporaryDirectory() as temp, patch.object(config, "RESEARCH_DIR", Path(temp)):
        recording = Path(run("recording"))
        meta = fileio.read_json(recording / "walkthrough.json")
        assert meta["synthetic"] and not meta["screen_recording"]
        assert meta["human_participants"] is None
        demo = Path(meta["demo"])
        page = (demo / "demo.html").read_text(encoding="utf-8")
        assert "SYNTHETIC DEMONSTRATION" in page and "Meyerland Houston" in page
        assert "Why it ranks here" in page and "location-uncertainty" in page
        decisions = fileio.read_jsonl(demo / config.DECISIONS_FILE, Decision)
        assert [d.action for d in decisions] == ["accept", "override", "dismiss"]
        traces = fileio.read_jsonl(demo / config.DECISION_PROVENANCE_FILE)
        assert len(traces) == len(decisions)
        assert all(t["incident_report_ids"] and t["incidents_sha256"] and t["run_manifest_sha256"] for t in traces)

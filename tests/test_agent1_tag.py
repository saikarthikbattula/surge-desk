"""Agent 1 acceptance tests: offline fixtures and deterministic clocks."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from surgedesk import config, data_humaid, embed, tag
from surgedesk.contracts import ContractError, Report, Tagged, is_actionable
from surgedesk.fileio import load_embeddings, read_json, read_jsonl, save_embeddings, write_jsonl
from surgedesk.geo import Gazetteer, PlaceMatcher, SVI
from surgedesk.replay import busiest_window_start, iter_batches
from surgedesk.testing import skip, load_fixture, fixture_dir
from surgedesk.textutil import mask_privacy, parse_iso, to_iso

EVENTS = (config.FIXTURE_EVENT, config.FIXTURE_TRAIN_EVENT)


@contextmanager
def _outputs(directory):
    directory = Path(directory)
    with patch.object(config, "OUTPUTS", directory), patch.object(config, "CLASSIFIER_PATH", directory / "classifier.joblib"):
        yield directory


def _gaz():
    return Gazetteer(config.FIXTURES / "gazetteer")


def _require_classifier():
    if importlib.util.find_spec("sklearn") is None:
        skip("scikit-learn not installed")


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


def test_load_event_matches_fixture():
    discovered = data_humaid.discover_event_files(config.FIXTURES / "humaid")
    for event in EVENTS:
        reports = data_humaid.load_event(event, discovered[event], source="synthetic")
        assert [r.to_dict() for r in reports] == load_fixture(config.REPORTS_FILE, event=event)


def test_discover_ignores_other_tsv():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "x_train.tsv").touch()
        (root / "labeled_5_set1.tsv").touch()
        assert list(data_humaid.discover_event_files(root)) == ["x"]
        try:
            data_humaid.discover_event_files(root / "absent")
        except FileNotFoundError as exc:
            assert "absent" in str(exc)
        else:
            raise AssertionError("Missing input directory accepted")


def test_loader_precedence_and_validation():
    sample = load_fixture(config.REPORTS_FILE, Report)[0]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for split, text in [("test", "test text"), ("dev", "dev text"), ("train", "train text")]:
            with (root / f"example_{split}.tsv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                writer.writerow([" ID ", " Text ", " Label "])
                writer.writerow([sample.report_id, text, sample.gold_label])
                writer.writerow(["invalid", "skip", sample.gold_label])
                writer.writerow([str(int(sample.report_id) + 1), "   ", sample.gold_label])
        reports = data_humaid.load_event("example", list(root.glob("*.tsv")), "synthetic")
        assert len(reports) == 1 and reports[0].text == "train text"
        bad = root / "bad_train.tsv"
        bad.write_text("wrong\nvalue\n", encoding="utf-8")
        try:
            data_humaid.load_event("bad", [bad])
        except ContractError as exc:
            assert "wrong" in str(exc)
        else:
            raise AssertionError("Missing columns accepted")


def test_build_reports_cache_and_summary():
    with tempfile.TemporaryDirectory() as temp, _outputs(temp):
        summary = data_humaid.build_reports(humaid_dir=config.FIXTURES / "humaid", source="synthetic")
        assert summary.event.tolist() == sorted(EVENTS)
        assert read_json(config.OUTPUTS / "events.json") == sorted(EVENTS)
        target = config.event_file(config.FIXTURE_EVENT, config.REPORTS_FILE)
        original = target.read_bytes()
        with patch.object(data_humaid, "load_event", side_effect=AssertionError("Cache not used")):
            data_humaid.build_reports(humaid_dir=config.FIXTURES / "humaid", source="synthetic")
        assert target.read_bytes() == original


def test_fixture_notebook_preserves_production_outputs():
    """Running the actual fixture cells must not replace the real event registry."""
    _require_classifier()
    if importlib.util.find_spec("IPython") is None:
        skip("IPython display support not installed for notebook regression check")
    notebook = json.loads((config.ROOT / "notebooks" / "01_tag.ipynb").read_text())
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        production, stub = root / "production", root / "stub"
        production.mkdir()
        (production / "events.json").write_text(json.dumps([config.DEMO_EVENT, "existing_train_event"]))
        before = {str(path.relative_to(production)): path.read_bytes()
                  for path in production.rglob("*") if path.is_file()}
        scope = {}
        with patch.object(config, "OUTPUTS", production), patch.object(config, "STUB_OUTPUTS", stub), \
             patch.object(config, "HUMAID_DIR", root / "missing-data"), \
             patch.object(config, "MINILM_DIR", root / "missing-model"), \
             patch("IPython.display.display"), redirect_stdout(StringIO()):
            try:
                # Bootstrap, setup, ingestion, embeddings, training and tagging.
                for cell in [notebook["cells"][1], *notebook["cells"][3:8]]:
                    source = cell["source"]
                    source = "".join(source) if isinstance(source, list) else source
                    exec(compile(source, "01_tag.ipynb", "exec"), scope)
            finally:
                if scope.get("fixture_model_dir") is not None:
                    scope["fixture_model_dir"].cleanup()
        after = {str(path.relative_to(production)): path.read_bytes()
                 for path in production.rglob("*") if path.is_file()}
        assert before == after, "Fixture walkthrough changed production outputs"
        assert read_json(stub / "events.json") == sorted(EVENTS)
        assert config.OUTPUTS != stub
        for event in EVENTS:
            assert read_jsonl(stub / event / config.REPORTS_FILE, Report)
            assert read_jsonl(stub / event / config.TAGGED_FILE, Tagged)


def test_place_matcher_matches_fixture():
    gaz = _gaz()
    for event in EVENTS:
        matcher = PlaceMatcher(event, gaz)
        reports = load_fixture(config.REPORTS_FILE, Report, event)
        tags = load_fixture(config.TAGGED_FILE, Tagged, event)
        for report, row in zip(reports, tags):
            assert matcher.match(report.text) == (row.places, row.county_fips)


def test_place_matcher_reference_index():
    spec = importlib.util.spec_from_file_location("_agent1_fixture_reference", config.ROOT / "scripts" / "make_fixtures.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    gaz = _gaz()
    for event in EVENTS:
        states = config.states_for_event(event)
        index = gaz.place_index(states)
        assert index == reference.build_place_index(states)
        matcher = PlaceMatcher(event, gaz)
        texts = [r.text for r in load_fixture(config.REPORTS_FILE, Report, event)]
        texts += ["Harris County and Houston Houston #KatyFlood", "Mission texas and Victoria TX", "mission accomplished"]
        for text in texts:
            assert matcher.match(text) == reference.match_places(text, index, states)


def test_place_matcher_traps():
    matcher = PlaceMatcher(config.FIXTURE_EVENT, _gaz())
    assert matcher.match("Houston") == (["Houston"], "48201")
    assert matcher.match("mission accomplished") == ([], None)
    assert matcher.match("Mission TX flooding") == (["Mission"], "48215")
    assert matcher.match("#KatyFlood")[0] == ["Katy"]
    assert matcher.match("Mission texas flooding") == (["Mission"], "48215")
    with patch("surgedesk.geo.Gazetteer", side_effect=AssertionError("Unnecessary Gazetteer load")):
        assert PlaceMatcher("kerala_floods_2018").match("Houston") == ([], None)


def test_svi_fixture():
    svi = SVI(config.FIXTURES / "svi" / "svi_counties.csv")
    assert svi.available
    assert svi.get("01001") == 0.39
    assert svi.get("48201") == 0.84
    for code in ("48039", "99999", None):
        assert svi.get(code) is None


def test_svi_missing_and_tracts():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        output = StringIO()
        with redirect_stdout(output):
            missing = SVI(root / "absent.csv")
            assert missing.get("48201") is None
            assert missing.get(None) is None
        assert not missing.available and output.getvalue().count("Warning:") == 1
        target = root / "tracts.csv"
        target.write_text("FIPS,RPL_THEMES\n01001020100,0.2\n01001020200,0.6\n01001020300,-999\n48201000100,-999\n", encoding="utf-8")
        svi = SVI(target)
        assert abs(svi.get("01001") - 0.4) < 1e-9
        assert svi.get("48201") is None
        target.write_text("STCNTY,FIPS,RPL_THEMES\n1001,99999,0.39\n", encoding="utf-8")
        assert SVI(target).get("01001") == 0.39


def test_vuln_terms_match_fixture():
    for event in EVENTS:
        reports = load_fixture(config.REPORTS_FILE, Report, event)
        tags = load_fixture(config.TAGGED_FILE, Tagged, event)
        for report, row in zip(reports, tags):
            assert tag.find_vuln_terms(report.text) == row.vuln_terms


def test_iter_batches_fake_clock():
    for event in EVENTS:
        reports = load_fixture(config.REPORTS_FILE, Report, event)
        assert busiest_window_start(reports) in [parse_iso(r.timestamp) for r in reports]
        for speed in (1, 20, "max"):
            fake = FakeClock()
            batches = list(iter_batches(reports, speed, clock=fake.clock, sleep=fake.sleep))
            assert [r.report_id for _, batch in batches for r in batch] == [r.report_id for r in reports]
            assert all(batch for _, batch in batches)
            assert [scheduled for scheduled, _ in batches] == sorted(scheduled for scheduled, _ in batches)
            if speed == "max":
                assert not fake.sleeps


def test_replay_boundaries_and_duration():
    report = load_fixture(config.REPORTS_FILE, Report)[0]
    start = parse_iso(report.timestamp)
    reports = [replace(report, report_id=str(i), timestamp=to_iso(start + timedelta(seconds=s)))
               for i, s in enumerate((0, 59, 60, 119, 120))]
    fake = FakeClock()
    batches = list(iter_batches(reports, 1, duration_s=10, batch_seconds=5, clock=fake.clock, sleep=fake.sleep))
    assert [[r.report_id for r in rows] for _, rows in batches] == [["0", "1"], ["2", "3"]]
    assert [t for t, _ in batches] == [105.0, 110.0]
    fake = FakeClock()
    partial = list(iter_batches(reports, 1, duration_s=7, batch_seconds=5, clock=fake.clock, sleep=fake.sleep))
    assert [[r.report_id for r in rows] for _, rows in partial] == [["0", "1"], ["2"]]
    assert partial[-1][0] == 107.0
    fake = FakeClock()
    maximum = list(iter_batches(reports, "max", duration_s=0.5, batch_seconds=0.25, clock=fake.clock, sleep=fake.sleep))
    assert [r.report_id for _, rows in maximum for r in rows] == ["0", "1", "2", "3"]
    assert not fake.sleeps
    assert list(iter_batches([], 1)) == []
    assert list(iter_batches(reports, 1, duration_s=0)) == []
    # The half-open one-minute windows tie; the earliest report wins.
    assert busiest_window_start(reports, hours=1 / 60) == start


def test_embedder_shape():
    if not (config.MINILM_DIR / "config.json").is_file():
        skip("MiniLM not downloaded")
    encoder = embed.Embedder()
    vectors = encoder.encode(["Flooding in Houston", "Need rescue @someone https://example.org", "Power outage"])
    assert vectors.shape == (3, config.EMBED_DIM) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert encoder.encode([]).shape == (0, config.EMBED_DIM)
    private_text = "Need rescue @someone https://example.org"
    masked_pair = encoder.encode([private_text, mask_privacy(private_text)])
    assert np.allclose(masked_pair[0], masked_pair[1], atol=1e-6)
    single = encoder.encode(["Flooding in Houston"])
    assert np.allclose(single[0], vectors[0], atol=1e-5)
    # Exercise classifier training and live tagging with actual MiniLM vectors.
    _require_classifier()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train_reports = load_fixture(config.REPORTS_FILE, Report, config.FIXTURE_TRAIN_EVENT)
        train_dir = root / config.FIXTURE_TRAIN_EVENT
        write_jsonl(train_dir / config.REPORTS_FILE, train_reports)
        save_embeddings(train_dir / config.EMBEDDINGS_FILE, encoder.encode([r.text for r in train_reports]))
        classifier = root / "classifier.joblib"
        tag.train_classifier([config.FIXTURE_TRAIN_EVENT], allow_fixture=True, classifier_path=classifier, outputs_root=root)
        tagger = tag.Tagger(config.FIXTURE_EVENT, classifier_path=classifier, embedder=encoder,
                            matcher=PlaceMatcher(config.FIXTURE_EVENT, _gaz()))
        sample = load_fixture(config.REPORTS_FILE, Report)[:3]
        tagged, live_vectors = tagger.tag_batch(sample)
        assert live_vectors.shape == (len(sample), config.EMBED_DIM)
        assert [row.report_id for row in tagged] == [r.report_id for r in sample]
        for row in tagged:
            row.validate()


def test_embedding_cache():
    with tempfile.TemporaryDirectory() as temp, _outputs(temp):
        event = config.FIXTURE_EVENT
        reports = load_fixture(config.REPORTS_FILE, Report)
        write_jsonl(config.event_file(event, config.REPORTS_FILE), reports)
        expected = load_fixture(config.EMBEDDINGS_FILE)
        class Encoder:
            threads, device = config.N_THREADS, "cpu"
            calls = 0
            def encode(self, texts):
                self.calls += 1
                assert texts == [r.text for r in reports]
                return expected
        encoder = Encoder()
        assert np.array_equal(embed.embed_event(event, embedder=encoder), expected)
        assert np.array_equal(embed.embed_event(event, embedder=encoder), expected)
        assert encoder.calls == 1
        save_embeddings(config.event_file(event, config.EMBEDDINGS_FILE), expected[:-1])
        embed.embed_event(event, embedder=encoder)
        assert encoder.calls == 2
        timing = read_json(config.event_file(event, "embed_timing.json"))
        assert timing["n"] == len(reports) and timing["seconds"] >= 0


def test_training_guards():
    for events, allow in [([config.DEMO_EVENT], True), ([config.FIXTURE_TRAIN_EVENT], False)]:
        try:
            tag.train_classifier(events, allow_fixture=allow)
        except AssertionError:
            pass
        else:
            raise AssertionError("Training guard failed")


def test_tagger_fixture():
    _require_classifier()
    with tempfile.TemporaryDirectory() as temp:
        classifier = Path(temp) / "classifier.joblib"
        metrics = tag.train_classifier([config.FIXTURE_TRAIN_EVENT], allow_fixture=True,
                                      classifier_path=classifier, outputs_root=config.FIXTURES / "outputs")
        assert metrics["dev_macro_f1"] is None
        reports = load_fixture(config.REPORTS_FILE, Report)
        expected = load_fixture(config.TAGGED_FILE, Tagged)
        embeddings = load_fixture(config.EMBEDDINGS_FILE)
        tagger = tag.Tagger(config.FIXTURE_EVENT, classifier_path=classifier,
                            matcher=PlaceMatcher(config.FIXTURE_EVENT, _gaz()))
        rows, used = tagger.tag_batch(reports, embeddings)
        assert np.array_equal(used, embeddings) and tagger.embedder is None
        for i, (row, ref) in enumerate(zip(rows, expected)):
            row.validate()
            assert row.report_id == reports[i].report_id and row.emb_row == i
            assert (row.places, row.county_fips, row.vuln_terms) == (ref.places, ref.county_fips, ref.vuln_terms)
            assert row.is_actionable == is_actionable(row.pred_category)
        without_gold, _ = tagger.tag_batch([replace(r, gold_label=None) for r in reports], embeddings)
        assert [t.to_dict() for t in without_gold] == [t.to_dict() for t in rows]
        shifted, _ = tagger.tag_batch(reports[:2], embeddings[:2], start_row=7)
        assert [t.emb_row for t in shifted] == [7, 8]
        assert tagger.tag_batch([])[0] == []


def test_tag_event_and_evaluation():
    _require_classifier()
    with tempfile.TemporaryDirectory() as temp, _outputs(temp):
        event = config.FIXTURE_EVENT
        tag.train_classifier([config.FIXTURE_TRAIN_EVENT], allow_fixture=True,
                             classifier_path=config.CLASSIFIER_PATH, outputs_root=config.FIXTURES / "outputs")
        write_jsonl(config.event_file(event, config.REPORTS_FILE), load_fixture(config.REPORTS_FILE, Report))
        save_embeddings(config.event_file(event, config.EMBEDDINGS_FILE), load_fixture(config.EMBEDDINGS_FILE))
        factory = tag.Tagger
        def fixture_tagger(e):
            return factory(e, classifier_path=config.CLASSIFIER_PATH, matcher=PlaceMatcher(e, _gaz()))
        with patch.object(tag, "Tagger", side_effect=fixture_tagger):
            first = tag.tag_event(event)
        with patch.object(tag, "Tagger", side_effect=AssertionError("Cache not used")):
            assert tag.tag_event(event) == first
        # Same-length stale files must be rejected using report IDs/embedding row alignment.
        write_jsonl(config.event_file(event, config.TAGGED_FILE), list(reversed(first)))
        with patch.object(tag, "Tagger", side_effect=fixture_tagger):
            assert tag.tag_event(event) == first
        stats = read_json(config.event_file(event, "tag_stats.json"))
        assert stats["n"] == len(first)
        result = tag.evaluate_classifier(event)
        assert result["n"] == len(first)
        assert sum(map(sum, result["confusion"])) == len(first)
        assert 0 <= result["macro_f1"] <= 1


def test_embedding_endianness_roundtrip():
    reference = load_fixture(config.EMBEDDINGS_FILE)
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / config.EMBEDDINGS_FILE
        for source_dtype in ("<f4", ">f4"):
            source = reference.astype(source_dtype)
            before = source.tobytes()
            save_embeddings(target, source)
            stored = np.load(target, allow_pickle=False)
            assert stored.dtype.str == ">f4"
            assert stored.shape == reference.shape
            assert np.array_equal(stored, reference)
            assert source.tobytes() == before
            restored = load_embeddings(target)
            assert restored.dtype == np.float32 and restored.dtype.isnative
            assert np.array_equal(restored, reference)
            # Old little-endian fixture files and big-endian files both remain readable.
            np.save(target, source)
            restored = load_embeddings(target)
            assert restored.dtype == np.float32 and restored.dtype.isnative
            assert np.array_equal(restored, reference)
        save_embeddings(target, np.empty((0, config.EMBED_DIM), dtype=np.float32))
        assert np.load(target, allow_pickle=False).dtype.str == ">f4"
        assert load_embeddings(target).shape == (0, config.EMBED_DIM)


def test_embedding_loader_rejects_invalid_arrays():
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / config.EMBEDDINGS_FILE
        invalid = [np.zeros((2, config.EMBED_DIM), dtype=dtype)
                   for dtype in ("<f8", ">f8", "f2", "i4", "c8")]
        invalid.append(np.zeros(config.EMBED_DIM, dtype=">f4"))
        for values in invalid:
            np.save(target, values)
            try:
                load_embeddings(target)
            except ContractError:
                pass
            else:
                raise AssertionError(f"Invalid embedding array accepted: {values.dtype} {values.shape}")

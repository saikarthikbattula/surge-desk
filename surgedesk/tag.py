"""Embedding-only classifier training and report tagging (Agent 1)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, stubs
from .contracts import CATEGORIES, ContractError, Report, Tagged, VULN_TERMS, is_actionable
from .embed import Embedder
from .fileio import load_embeddings, read_jsonl, write_json, write_jsonl
from .geo import PlaceMatcher
from .textutil import to_iso


def find_vuln_terms(text: str) -> list[str]:
    return stubs.find_terms(text, VULN_TERMS)


def _check_embeddings(reports, embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(reports), config.EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ContractError(f"Expected finite embeddings of shape {(len(reports), config.EMBED_DIM)}, got {embeddings.shape}")
    return embeddings


def train_classifier(train_events: list[str], allow_fixture: bool = False,
                     classifier_path: Path = config.CLASSIFIER_PATH,
                     outputs_root: Path = config.OUTPUTS) -> dict:
    assert config.DEMO_EVENT not in train_events, "Never train on config.DEMO_EVENT"
    assert allow_fixture or not any(e.startswith("fixture") for e in train_events), "Fixture training requires allow_fixture=True"
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    matrices, labels = [], []
    for event in train_events:
        directory = Path(outputs_root) / event
        reports = read_jsonl(directory / config.REPORTS_FILE, Report)
        embeddings = _check_embeddings(reports, load_embeddings(directory / config.EMBEDDINGS_FILE))
        keep = [i for i, r in enumerate(reports) if r.gold_label is not None]
        matrices.append(embeddings[keep])
        labels.extend(reports[i].gold_label for i in keep)
    if not labels:
        raise ValueError("No labeled training reports found")
    if len(set(labels)) < 2:
        raise ValueError("Classifier training requires at least two categories")
    X, y = np.concatenate(matrices), np.asarray(labels)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)
    dev_macro_f1 = None
    if len(y) >= 200:
        stratify = y if min(Counter(labels).values()) >= 2 else None
        try:
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, stratify=stratify, random_state=config.SEED)
        except ValueError:
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=config.SEED)
        model.fit(X_train, y_train)
        dev_macro_f1 = float(f1_score(y_dev, model.predict(X_dev), average="macro", zero_division=0))
    model.fit(X, y)
    meta = dict(classes=model.classes_.tolist(), train_events=list(train_events),
                trained_at=to_iso(datetime.now(timezone.utc)), dev_macro_f1=dev_macro_f1)
    classifier_path = Path(classifier_path)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, **meta), classifier_path)
    write_json(classifier_path.parent / "classifier_meta.json", meta)
    return meta


class Tagger:
    def __init__(self, event: str, classifier_path: Path = config.CLASSIFIER_PATH,
                 embedder: Embedder | None = None, matcher: PlaceMatcher | None = None):
        import joblib

        artifact = joblib.load(classifier_path)
        self.model = artifact["model"]
        self.classes = np.asarray(artifact["classes"])
        if not np.array_equal(self.classes, self.model.classes_):
            raise ContractError("Classifier metadata classes do not match model.classes_")
        self.event, self.embedder = event, embedder
        self.matcher = matcher if matcher is not None else PlaceMatcher(event)

    def tag_batch(self, reports: list[Report], embeddings: np.ndarray | None = None,
                  start_row: int = 0) -> tuple[list[Tagged], np.ndarray]:
        if isinstance(start_row, bool) or not isinstance(start_row, int) or start_row < 0:
            raise ValueError("start_row must be an integer >= 0")
        if embeddings is None:
            if not reports:
                return [], np.empty((0, config.EMBED_DIM), dtype=np.float32)
            if self.embedder is None:
                self.embedder = Embedder()
            embeddings = self.embedder.encode([r.text for r in reports])
        embeddings = _check_embeddings(reports, embeddings)
        if not reports:
            return [], embeddings
        probabilities = self.model.predict_proba(embeddings)
        indices = probabilities.argmax(axis=1)
        tagged = []
        for i, report in enumerate(reports):
            category = str(self.classes[indices[i]])
            places, county = self.matcher.match(report.text)
            row = Tagged(report.report_id, start_row + i, category, round(float(probabilities[i, indices[i]]), 6),
                         places, county, find_vuln_terms(report.text), is_actionable(category))
            row.validate()
            tagged.append(row)
        return tagged, embeddings


def tag_event(event: str, force: bool = False) -> list[Tagged]:
    reports = read_jsonl(config.event_file(event, config.REPORTS_FILE), Report)
    embeddings = _check_embeddings(reports, load_embeddings(config.event_file(event, config.EMBEDDINGS_FILE)))
    target = config.event_file(event, config.TAGGED_FILE)
    tagged = None
    if target.exists() and not force:
        cached = read_jsonl(target, Tagged)
        if len(cached) == len(reports) and all(t.report_id == r.report_id and t.emb_row == i
                                               for i, (r, t) in enumerate(zip(reports, cached))):
            tagged = cached
    if tagged is None:
        tagged, _ = Tagger(event).tag_batch(reports, embeddings)
        write_jsonl(target, tagged)
    n = len(tagged)
    actionable = [t for t in tagged if t.is_actionable]
    metrics = dict(n=n, actionable_share=len(actionable) / n if n else 0.0,
                   place_coverage_actionable=sum(t.county_fips is not None for t in actionable) / len(actionable) if actionable else 0.0,
                   vuln_share=sum(bool(t.vuln_terms) for t in tagged) / n if n else 0.0,
                   category_counts=dict(sorted(Counter(t.pred_category for t in tagged).items())))
    write_json(config.event_file(event, "tag_stats.json"), metrics)
    return tagged


def evaluate_classifier(event: str = config.DEMO_EVENT) -> dict:
    import joblib
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    artifact = joblib.load(config.CLASSIFIER_PATH)
    assert event not in artifact["train_events"], "Evaluation event was used for training"
    reports = read_jsonl(config.event_file(event, config.REPORTS_FILE), Report)
    tagged = read_jsonl(config.event_file(event, config.TAGGED_FILE), Tagged)
    if len(reports) != len(tagged) or any(r.report_id != t.report_id or t.emb_row != i
                                        for i, (r, t) in enumerate(zip(reports, tagged))):
        raise ContractError("Report/tag alignment mismatch during evaluation")
    pairs = [(r.gold_label, t.pred_category) for r, t in zip(reports, tagged) if r.gold_label is not None]
    labels = list(CATEGORIES)
    if pairs:
        gold, predicted = zip(*pairs)
        per_class = f1_score(gold, predicted, labels=labels, average=None, zero_division=0)
        metrics = dict(n=len(pairs), accuracy=float(accuracy_score(gold, predicted)),
                       macro_f1=float(f1_score(gold, predicted, average="macro", zero_division=0)),
                       per_class_f1={label: float(score) for label, score in zip(labels, per_class)},
                       labels=labels, confusion=confusion_matrix(gold, predicted, labels=labels).tolist())
    else:
        metrics = dict(n=0, accuracy=None, macro_f1=None, per_class_f1={c: None for c in labels},
                       labels=labels, confusion=[[0] * len(labels) for _ in labels])
    write_json(config.event_file(event, "classifier_eval.json"), metrics)
    return metrics

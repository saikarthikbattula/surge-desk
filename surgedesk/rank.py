"""Offline incident urgency model, snapshot ranking, and live scoring API."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, fairness, fileio
from .contracts import FEATURE_KEYS, RANK_FEATURE_KEYS, Incident
from .features import _snapshot_row, build_snapshots, feature_definition, feature_keys, snapshot_columns
from .group import GroupState, SVILookup, group_event, load_stage1
from .textutil import parse_iso

assert "gold" not in " ".join(FEATURE_KEYS)

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", random_state=config.SEED)


def _target_description(objective):
    return ("HumAID requests_or_urgent_needs or missing_or_found_people; a category proxy, not independent dispatch-urgency labels"
            if objective == "assistance_v1" else
            "Legacy HumAID requests, injuries/deaths, or missing-person category proxy")


def validate_ranked_outputs(event, incidents):
    """Old schemas stay readable, but cannot silently serve a new objective."""
    if any(list(i.features) != feature_keys(event) for i in incidents):
        raise ValueError("Ranked outputs use a different objective/schema. Run stage ranker, then stage rank --force before brief/eval/demo.")


def _check_model(model, event):
    if model.meta.get("objective", "legacy") != config.ranking_objective(event):
        raise ValueError("Ranker objective changed. Retrain with python -m surgedesk.pipeline --stage ranker before rank --force.")
    if not event.startswith("fixture") and any(e.startswith("fixture") for e in model.meta["train_events"]):
        raise ValueError("Fixture-trained ranker cannot score a real event; run stage ranker on non-demo real training events.")
    expected = feature_definition(event)
    if expected is not None and model.meta.get("feature_definition") != expected:
        raise ValueError("Ranker assistance feature logic changed. Retrain stage ranker, then run stage rank --force.")


def _priority_scores(rows, scores, objective):
    scores = np.asarray(scores, dtype=float)
    if objective == "legacy":
        return scores
    # Preserve every incident. Only current-need cues enter the upper band;
    # model probability orders incidents within each explicit policy band.
    current = rows["share_current_assistance"].to_numpy() > 0
    return np.where(current, config.ASSISTANCE_SCORE_MIN + (1 - config.ASSISTANCE_SCORE_MIN) * scores,
                    config.CONTEXT_SCORE_MAX * scores)


def _signature(event, src, meta, model_path):
    paths = [src / name for name in (config.REPORTS_FILE, config.TAGGED_FILE, config.EMBEDDINGS_FILE)]
    paths += [config.PACKAGE_DIR / name for name in ("rank.py", "features.py", "group.py", "config.py", "contracts.py", "stubs.py")]
    if model_path is not None:
        paths.append(Path(model_path))
    if config.ranking_objective(event) == "assistance_v1":
        paths += [config.LEGACY_RANKER_DIR / config.RANKER_PATH.name,
                  config.LEGACY_RANKER_DIR / config.RANKER_META_PATH.name]
    digests = {}
    for path in paths:
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[str(path.resolve())] = digest.hexdigest()
    return {"objective": config.ranking_objective(event), "model_meta": meta, "input_hashes": digests}


def default_svi_lookup() -> SVILookup:
    try:
        from .geo import SVI
    except ImportError:
        print("Warning: surgedesk.geo unavailable; SVI will be unknown")
        return lambda f: None
    return SVI().get


def train_ranker(train_events: list[str], svi_lookup: SVILookup | None = None,
                 outputs_root: Path = config.OUTPUTS, ranker_path: Path = config.RANKER_PATH,
                 allow_fixture: bool = False) -> dict:
    assert config.DEMO_EVENT not in train_events, "Never train on the demo event"
    assert allow_fixture or not any(e.startswith("fixture") for e in train_events), "Fixture training requires allow_fixture=True"
    objectives = {config.ranking_objective(e) for e in train_events}
    if len(objectives) != 1:
        raise ValueError("Training events must share one ranking objective")
    objective = objectives.pop()
    keys = RANK_FEATURE_KEYS if objective == "assistance_v1" else FEATURE_KEYS
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBClassifier

    lookup = svi_lookup if svi_lookup is not None else default_svi_lookup()
    ranker_path = Path(ranker_path)
    frames = []
    for event in train_events:
        src = Path(outputs_root) / event
        # Fixture sources are frozen; generated training artifacts go beside the temporary model.
        out = ranker_path.parent / event if allow_fixture else src
        state = group_event(event, src_dir=src, out_dir=out)
        reports = load_stage1(event, src)[0]
        frame = build_snapshots(event, state, reports, lookup, out_dir=out)
        frame = frame.loc[frame["urgent_proxy"].notna()].copy()
        frame["_group"] = event + ":" + frame["incident_id"]
        frames.append(frame)
    if not frames or not any(len(f) for f in frames):
        raise ValueError("No labeled incident snapshots available for ranker training")
    frame = pd.concat(frames, ignore_index=True)
    X = frame[keys].astype(float)
    y = frame["urgent_proxy"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Ranker training requires both urgent and non-urgent incidents")

    def fit(indices):
        labels = y.iloc[indices]
        pos = int(labels.sum())
        params = dict(XGB_PARAMS, n_jobs=config.N_THREADS,
                      scale_pos_weight=(len(labels) - pos) / pos if pos else 1.0)
        model = XGBClassifier(**params)
        model.fit(X.iloc[indices], labels)
        return model, params

    dev_auc = dev_ap = None
    if frame["_group"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
        train_idx, dev_idx = next(splitter.split(X, y, groups=frame["_group"]))
        if y.iloc[train_idx].nunique() > 1 and y.iloc[dev_idx].nunique() > 1:
            dev_model, _ = fit(train_idx)
            probs = dev_model.predict_proba(X.iloc[dev_idx])[:, 1]
            dev_auc = float(roc_auc_score(y.iloc[dev_idx], probs))
            dev_ap = float(average_precision_score(y.iloc[dev_idx], probs))
    model, params = fit(np.arange(len(frame)))
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    previous_meta = ranker_path.parent / config.RANKER_META_PATH.name
    if not allow_fixture and objective == "assistance_v1" and ranker_path.exists() and previous_meta.exists():
        old = fileio.read_json(previous_meta)
        if old.get("objective", "legacy") == "legacy":
            archive = config.LEGACY_RANKER_DIR
            archive.mkdir(parents=True, exist_ok=True)
            if not (archive / ranker_path.name).exists() or not (archive / previous_meta.name).exists():
                shutil.copy2(ranker_path, archive / ranker_path.name)
                shutil.copy2(previous_meta, archive / previous_meta.name)
    model.save_model(ranker_path)
    meta = dict(feature_keys=list(keys), objective=objective, target_definition=_target_description(objective),
                score_interpretation="Explicit assistance-priority bands with model probability ordering within bands" if objective == "assistance_v1" else "Legacy model score",
                priority_bands={"current_min": config.ASSISTANCE_SCORE_MIN, "context_max": config.CONTEXT_SCORE_MAX,
                                "context_fairness_cap": config.CONTEXT_ADJUSTED_MAX} if objective == "assistance_v1" else None,
                feature_definition=feature_definition(train_events[0]),
                train_events=list(train_events), n_rows=len(frame),
                pos_rate=float(y.mean()), dev_auc=dev_auc, dev_ap=dev_ap, params=params,
                trained_at=datetime.now(timezone.utc).isoformat())
    fileio.write_json(ranker_path.parent / config.RANKER_META_PATH.name, meta)
    return meta


class Ranker:
    def __init__(self, path: Path = config.RANKER_PATH):
        from xgboost import Booster

        path = Path(path)
        self.meta = fileio.read_json(path.parent / config.RANKER_META_PATH.name)
        self.path = path
        self.keys = self.meta["feature_keys"]
        expected = RANK_FEATURE_KEYS if self.meta.get("objective", "legacy") == "assistance_v1" else FEATURE_KEYS
        assert self.keys == expected, "Ranker feature order differs; retrain stage ranker"
        if self.meta.get("objective", "legacy") == "assistance_v1":
            definition = feature_definition(self.meta["train_events"][0])
            if definition is None or self.meta.get("feature_definition") != definition:
                raise ValueError("Ranker assistance feature logic changed. Retrain stage ranker, then run stage rank --force.")
        self.booster = Booster(params={"nthread": config.N_THREADS})
        self.booster.load_model(path)
        self.booster.set_param({"nthread": config.N_THREADS})

    def _matrix(self, X: pd.DataFrame):
        from xgboost import DMatrix

        return DMatrix(X[self.keys].astype(float).values, feature_names=self.keys,
                       nthread=config.N_THREADS)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.empty(0, dtype=float)
        return np.clip(self.booster.predict(self._matrix(X)), 0.0, 1.0)

    def explain(self, X: pd.DataFrame, k: int = 5) -> list[list[list]]:
        if X.empty:
            return []
        values = self.booster.predict(self._matrix(X), pred_contribs=True)[:, :-1]
        count = min(max(k, 0), 5, len(self.keys))
        return [[[self.keys[j], round(float(row[j]), 6)]
                 for j in sorted(range(len(row)), key=lambda j: -abs(float(row[j])))[:count]]
                for row in values]


def rows_to_incidents(event: str, rows: pd.DataFrame, scores, explanations) -> list[Incident]:
    assert len(rows) == len(scores) == len(explanations), "Ranking row counts differ"
    records = rows.to_dict("records")
    rounded = [round(float(s), 6) for s in scores]
    order = sorted(range(len(records)), key=lambda j: (
        -rounded[j], parse_iso(records[j]["first_seen"]), records[j]["incident_id"]))
    out = []
    for rank, j in enumerate(order, 1):
        row = records[j]
        svi = row["svi"]
        label = row["urgent_proxy"]
        inc = Incident(
            incident_id=row["incident_id"], event=event, snapshot_time=row["snapshot_time"],
            report_ids=json.loads(row["report_ids"]), first_seen=row["first_seen"], last_seen=row["last_seen"],
            place=None if pd.isna(row["place"]) else row["place"],
            county_fips=None if pd.isna(row["county_fips"]) else row["county_fips"],
            svi=None if pd.isna(svi) or svi < 0 else float(svi),
            features={key: float(row[key]) for key in feature_keys(event)},
            score=rounded[j], score_adj=rounded[j], lam=0.0, rank=rank, rank_adj=rank,
            shap_top=explanations[j], urgent_proxy=None if pd.isna(label) else int(label),
        )
        inc.validate()
        out.append(inc)
    return out


def rank_now(state: GroupState, now: datetime, ranker: Ranker, svi_lookup: SVILookup,
             lam: float = 0.0) -> list[Incident]:
    _check_model(ranker, state.event)
    rows = pd.DataFrame([_snapshot_row(inc, now, svi_lookup) for inc in state.open_incidents(now)],
                        columns=snapshot_columns(state.event))
    if rows.empty:
        return []
    scores = _priority_scores(rows, ranker.score(rows), config.ranking_objective(state.event))
    return fairness.apply(rows_to_incidents(state.event, rows, scores, ranker.explain(rows)), lam)


def rank_event(event: str = config.DEMO_EVENT, force: bool = False, svi_lookup: SVILookup | None = None,
               src_dir: Path | None = None, out_dir: Path | None = None,
               ranker: Ranker | None = None) -> list[Incident]:
    out = Path(out_dir) if out_dir is not None else config.event_dir(event)
    src = Path(src_dir) if src_dir is not None else config.event_dir(event)
    path = out / config.INCIDENTS_FILE
    model = ranker if ranker is not None else None
    if path.exists() and not force:
        cached = fileio.read_jsonl(path, Incident)
        validate_ranked_outputs(event, cached)
        if config.ranking_objective(event) == "assistance_v1":
            manifest = out / config.RANK_PROVENANCE_FILE
            meta = model.meta if model else fileio.read_json(config.RANKER_META_PATH)
            from types import SimpleNamespace
            _check_model(model if model else SimpleNamespace(meta=meta), event)
            signature = _signature(event, src, meta, getattr(model, "path", config.RANKER_PATH))
            if not manifest.exists() or fileio.read_json(manifest).get("signature") != signature:
                raise ValueError("Ranked outputs are stale after model/input changes; rerun stage rank --force.")
        return cached
    lookup = svi_lookup if svi_lookup is not None else default_svi_lookup()
    state = group_event(event, src_dir=src_dir, out_dir=out)
    reports = load_stage1(event, src_dir)[0]
    rows = build_snapshots(event, state, reports, lookup, out_dir=out)
    incidents, score_rows = [], []
    if not rows.empty:
        model = model if model is not None else Ranker()
        _check_model(model, event)
        raw_scores, explanations = model.score(rows), model.explain(rows)
        scores = _priority_scores(rows, raw_scores, config.ranking_objective(event))
        legacy_scores = None
        legacy_path = config.LEGACY_RANKER_DIR / config.RANKER_PATH.name
        if config.ranking_objective(event) == "assistance_v1" and legacy_path.exists():
            previous = Ranker(legacy_path)
            if previous.meta.get("objective", "legacy") != "legacy":
                raise ValueError("Archived comparison model is not a legacy ranker; restore the preserved legacy model before ranking.")
            if event in previous.meta["train_events"] or config.DEMO_EVENT in previous.meta["train_events"]:
                raise ValueError("Legacy comparison model was trained on the demo event")
            legacy_scores = previous.score(rows)
        scored = rows.copy()
        scored["_score"] = scores
        scored["_model_score"] = raw_scores
        scored["_legacy_score"] = legacy_scores if legacy_scores is not None else [None] * len(rows)
        scored["_explanation"] = explanations
        for _, snapshot in scored.groupby("snapshot_time", sort=True):
            incidents.extend(rows_to_incidents(event, snapshot, snapshot["_score"].tolist(),
                                               snapshot["_explanation"].tolist()))
            for row in snapshot.to_dict("records"):
                score_rows.append(dict(incident_id=row["incident_id"], snapshot_time=row["snapshot_time"],
                                       model_score=row["_model_score"], priority_score=row["_score"],
                                       legacy_model_score=row["_legacy_score"]))
    fileio.write_jsonl(path, incidents)
    fileio.write_jsonl(out / config.RANK_SCORES_FILE, score_rows)
    fileio.write_json(out / config.RANK_PROVENANCE_FILE,
                     {"signature": _signature(event, src, model.meta if model else {}, getattr(model, "path", None)),
                      "objective": config.ranking_objective(event), "target_definition": _target_description(config.ranking_objective(event)),
                      "legacy_model_comparison": bool(score_rows and score_rows[0]["legacy_model_score"] is not None)})
    return incidents


def pick_demo_snapshot(incidents: list[Incident]) -> str:
    if os.environ.get("SURGEDESK_SNAPSHOT"):
        return os.environ["SURGEDESK_SNAPSHOT"]
    totals: dict[str, float] = {}
    for inc in incidents:
        totals[inc.snapshot_time] = totals.get(inc.snapshot_time, 0.0) + inc.features["reports_last_60m"]
    if not totals:
        raise ValueError("Cannot select a snapshot from an empty incident list")
    return min(totals, key=lambda snap: (-totals[snap], parse_iso(snap)))

"""Offline incident urgency model, snapshot ranking, and live scoring API."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, fairness, fileio
from .contracts import FEATURE_KEYS, Incident
from .features import SNAPSHOT_COLUMNS, _snapshot_row, build_snapshots
from .group import GroupState, SVILookup, group_event, load_stage1
from .textutil import parse_iso

assert "gold" not in " ".join(FEATURE_KEYS)

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", random_state=config.SEED)


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
    X = frame[FEATURE_KEYS].astype(float)
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
    model.save_model(ranker_path)
    meta = dict(feature_keys=list(FEATURE_KEYS), train_events=list(train_events), n_rows=len(frame),
                pos_rate=float(y.mean()), dev_auc=dev_auc, dev_ap=dev_ap, params=params,
                trained_at=datetime.now(timezone.utc).isoformat())
    fileio.write_json(ranker_path.parent / config.RANKER_META_PATH.name, meta)
    return meta


class Ranker:
    def __init__(self, path: Path = config.RANKER_PATH):
        from xgboost import Booster

        path = Path(path)
        self.meta = fileio.read_json(path.parent / config.RANKER_META_PATH.name)
        assert self.meta["feature_keys"] == FEATURE_KEYS, "Ranker feature order differs"
        self.booster = Booster(params={"nthread": config.N_THREADS})
        self.booster.load_model(path)
        self.booster.set_param({"nthread": config.N_THREADS})

    def _matrix(self, X: pd.DataFrame):
        from xgboost import DMatrix

        return DMatrix(X[FEATURE_KEYS].astype(float).values, feature_names=FEATURE_KEYS,
                       nthread=config.N_THREADS)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.empty(0, dtype=float)
        return np.clip(self.booster.predict(self._matrix(X)), 0.0, 1.0)

    def explain(self, X: pd.DataFrame, k: int = 5) -> list[list[list]]:
        if X.empty:
            return []
        values = self.booster.predict(self._matrix(X), pred_contribs=True)[:, :-1]
        count = min(max(k, 0), 5, len(FEATURE_KEYS))
        return [[[FEATURE_KEYS[j], round(float(row[j]), 6)]
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
            features={key: float(row[key]) for key in FEATURE_KEYS},
            score=rounded[j], score_adj=rounded[j], lam=0.0, rank=rank, rank_adj=rank,
            shap_top=explanations[j], urgent_proxy=None if pd.isna(label) else int(label),
        )
        inc.validate()
        out.append(inc)
    return out


def rank_now(state: GroupState, now: datetime, ranker: Ranker, svi_lookup: SVILookup,
             lam: float = 0.0) -> list[Incident]:
    rows = pd.DataFrame([_snapshot_row(inc, now, svi_lookup) for inc in state.open_incidents(now)],
                        columns=SNAPSHOT_COLUMNS)
    if rows.empty:
        return []
    return fairness.apply(rows_to_incidents(state.event, rows, ranker.score(rows), ranker.explain(rows)), lam)


def rank_event(event: str = config.DEMO_EVENT, force: bool = False, svi_lookup: SVILookup | None = None,
               src_dir: Path | None = None, out_dir: Path | None = None,
               ranker: Ranker | None = None) -> list[Incident]:
    out = Path(out_dir) if out_dir is not None else config.event_dir(event)
    path = out / config.INCIDENTS_FILE
    if path.exists() and not force:
        return fileio.read_jsonl(path, Incident)
    lookup = svi_lookup if svi_lookup is not None else default_svi_lookup()
    state = group_event(event, src_dir=src_dir, out_dir=out)
    reports = load_stage1(event, src_dir)[0]
    rows = build_snapshots(event, state, reports, lookup, out_dir=out)
    incidents = []
    if not rows.empty:
        model = ranker if ranker is not None else Ranker()
        scores, explanations = model.score(rows), model.explain(rows)
        scored = rows.copy()
        scored["_score"] = scores
        scored["_explanation"] = explanations
        for _, snapshot in scored.groupby("snapshot_time", sort=True):
            incidents.extend(rows_to_incidents(event, snapshot, snapshot["_score"].tolist(),
                                               snapshot["_explanation"].tolist()))
    fileio.write_jsonl(path, incidents)
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

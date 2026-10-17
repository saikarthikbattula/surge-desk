"""Reproducible evaluation. Pitch metrics are computed here, never in notebooks."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, envcheck, fileio
from .contracts import BRIEF_FIELDS, HAND_LABEL_COLUMNS, PURITY_COLUMNS, Briefing, Incident, Report, Tagged
from .pipeline import StageUnavailable, pick_snapshot, require_module
from .textutil import content_words, mask_privacy, parse_iso, to_iso


def _dir(event, src_dir):
    d = Path(src_dir) if src_dir is not None else config.event_dir(event)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(d, name, value):
    value = _clean(value)
    fileio.write_json(d / name, value)
    return value


def _skipped(reason):
    return {"status": "skipped", "reason": str(reason)}


def measured_caption(info=None):
    info = info or envcheck.collect()
    if info["arch"] != "s390x":
        return "Development run, NOT LinuxONE"
    ram = info.get("ram_gb")
    return f'Measured on IBM LinuxONE ({info["arch"]}), {info["cores"]} vCPU, {round(ram) if ram else "unknown"} GB'


def _figure(d, name, title, draw, info=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    draw(ax)
    ax.set_title(title)
    fig.supxlabel(measured_caption(info), fontsize=9)
    plots = d / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / name, dpi=140)


def _read_labels(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "report_id" in frame and frame.report_id.str.strip().str.match(
        r"^[+-]?\d+(?:\.\d*)?[eE][+-]?\d+$"
    ).any():
        raise ValueError(f"{path}: report_id is in scientific notation; re-export with the report_id column formatted as plain text")
    return frame


def _hand_labels():
    path = config.LABELS_DIR / "hand_labels.csv"
    if not path.exists():
        return None
    labels = _read_labels(path)
    missing = set(HAND_LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    keep = labels.labeler.str.strip().ne("") & labels[BRIEF_FIELDS].apply(
        lambda row: any(str(v).strip() for v in row), axis=1)
    labels = labels.loc[keep].copy()
    if labels.report_id.duplicated().any():
        raise ValueError(f"{path}: duplicate report_id labels")
    return labels


def make_label_sheet(event=config.DEMO_EVENT, n=config.LABEL_SAMPLE_N,
                     n_actionable=config.LABEL_SAMPLE_ACTIONABLE, seed=config.SEED,
                     src_dir: Path | None = None, force: bool = False) -> Path:
    """Sample without replacement; never overwrite human work without force."""
    path = config.LABELS_DIR / "to_label.csv"
    if path.exists() and not force:
        return path
    if n < 0 or n_actionable < 0 or n_actionable > n:
        raise ValueError("Require 0 <= n_actionable <= n")
    d = _dir(event, src_dir)
    reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
    tags = {t.report_id: t for t in fileio.read_jsonl(d / config.TAGGED_FILE, Tagged)}
    rng = np.random.default_rng(seed)
    selected = []
    for actionable, count in ((True, n_actionable), (False, n - n_actionable)):
        candidates = [r for r in reports if r.report_id in tags and tags[r.report_id].is_actionable == actionable]
        selected.extend(candidates[int(i)] for i in rng.choice(len(candidates), min(count, len(candidates)), replace=False))
    rows = []
    for idx, report in enumerate(selected):
        row = dict.fromkeys(HAND_LABEL_COLUMNS, "")
        row.update(report_id=report.report_id, text=mask_privacy(report.text), labeler=f"P{idx % 4 + 1}")
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=HAND_LABEL_COLUMNS).to_csv(path, index=False)
    return path


def make_purity_sheet(event=config.DEMO_EVENT, n=config.PURITY_SAMPLE_N, seed=config.SEED,
                      src_dir: Path | None = None, force: bool = False) -> Path:
    path = config.LABELS_DIR / "purity_to_label.csv"
    if path.exists() and not force:
        return path
    if n < 0:
        raise ValueError("n must be nonnegative")
    d = _dir(event, src_dir)
    groups = [g for g in fileio.read_jsonl(d / "groups.jsonl") if len(g["report_ids"]) >= 2]
    reports = {r.report_id: r for r in fileio.read_jsonl(d / config.REPORTS_FILE, Report)}
    rng = np.random.default_rng(seed)
    rows = []
    for idx in rng.choice(len(groups), min(n, len(groups)), replace=False):
        group = groups[int(idx)]
        for rid in group["report_ids"]:
            rows.append(dict(incident_id=group["incident_id"], report_id=rid,
                             text=mask_privacy(reports[rid].text), same_incident="", labeler=""))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PURITY_COLUMNS).to_csv(path, index=False)
    return path


def extraction_match(hand, predicted):
    """Empty values agree; otherwise match at least half of the hand content words."""
    hand = str(hand).strip() if hand is not None else ""
    predicted = str(predicted).strip() if predicted is not None else ""
    if not hand or not predicted:
        return not hand and not predicted
    words = content_words(hand)
    return bool(words) and sum(w in set(content_words(predicted)) for w in words) / len(words) >= 0.5


def eval_classifier(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    path = d / "classifier_eval.json"
    if not path.exists():
        return _save(d, "classifier_metrics.json", _skipped("classifier_eval.json not available yet (owner: Agent 1)"))
    result = fileio.read_json(path)
    confusion = np.asarray(result["confusion"])
    labels = result["labels"]
    if confusion.shape != (len(labels), len(labels)):
        raise ValueError("classifier_eval.json: confusion shape does not match labels")

    def draw(ax):
        image = ax.imshow(confusion, cmap="Blues")
        ax.figure.colorbar(image, ax=ax)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    _figure(d, "confusion.png", "Held-out event classifier confusion", draw)
    return _save(d, "classifier_metrics.json", result)


def eval_extraction(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    labels = _hand_labels()
    path = d / "extraction_preds.jsonl"
    if labels is None or labels.empty or not path.exists():
        return _save(d, "extraction_metrics.json", _skipped("Awaiting hand labels and extraction_preds.jsonl (owner: Agent 3)"))
    reports_path = d / config.REPORTS_FILE
    if not reports_path.exists():
        return _save(d, "extraction_metrics.json", _skipped("reports.jsonl not available yet (owner: Agent 1)"))
    report_ids = {r.report_id for r in fileio.read_jsonl(reports_path, Report)}
    labels = labels[labels.report_id.isin(report_ids)]
    if labels.empty:
        return _save(d, "extraction_metrics.json", _skipped("No labeled reports for this event"))
    predictions = {p["report_id"]: p for p in fileio.read_jsonl(path)}
    missing = set(labels.report_id) - set(predictions)
    if missing:
        return _save(d, "extraction_metrics.json", _skipped(f"Missing predictions for {len(missing)} labeled reports (owner: Agent 3)"))
    matched = [(row, predictions[row.report_id]) for row in labels.itertuples(index=False)]
    result = {}
    for field in BRIEF_FIELDS:
        pairs = [(getattr(row, field).strip(), pred["fields"][field]) for row, pred in matched]
        nulls = [not hand for hand, _ in pairs]
        nonnull = [fv["value"] is not None and str(fv["value"]).strip() != "" for _, fv in pairs]
        result[field] = {
            "match_rate": sum(extraction_match(hand, fv["value"]) for hand, fv in pairs) / len(pairs),
            "null_agreement": sum(not hand and not nonnull[idx] for idx, (hand, _) in enumerate(pairs)) / sum(nulls) if any(nulls) else None,
            "predicted_non_null_rate": sum(nonnull) / len(pairs),
            "grounded_rate": sum(bool(fv["grounded"]) and nonnull[idx] for idx, (_, fv) in enumerate(pairs)) / sum(nonnull) if any(nonnull) else None,
        }
    result.update(json_valid_rate=sum(bool(p["timing"].get("ok", False)) for _, p in matched) / len(matched),
                  mean_seconds_per_report=float(np.mean([p["timing"]["seconds"] for _, p in matched])),
                  n_labeled=len(matched))
    pd.DataFrame([dict(field=f, **result[f]) for f in BRIEF_FIELDS]).to_csv(d / "extraction_metrics.csv", index=False)
    return _save(d, "extraction_metrics.json", result)


def eval_grouping(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    path = d / "group_stats.json"
    result = fileio.read_json(path) if path.exists() else _skipped("group_stats.json not available yet (owner: Agent 2)")
    purity = config.LABELS_DIR / "purity_labels.csv"
    result["purity"] = None
    if purity.exists():
        rows = _read_labels(purity)
        if not set(PURITY_COLUMNS).issubset(rows.columns):
            raise ValueError("purity_labels.csv: missing required columns")
        incidents_path = d / config.INCIDENTS_FILE
        if incidents_path.exists():
            ids = {i.incident_id for i in fileio.read_jsonl(incidents_path, Incident)}
            rows = rows[rows.incident_id.isin(ids)]
            values = rows.same_incident.str.strip().str.lower()
            if len(values):
                result["purity"] = float(values.isin({"y", "yes", "1", "true"}).mean())
    return _save(d, "grouping_metrics.json", result)


def ranking_metrics(incidents: list[Incident], order: list[str], k=config.TOP_K_EVAL) -> dict:
    """Binary relevance at k, including an ideal DCG over the complete snapshot."""
    by_id = {i.incident_id: i for i in incidents}
    if len(order) != len(by_id) or set(order) != set(by_id):
        raise ValueError("Ranking order must be a permutation of snapshot incident IDs")
    k = min(k, len(incidents))
    urgent = sum(i.urgent_proxy == 1 for i in incidents)
    if not urgent or k <= 0:
        return _skipped("No urgent incidents at snapshot")
    relevance = [int(by_id[iid].urgent_proxy == 1) for iid in order[:k]]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal = sum(1 / math.log2(idx + 2) for idx in range(min(k, urgent)))
    return {"p_at_10": sum(relevance) / k, "r_at_10": sum(relevance) / urgent,
            "ndcg_at_10": dcg / ideal}


def time_to_surface(incidents: list[Incident], orders_by_snapshot: dict[str, list[str]],
                    k=config.TOP_K_EVAL) -> list[dict]:
    """A missing top-k appearance is censored, never replaced with zero minutes."""
    first_seen = {}
    urgent = set()
    for i in incidents:
        if i.incident_id not in first_seen or parse_iso(i.first_seen) < parse_iso(first_seen[i.incident_id]):
            first_seen[i.incident_id] = i.first_seen
        if i.urgent_proxy == 1:
            urgent.add(i.incident_id)
    surfaced = {}
    for snap in sorted(orders_by_snapshot, key=parse_iso):
        for iid in orders_by_snapshot[snap][:k]:
            if iid in urgent and iid not in surfaced:
                surfaced[iid] = snap
    rows = []
    for iid in sorted(urgent):
        snap = surfaced.get(iid)
        minutes = (parse_iso(snap) - parse_iso(first_seen[iid])).total_seconds() / 60 if snap else None
        if minutes is not None and minutes < 0:
            raise ValueError(f"{iid}: surfaced before first_seen")
        rows.append({"incident_id": iid, "first_seen": first_seen[iid], "surface_snapshot": snap,
                     "minutes": minutes, "censored": snap is None})
    return rows


def eval_ranking(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    path = d / config.INCIDENTS_FILE
    if not path.exists():
        return _save(d, "ranking_eval.json", _skipped("incidents.jsonl not available yet (owner: Agent 2)"))
    try:
        baselines = require_module("baselines")
    except StageUnavailable as exc:
        return _save(d, "ranking_eval.json", _skipped(exc))
    reports_path = d / config.REPORTS_FILE
    if not reports_path.exists():
        return _save(d, "ranking_eval.json", _skipped("reports.jsonl not available yet (owner: Agent 1)"))
    incidents = fileio.read_jsonl(path, Incident)
    texts = {r.report_id: mask_privacy(r.text) for r in fileio.read_jsonl(reports_path, Report)}
    snapshots = sorted({i.snapshot_time for i in incidents})
    methods = {"SurgeDesk": lambda rows: baselines.order_surgedesk(
                   sorted(rows, key=lambda i: i.rank)),
               "FIFO": baselines.order_fifo, "Volume": baselines.order_volume,
               "Keyword": lambda rows: baselines.order_keyword(rows, texts)}
    metric_rows, surface_rows, means, times = [], [], {}, {}
    for name, order in methods.items():
        orders = {}
        for snap in snapshots:
            rows = [i for i in incidents if i.snapshot_time == snap]
            # Evaluate lambda=0 from stored raw ranks even if a caller supplied adjusted records.
            if name == "SurgeDesk":
                from dataclasses import replace
                rows = [replace(i, rank_adj=i.rank, score_adj=i.score, lam=0.0) for i in rows]
            orders[snap] = order(rows)
            metrics = ranking_metrics(rows, orders[snap])
            if metrics.get("status") != "skipped":
                metric_rows.append(dict(method=name, snapshot_time=snap,
                                        k=min(config.TOP_K_EVAL, len(rows)), **metrics))
        selected = [r for r in metric_rows if r["method"] == name]
        means[name] = {key: float(np.mean([r[key] for r in selected])) if selected else None
                       for key in ("p_at_10", "r_at_10", "ndcg_at_10")}
        surface = time_to_surface(incidents, orders)
        surface_rows.extend(dict(method=name, **r) for r in surface)
        observed = [r["minutes"] for r in surface if not r["censored"]]
        times[name] = {"median_min": float(np.median(observed)) if observed else None,
                       "share_surfaced": len(observed) / len(surface) if surface else None}
    pd.DataFrame(metric_rows, columns=["method", "snapshot_time", "k", "p_at_10", "r_at_10", "ndcg_at_10"]).to_csv(d / "ranking_metrics.csv", index=False)
    pd.DataFrame(surface_rows, columns=["method", "incident_id", "first_seen", "surface_snapshot", "minutes", "censored"]).to_csv(d / "time_to_surface.csv", index=False)

    def draw(ax):
        ax.bar(list(times), [times[m]["median_min"] if times[m]["median_min"] is not None else float("nan") for m in times])
        ax.set_ylabel("Median minutes (surfaced incidents only)")

    _figure(d, "time_to_surface.png", "Time to first top-k appearance; censored rows excluded", draw)
    return _save(d, "ranking_eval.json", {"ranking": means, "time_to_surface": times})


def eval_fairness(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    path = d / config.INCIDENTS_FILE
    if not path.exists():
        return _save(d, "fairness_metrics.json", _skipped("incidents.jsonl not available yet (owner: Agent 2)"))
    try:
        fairness = require_module("fairness")
        incidents = fileio.read_jsonl(path, Incident)
        if not incidents:
            return _save(d, "fairness_metrics.json", _skipped("No incident snapshots"))
        snap = pick_snapshot(incidents, d, fixture=event.startswith("fixture"))
    except StageUnavailable as exc:
        return _save(d, "fairness_metrics.json", _skipped(exc))
    tables = []
    for s in sorted({i.snapshot_time for i in incidents}):
        table = fairness.fairness_table([i for i in incidents if i.snapshot_time == s]).copy()
        table["snapshot_time"] = s
        tables.append(table)
    all_rows = pd.concat(tables, ignore_index=True)
    numeric = [c for c in all_rows.columns if c not in ("lam", "snapshot_time")]
    mean = all_rows.groupby("lam", as_index=False)[numeric].mean()
    mean["scope"] = "mean"
    demo = all_rows[all_rows.snapshot_time == snap].copy()
    demo["scope"] = "demo"
    pd.concat([demo, mean], ignore_index=True).to_csv(d / "fairness.csv", index=False)

    def draw(ax):
        ax.plot(mean.lam, mean.precision_at_k, marker="o", label="Precision@10")
        ax.plot(mean.lam, mean.high_svi_share_top_k, marker="o", label="High-SVI share")
        ax.set_xlabel("λ")
        ax.set_ylim(0, 1)
        ax.legend()

    _figure(d, "fairness.png", "Mean over snapshots: ranking and vulnerability tradeoff", draw)
    return _save(d, "fairness_metrics.json", {"demo_snapshot": snap,
                 "demo": demo.drop(columns="scope").to_dict("records"),
                 "mean": mean.drop(columns="scope").to_dict("records")})


def briefing_metrics(briefs: list[Briefing]) -> dict:
    fields = [field for b in briefs for field in b.fields.values()]
    sentences = [s for b in briefs for s in b.sentences]
    return {"n": len(briefs),
            "fields_grounded_share": sum(f["grounded"] for f in fields) / len(fields) if fields else None,
            "sentences_accepted_share": sum(s["accepted"] for s in sentences) / len(sentences) if sentences else None,
            "fallback_share": sum(b.used_fallback for b in briefs) / len(briefs) if briefs else None}


def _markdown_table(records):
    if not records:
        return "Not available yet.\n"
    columns = list(dict.fromkeys(k for row in records for k in row))

    def cell(value):
        if value is None:
            return "not available yet"
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        return mask_privacy(text).replace("|", "\\|").replace("\n", " ")

    return ("| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" +
            "\n".join("| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in records) + "\n")


def _results(summary, sections, info):
    caption = measured_caption(info)
    if info["arch"] != "s390x":
        caption = f"**{caption}**"
    text = f"{caption}, {to_iso(datetime.now(timezone.utc))}\n\n# SurgeDesk measured results\n\n"
    text += "Generated by `surgedesk.evaluate`; fixtures and stub outputs are development evidence only.\n\n"
    text += _markdown_table([dict(event=summary["event"], **summary["measured_on"])])
    for title, keys in (("Data and grouping", ["n_reports", "n_actionable", "n_incidents", "singleton_share", "purity"]),
                        ("Classifier and ranker", ["classifier_macro_f1", "ranker_dev_auc"])):
        text += f"\n## {title}\n\n" + _markdown_table([{k: summary[k] for k in keys}])
    for title, key in (("Ranking (mean over eligible snapshots)", "ranking"), ("Time to surface", "time_to_surface")):
        text += f"\n## {title}\n\n" + _markdown_table([dict(method=m, **v) for m, v in summary[key].items()])
    text += "\nTime-to-surface medians include surfaced incidents only; share_surfaced includes censored incidents. Snapshots without any urgent proxy are excluded from ranking metrics.\n"
    text += "\n## Fairness at demo snapshot\n\n" + _markdown_table(summary["fairness_demo_snapshot"])
    text += "\n## Fairness mean over snapshots\n\n" + _markdown_table(sections["fairness"].get("mean", []))
    text += "\n## Grouping details\n\n" + _markdown_table([sections["grouping"]])
    extraction = summary["extraction"]
    text += "\n## Extraction\n\n"
    if extraction.get("status") == "skipped":
        text += mask_privacy(extraction["reason"]) + "\n"
    else:
        text += _markdown_table([dict(field=f, **extraction[f]) for f in BRIEF_FIELDS])
        text += _markdown_table([{k: v for k, v in extraction.items() if k not in BRIEF_FIELDS}])
    text += "\nGrounded rate is conditional on a non-null prediction; null agreement is conditional on a null hand label. Empty denominators are unavailable.\n"
    text += "\n## Briefings\n\n" + _markdown_table([summary["briefings"]])
    text += "\n## Availability\n\n" + _markdown_table([dict(evaluation=k, **v) for k, v in sections.items() if v.get("status") == "skipped"])
    fileio._atomic_write_text(config.DOCS / "RESULTS.md", text)


def run_all(event=config.DEMO_EVENT, src_dir: Path | None = None) -> dict:
    d = _dir(event, src_dir)
    # Samples are prepared before evaluation, but missing upstream files never block other metrics.
    for sample in (make_label_sheet, make_purity_sheet):
        try:
            sample(event, src_dir=d)
        except FileNotFoundError:
            pass
    labels = _hand_labels()
    extraction_error = None
    if labels is not None and not labels.empty and (d / config.REPORTS_FILE).exists():
        report_ids = {r.report_id for r in fileio.read_jsonl(d / config.REPORTS_FILE, Report)}
        ids = [rid for rid in labels.report_id if rid in report_ids]
        pred_path = d / "extraction_preds.jsonl"
        covered = {p["report_id"] for p in fileio.read_jsonl(pred_path)} if pred_path.exists() else set()
        if set(ids) - covered:
            try:
                require_module("brief").extract_for_eval(event, ids, src_dir=d, out_dir=d)
            except (StageUnavailable, FileNotFoundError) as exc:
                extraction_error = _skipped(exc)
    sections = {"classifier": eval_classifier(event, d), "extraction": eval_extraction(event, d),
                "grouping": eval_grouping(event, d), "ranking": eval_ranking(event, d),
                "fairness": eval_fairness(event, d)}
    if extraction_error:
        sections["extraction"] = _save(d, "extraction_metrics.json", extraction_error)
    reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report) if (d / config.REPORTS_FILE).exists() else None
    tags = fileio.read_jsonl(d / config.TAGGED_FILE, Tagged) if (d / config.TAGGED_FILE).exists() else None
    incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident) if (d / config.INCIDENTS_FILE).exists() else None
    briefs = fileio.read_jsonl(d / config.BRIEFINGS_FILE, Briefing) if (d / config.BRIEFINGS_FILE).exists() else []
    meta = fileio.read_json(config.RANKER_META_PATH) if config.RANKER_META_PATH.exists() else {}
    info = envcheck.collect()
    summary = {"event": event, "measured_on": {k: info[k] for k in ("arch", "cores", "ram_gb", "nnpa_flag")},
               "n_reports": len(reports) if reports is not None else None,
               "n_actionable": sum(t.is_actionable for t in tags) if tags is not None else None,
               "n_incidents": sections["grouping"].get("n_incidents", len({i.incident_id for i in incidents}) if incidents is not None else None),
               "singleton_share": sections["grouping"].get("singleton_share"),
               "classifier_macro_f1": sections["classifier"].get("macro_f1"), "ranker_dev_auc": meta.get("dev_auc"),
               "ranking": sections["ranking"].get("ranking", {}),
               "time_to_surface": sections["ranking"].get("time_to_surface", {}),
               "fairness_demo_snapshot": sections["fairness"].get("demo", []),
               "extraction": sections["extraction"], "briefings": briefing_metrics(briefs),
               "purity": sections["grouping"].get("purity")}
    summary = _save(d, config.SUMMARY_FILE, summary)
    _results(summary, sections, info)
    return summary

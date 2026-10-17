"""Offline coordinator review cards and an explicitly labelled historical replay."""
from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from . import config, fileio
from .contracts import Briefing, Decision, FEATURE_PLAIN, Incident, Report, Tagged
from .pipeline import StageUnavailable, pick_snapshot, require_module
from .textutil import mask_privacy, to_iso

BANNER = "Recommendations only. A human makes every dispatch decision."


@dataclass
class DemoData:
    event: str
    snapshots: list[str]
    incidents_by_snapshot: dict[str, list[Incident]]
    briefings: dict[tuple[str, str], Briefing]
    texts: dict[str, str]
    demo_snapshot: str


def _source(event, stub, src_dir):
    return Path(src_dir) if src_dir is not None else config.event_dir(
        config.FIXTURE_EVENT if stub else event, stub=stub)


def load_demo(event: str = config.DEMO_EVENT, stub: bool = False,
              src_dir: Path | None = None) -> DemoData:
    d = _source(event, stub, src_dir)
    incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident)
    reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
    briefs = fileio.read_jsonl(d / config.BRIEFINGS_FILE, Briefing) if (d / config.BRIEFINGS_FILE).exists() else []
    incident_lookup = {(i.incident_id, i.snapshot_time): i for i in incidents}
    try:
        cache_key = require_module("brief").cache_key
    except StageUnavailable:
        cache_key = None
    # IDs may be reused when regrouping. Never attach a cached briefing to new members.
    briefs = [b for b in briefs if (incident := incident_lookup.get((b.incident_id, b.snapshot_time)))
              and (stub or cache_key is None or b.cache_key == cache_key(incident, b.model_file))]
    actual = config.FIXTURE_EVENT if stub else event
    snapshots = sorted({i.snapshot_time for i in incidents})
    return DemoData(actual, snapshots,
                    {s: [i for i in incidents if i.snapshot_time == s] for s in snapshots},
                    {(b.incident_id, b.snapshot_time): b for b in briefs},
                    {r.report_id: mask_privacy(r.text) for r in reports},
                    pick_snapshot(incidents, d, fixture=stub or actual.startswith("fixture")))


def _h(value):
    return html.escape(mask_privacy(str(value)), quote=True)


def _adjust(incidents, lam):
    try:
        return require_module("fairness").apply(incidents, lam)
    except StageUnavailable:
        if lam != 0.0:
            raise
        # Stored lambda-zero outputs remain reviewable before Agent 2 arrives.
        ordered = sorted(incidents, key=lambda i: (-i.score, i.first_seen, i.incident_id))
        return [replace(i, score_adj=i.score, lam=0.0, rank_adj=n)
                for n, i in enumerate(ordered, 1)]


def _card(incident, data, baseline, reports):
    i = incident
    delta = baseline[i.incident_id] - i.rank_adj
    movement = f"▲{delta}" if delta > 0 else f"▼{-delta}" if delta < 0 else "—"
    svi = f"SVI {i.svi:.2f}" if i.svi is not None else "SVI unknown"
    parts = [f'<article class="incident-card" data-incident-id="{_h(i.incident_id)}">',
             f'<h3>#{i.rank_adj} {_h(i.place or "Unknown place")} <small>{movement}</small></h3>',
             f'<p>{_h(i.incident_id)} · County {_h(i.county_fips or "unknown")} · <b>{svi}</b></p>',
             f'<p>Score {i.score:.4f} · Adjusted {i.score_adj:.4f} · '
             f'{int(i.features["n_reports"])} reports</p><h4>Why it ranks here</h4>']
    scale = max((abs(v) for _, v in i.shap_top), default=1) or 1
    for key, contribution in i.shap_top:
        parts.append(f'<div>{_h(FEATURE_PLAIN[key])}: {contribution:+.4f} '
                     f'<meter min="0" max="1" value="{abs(contribution) / scale}"></meter></div>')
    b = data.briefings.get((i.incident_id, i.snapshot_time))
    if b:
        parts.append('<h4>Granite fields</h4><table><tr><th>Field</th><th>Value</th><th>Quoted evidence</th></tr>')
        for field, value in b.fields.items():
            parts.append(f'<tr><td>{_h(field)}</td><td>{_h(value["value"] or "unknown")}</td>'
                         f'<td><i>“{_h(value["evidence"] or "")}”</i></td></tr>')
        parts.append(f'</table><p>{_h(b.briefing)}</p>')
        if b.used_fallback:
            parts.append('<span class="badge">template</span>')
        rejected = [s for s in b.sentences if not s["accepted"]]
        parts.append(f'<details><summary>{len(rejected)} sentences rejected</summary>')
        parts.extend(f'<p>{_h(s["text"])} — {_h(s["reject_reason"])}</p>' for s in rejected)
        parts.append('</details>')
    else:
        parts.append(f'<p>Granite briefs only the top {config.TOP_N_BRIEF} at the demo snapshot</p>')
    parts.append(f'<details><summary>{len(i.report_ids)} member reports</summary>')
    for rid in i.report_ids:
        timestamp = reports[rid].timestamp if rid in reports else "timestamp unavailable"
        parts.append(f'<p><small>{_h(timestamp)} · {_h(rid)}</small><br>'
                     f'{_h(data.texts.get(rid, "Report unavailable"))}</p>')
    parts.append('</details></article>')
    return "\n".join(parts)


def _counts(reports, tags, snapshot, incidents, shown):
    ids = {r.report_id for r in reports.values() if r.timestamp <= snapshot}
    actionable = sum(t.is_actionable for t in tags if t.report_id in ids)
    return f"{len(ids)} reports so far / {actionable} actionable / {len(incidents)} open incidents / {shown} shown"


def render_html(event: str = config.DEMO_EVENT, snapshot: str | None = None, lam: float = 0.0,
                stub: bool = False, top: int = 10, src_dir: Path | None = None) -> str:
    data = load_demo(event, stub, src_dir)
    d = _source(event, stub, src_dir)
    snap = snapshot or data.demo_snapshot
    if snap not in data.incidents_by_snapshot:
        raise ValueError(f"Snapshot {snap!r} is not available")
    at = data.incidents_by_snapshot[snap]
    baseline = {i.incident_id: i.rank_adj for i in _adjust(at, 0.0)}
    adjusted = _adjust(at, lam)[:max(0, top)]
    reports = {r.report_id: r for r in fileio.read_jsonl(d / config.REPORTS_FILE, Report)}
    tags = fileio.read_jsonl(d / config.TAGGED_FILE, Tagged) if (d / config.TAGGED_FILE).exists() else []
    return (f'<section><h2>{_h(data.event)} · {_h(snap)} · λ={lam:g}</h2>'
            f'<p role="alert"><strong>{BANNER}</strong></p>'
            f'<p>{_counts(reports, tags, snap, at, len(adjusted))}</p>'
            + "\n".join(_card(i, data, baseline, reports) for i in adjusted) + '</section>')


def _log_decision(src_dir: Path, incident: Incident, action: str, new_rank: int | None = None,
                  note: str = "", reviewer: str = "", lam: float = 0.0) -> Decision:
    if action == "override" and (isinstance(new_rank, bool) or not isinstance(new_rank, int) or new_rank < 1):
        raise ValueError("Override requires new_rank (an integer >= 1)")
    decision = Decision(incident.incident_id, incident.event, incident.snapshot_time, action,
                        new_rank, mask_privacy(note), mask_privacy(reviewer),
                        to_iso(datetime.now(timezone.utc)), lam)
    decision.validate()
    fileio.append_jsonl(Path(src_dir) / config.DECISIONS_FILE, decision)
    return decision


def dashboard(event: str = config.DEMO_EVENT, snapshot: str | None = None, reviewer: str = "",
              stub: bool = False, src_dir: Path | None = None) -> "ipywidgets.Widget":
    import ipywidgets as widgets

    data = load_demo(event, stub, src_dir)
    d = _source(event, stub, src_dir)
    reports = {r.report_id: r for r in fileio.read_jsonl(d / config.REPORTS_FILE, Report)}
    tags = fileio.read_jsonl(d / config.TAGGED_FILE, Tagged) if (d / config.TAGGED_FILE).exists() else []
    choice = widgets.Dropdown(options=data.snapshots, value=snapshot or data.demo_snapshot, description="Snapshot")
    slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.25, description="λ")
    who = widgets.Text(value=reviewer, description="Reviewer")
    counts = widgets.HTML()
    cards = widgets.VBox()

    def refresh(change=None):
        at = data.incidents_by_snapshot[choice.value]
        try:
            baseline = {i.incident_id: i.rank_adj for i in _adjust(at, 0.0)}
            adjusted = _adjust(at, slider.value)[:config.TOP_K_EVAL]
        except StageUnavailable as exc:
            cards.children = (widgets.HTML(f'<p>{_h(exc)}</p>'),)
            return
        counts.value = _counts(reports, tags, choice.value, at, len(adjusted))
        rendered = []
        for i in adjusted:
            note = widgets.Textarea(description="Note")
            new_rank = widgets.IntText(value=0, description="New rank")
            logged = widgets.HTML()
            buttons = []
            for action in ("accept", "override", "dismiss"):
                button = widgets.Button(description=action.title())

                def click(_, inc=i, act=action, note_box=note, rank_box=new_rank, status=logged):
                    try:
                        decision = _log_decision(d, inc, act, rank_box.value if act == "override" else None,
                                                 note_box.value, who.value, inc.lam)
                        status.value = f'<pre>Logged: {_h(json.dumps(decision.to_dict()))}</pre>'
                    except (ValueError, OSError) as exc:
                        status.value = f'<p>{_h(exc)}</p>'

                button.on_click(click)
                buttons.append(button)
            rendered.append(widgets.VBox([widgets.HTML(_card(i, data, baseline, reports)), note,
                                           new_rank, widgets.HBox(buttons), logged]))
        cards.children = tuple(rendered)

    choice.observe(refresh, names="value")
    slider.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HTML(f'<h2>{_h(data.event)}</h2>'),
                         widgets.HBox([choice, slider, who]), counts,
                         widgets.HTML(f'<strong>{BANNER}</strong>'), cards])


def live_replay(event: str = config.DEMO_EVENT, speed: float = 5.0, duration_s: float = 60.0,
                top: int = 10) -> "ipywidgets.Widget":
    import ipywidgets as widgets
    from .benchmark import iter_stream, latency_stats

    if speed <= 0 or duration_s <= 0:
        raise ValueError("speed and duration_s must be positive")
    # Resolve upstream modules before starting the background worker, so errors stay visible.
    for name in ("replay", "tag", "group", "rank"):
        require_module(name, "live replay")
    view, counters = widgets.HTML(), widgets.HTML(value="Starting historical replay…")
    stop = widgets.Button(description="Stop replay")
    cancelled = threading.Event()
    stop.on_click(lambda _: cancelled.set())

    def update():
        latencies = []
        try:
            for row, incidents in iter_stream(event, speed, duration_s, cancelled.is_set):
                latencies.append(row["latency_s"])
                stats = latency_stats(latencies)
                counters.value = (f'{row["reports_in"]} reports in / {row["actionable"]} actionable / '
                                  f'{row["open_incidents"]} open incidents / '
                                  f'last batch {row["latency_s"] * 1000:.1f} ms / '
                                  f'running p95 {stats["p95_latency_s"] * 1000:.1f} ms')
                body = []
                for i in incidents[:max(0, top)]:
                    reason = FEATURE_PLAIN[i.shap_top[0][0]] if i.shap_top else "No explanation available"
                    body.append(f'<tr><td>{i.rank_adj}</td><td>{_h(i.place or "Unknown")}</td>'
                                f'<td>{int(i.features["n_reports"])}</td><td>{i.score_adj:.4f}</td>'
                                f'<td>{_h(reason)}</td></tr>')
                view.value = ('<table><tr><th>Rank</th><th>Place</th><th>Reports</th><th>Score</th>'
                              '<th>Top reason</th></tr>' + ''.join(body) + '</table>')
            counters.value += " · Replay stopped" if cancelled.is_set() else " · Replay complete"
        except Exception as exc:
            counters.value = f'<p>Replay failed: {_h(exc)}</p>'
        finally:
            stop.disabled = True

    panel = widgets.VBox([widgets.HTML(f'<strong>{BANNER}</strong>'), counters, view, stop,
                          widgets.HTML(f'Replay of real {_h(event)} tweets at {speed:g}x '
                                       f'(1x = {config.REPLAY_COMPRESSION:g}x real time)')])
    threading.Thread(target=update, daemon=True, name="surgedesk-replay").start()
    return panel

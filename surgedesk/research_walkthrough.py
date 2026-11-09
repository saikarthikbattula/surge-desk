"""Record an actual synthetic API/terminal walkthrough with local HTML playback.

This records automated fixture actions, not human usability observations or a
screen recording. It uses no external player service or paid API.
"""
import json
import time

from . import config, fileio
from .contracts import Briefing, Incident, Report, feature_reason
from .provenance import capture
from .textutil import mask_privacy


def run(run_id):
    from .research import _new_run
    from .research_demo import run as make_demo
    from .ui import _log_decision
    from pathlib import Path
    directory = _new_run(run_id)
    events = []
    start = time.perf_counter()
    def say(text):
        message = mask_privacy(text) + "\r\n"
        events.append([time.perf_counter() - start, "o", message])
        print(message, end="", flush=True)
    say("SurgeDesk: automated synthetic walkthrough")
    say("Simulated scores and scripted briefings. No live emergency feed or model-quality claim.")
    demo = Path(make_demo(run_id + "-demo"))
    reports = {r.report_id: r for r in fileio.read_jsonl(demo / config.REPORTS_FILE, Report)}
    if not all(r.is_synthetic for r in reports.values()):
        raise ValueError("Only synthetic reports may appear in the recording")
    snapshots = fileio.read_jsonl(demo / config.INCIDENTS_FILE, Incident)
    snapshot = max(i.snapshot_time for i in snapshots)
    incidents = sorted([i for i in snapshots if i.snapshot_time == snapshot], key=lambda i: i.rank_adj)
    briefings = {b.incident_id: b for b in fileio.read_jsonl(demo / config.BRIEFINGS_FILE, Briefing) if b.snapshot_time == snapshot}
    say(f"Created local synthetic demo with {len(reports)} reports and {len(incidents)} incidents.")
    incident = incidents[0]
    say(f"Inspect incident {incident.incident_id}: rank {incident.rank_adj}, place {incident.place or 'unknown'}.")
    say("Location: a detected place does not establish precise coordinates; county assignment may be approximate.")
    for key, value in incident.shap_top:
        say(f"Ranking reason: {feature_reason(key, incident.features)}; contribution {value:+.4f}")
    for rid in incident.report_ids:
        say(f"Member {rid}: {reports[rid].text}")
    briefing = briefings.get(incident.incident_id)
    if briefing:
        say("Briefing: " + briefing.briefing)
        for name, field in briefing.fields.items():
            if field["evidence"]:
                say(f"Evidence for {name}: {field['evidence']}")
    for action, chosen in zip(("accept", "override", "dismiss"), incidents):
        decision = _log_decision(demo, chosen, action, new_rank=1 if action == "override" else None,
                                 note="Automated synthetic walkthrough action; not a human usability observation",
                                 reviewer="automated_fixture_walkthrough")
        say(f"Logged {decision.action} for {decision.incident_id}.")
    say("Decision records and source-provenance sidecars are saved locally.")
    say("Open the synthetic HTML to inspect cards; use notebook 06 for interactive human decisions.")
    header = {"version": 2, "width": 110, "height": 35, "title": "SurgeDesk synthetic terminal walkthrough",
              "env": {"TERM": "xterm-256color"}}
    fileio._atomic_write_text(directory / config.RESEARCH_WALKTHROUGH_CAST, "\n".join(json.dumps(v) for v in [header, *events]) + "\n")
    payload = json.dumps(events).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>SurgeDesk synthetic walkthrough</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;color:#16283b;background:#f4f7fb}pre{white-space:pre-wrap;background:#111827;color:#ecf0f6;padding:1.5rem;border-radius:12px;line-height:1.6}button{padding:.6rem 1rem;margin-right:1rem}</style>
    <h1>SurgeDesk synthetic walkthrough</h1><p>Recorded automated terminal/API actions on synthetic fixtures. This is not a screen recording, usability session, live feed, or trained-model benchmark.</p>
    <button id="play">Replay recorded output</button><button id="all">Show transcript</button><pre id="terminal" aria-live="polite"></pre><script>
    const events=PAYLOAD; let timers=[]; const out=document.getElementById('terminal');
    function stop(){timers.forEach(clearTimeout);timers=[];out.textContent='';}
    document.getElementById('play').onclick=()=>{stop();events.forEach(e=>timers.push(setTimeout(()=>{out.textContent+=e[2]},e[0]*1000)))};
    document.getElementById('all').onclick=()=>{stop();out.textContent=events.map(e=>e[2]).join('')};
    out.textContent=events.map(e=>e[2]).join('');</script></html>""".replace("PAYLOAD", payload)
    fileio._atomic_write_text(directory / config.RESEARCH_WALKTHROUGH_HTML, page)
    fileio.write_json(directory / config.RESEARCH_WALKTHROUGH_META, {"synthetic": True, "type": "automated terminal/API recording",
                       "screen_recording": False, "human_participants": None,
                       "duration_seconds": events[-1][0], "demo": str(demo), "provenance": capture()})
    return str(directory)

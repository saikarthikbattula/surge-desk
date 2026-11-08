"""Isolated, explicitly synthetic demonstration using stored fixture scenarios."""
from . import artifacts, config, fileio
from .contracts import Report, Tagged, Incident, Briefing
from .provenance import capture, tree_hash


def run(run_id):
    from .research import _new_run
    from .ui import render_html
    directory = _new_run(run_id)
    fixture = config.FIXTURES / "outputs" / config.FIXTURE_EVENT
    reports = fileio.read_jsonl(fixture / config.REPORTS_FILE, Report)
    if not all(r.is_synthetic for r in reports):
        raise ValueError("Demo accepts synthetic fixtures only")
    tags = fileio.read_jsonl(fixture / config.TAGGED_FILE, Tagged)
    embeddings = fileio.load_embeddings(fixture / config.EMBEDDINGS_FILE)
    incidents = fileio.read_jsonl(fixture / config.INCIDENTS_FILE, Incident)
    briefings = fileio.read_jsonl(fixture / config.BRIEFINGS_FILE, Briefing)
    files = {config.REPORTS_FILE: reports, config.TAGGED_FILE: tags,
             config.INCIDENTS_FILE: incidents, config.BRIEFINGS_FILE: briefings}
    for name, rows in files.items():
        fileio.write_jsonl(directory / name, rows)
    fileio.save_embeddings(directory / config.EMBEDDINGS_FILE, embeddings)
    fileio.write_json(directory / config.RESEARCH_DEMO_PROVENANCE_FILE, {"synthetic": True, "provenance": capture(),
                                                          "fixture_checksums": tree_hash(fixture),
                                                          "models": "stored synthetic fixtures; no inference or trained-model quality claim"})
    artifacts.complete(directory / config.STAGE_MANIFEST_FILE, {"synthetic": True},
                       [directory / n for n in files] + [directory / config.EMBEDDINGS_FILE, directory / config.RESEARCH_DEMO_PROVENANCE_FILE])
    content = render_html(config.FIXTURE_EVENT, stub=True, src_dir=directory)
    style = """body{font:16px/1.6 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;background:#f3f6fa;color:#17283d}
    article{background:white;padding:1.5rem;border:1px solid #cbd5e1;border-radius:12px;margin:1rem 0}
    h1,h2,h3{line-height:1.2}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #ddd}
    details{margin:.6rem 0}summary{cursor:pointer}.badge{background:#e6edf8;padding:.2rem .5rem}.location-uncertainty{color:#785000}
    """
    fileio._atomic_write_text(directory / config.RESEARCH_DEMO_HTML_FILE, '<!doctype html><html lang="en"><meta charset="utf-8">'
                             '<meta name="viewport" content="width=device-width,initial-scale=1"><title>SurgeDesk synthetic demo</title>'
                             f'<style>{style}</style><h1>SurgeDesk</h1><p><strong>SYNTHETIC DEMONSTRATION</strong> — '
                             'Historical-style examples, simulated scores, scripted briefings. No live emergency feed or model inference.</p>'
                             '<p>Expand member reports and quoted evidence. For Accept / Override / Dismiss logging, '
                             'open notebooks/06_research_demo.ipynb.</p>' + content + '</html>')
    return str(directory)

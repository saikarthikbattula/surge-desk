"""Private development review exports and computed review summaries."""
from collections import Counter

import numpy as np

from . import config, fileio, research_data
from .geo import Gazetteer, PlaceMatcher
from .textutil import mask_privacy


def duplicate_sample(manifest, output):
    from pathlib import Path
    from .data_humaid import load_event
    research_data.validate_manifest(manifest, require_frozen=False)
    by_key = {row["key"]: row for row in manifest["rows"]}
    sources, output_rows = {}, []
    for pair in manifest["near_duplicates"]:
        item = dict(pair)
        for side in ("left", "right"):
            row = by_key[pair[side]]
            source = row["source_file"]
            if source not in sources:
                sources[source] = {r.report_id: r for r in load_event(row["event"], [Path(manifest["root"]) / source])}
            item[side + "_report"] = {"report_id": row["report_id"], "partition": row["partition"],
                                     "text": mask_privacy(sources[source][row["report_id"]].text)}
        output_rows.append(item)
    research_data.save_new(output, {"near_duplicates": output_rows,
                                   "instructions": "Review masked text; transfer decisions and notes to draft manifest before freezing"})
    return {"pairs": len(output_rows), "output": str(output)}


def location_sample(manifest, output, gazetteer=config.GAZETTEER_DIR):
    from .provenance import capture, tree_hash
    selected = research_data.selected_reports(manifest, "dev")
    rng = np.random.default_rng(config.SEED)
    indices = rng.permutation(len(selected))[:config.RESEARCH_REVIEW_LIMIT]
    gaz = Gazetteer(gazetteer)
    matchers, rows = {}, []
    for index in indices:
        row, report = selected[index]
        if report.event not in matchers:
            matchers[report.event] = PlaceMatcher(report.event, gaz)
        matcher = matchers[report.event]
        rows.append({"key": row["key"], "report_id": report.report_id, "event": report.event,
                     "text": mask_privacy(report.text), "location": matcher.match_with_evidence(report.text),
                     "reviewer": "", "review_kind": "", "outcome": "", "error_categories": [], "note": "",
                     "sample_seed": config.SEED, "sampling": "uniform development reports without replacement",
                     "manifest_fingerprint": manifest["fingerprint"]})
    research_data.save_new(output, {"synthetic": manifest["synthetic"], "locations": rows,
                                   "gazetteer_checksums": tree_hash(gazetteer), "policy": config.LOCATION_POLICY,
                                   "provenance": capture()})
    return {"sampled": len(rows), "output": str(output)}


def summarize(path):
    document = fileio.read_json(path)
    rows = document.get("locations", document.get("classification_errors", []))
    reviewed = [r for r in rows if r.get("reviewer", "").strip() and r.get("outcome") in {"correct", "incorrect", "uncertain"}]
    counts = Counter(mask_privacy(category) for row in reviewed for category in row["error_categories"])
    return {"sampled": len(rows), "reviewed": len(reviewed),
            "review_kinds": dict(Counter(r.get("review_kind", "unspecified") for r in reviewed)),
            "outcomes": dict(Counter(r["outcome"] for r in reviewed)),
            "prioritized_errors": [{"category": k, "count": v} for k, v in counts.most_common()],
            "agreement": {"status": "unavailable", "reason": "Solo review is not inter-rater agreement"},
            "limitation": "Internal development review, not expert emergency-response validation"}

"""Split-preserving HumAID manifests and reviewable leakage ledgers."""
from __future__ import annotations

import math
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path

from . import config, fileio
from .data_humaid import discover_event_files, load_event
from .provenance import digest, file_hash
from .textutil import mask_privacy


def normalized(text):
    return " ".join(re.findall(r"\w+", mask_privacy(text).casefold()))


def _signature(manifest):
    # The local mount may change; relative source names and content hashes may not.
    return digest({k: v for k, v in manifest.items() if k not in {"fingerprint", "root"}})


def inventory(root):
    return {event: [{"file": str(p.relative_to(root)).replace("\\", "/"),
                     "split": p.stem.rsplit("_", 1)[1], "sha256": file_hash(p),
                     "n_valid_unique_ids": len(load_event(event, [p])), **audit_file(p)} for p in paths]
            for event, paths in discover_event_files(Path(root)).items()}


def audit_file(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        columns = {c.strip().lower(): c for c in reader.fieldnames or []}
        identifier = next((columns[c] for c in ("tweet_id", "id") if c in columns), None)
        text = next((columns[c] for c in ("tweet_text", "text") if c in columns), None)
        if identifier is None or text is None:
            raise ValueError(f"Missing ID/text columns in {Path(path).name}")
        seen, counts = set(), Counter()
        for row in reader:
            counts["raw_rows"] += 1
            rid = (row.get(identifier) or "").strip()
            if not rid.isascii() or not rid.isdigit():
                counts["invalid_id_rows"] += 1
            if not (row.get(text) or "").strip():
                counts["empty_text_rows"] += 1
            if rid in seen:
                counts["repeated_id_rows"] += 1
            seen.add(rid)
        return {k: counts[k] for k in ("raw_rows", "invalid_id_rows", "empty_text_rows", "repeated_id_rows")}


def near_pairs(rows, texts):
    """Exact Jaccard threshold candidates via globally ordered token prefixes.

    No semantic duplicate claim: short paraphrases may have no token overlap.
    Only cross-partition pairs need human leakage decisions.
    """
    sets = [set(normalized(t).split()) for t in texts]
    frequencies = Counter(t for tokens in sets for t in tokens)
    postings = defaultdict(list)
    pairs = []
    threshold = config.RESEARCH_NEAR_DUPLICATE
    for i, tokens in enumerate(sets):
        if not tokens:
            continue
        ordered = sorted(tokens, key=lambda t: (frequencies[t], t))
        prefix = ordered[:len(tokens) - math.ceil(threshold * len(tokens)) + 1]
        candidates = set(j for token in prefix for j in postings[token])
        for j in sorted(candidates):
            if rows[j]["partition"] == rows[i]["partition"]:
                continue
            similarity = len(tokens & sets[j]) / len(tokens | sets[j])
            if similarity >= threshold and normalized(texts[i]) != normalized(texts[j]):
                pairs.append({"left": rows[j]["key"], "right": rows[i]["key"],
                              "jaccard": similarity, "decision": "pending", "note": ""})
        for token in prefix:
            postings[token].append(i)
    return pairs


def make_manifest(root, partitions, rationale, synthetic=False):
    root = Path(root).resolve()
    found = discover_event_files(root)
    if not rationale.strip():
        raise ValueError("Record an event-selection rationale")
    if set(partitions) - set(config.RESEARCH_PARTITIONS):
        raise ValueError("Unknown partition")
    memberships = [e for events in partitions.values() for e in events]
    if len(set(memberships)) != len(memberships):
        raise ValueError("Each event must belong to exactly one partition")
    if set(memberships) - found.keys():
        raise ValueError("Partition names an undiscovered event")
    if config.DEMO_EVENT in partitions.get("train", []):
        raise ValueError("Never train on config.DEMO_EVENT")
    if config.DEMO_EVENT in partitions.get("test", []):
        raise ValueError("Previously inspected demo event belongs in historical or dev")
    if not partitions.get("train") or not partitions.get("dev"):
        raise ValueError("Training and development events are required")
    if not synthetic and any(e.startswith("fixture") for e in memberships):
        raise ValueError("Fixture events require explicit synthetic mode")
    rows, texts, sources = [], [], {}
    # Protect reserved data: exclude duplicate training rows before held-out rows.
    for partition in ("test", "dev", "historical", "train"):
        for event in sorted(partitions.get(partition, [])):
            for path in sorted(found[event], key=lambda p: ({"train": 0, "dev": 1, "test": 2}[p.stem.rsplit("_", 1)[1]], str(p))):
                source = str(path.relative_to(root)).replace("\\", "/")
                sources[source] = file_hash(path)
                for report in load_event(event, [path], source="synthetic" if synthetic else "humaid"):
                    key = digest([source, report.report_id])
                    rows.append({"key": key, "report_id": report.report_id, "event": event,
                                 "partition": partition, "source_file": source,
                                 "original_split": path.stem.rsplit("_", 1)[1],
                                 "text_sha256": digest(normalized(report.text)),
                                 "gold_label": report.gold_label})
                    texts.append(report.text)
    excluded, seen_ids, seen_texts = [], {}, {}
    for row in rows:
        prior = seen_ids.get(row["report_id"]) or seen_texts.get(row["text_sha256"])
        if prior:
            excluded.append({"key": row["key"], "reason": "duplicate ID or normalized text",
                             "retained_key": prior})
        else:
            seen_ids[row["report_id"]] = row["key"]
            seen_texts[row["text_sha256"]] = row["key"]
    excluded_keys = {x["key"] for x in excluded}
    active = [(r, t) for r, t in zip(rows, texts) if r["key"] not in excluded_keys]
    return {"schema": config.RESEARCH_SCHEMA, "state": "draft", "synthetic": synthetic,
            "root": str(root), "rationale": rationale, "partitions": partitions,
            "sources": sources, "rows": rows, "exclusions": excluded,
            "source_audit": {source: audit_file(root / source) for source in sources},
            "near_duplicates": near_pairs([r for r, _ in active], [t for _, t in active]),
            "near_duplicate_method": "token-set Jaccard; misses semantic paraphrases",
            "near_duplicate_threshold": config.RESEARCH_NEAR_DUPLICATE,
            "assistance_labels": list(config.RESEARCH_ASSISTANCE_LABELS),
            "generalization_limit": None if partitions.get("test") else "No reserved unseen evaluation events"}


def validate_manifest(manifest, require_frozen=True):
    if manifest.get("schema") != config.RESEARCH_SCHEMA:
        raise ValueError("Unsupported manifest schema")
    if require_frozen and (manifest.get("state") != "frozen" or manifest.get("fingerprint") != _signature(manifest)):
        raise ValueError("Manifest is not frozen or its contents changed; review and freeze again")
    root = Path(manifest["root"])
    for relative, expected in manifest["sources"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Source data changed or missing: {relative}; rebuild manifest")
    events = [e for values in manifest["partitions"].values() for e in values]
    if len(events) != len(set(events)) or config.DEMO_EVENT in manifest["partitions"].get("train", []):
        raise ValueError("Invalid event partitions")
    keys = {r["key"] for r in manifest["rows"]}
    if len(keys) != len(manifest["rows"]) or any(x["key"] not in keys for x in manifest["exclusions"]):
        raise ValueError("Invalid row/exclusion keys")
    excluded = {x["key"] for x in manifest["exclusions"]}
    active = [r for r in manifest["rows"] if r["key"] not in excluded]
    for row in manifest["rows"]:
        if (row["event"] not in manifest["partitions"].get(row["partition"], [])
                or row["source_file"] not in manifest["sources"]
                or row["key"] != digest([row["source_file"], row["report_id"]])):
            raise ValueError("Row provenance is inconsistent with partitions/sources")
    if len({r["report_id"] for r in active}) != len(active) or len({r["text_sha256"] for r in active}) != len(active):
        raise ValueError("Duplicate IDs/text remain active; resolve exclusions")


def freeze(manifest):
    validate_manifest(manifest, require_frozen=False)
    known = {r["key"] for r in manifest["rows"]}
    for pair in manifest["near_duplicates"]:
        if pair["left"] not in known or pair["right"] not in known:
            raise ValueError("Near-duplicate pair names unknown rows")
        decision = pair["decision"]
        if decision not in {"distinct", "exclude_left", "exclude_right"} or not pair["note"].strip():
            raise ValueError("Resolve every near-duplicate pair with a decision and note before freezing")
        if decision.startswith("exclude_"):
            key = pair[decision.removeprefix("exclude_")]
            if key not in known:
                raise ValueError("Unknown exclusion key")
            if not any(x["key"] == key for x in manifest["exclusions"]):
                manifest["exclusions"].append({"key": key, "reason": pair["note"]})
    manifest["state"] = "frozen"
    manifest["fingerprint"] = _signature(manifest)
    return manifest


def selected_reports(manifest, partition, events=None):
    validate_manifest(manifest)
    if partition not in config.RESEARCH_PARTITIONS:
        raise ValueError("Unknown partition")
    allowed = set(manifest["partitions"].get(partition, []))
    events = allowed if events is None else set(events)
    if not events <= allowed:
        raise ValueError("Requested events are outside the selected partition")
    excluded = {x["key"] for x in manifest["exclusions"]}
    cached = {}
    result = []
    for row in manifest["rows"]:
        if row["partition"] != partition or row["event"] not in events or row["key"] in excluded:
            continue
        source = row["source_file"]
        if source not in cached:
            cached[source] = {r.report_id: r for r in load_event(row["event"], [Path(manifest["root"]) / source],
                                                              source="synthetic" if manifest["synthetic"] else "humaid")}
        report = cached[source][row["report_id"]]
        if digest(normalized(report.text)) != row["text_sha256"] or report.gold_label != row["gold_label"]:
            raise ValueError("Manifest row does not match source")
        result.append((row, report))
    return result


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fileio.write_json(path, value)


def reconstruct_manifest(specification, root):
    """Rebuild a private manifest from a public text-free workload specification."""
    manifest = make_manifest(root, specification["partitions"], specification["rationale"], specification.get("synthetic", False))
    if manifest["sources"] != specification["tsv_checksums"]:
        raise ValueError("TSV checksums differ from the public benchmark; use its pinned preparation source")
    excluded = set(specification["excluded_keys"])
    for pair in manifest["near_duplicates"]:
        if pair["right"] in excluded:
            pair["decision"] = "exclude_right"
        elif pair["left"] in excluded:
            pair["decision"] = "exclude_left"
        else:
            pair["decision"] = "distinct"
        pair["note"] = "Reconstructed from published non-text exclusion keys; original private review notes are not redistributed"
    known = {row["key"] for row in manifest["rows"]}
    if not excluded <= known:
        raise ValueError("Published exclusions refer to unknown rows")
    for key in sorted(excluded):
        if not any(x["key"] == key for x in manifest["exclusions"]):
            manifest["exclusions"].append({"key": key, "reason": "Published benchmark exclusion"})
    manifest = freeze(manifest)
    from .provenance import report_identity
    for role, expected in specification["workload_fingerprints"].items():
        if report_identity([r for _, r in selected_reports(manifest, role)]) != expected:
            raise ValueError("Reconstructed workload differs from the published benchmark")
    return manifest

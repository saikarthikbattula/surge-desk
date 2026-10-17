"""HumAID ingestion and time-ordered report handoff (Agent 1)."""
from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config
from .contracts import ContractError, Report, normalize_label
from .fileio import read_jsonl, write_json, write_jsonl
from .textutil import parse_iso, to_iso, tweet_id_to_datetime


def discover_event_files(humaid_dir: Path = config.HUMAID_DIR) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(humaid_dir).rglob("*.tsv")):
        match = re.fullmatch(r"(.+)_(train|dev|test)\.tsv", path.name)
        if match:
            found.setdefault(match[1], []).append(path)
    if not found:
        raise FileNotFoundError(f"No <event>_{{train,dev,test}}.tsv files found in {humaid_dir}")
    return {e: sorted(found[e], key=lambda p: (p.name, str(p))) for e in sorted(found)}


def load_event(event: str, files: list[Path], source: str = "humaid") -> list[Report]:
    priority = {"train": 0, "dev": 1, "test": 2}
    ordered = sorted(map(Path, files), key=lambda p: (priority.get(p.stem.rsplit("_", 1)[-1], 3), str(p)))
    rows: dict[str, Report] = {}
    for path in ordered:
        frame = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                            keep_default_na=False, on_bad_lines="warn")
        frame.columns = [c.strip().lower() for c in frame.columns]
        columns = []
        for aliases in (("tweet_id", "id"), ("tweet_text", "text"), ("class_label", "label")):
            chosen = next((a for a in aliases if a in frame.columns), None)
            if chosen is None:
                raise ContractError(f"{path}: missing {aliases}; columns found: {list(frame.columns)}")
            columns.append(chosen)
        for rid, text, label in frame[columns].itertuples(index=False, name=None):
            rid, text = rid.strip(), text.strip()
            if not rid.isascii() or not rid.isdigit() or not text or rid in rows:
                continue
            report = Report(rid, event, text, to_iso(tweet_id_to_datetime(rid)), source,
                            normalize_label(label), source == "synthetic")
            report.validate()
            rows[rid] = report
    return sorted(rows.values(), key=lambda r: (r.timestamp, r.report_id))


def build_reports(events: list[str] | None = None, humaid_dir: Path = config.HUMAID_DIR,
                  force: bool = False, source: str = "humaid") -> pd.DataFrame:
    discovered = discover_event_files(humaid_dir)
    selected = sorted(discovered if events is None else set(events))
    missing = set(selected) - discovered.keys()
    if missing:
        raise FileNotFoundError(f"Events {sorted(missing)} not found in {humaid_dir}; available: {list(discovered)}")
    summary = []
    for event in selected:
        target = config.event_file(event, config.REPORTS_FILE)
        if target.exists() and not force:
            reports = read_jsonl(target, Report)
        else:
            reports = load_event(event, discovered[event], source)
            write_jsonl(target, reports)
        if any(not 2010 <= parse_iso(r.timestamp).year <= 2025 for r in reports):
            warnings.warn(f"{event}: report timestamps outside 2010-2025", stacklevel=2)
        summary.append(dict(event=event, n_reports=len(reports),
                            first=reports[0].timestamp if reports else None,
                            last=reports[-1].timestamp if reports else None,
                            states=json.dumps(config.states_for_event(event)),
                            label_counts=json.dumps(dict(sorted(Counter(r.gold_label for r in reports).items())))))
    frame = pd.DataFrame(summary, columns=["event", "n_reports", "first", "last", "states", "label_counts"])
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_json(config.OUTPUTS / "events.json", selected)
    frame.to_csv(config.OUTPUTS / "events_summary.csv", index=False)
    print(frame.to_string(index=False))
    return frame

"""Explicit network preparation only; all SurgeDesk inference remains offline.

Downloads pinned author-hosted assets. Never executes remote dataset code.
Raw JSON, converted TSVs, and checksums stay in gitignored local directories.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgedesk import config, fileio
from surgedesk.provenance import file_hash


def fetch(url, target, expected_hash=None):
    target = Path(target)
    if target.exists() and expected_hash is not None and file_hash(target) == expected_hash:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".download")
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=config.RESEARCH_DOWNLOAD_TIMEOUT) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
        if target.exists():
            if file_hash(target) != file_hash(temporary):
                raise FileExistsError(f"Existing asset differs from pinned source; preserve it in a separate root: {target}")
        else:
            os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_model():
    root = config.MINILM_DIR
    previous_path = root / config.RESEARCH_SOURCE_MANIFEST_FILE
    previous = fileio.read_json(previous_path) if previous_path.exists() else {}
    if previous and previous.get("revision") != config.RESEARCH_MODEL_REVISION:
        raise ValueError("Model root was prepared from another revision; use a separate root")
    source = {"repository": config.RESEARCH_MODEL_ID, "revision": config.RESEARCH_MODEL_REVISION,
              "files": {}, "kind": "local MiniLM preparation; no paid inference API"}
    for name in config.RESEARCH_MODEL_FILES:
        target = root / name
        fetch(f"{config.RESEARCH_HUB_URL}/{config.RESEARCH_MODEL_ID}/resolve/{config.RESEARCH_MODEL_REVISION}/{name}", target,
              previous.get("files", {}).get(name))
        source["files"][name] = file_hash(target)
        print(f"Prepared model file: {name}", flush=True)
    fileio.write_json(root / config.RESEARCH_SOURCE_MANIFEST_FILE, source)


def prepare_data(events=None):
    base = f"{config.RESEARCH_HUB_URL}/datasets/{config.RESEARCH_DATASET_ID}/resolve/{config.RESEARCH_DATASET_REVISION}"
    api = f"{config.RESEARCH_HUB_URL}/api/datasets/{config.RESEARCH_DATASET_ID}/revision/{config.RESEARCH_DATASET_REVISION}"
    with urllib.request.urlopen(api, timeout=config.RESEARCH_DOWNLOAD_TIMEOUT) as response:
        metadata = json.load(response)
    files = [s["rfilename"] for s in metadata["siblings"] if re.fullmatch(r"[a-z0-9_]+/(train|dev|test)\.json", s["rfilename"])]
    if events is not None:
        if set(events) - {p.split("/")[0] for p in files}:
            raise ValueError("Requested event is absent from pinned author mirror")
        files = [p for p in files if p.split("/")[0] in events]
    root = config.HUMAID_DIR
    previous_path = root / config.RESEARCH_SOURCE_MANIFEST_FILE
    previous = fileio.read_json(previous_path) if previous_path.exists() else {}
    if previous and previous.get("revision") != config.RESEARCH_DATASET_REVISION:
        raise ValueError("Data root was prepared from another revision; use a separate root")
    source = {"repository": config.RESEARCH_DATASET_ID, "revision": config.RESEARCH_DATASET_REVISION,
              "scope": "Author event mirror; not asserted identical to historical Set 1",
              "conversion": "Preserve IDs/labels; replace embedded tabs and newlines with spaces for unquoted TSV compatibility",
              "files": dict(previous.get("files", {}))}
    for relative in sorted(files):
        original = root / config.RESEARCH_SOURCE_JSON_NAME / relative
        fetch(f"{base}/{relative}", original, previous.get("files", {}).get(relative, {}).get("json_sha256"))
        raw = original.read_text(encoding="utf-8")
        try:
            values = json.loads(raw)
            if isinstance(values, dict):
                values = [values]
        except json.JSONDecodeError:
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        event, split = relative.split("/")
        target = root / event / (event + "_" + Path(split).stem + ".tsv")
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator=config.RESEARCH_TSV_LINE_ENDING)
        writer.writerow(["tweet_id", "tweet_text", "class_label"])
        for value in values:
            text = value["tweet_text"].replace("\t", " ").replace("\r", " ").replace("\n", " ")
            writer.writerow([str(value["tweet_id"]), text, value["class_label"]])
        canonical = output.getvalue().encode("utf-8")
        if target.exists() and target.read_bytes() != canonical:
            raise FileExistsError(f"Existing TSV differs; preserve it and use a separate SURGEDESK_ROOT: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".convert")
            try:
                with os.fdopen(fd, "wb") as converted:
                    converted.write(canonical)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        source["files"][relative] = {"json_sha256": file_hash(original), "tsv_sha256": file_hash(target), "rows": len(values)}
        print(f"Prepared {event}/{Path(split).stem}: {len(values)} rows", flush=True)
    fileio.write_json(root / config.RESEARCH_SOURCE_MANIFEST_FILE, source)


def prepare_gazetteer():
    root = config.GAZETTEER_DIR
    year = config.RESEARCH_GAZETTEER_YEAR
    source = {"source": config.RESEARCH_CENSUS_URL, "year": year, "files": {}}
    previous_path = root / config.RESEARCH_SOURCE_MANIFEST_FILE
    previous = fileio.read_json(previous_path) if previous_path.exists() else {}
    for kind in ("place", "counties"):
        stem = f"{year}_Gaz_{kind}_national"
        archive = root / (stem + ".zip")
        fetch(f"{config.RESEARCH_CENSUS_URL}/{year}_Gazetteer/{stem}.zip", archive,
              previous.get("files", {}).get(stem, {}).get("zip_sha256"))
        with zipfile.ZipFile(archive) as zipped:
            names = [n for n in zipped.namelist() if Path(n).name == stem + ".txt"]
            if len(names) != 1:
                raise ValueError("Census archive does not contain exactly one expected national table")
            content = zipped.read(names[0])
        target = root / (stem + ".txt")
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("Preserve the existing Gazetteer in a separate root")
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=root, suffix=".extract")
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(content)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        source["files"][stem] = {"zip_sha256": file_hash(archive), "table_sha256": file_hash(target)}
        print(f"Prepared Gazetteer: {stem}", flush=True)
    fileio.write_json(root / config.RESEARCH_SOURCE_MANIFEST_FILE, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--gazetteer-only", action="store_true")
    parser.add_argument("--events", nargs="+")
    args = parser.parse_args()
    if sum((args.models_only, args.data_only, args.gazetteer_only)) > 1:
        parser.error("Choose only one preparation-only mode")
    if args.gazetteer_only:
        prepare_gazetteer()
        return
    if not args.data_only:
        prepare_model()
    if not args.models_only:
        prepare_data(args.events)


if __name__ == "__main__":
    main()

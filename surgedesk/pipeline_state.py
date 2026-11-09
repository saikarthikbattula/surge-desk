"""Whole-stage recovery receipts for the legacy integration runner."""
from pathlib import Path
import time
import shutil
from uuid import uuid4

from . import config, fileio
from .provenance import digest, file_hash, tree_hash


def _events():
    path = config.OUTPUTS / "events.json"
    return fileio.read_json(path) if path.exists() else []


def _identity(stage, event):
    names = {"data": (), "embed": (config.REPORTS_FILE,),
             "classifier": (config.REPORTS_FILE, config.EMBEDDINGS_FILE),
             "tag": (config.REPORTS_FILE, config.EMBEDDINGS_FILE),
             "ranker": (config.REPORTS_FILE, config.TAGGED_FILE, config.EMBEDDINGS_FILE),
             "rank": (config.REPORTS_FILE, config.TAGGED_FILE, config.EMBEDDINGS_FILE),
             "brief": (config.REPORTS_FILE, config.INCIDENTS_FILE),
             "eval": (), "benchmark": ()}[stage]
    events = _events() if stage in {"embed", "classifier", "tag", "ranker"} else [event]
    inputs = {str(p): file_hash(p) for e in events for name in names
              if (p := config.OUTPUTS / e / name).exists()}
    models = {"embed": [config.MINILM_DIR], "tag": [config.CLASSIFIER_PATH, config.GAZETTEER_DIR],
              "rank": [config.RANKER_PATH, config.RANKER_META_PATH, config.SVI_PATH],
              "brief": [config.GRANITE_PATH]}.get(stage, [])
    return {"stage": stage, "event": event, "events": events, "inputs": inputs,
            "models": {str(p): tree_hash(p) for p in models},
            "raw_data": tree_hash(config.HUMAID_DIR) if stage == "data" else None,
            "code": tree_hash(config.PACKAGE_DIR),
            "settings": {k: str(v) for k, v in vars(config).items() if k.isupper()}}


def _outputs(stage, event):
    if stage == "classifier":
        return [config.CLASSIFIER_PATH, config.CLASSIFIER_PATH.with_name("classifier_meta.json")]
    if stage == "ranker":
        return [config.RANKER_PATH, config.RANKER_META_PATH]
    events = _events() if stage in {"data", "embed", "tag"} else [event]
    paths = [config.OUTPUTS / e / name for e in events for name in config.PIPELINE_STAGE_FILES.get(stage, ())]
    if stage == "data":
        paths.append(config.OUTPUTS / "events.json")
    return paths


def validate(stage, event):
    """Read-only freshness check for artifacts produced by this runner."""
    receipt_path = config.PIPELINE_RECEIPTS / (digest([stage, event]) + ".json")
    if not receipt_path.exists():
        return False  # Imported historical artifacts have no managed receipt.
    prior = fileio.read_json(receipt_path)
    paths = _outputs(stage, event)
    if (prior.get("schema") != config.RESEARCH_SCHEMA or prior.get("identity") != digest(_identity(stage, event))
            or prior.get("outputs") != {str(p): file_hash(p) for p in paths}):
        raise ValueError(f"Managed {stage} artifacts are stale; rerun stage {stage}")
    return True


def run(stage, event, force, execute):
    # Evaluation depends on mutable private labels; benchmarks must measure work.
    if stage in {"eval", "benchmark"}:
        return execute(stage, event, force)
    receipt_path = config.PIPELINE_RECEIPTS / (digest([stage, event]) + ".json")
    identity = _identity(stage, event)
    expected = _outputs(stage, event)
    try:
        prior = fileio.read_json(receipt_path)
    except (OSError, ValueError):
        prior = None
    if not force and expected and prior and prior.get("identity") == digest(identity):
        try:
            current = {str(p): file_hash(p) for p in expected}
            if prior.get("outputs") == current:
                print(f"[{stage}] reused validated completed stage", flush=True)
                return prior["result"]
        except OSError:
            pass
    start = time.perf_counter()
    existing = [p for p in expected if p.is_file()]
    if existing:
        archive = config.PIPELINE_ARCHIVES / uuid4().hex
        archive.mkdir(parents=True, exist_ok=False)
        records = []
        for path in existing:
            copy = archive / (digest(str(path)) + "-" + path.name)
            shutil.copy2(path, copy)
            records.append({"original": str(path), "archived": copy.name, "sha256": file_hash(copy)})
        fileio.write_json(archive / config.PIPELINE_ARCHIVE_INDEX_FILE, {"stage": stage, "event": event, "files": records})
        print(f"[{stage}] preserved previous outputs in {archive}", flush=True)
    # Bypass legacy existence-only caches if a receipt cannot prove freshness.
    result = execute(stage, event, True)
    outputs = _outputs(stage, event)
    if outputs and all(p.exists() for p in outputs):
        fileio.write_json(receipt_path, {"schema": config.RESEARCH_SCHEMA,
                            "identity": digest(_identity(stage, event)),
                            "outputs": {str(p): file_hash(p) for p in outputs},
                            "seconds": time.perf_counter() - start, "result": result})
    return result

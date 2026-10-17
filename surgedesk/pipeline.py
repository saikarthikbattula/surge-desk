"""Stage CLI used by the frozen shell runners; imports upstream stages lazily."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

from . import config, fileio
from .contracts import Incident, Report


OWNERS = {"data_humaid": 1, "embed": 1, "tag": 1, "replay": 1, "geo": 1,
          "group": 2, "features": 2, "rank": 2, "fairness": 2, "baselines": 2,
          "llm": 3, "brief": 3, "grounding": 3, "evaluate": 4, "benchmark": 4}
STAGES = ("data", "embed", "classifier", "tag", "ranker", "rank", "brief", "eval", "benchmark")


class StageUnavailable(RuntimeError):
    """An upstream stage has not been delivered yet."""


def require_module(name: str, stage: str | None = None):
    try:
        return importlib.import_module(f"surgedesk.{name}")
    except ImportError as exc:
        raise StageUnavailable(
            f"stage {stage or name} not implemented yet (owner: Agent {OWNERS[name]}): "
            f"module {name} or its dependency is unavailable ({exc})"
        ) from None


def pick_snapshot(incidents: list[Incident], src_dir, fixture: bool = False) -> str:
    path = src_dir / "demo_snapshot.txt"
    snapshots = sorted({i.snapshot_time for i in incidents})
    if not snapshots:
        raise ValueError("No incident snapshots available; run stage rank first")
    snap = path.read_text().strip() if path.exists() else (
        snapshots[-1] if fixture else require_module("rank").pick_demo_snapshot(incidents)
    )
    if snap not in snapshots:
        raise ValueError(f"Demo snapshot {snap!r} is absent from incidents.jsonl")
    return snap


def _events() -> list[str]:
    events = fileio.read_json(config.OUTPUTS / "events.json")
    if not isinstance(events, list) or not all(isinstance(e, str) for e in events):
        raise ValueError("events.json must contain a list of event names")
    return sorted(events, key=lambda e: (e != config.DEMO_EVENT, e))


def run_stage(stage: str, event: str, force: bool = False):
    if stage == "data":
        require_module("data_humaid", stage).build_reports(force=force)
        events = _events()
        if config.DEMO_EVENT not in events:
            raise ValueError(f"Demo event {config.DEMO_EVENT!r} absent; events={events}; set SURGEDESK_EVENT")
        return {"events": events}
    if stage in ("embed", "tag", "classifier", "ranker"):
        mod = require_module({"embed": "embed", "tag": "tag", "classifier": "tag", "ranker": "rank"}[stage], stage)
        events = _events()
        train = config.train_events(events)
        assert config.DEMO_EVENT not in train, "Never train on the demo event"
        if stage == "embed":
            return {e: {"n_embeddings": len(mod.embed_event(e, force=force))} for e in events}
        if stage == "tag":
            counts = {e: len(mod.tag_event(e, force=force)) for e in events}
            return {"tagged": counts, "classifier": mod.evaluate_classifier(config.DEMO_EVENT)}
        if stage == "classifier":
            return mod.train_classifier(train)
        return mod.train_ranker(train)
    if stage == "rank":
        return {"incident_snapshots": len(require_module("rank", stage).rank_event(event, force=force))}
    if stage == "brief":
        rank = require_module("rank", stage)
        fairness = require_module("fairness", stage)
        brief = require_module("brief", stage)
        d = config.event_dir(event)
        incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident)
        snap = rank.pick_demo_snapshot(incidents)
        fileio._atomic_write_text(d / "demo_snapshot.txt", snap + "\n")
        at = fairness.apply([i for i in incidents if i.snapshot_time == snap], 0.0)
        return {"snapshot": snap, "briefings": len(brief.brief_incidents(event, at, force=force))}
    return require_module("evaluate" if stage == "eval" else "benchmark", stage).run_all(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--event", default=config.DEMO_EVENT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args(argv)
    if not args.stub and args.stage is None:
        parser.error("--stage is required unless --stub is used")
    stage = "stub" if args.stub else args.stage
    owner = {"data": 1, "embed": 1, "classifier": 1, "tag": 1,
             "ranker": 2, "rank": 2, "brief": 3}.get(stage, 4)
    start = time.monotonic()
    print(f"[{stage}] start", flush=True)
    try:
        if args.stub:
            from .stubs import run_stub_pipeline
            reports = fileio.read_jsonl(config.FIXTURES / "outputs" / config.FIXTURE_EVENT /
                                       config.REPORTS_FILE, Report)
            result = run_stub_pipeline(config.FIXTURE_EVENT, reports)
        else:
            result = run_stage(stage, args.event, args.force)
        print(json.dumps(result, default=str, ensure_ascii=False))
        print(f"[{stage}] done in {time.monotonic() - start:.3f}s")
        return 0
    except Exception as exc:
        message = " ".join(str(exc).splitlines())
        print(f"[{stage}] failed (owner: Agent {owner}): {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

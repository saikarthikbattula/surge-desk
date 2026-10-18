"""Local-model measurements; captions describe the machine that actually ran them."""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import config, envcheck, fileio
from .contracts import Incident, Report, Tagged
from .evaluate import _clean, _figure, _save, _skipped, measured_caption
from .pipeline import StageUnavailable, pick_snapshot, require_module
from .textutil import mask_privacy, parse_iso


def latency_stats(latencies):
    return {"p50_latency_s": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else None}


def iter_stream(event, speed, run_seconds, cancelled=lambda: False):
    """Shared benchmark/demo engine. No cached embeddings or batch model scores."""
    replay = require_module("replay", "stream")
    tag = require_module("tag", "stream")
    group = require_module("group", "stream")
    rank = require_module("rank", "stream")
    reports = fileio.read_jsonl(config.event_dir(event) / config.REPORTS_FILE, Report)
    if not reports:
        return
    reports = sorted(reports, key=lambda r: (r.timestamp, r.report_id))
    start = replay.busiest_window_start(reports)
    state = group.GroupState(event)
    tagger, ranker = tag.Tagger(event), rank.Ranker()
    svi = rank.default_svi_lookup()
    received = actionable = 0
    began = time.monotonic()
    for idx, (scheduled, batch) in enumerate(replay.iter_batches(reports, speed, start, run_seconds)):
        if cancelled():
            break
        processing = time.monotonic()
        # Live inference never receives evaluation targets.
        batch = [replace(r, gold_label=None) for r in batch]
        if not batch:
            continue
        tags, vectors = tagger.tag_batch(batch)
        if len(batch) != len(tags) or len(vectors) != len(batch):
            raise ValueError("tag_batch returned misaligned reports/tags/embeddings (owner: Agent 1)")
        for report, tagged, vector in zip(batch, tags, vectors):
            if report.report_id != tagged.report_id:
                raise ValueError("tag_batch returned mismatched report IDs (owner: Agent 1)")
            state.add(report, tagged, vector)
        now = parse_iso(batch[-1].timestamp)
        incidents = rank.rank_now(state, now, ranker, svi)
        end = time.monotonic()
        received += len(batch)
        actionable += sum(t.is_actionable for t in tags)
        yield {"speed": speed, "batch_idx": idx, "n_reports": len(batch),
               "process_s": end - processing, "latency_s": end - scheduled,
               "open_incidents": len(incidents), "reports_in": received,
               "actionable": actionable, "elapsed_s": end - began}, incidents


def bench_stream(event=config.DEMO_EVENT, speeds=config.BENCH_SPEEDS,
                 run_seconds=config.BENCH_RUN_SECONDS) -> pd.DataFrame:
    if run_seconds <= 0:
        raise ValueError("run_seconds must be positive")
    d = config.event_dir(event)
    rows, summaries = [], []
    for speed in speeds:
        if speed != "max" and float(speed) <= 0:
            raise ValueError("Each speed must be positive or 'max'")
        current = [row for row, _ in iter_stream(event, speed, run_seconds)]
        rows.extend(current)
        stats = latency_stats([r["latency_s"] for r in current])
        p95 = stats["p95_latency_s"]
        summaries.append({"speed": speed,
                          "reports_per_sec": sum(r["n_reports"] for r in current) / current[-1]["elapsed_s"] if current else None,
                          **stats, "kept_up": p95 < 2 * config.BATCH_SECONDS if p95 is not None else None})
    frame = pd.DataFrame(rows, columns=["speed", "batch_idx", "n_reports", "process_s", "latency_s", "open_incidents"])
    summary = pd.DataFrame(summaries, columns=["speed", "reports_per_sec", "p50_latency_s", "p95_latency_s", "kept_up"])
    frame.to_csv(d / "benchmark_stream.csv", index=False)
    summary.to_csv(d / "benchmark_stream_summary.csv", index=False)

    def draw(ax):
        for speed, batches in frame.groupby("speed", sort=False):
            ax.plot(batches.batch_idx, batches.latency_s, marker=".", label=f"{speed}x" if speed != "max" else "max")
        ax.axhline(2 * config.BATCH_SECONDS, linestyle="--", color="grey", label="Kept-up threshold")
        ax.set_xlabel("Batch index")
        ax.set_ylabel("End-to-schedule latency (s)")
        ax.legend()

    _figure(d, "benchmark_stream.png", "Historical replay: live embedding, grouping and ranking", draw)
    return frame


def bench_granite(event=config.DEMO_EVENT, n=10) -> pd.DataFrame:
    import psutil

    if n <= 0:
        raise ValueError("n must be positive")
    brief = require_module("brief", "Granite benchmark")
    llm_module = require_module("llm", "Granite benchmark")
    d = config.event_dir(event)
    incidents = fileio.read_jsonl(d / config.INCIDENTS_FILE, Incident)
    snap = pick_snapshot(incidents, d, fixture=event.startswith("fixture"))
    at = sorted([i for i in incidents if i.snapshot_time == snap], key=lambda i: i.rank)[:n]
    reports = {r.report_id: r for r in fileio.read_jsonl(d / config.REPORTS_FILE, Report)}
    tags = {t.report_id: t for t in fileio.read_jsonl(d / config.TAGGED_FILE, Tagged)}
    embeddings = fileio.load_embeddings(d / config.EMBEDDINGS_FILE)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = threading.Event()

    def watch():
        while not done.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    watcher = threading.Thread(target=watch, daemon=True, name="surgedesk-rss")
    watcher.start()
    rows = []
    try:
        llm = llm_module.get_llm(log_path=d / config.LLM_LOG_FILE)
        load_s = llm.load_s
        for incident in at:
            reps = brief.select_representatives(incident, reports, tags, embeddings)
            # Direct calls deliberately bypass brief_incidents' cache.
            fields, extraction = brief.extract_fields(llm, reps)
            _, briefing = brief.write_briefing(llm, incident, fields)
            extract_s, brief_s = extraction["seconds"], briefing["seconds"]
            tokens = extraction["completion_tokens"] + briefing["completion_tokens"]
            seconds = extract_s + brief_s
            peak[0] = max(peak[0], process.memory_info().rss)
            rows.append({"incident_id": incident.incident_id, "extract_s": extract_s, "brief_s": brief_s,
                         "completion_tokens": tokens, "tokens_per_s": tokens / seconds if seconds > 0 else None})
            print(f'Granite benchmark {len(rows)}/{len(at)}: {incident.incident_id} saved', flush=True)
    finally:
        done.set()
        watcher.join()
    info = envcheck.collect()
    frame = pd.DataFrame(rows, columns=["incident_id", "extract_s", "brief_s", "completion_tokens", "tokens_per_s"])
    frame.to_csv(d / "benchmark_granite.csv", index=False)
    summary = {"load_s": load_s, "peak_rss_bytes": peak[0], "rss_sampling_interval_s": 0.05,
               "model_file": getattr(llm, "model_file", config.GRANITE_PATH.name),
               "configured_threads": config.N_THREADS, "configured_context_tokens": config.GRANITE_CTX,
               "extract_max_tokens": config.EXTRACT_MAX_TOKENS, "brief_max_tokens": config.BRIEF_MAX_TOKENS,
               "timing_note": "Uncached model calls; tokens_per_s is completion tokens divided by total call seconds, including prompt processing. Model load is recorded separately.",
               "nnpa_flag": info["nnpa_flag"], "zdnn_libs": info["zdnn_libs"], "n": len(frame),
               "mean_extract_s": float(frame.extract_s.mean()), "mean_brief_s": float(frame.brief_s.mean()),
               "tokens_per_s": float(frame.completion_tokens.sum() / (frame.extract_s.sum() + frame.brief_s.sum())) if (frame.extract_s.sum() + frame.brief_s.sum()) > 0 else None,
               "measured_on": {k: info[k] for k in ("arch", "cores", "ram_gb")},
               "caption": measured_caption(info)}
    _save(d, "benchmark_granite_summary.json", summary)

    def draw(ax):
        ax.bar(frame.incident_id, frame.extract_s, label="Extraction")
        ax.bar(frame.incident_id, frame.brief_s, bottom=frame.extract_s, label="Briefing")
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.set_ylabel("Uncached seconds")
        ax.legend()

    _figure(d, "benchmark_granite.png", "Granite extraction and briefing", draw, info)
    return frame


def bench_embed_devices(event=config.DEMO_EVENT) -> dict:
    d = config.event_dir(event)
    try:
        embed = require_module("embed", "embedding device benchmark")
        if not hasattr(embed, "compare_devices"):
            return _save(d, "benchmark_embed.json", _skipped("embed.compare_devices not implemented yet (owner: Agent 1)"))
        reports = fileio.read_jsonl(d / config.REPORTS_FILE, Report)
        result = embed.compare_devices([mask_privacy(r.text) for r in reports[:512]], n=512)
    except (StageUnavailable, FileNotFoundError, NotImplementedError) as exc:
        result = _skipped(exc)
    return _save(d, "benchmark_embed.json", result)


def run_all(event=config.DEMO_EVENT) -> dict:
    info = envcheck.collect()
    if info["threads_setting"] != 3:
        print(f'WARNING: benchmark threads_setting={info["threads_setting"]}; expected 3. Use bash run_benchmark.sh.', flush=True)
    d = config.event_dir(event)
    results = {"measured_on": {k: info[k] for k in ("arch", "cores", "ram_gb", "nnpa_flag")},
               "caption": measured_caption(info)}
    try:
        for name, fn in (("stream", bench_stream), ("granite", bench_granite), ("embed", bench_embed_devices)):
            try:
                value = fn(event)
                results[name] = {"status": "ok", "n_rows": len(value)} if isinstance(value, pd.DataFrame) else value
            except (StageUnavailable, FileNotFoundError) as exc:
                results[name] = _skipped(exc)
                print(f"[benchmark {name}] {exc}", flush=True)
    finally:
        # Avoid importing an absent LLM module solely to unload it.
        import sys
        loaded = sys.modules.get("surgedesk.llm")
        if loaded is not None:
            loaded.unload()
    fileio.write_json(d / "benchmark_summary.json", _clean(results))
    return _clean(results)

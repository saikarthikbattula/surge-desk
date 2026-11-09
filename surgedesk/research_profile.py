"""Measured local embedding batches and cache runs, with a cProfile artifact."""
import cProfile
import io
import pstats
import importlib.util
import subprocess

import numpy as np

from . import config, fileio, research_data
from .provenance import Measurement, capture, file_hash, report_identity
from .textutil import mask_privacy


def run(manifest, run_id, partition="dev"):
    from .research import _embeddings, _new_run
    selected = research_data.selected_reports(manifest, partition)
    if not selected:
        raise ValueError("No reports selected for profiling")
    rng = np.random.default_rng(config.SEED)
    indices = sorted(rng.choice(len(selected), min(config.RESEARCH_PROFILE_LIMIT, len(selected)), replace=False).tolist())
    selected = [selected[i] for i in indices]
    reports = [r for _, r in selected]
    directory = _new_run(run_id)
    profiler = cProfile.Profile()
    rows = []
    reference = None
    baseline = {"status": "unavailable", "reason": "Baseline measurement not completed"}
    profiler.enable()
    try:
        for batch in config.RESEARCH_BATCHES:
            for repeat in range(config.RESEARCH_REPEAT):
                for mode in ("cold", "warm"):
                    with Measurement() as measurement:
                        embeddings, timing = _embeddings(reports, mode, batch)
                    if reference is None:
                        reference = embeddings.copy()
                    rows.append({"batch_size": batch, "repeat": repeat, "n": len(embeddings),
                                 **timing, **measurement.as_dict(),
                                 "max_abs_diff": float(np.max(np.abs(embeddings - reference))),
                                 "min_cosine": float(np.min(np.sum(embeddings * reference, axis=1))),
                                 "reports_per_second": len(reports) / measurement.seconds})
                    fileio.write_json(directory / config.RESEARCH_PROFILE_FILE, {"status": "running", "rows": rows})
        # Run the preserved repository implementation on exactly the same text,
        # weights, threads and default batch; compare encode time to encode time.
        source = subprocess.check_output(["git", "show", config.RESEARCH_BASELINE_COMMIT + ":surgedesk/embed.py"], cwd=config.ROOT)
        baseline_path = directory / config.RESEARCH_BASELINE_EMBED_FILE
        baseline_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("surgedesk._profile_baseline", baseline_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with Measurement() as load:
            encoder = module.Embedder(model_dir=config.MINILM_DIR, threads=config.N_THREADS)
        baseline_rows = []
        for repeat in range(config.RESEARCH_REPEAT):
            with Measurement() as measured:
                values = encoder.encode([r.text for r in reports], batch_size=config.EMBED_BATCH)
            baseline_rows.append({"repeat": repeat, **measured.as_dict(), "n": len(values),
                                   "max_abs_diff": float(np.max(np.abs(values - reference))),
                                   "min_cosine": float(np.min(np.sum(values * reference, axis=1)))})
        baseline = {"status": "complete", "commit": config.RESEARCH_BASELINE_COMMIT,
                    "source_sha256": file_hash(baseline_path), "batch_size": config.EMBED_BATCH,
                    "model_load": load.as_dict(), "rows": baseline_rows,
                    "comparison": "Baseline encode seconds versus current embedding_seconds; current shared foundation/config used with identical recorded settings"}
    except (OSError, ImportError, ValueError) as exc:
        fileio.write_json(directory / config.RESEARCH_PROFILE_FILE, {"status": "unavailable", "rows": rows,
                            "reason": mask_privacy(str(exc)), "provenance": capture()})
        raise
    finally:
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(40)
        fileio._atomic_write_text(directory / config.RESEARCH_PROFILE_STATS_FILE, stream.getvalue())
    result = {"status": "complete", "rows": rows, "baseline": baseline,
              "report_keys": [r["key"] for r, _ in selected], "report_identity": report_identity(reports),
              "manifest_fingerprint": manifest["fingerprint"],
              "provenance": capture(), "limitation": "Application-cache cold; OS caches not cleared. RSS sampled in one process."}
    fileio.write_json(directory / config.RESEARCH_PROFILE_FILE, result)
    return str(directory)

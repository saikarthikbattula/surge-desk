"""Offline research CLI. Run `python -m surgedesk.research --help`."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from . import config

# CLI callers should also set these before Python starts for existing BLAS pools.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, str(config.N_THREADS))

import numpy as np
from . import artifacts, fileio, research_data
from .contracts import CATEGORIES
from .provenance import Measurement, capture, digest, file_hash, report_identity, tree_hash
from .textutil import mask_privacy


def metrics(gold, predictions):
    gold, predictions = list(gold), list(predictions)
    if len(gold) != len(predictions):
        raise ValueError("Prediction alignment mismatch")
    labels = list(CATEGORIES)
    if set(gold + predictions) - set(labels):
        raise ValueError("Unknown category")
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(gold, predictions):
        matrix[labels.index(actual)][labels.index(predicted)] += 1
    per_class = {}
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        true_positive = matrix[i][i]
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else None
        f1 = 2 * true_positive / (support + predicted_count) if support + predicted_count else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    # Fixed taxonomy, including absent classes, makes event scores comparable.
    return {"n": len(gold), "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(gold) if gold else None,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels) if gold else None,
            "macro_definition": "mean over fixed contract taxonomy; absent classes have F1 zero",
            "per_class": per_class, "labels": labels, "confusion": matrix,
            "missing_gold_categories": [c for c in labels if per_class[c]["support"] == 0],
            "assistance_recall": {c: per_class[c]["recall"] for c in config.RESEARCH_ASSISTANCE_LABELS}}


def _new_run(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."} or any(c in run_id for c in "/\\:"):
        raise ValueError("Run ID must be a single safe directory name")
    directory = config.RESEARCH_DIR / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _atomic_model(path, artifact):
    import joblib
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(artifact, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _embeddings(reports, mode="warm", batch_size=None):
    from .embed import Embedder
    identity = {"reports": report_identity(reports), "model": tree_hash(config.MINILM_DIR),
                "tokens": config.EMBED_MAX_TOKENS, "dimension": config.EMBED_DIM,
                "implementation": file_hash(Path(__file__).with_name("embed.py"))}
    if not identity["model"]:
        raise FileNotFoundError(f"Local MiniLM weights missing: {config.MINILM_DIR}")
    cache = config.RESEARCH_DIR / config.RESEARCH_EMBED_CACHE_NAME / digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    path, manifest = cache / config.EMBEDDINGS_FILE, cache / config.EMBED_MANIFEST_FILE
    start = time.perf_counter()
    if mode == "warm" and artifacts.valid(manifest, identity, [path]):
        result = fileio.load_embeddings(path)
        if result.shape == (len(reports), config.EMBED_DIM) and np.isfinite(result).all():
            return result, {"cache_hit": True, "cache_mode": mode, "model_load_seconds": 0.0,
                            "embedding_seconds": time.perf_counter() - start}
    load_start = time.perf_counter()
    encoder = Embedder()
    load_seconds = time.perf_counter() - load_start
    encode_start = time.perf_counter()
    result = encoder.encode([r.text for r in reports], batch_size=batch_size or config.EMBED_BATCH)
    if result.shape != (len(reports), config.EMBED_DIM) or not np.isfinite(result).all():
        raise ValueError("Encoder returned invalid embeddings")
    seconds = time.perf_counter() - encode_start
    fileio.save_embeddings(path, result)
    artifacts.complete(manifest, identity, [path])
    return result, {"cache_hit": False, "cache_mode": mode, "model_load_seconds": load_seconds,
                    "embedding_seconds": seconds}


def train(manifest, run_id, kind="tfidf", regularization=1.0, class_weight="balanced"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if kind not in {"tfidf", "minilm"} or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Invalid model configuration")
    if class_weight not in {None, "balanced"}:
        raise ValueError("Unsupported class weighting")
    selected = research_data.selected_reports(manifest, "train")
    if not selected:
        raise ValueError("No training rows after exclusions")
    rows, reports = zip(*selected)
    if config.DEMO_EVENT in {r.event for r in reports}:
        raise ValueError("Never train on config.DEMO_EVENT")
    labels = [r.gold_label for r in reports]
    if None in labels or len(set(labels)) < 2:
        raise ValueError("Training requires labeled rows from at least two classes")
    directory = _new_run(run_id)
    fileio.write_json(directory / config.RESEARCH_MANIFEST_FILE, manifest)
    fileio.write_json(directory / config.RESEARCH_PROVENANCE_FILE, capture())
    vectorizer = None
    embedding = {"embedding_seconds": None, "model_load_seconds": None,
                 "reason": "TF-IDF has no embedding stage"}
    with Measurement() as measurement:
        if kind == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=config.RESEARCH_TFIDF_NGRAM,
                                         max_features=config.RESEARCH_TFIDF_MAX_FEATURES)
            features = vectorizer.fit_transform([mask_privacy(r.text) for r in reports])
        else:
            features, embedding = _embeddings(reports)
        model = LogisticRegression(C=regularization, class_weight=class_weight,
                                   max_iter=config.RESEARCH_MAX_ITER, random_state=config.SEED)
        model.fit(features, labels)
    artifact = {"model": model, "vectorizer": vectorizer, "kind": kind,
                "classes": model.classes_.tolist(), "train_events": sorted({r.event for r in reports}),
                "manifest_fingerprint": manifest["fingerprint"], "train_keys": [r["key"] for r in rows],
                "model_files": tree_hash(config.MINILM_DIR) if kind == "minilm" else {},
                "configuration": {"C": regularization, "class_weight": class_weight, "seed": config.SEED,
                                  "threads": config.N_THREADS, "kind": kind,
                                  "max_iter": config.RESEARCH_MAX_ITER,
                                  "tfidf_max_features": config.RESEARCH_TFIDF_MAX_FEATURES if kind == "tfidf" else None,
                                  "tfidf_ngram_range": config.RESEARCH_TFIDF_NGRAM if kind == "tfidf" else None,
                                  "embedding_max_tokens": config.EMBED_MAX_TOKENS if kind == "minilm" else None,
                                  "embedding_batch_size": config.EMBED_BATCH if kind == "minilm" else None},
                "training": measurement.as_dict(), "embedding": embedding}
    path = directory / config.RESEARCH_MODEL_FILE
    _atomic_model(path, artifact)
    fileio.write_json(directory / config.RESEARCH_TRAINING_FILE, {k: v for k, v in artifact.items() if k not in {"model", "vectorizer"}})
    artifacts.complete(directory / config.STAGE_MANIFEST_FILE,
                       {"manifest": manifest["fingerprint"], "configuration": artifact["configuration"]},
                       [path, directory / config.RESEARCH_TRAINING_FILE, directory / config.RESEARCH_MANIFEST_FILE])
    return directory


def _load_model(directory, manifest):
    import joblib
    artifacts.verify(directory / config.STAGE_MANIFEST_FILE)
    artifact = joblib.load(directory / config.RESEARCH_MODEL_FILE)
    if artifact["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Classifier and manifest differ; train with this frozen manifest")
    if artifact["kind"] == "minilm" and artifact["model_files"] != tree_hash(config.MINILM_DIR):
        raise ValueError("MiniLM files changed; train again")
    return artifact


def evaluate(manifest, model_dir, run_id, partition="dev", events=None, cache_mode="warm", selection=None):
    evaluation_started = time.perf_counter()
    research_data.validate_manifest(manifest)
    model_dir = Path(model_dir)
    artifact = _load_model(model_dir, manifest)
    if partition == "train":
        raise ValueError("Training scores are not evaluation evidence")
    if partition == "test":
        if selection is None:
            raise ValueError("Reserved evaluation requires a frozen selection file")
        chosen = fileio.read_json(selection)
        if chosen["manifest_fingerprint"] != manifest["fingerprint"] or file_hash(model_dir / config.RESEARCH_MODEL_FILE) not in chosen["model_hashes"]:
            raise ValueError("Model or manifest is outside the frozen final comparison")
        access = config.RESEARCH_DIR / config.RESEARCH_RESERVED_ACCESS_NAME / (manifest["fingerprint"] + ".json")
        selection_hash = file_hash(selection)
        if access.exists() and fileio.read_json(access)["selection_sha256"] != selection_hash:
            raise ValueError("Reserved results were opened under a different selection")
        if not access.exists():
            fileio.write_json(access, {"selection_sha256": selection_hash,
                                      "manifest_fingerprint": manifest["fingerprint"], "provenance": capture()})
    selected = research_data.selected_reports(manifest, partition, events)
    if not selected:
        raise ValueError("No labeled evaluation rows in this partition")
    rows, reports = zip(*selected)
    if set(artifact["train_events"]) & {r.event for r in reports}:
        raise ValueError("Evaluation event was used for training")
    directory = _new_run(run_id)
    embedding = {"embedding_seconds": None, "model_load_seconds": None,
                 "reason": "TF-IDF has no embedding stage"}
    with Measurement() as measurement:
        if artifact["kind"] == "tfidf":
            features = artifact["vectorizer"].transform([mask_privacy(r.text) for r in reports])
        else:
            features, embedding = _embeddings(reports, mode=cache_mode)
        predict_start = time.perf_counter()
        probabilities = artifact["model"].predict_proba(features)
        predictions = artifact["model"].classes_[probabilities.argmax(axis=1)].tolist()
        predict_seconds = time.perf_counter() - predict_start
    output = [{"key": row["key"], "event": r.event, "report_id": r.report_id,
               "gold_label": r.gold_label, "prediction": pred,
               "confidence": float(max(probs))} for row, r, pred, probs in zip(rows, reports, predictions, probabilities)]
    labeled = [r for r in output if r["gold_label"] is not None]
    per_event = {e: metrics([r["gold_label"] for r in labeled if r["event"] == e],
                            [r["prediction"] for r in labeled if r["event"] == e]) for e in sorted({r.event for r in reports})}
    result = {"schema": config.RESEARCH_SCHEMA, "synthetic": manifest["synthetic"], "partition": partition,
              "manifest_fingerprint": manifest["fingerprint"], "model_sha256": file_hash(model_dir / config.RESEARCH_MODEL_FILE),
              "configuration": artifact["configuration"], "provenance": capture(), "per_event": per_event,
              "pooled": metrics([r["gold_label"] for r in labeled], [r["prediction"] for r in labeled]),
              "mean_event_macro_f1": float(np.mean([x["macro_f1"] for x in per_event.values() if x["macro_f1"] is not None])) if labeled else None,
              "timing": {**measurement.as_dict(), **embedding, "prediction_seconds": predict_seconds,
                         "inference_seconds": max(0.0, measurement.seconds - (embedding.get("model_load_seconds") or 0.0)),
                         "total_processing_seconds": time.perf_counter() - evaluation_started,
                         "total_processing_definition": "validation/model/data loading through predictions; excludes report export",
                         "reports_per_second": len(reports) / measurement.seconds},
              "uncertainty": {"status": "unavailable", "reason": "No uncertainty interval computed; report event sample sizes"},
              "interpretation": "Humanitarian category agreement; not validated operational urgency"}
    fileio.write_jsonl(directory / config.RESEARCH_PREDICTIONS_FILE, output)
    fileio.write_json(directory / config.RESEARCH_RESULT_FILE, result)
    fileio.write_json(directory / config.RESEARCH_MANIFEST_FILE, manifest)
    _write_tables(directory, result)
    artifacts.complete(directory / config.STAGE_MANIFEST_FILE,
                       {"model": result["model_sha256"], "manifest": manifest["fingerprint"], "partition": partition},
                       [directory / config.RESEARCH_RESULT_FILE, directory / config.RESEARCH_PREDICTIONS_FILE,
                        directory / config.RESEARCH_MANIFEST_FILE])
    return directory


def _write_tables(directory, result):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["event", "category", "precision", "recall", "f1", "support"])
    for event, values in result["per_event"].items():
        for label, row in values["per_class"].items():
            writer.writerow([event, label, row["precision"], row["recall"], row["f1"], row["support"]])
    fileio._atomic_write_text(directory / config.RESEARCH_CLASS_TABLE_FILE, buffer.getvalue())
    lines = ["# SurgeDesk experiment", "", "Synthetic fixtures: not real-disaster evidence." if result["synthetic"] else result["interpretation"],
             "", f"Partition: {result['partition']}; model SHA-256: {result['model_sha256']}", "",
             "| Event | Labeled reports | Macro-F1 |", "| --- | ---: | ---: |"]
    for event, values in result["per_event"].items():
        lines.append(f"| {event} | {values['n']} | {values['macro_f1']} |")
    lines.extend(["", f"Equal-weight mean event macro-F1: {result['mean_event_macro_f1']}",
                  f"Pooled macro-F1: {result['pooled']['macro_f1']}",
                  "", "Macro-F1 uses the fixed contract taxonomy, including zero F1 for absent categories.",
                  "Recall is unavailable when a category has no gold support.", "", "Timing and memory:",
                  "```json", json.dumps(result["timing"], indent=2), "```",
                  "", "Uncertainty intervals: unavailable. Laptop and LinuxONE runs must be compared separately."])
    fileio._atomic_write_text(directory / config.RESEARCH_SUMMARY_FILE, "\n".join(lines) + "\n")
    plot_confusion(directory / config.RESEARCH_CONFUSION_FILE, result)


def plot_confusion(path, result):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12, 10), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    matrix = result["pooled"]["confusion"]
    image = ax.imshow(matrix, cmap="Blues")
    names = [label.replace("_", "\n") for label in result["pooled"]["labels"]]
    ax.set_xticks(range(len(names)), names, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Dataset gold category")
    title = ("SYNTHETIC FIXTURE — " if result["synthetic"] else "Local experiment — ") + f"{result['partition']} pooled confusion; n={result['pooled']['n']}"
    ax.set_title(title)
    arch = result["provenance"]["environment"]["arch"]
    caption = f"Recorded architecture: {arch}. " + ("LinuxONE run." if arch == "s390x" else "Development run, not LinuxONE.")
    figure.supxlabel(caption, fontsize=9)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=7,
                    color="white" if value > max(map(max, matrix), default=0) / 2 else "black")
    figure.colorbar(image, ax=ax)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, metadata={"Title": title, "Description": caption})


def compare(left, right):
    for path in (left, right):
        artifacts.verify(Path(path) / config.STAGE_MANIFEST_FILE)
    a, b = [fileio.read_json(Path(p) / config.RESEARCH_RESULT_FILE) for p in (left, right)]
    for key in ("manifest_fingerprint", "partition"):
        if a[key] != b[key]:
            raise ValueError(f"Comparison requires identical {key}")
    pa, pb = [fileio.read_jsonl(Path(p) / config.RESEARCH_PREDICTIONS_FILE) for p in (left, right)]
    if [(r["key"], r["gold_label"]) for r in pa] != [(r["key"], r["gold_label"]) for r in pb]:
        raise ValueError("Comparison workloads or their ordering differ")
    hardware_keys = ("arch", "cores", "ram_gb", "threads_setting")
    same_hardware = all(a["provenance"]["environment"][k] == b["provenance"]["environment"][k] for k in hardware_keys)
    same_hardware &= bool(a["provenance"].get("machine_token")) and a["provenance"].get("machine_token") == b["provenance"].get("machine_token")
    same_hardware &= a["provenance"].get("thread_environment") == b["provenance"].get("thread_environment")
    same_environment = a["provenance"].get("installed_distributions") == b["provenance"].get("installed_distributions")
    return {"n": len(pa), "identical_predictions": [r["prediction"] for r in pa] == [r["prediction"] for r in pb],
            "identical_quality_metrics": a["pooled"] == b["pooled"] and a["per_event"] == b["per_event"],
            "macro_f1_delta": b["mean_event_macro_f1"] - a["mean_event_macro_f1"] if a["mean_event_macro_f1"] is not None and b["mean_event_macro_f1"] is not None else None,
            "per_class_f1_delta": {c: b["pooled"]["per_class"][c]["f1"] - a["pooled"]["per_class"][c]["f1"] for c in CATEGORIES},
            "same_hardware": same_hardware,
            "same_runtime_environment": same_environment,
            "runtime_ratio_left_over_right": a["timing"]["seconds"] / b["timing"]["seconds"] if same_hardware and same_environment else None,
            "changed_predictions": [{"key": x["key"], "before": x["prediction"], "after": y["prediction"],
                                      "gold_label": x["gold_label"]} for x, y in zip(pa, pb) if x["prediction"] != y["prediction"]]}


def review(manifest, evaluation_dir, output):
    directory = Path(evaluation_dir)
    artifacts.verify(directory / config.STAGE_MANIFEST_FILE)
    result = fileio.read_json(directory / config.RESEARCH_RESULT_FILE)
    if result["partition"] != "dev" or result["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Review sampling is restricted to matching development results")
    reports = {row["key"]: r for row, r in research_data.selected_reports(manifest, "dev")}
    predictions = fileio.read_jsonl(directory / config.RESEARCH_PREDICTIONS_FILE)
    errors = [p for p in predictions if p["gold_label"] != p["prediction"]]
    rng = np.random.default_rng(config.SEED)
    chosen = rng.permutation(len(errors))[:config.RESEARCH_REVIEW_LIMIT]
    rows = [{**errors[i], "text": mask_privacy(reports[errors[i]["key"]].text), "reviewer": "", "uncertain": "",
             "error_categories": [], "outcome": "", "review_kind": "", "note": "", "sample_seed": config.SEED,
             "sampling": "uniform without replacement from development classification errors",
             "model_sha256": result["model_sha256"], "manifest_fingerprint": manifest["fingerprint"]} for i in chosen]
    research_data.save_new(output, {"synthetic": manifest["synthetic"], "classification_errors": rows})
    return {"sampled": len(rows), "output": str(output)}


def select(manifest, candidates, baseline, output):
    research_data.validate_manifest(manifest)
    if (config.RESEARCH_DIR / config.RESEARCH_RESERVED_ACCESS_NAME / (manifest["fingerprint"] + ".json")).exists():
        raise ValueError("Reserved evaluation was opened; do not select again on the same benchmark")
    checked = []
    for candidate in candidates:
        directory = Path(candidate)
        artifacts.verify(directory / config.STAGE_MANIFEST_FILE)
        result = fileio.read_json(directory / config.RESEARCH_RESULT_FILE)
        if result["partition"] != "dev" or result["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Select only matching development experiments")
        if result["mean_event_macro_f1"] is None:
            raise ValueError("Cannot select a model without labeled development measurements")
        checked.append((directory, result))
    if not checked:
        raise ValueError("At least one development candidate is required")
    for candidate, _ in checked[1:]:
        compare(checked[0][0], candidate)
    best_path, best = min(checked, key=lambda pair: (-pair[1]["mean_event_macro_f1"], pair[1]["timing"]["inference_seconds"], str(pair[0])))
    baseline_hash = file_hash(Path(baseline) / config.RESEARCH_MODEL_FILE)
    if baseline_hash not in {r["model_sha256"] for _, r in checked}:
        raise ValueError("Include baseline development results among candidates")
    selection = {"manifest_fingerprint": manifest["fingerprint"], "selected_model_sha256": best["model_sha256"],
                 "selected_development_run": str(best_path), "baseline_model_sha256": baseline_hash,
                 "model_hashes": sorted({baseline_hash, best["model_sha256"]}),
                 "criterion": "development mean event macro-F1; exact ties use representation plus prediction time, excluding encoder load",
                 "candidate_results": {str(p): file_hash(p / config.RESEARCH_RESULT_FILE) for p, _ in checked}}
    research_data.save_new(output, selection)
    return selection


def smoke(run_id):
    root = _new_run(run_id)
    manifest = research_data.freeze(research_data.make_manifest(config.FIXTURES / "humaid",
         {"train": [config.FIXTURE_TRAIN_EVENT], "dev": [config.FIXTURE_EVENT]},
         "Offline synthetic smoke test; no unseen-event performance claim", synthetic=True))
    fileio.write_json(root / config.RESEARCH_MANIFEST_FILE, manifest)
    model = train(manifest, run_id + "-model")
    first = evaluate(manifest, model, run_id + "-first")
    second = evaluate(manifest, model, run_id + "-second")
    comparison = compare(first, second)
    if not comparison["identical_predictions"] or not comparison["identical_quality_metrics"]:
        raise ValueError("Fixture reproducibility check failed")
    review(manifest, first, root / config.RESEARCH_CLASS_REVIEW_FILE)
    from .research_review import location_sample
    location_sample(manifest, root / config.RESEARCH_LOCATION_REVIEW_FILE, config.FIXTURES / "gazetteer")
    from .research_demo import run
    demo = run(run_id + "-demo")
    result = {"synthetic": True, "status": "passed", "model": str(model), "first": str(first),
              "second": str(second), "comparison": comparison, "demo": demo,
              "real_evaluation": {"status": "pending", "reason": "Requires local data, weights, and reviewed event selection"}}
    fileio.write_json(root / config.RESEARCH_SMOKE_FILE, result)
    return result


def saved_results():
    """Read saved snapshots without loading models or displaying stale metrics."""
    summaries = []
    for path in sorted(config.RESEARCH_DIR.glob("*/" + config.RESEARCH_RESULT_FILE)):
        try:
            artifacts.verify(path.parent / config.STAGE_MANIFEST_FILE)
            result = fileio.read_json(path)
            summaries.append({"run": path.parent.name, "status": "available", "synthetic": result["synthetic"],
                              "partition": result["partition"], "model_sha256": result["model_sha256"],
                              "mean_event_macro_f1": result["mean_event_macro_f1"], "n": result["pooled"]["n"],
                              "hardware": result["provenance"]["environment"]["arch"],
                              "timing": result["timing"], "uncertainty": result["uncertainty"]})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summaries.append({"run": path.parent.name, "status": "unavailable", "reason": mask_privacy(str(exc))})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--data", type=Path, default=config.HUMAID_DIR)
    man = sub.add_parser("manifest")
    man.add_argument("--data", type=Path, default=config.HUMAID_DIR)
    for name in config.RESEARCH_PARTITIONS:
        man.add_argument("--" + name, nargs="*", default=[])
    man.add_argument("--rationale", required=True)
    man.add_argument("--synthetic", action="store_true")
    man.add_argument("--output", type=Path, required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--manifest", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    rebuild = sub.add_parser("reconstruct-manifest")
    rebuild.add_argument("--specification", type=Path, required=True)
    rebuild.add_argument("--data", type=Path, default=config.HUMAID_DIR)
    rebuild.add_argument("--output", type=Path, required=True)
    for command in ("train", "evaluate", "review", "select"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--manifest", type=Path, required=True)
        if command in {"train", "evaluate"}:
            cmd.add_argument("--run-id", required=True)
        if command == "train":
            cmd.add_argument("--kind", choices=("tfidf", "minilm"), default="tfidf")
            cmd.add_argument("--C", type=float, default=1.0)
            cmd.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
        if command == "evaluate":
            cmd.add_argument("--model", type=Path, required=True)
            cmd.add_argument("--partition", choices=("dev", "test", "historical"), default="dev")
            cmd.add_argument("--events", nargs="+")
            cmd.add_argument("--cache-mode", choices=("cold", "warm"), default="warm")
            cmd.add_argument("--selection", type=Path)
        if command in {"review", "select"}:
            cmd.add_argument("--output", type=Path, required=True)
        if command == "review":
            cmd.add_argument("--evaluation", type=Path, required=True)
        if command == "select":
            cmd.add_argument("--candidates", type=Path, nargs="+", required=True)
            cmd.add_argument("--baseline", type=Path, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("left", type=Path)
    comp.add_argument("right", type=Path)
    locations = sub.add_parser("review-locations")
    locations.add_argument("--manifest", type=Path, required=True)
    locations.add_argument("--gazetteer", type=Path, default=config.GAZETTEER_DIR)
    locations.add_argument("--output", type=Path, required=True)
    duplicate = sub.add_parser("review-duplicates")
    duplicate.add_argument("--manifest", type=Path, required=True)
    duplicate.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("review-summary")
    summary.add_argument("review_file", type=Path)
    profile = sub.add_parser("profile")
    profile.add_argument("--manifest", type=Path, required=True)
    profile.add_argument("--run-id", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--manifest", type=Path, required=True)
    sweep.add_argument("--kind", choices=("tfidf", "minilm"), required=True)
    sweep.add_argument("--run-id", required=True)
    for name in ("smoke", "demo", "walkthrough"):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = research_data.inventory(args.data)
        elif args.command == "manifest":
            value = research_data.make_manifest(args.data, {name: getattr(args, name) for name in config.RESEARCH_PARTITIONS}, args.rationale, args.synthetic)
            research_data.save_new(args.output, value)
            result = {"output": str(args.output), "rows": len(value["rows"]), "excluded": len(value["exclusions"]), "near_pairs": len(value["near_duplicates"])}
        elif args.command == "freeze":
            research_data.save_new(args.output, research_data.freeze(fileio.read_json(args.manifest)))
            result = {"output": str(args.output)}
        elif args.command == "reconstruct-manifest":
            value = research_data.reconstruct_manifest(fileio.read_json(args.specification), args.data)
            research_data.save_new(args.output, value)
            result = {"output": str(args.output), "fingerprint": value["fingerprint"]}
        elif args.command == "compare":
            result = compare(args.left, args.right)
        elif args.command == "review-locations":
            from .research_review import location_sample
            result = location_sample(fileio.read_json(args.manifest), args.output, args.gazetteer)
        elif args.command == "review-duplicates":
            from .research_review import duplicate_sample
            result = duplicate_sample(fileio.read_json(args.manifest), args.output)
        elif args.command == "review-summary":
            from .research_review import summarize
            result = summarize(args.review_file)
        elif args.command == "profile":
            from .research_profile import run
            result = run(fileio.read_json(args.manifest), args.run_id)
        elif args.command == "sweep":
            from .research_sweep import run
            result = run(fileio.read_json(args.manifest), args.run_id, args.kind)
        elif args.command == "smoke":
            result = smoke(args.run_id)
        elif args.command == "demo":
            from .research_demo import run
            result = run(args.run_id)
        elif args.command == "walkthrough":
            from .research_walkthrough import run
            result = run(args.run_id)
        else:
            manifest = fileio.read_json(args.manifest)
            if args.command == "train":
                result = str(train(manifest, args.run_id, args.kind, args.C, None if args.class_weight == "none" else args.class_weight))
            elif args.command == "evaluate":
                result = str(evaluate(manifest, args.model, args.run_id, args.partition, args.events, args.cache_mode, args.selection))
            elif args.command == "review":
                result = review(manifest, args.evaluation, args.output)
            else:
                result = select(manifest, args.candidates, args.baseline, args.output)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(mask_privacy(f"Research command failed: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

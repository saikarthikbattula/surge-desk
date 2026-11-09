"""Publish computed aggregate evidence; never export report text or annotations."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

import numpy as np

from . import artifacts, config, fileio, research_data
from .provenance import digest, file_hash, report_identity
from .research import compare, plot_confusion
from .research_review import summarize


def _read_result(directory):
    directory = Path(directory)
    artifacts.verify(directory / config.STAGE_MANIFEST_FILE)
    return fileio.read_json(directory / config.RESEARCH_RESULT_FILE)


def publish(manifest, baseline_dev, selected_dev, baseline_final, selected_final,
            selection, profile, classification_review=None, location_review=None, proposed_locations=None, sweep=None):
    research_data.validate_manifest(manifest)
    results = [_read_result(p) for p in (baseline_dev, selected_dev, baseline_final, selected_final)]
    bd, sd, bf, sf = results
    chosen = fileio.read_json(selection)
    if any(r["manifest_fingerprint"] != manifest["fingerprint"] for r in results):
        raise ValueError("All reported runs must use the same frozen manifest")
    if bd["partition"] != "dev" or sd["partition"] != "dev" or bf["partition"] != "test" or sf["partition"] != "test":
        raise ValueError("Report requires development selection and reserved-event results")
    if (bf["model_sha256"] != chosen["baseline_model_sha256"]
            or sf["model_sha256"] != chosen["selected_model_sha256"]
            or bd["model_sha256"] != chosen["baseline_model_sha256"]
            or sd["model_sha256"] != chosen["selected_model_sha256"]):
        raise ValueError("Final comparison differs from the saved selection")
    access = fileio.read_json(config.RESEARCH_DIR / config.RESEARCH_RESERVED_ACCESS_NAME / (manifest["fingerprint"] + ".json"))
    if access["selection_sha256"] != file_hash(selection):
        raise ValueError("Selection changed after opening reserved results")
    comparison = compare(baseline_final, selected_final)
    events = sorted(bf["per_event"])
    deltas = np.array([sf["per_event"][e]["macro_f1"] - bf["per_event"][e]["macro_f1"] for e in events])
    rng = np.random.default_rng(config.SEED)
    resampled = rng.choice(deltas, size=(config.RESEARCH_BOOTSTRAP_REPEATS, len(events)), replace=True).mean(axis=1)
    interval = np.quantile(resampled, [0.025, 0.975]).tolist()
    source = fileio.read_json(Path(manifest["root"]) / config.RESEARCH_SOURCE_MANIFEST_FILE)
    counts = Counter(r["partition"] for r in manifest["rows"] if r["key"] not in {x["key"] for x in manifest["exclusions"]})
    public_manifest = {"schema": manifest["schema"], "synthetic": manifest["synthetic"], "source_repository": source["repository"],
                       "source_revision": source["revision"], "scope": source["scope"],
                       "rationale": manifest["rationale"], "partitions": manifest["partitions"],
                       "active_counts": dict(counts), "tsv_checksums": manifest["sources"],
                       "excluded_keys": sorted(x["key"] for x in manifest["exclusions"]),
                       "private_manifest_fingerprint": manifest["fingerprint"],
                       "workload_fingerprints": {role: report_identity([r for _, r in research_data.selected_reports(manifest, role)]) for role in config.RESEARCH_PARTITIONS},
                       "privacy": "No report text, gold labels, or private review notes included",
                       "review_provenance": manifest.get("review_provenance"),
                       "near_duplicate_threshold": manifest["near_duplicate_threshold"]}
    quality = {"development": {"baseline": bd["mean_event_macro_f1"], "selected": sd["mean_event_macro_f1"]},
               "reserved": {"baseline": bf["mean_event_macro_f1"], "selected": sf["mean_event_macro_f1"],
                            "delta": comparison["macro_f1_delta"], "per_event": {e: {"n": bf["per_event"][e]["n"],
                                    "baseline_macro_f1": bf["per_event"][e]["macro_f1"], "selected_macro_f1": sf["per_event"][e]["macro_f1"],
                                    "delta": float(deltas[i])} for i, e in enumerate(events)},
                            "per_class": {c: {"baseline": bf["pooled"]["per_class"][c], "selected": sf["pooled"]["per_class"][c]} for c in bf["pooled"]["labels"]},
                            "event_bootstrap_delta_95_percent": interval,
                            "uncertainty_method": f"Paired event resampling, {config.RESEARCH_BOOTSTRAP_REPEATS} draws, seed {config.SEED}; few events, descriptive only"}}
    perf = fileio.read_json(Path(profile) / config.RESEARCH_PROFILE_FILE)
    if perf.get("manifest_fingerprint") != manifest["fingerprint"] or perf.get("status") != "complete":
        raise ValueError("Profiling did not complete on this manifest")
    reviews = {"classification": summarize(classification_review) if classification_review else None,
               "location": summarize(location_review) if location_review else None}
    experiments = []
    if sweep is not None:
        journal = fileio.read_json(Path(sweep) / config.RESEARCH_SWEEP_FILE)
        if journal["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Experiment history uses a different manifest")
        for attempt in journal["attempts"]:
            row = {k: attempt.get(k) for k in ("C", "class_weight", "status", "reason")}
            if attempt["status"] == "complete":
                result = _read_result(attempt["evaluation"])
                if result["partition"] != "dev" or result["manifest_fingerprint"] != manifest["fingerprint"]:
                    raise ValueError("Experiment history is not development-only")
                row.update(mean_event_macro_f1=result["mean_event_macro_f1"], assistance_recall=result["pooled"]["assistance_recall"])
            experiments.append(row)
    location_changes = None
    if location_review and proposed_locations:
        original = fileio.read_json(location_review)["locations"]
        proposed = fileio.read_json(proposed_locations)["locations"]
        if [r["key"] for r in original] != [r["key"] for r in proposed]:
            raise ValueError("Location comparison samples differ")
        changed = [i for i, (a, b) in enumerate(zip(original, proposed))
                   if a["location"]["places"] != b["location"]["places"] or a["location"]["county_fips"] != b["location"]["county_fips"]]
        location_changes = {"sampled": len(original), "changed": len(changed),
                            "changed_previously_flagged": sum(original[i].get("outcome") == "incorrect" for i in changed),
                            "limitation": "Agent-assisted development audit; not independent county ground truth or a generalization estimate"}
    evidence = {"manifest": public_manifest, "quality": quality, "baseline_configuration": bf["configuration"],
                "selected_configuration": sf["configuration"], "baseline_timing": bf["timing"], "selected_timing": sf["timing"],
                "same_hardware": comparison["same_hardware"], "same_runtime_environment": comparison["same_runtime_environment"],
                "profile": perf, "reviews": reviews, "location_changes": location_changes,
                "experiment_history": experiments,
                "provenance": {"baseline": {k: bf["provenance"].get(k) for k in ("commit", "branch", "seed", "threads", "source_files")},
                               "selected": {k: sf["provenance"].get(k) for k in ("commit", "branch", "seed", "threads", "source_files")}},
                "model_checksums": {"baseline": bf["model_sha256"], "selected": sf["model_sha256"]}}
    for name, result in (("baseline", bf), ("selected", sf)):
        environment = result["provenance"]["environment"]
        evidence["provenance"][name]["hardware"] = {k: environment.get(k) for k in ("arch", "byteorder", "python", "cores", "physical_cores", "ram_gb", "nnpa_flag", "threads_setting")}
        evidence["provenance"][name]["packages"] = environment["packages"]
    # Profiling has private row keys and source paths; export only aggregate rows.
    evidence["profile"] = {k: perf[k] for k in ("status", "rows", "baseline", "manifest_fingerprint", "limitation")}
    fileio.write_json(config.RESEARCH_PUBLIC_MANIFEST, public_manifest)
    fileio.write_json(config.RESEARCH_PUBLIC_EVIDENCE, evidence)
    plot_confusion(config.RESEARCH_PUBLIC_FIGURES / config.RESEARCH_BASELINE_CONFUSION, bf)
    plot_confusion(config.RESEARCH_PUBLIC_FIGURES / config.RESEARCH_SELECTED_CONFUSION, sf)
    lines = ["# SurgeDesk research comparison", "", "Generated from verified local experiment artifacts; not historical Set 1 results.",
             "", f"Source: `{source['repository']}` revision `{source['revision']}`. Raw reports and review notes are not redistributed.",
             "", f"Private workload manifest: `{manifest['fingerprint']}`.", "", "## Held-out event classification", "",
             "Macro-F1 is the unweighted mean over the fixed contract taxonomy, including zero F1 for absent classes. The aggregate is an equal-weight mean over events.",
             "", "| Event | Reports | Baseline MiniLM LR | Selected model | Delta |", "| --- | ---: | ---: | ---: | ---: |"]
    for event, row in quality["reserved"]["per_event"].items():
        lines.append(f"| {event} | {row['n']} | {row['baseline_macro_f1']:.4f} | {row['selected_macro_f1']:.4f} | {row['delta']:+.4f} |")
    lines.extend(["", f"Mean event macro-F1: {bf['mean_event_macro_f1']:.4f} → {sf['mean_event_macro_f1']:.4f}; delta {comparison['macro_f1_delta']:+.4f}.",
                  f"Paired event-bootstrap delta interval: [{interval[0]:+.4f}, {interval[1]:+.4f}]. {quality['reserved']['uncertainty_method']}.",
                  "", f"Selected configuration: {sf['configuration']['kind']} logistic regression, C={sf['configuration']['C']}, class weight={sf['configuration']['class_weight']}, seed={sf['configuration']['seed']}. Full settings are in RESEARCH_EVIDENCE.json. Selection used development data before reserved results were opened.",
                  "", "## Per-category held-out tradeoffs", "", "| Category | Support | Baseline F1 | Selected F1 | Baseline recall | Selected recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for category, values in quality["reserved"]["per_class"].items():
        a, b = values["baseline"], values["selected"]
        ar, br = (f"{v['recall']:.4f}" if v["recall"] is not None else "unavailable" for v in (a, b))
        lines.append(f"| {category} | {a['support']} | {a['f1']:.4f} | {b['f1']:.4f} | {ar} | {br} |")
    lines.extend(["", "Confusion matrices: [baseline](research_figures/baseline_confusion.png), [selected](research_figures/selected_confusion.png).",
                  "", "## Development experiment history", "", "| C | Class weight | Status | Mean event macro-F1 |", "| ---: | --- | --- | ---: |"])
    for experiment in experiments:
        score = experiment.get("mean_event_macro_f1")
        lines.append(f"| {experiment['C']} | {experiment['class_weight'] or 'none'} | {experiment['status']} | {score if score is not None else 'unavailable'} |")
    lines.extend(["", "The final configuration was selected from development results. The reserved results were not used to choose additional configurations."])
    lines.extend(["", "## Runtime and memory", "", "These are laptop measurements, not LinuxONE results. Process RSS is sampled and excludes children. OS caches were not cleared.",
                  "", f"Baseline final processing-through-prediction seconds: {bf['timing']['total_processing_seconds']:.3f}; selected: {sf['timing']['total_processing_seconds']:.3f}.",
                  f"Sampled peak RSS bytes: baseline {bf['timing']['peak_rss_bytes']}; selected {sf['timing']['peak_rss_bytes']}.",
                  "The classifier comparison includes each representation's own feature costs; grouping still requires embeddings when using TF-IDF classification.",
                  "", "| Profile batch | Cold encoding median seconds | Warm artifact median seconds |", "| ---: | ---: | ---: |"])
    for batch in config.RESEARCH_BATCHES:
        cold = [r["embedding_seconds"] for r in perf["rows"] if r["batch_size"] == batch and not r["cache_hit"]]
        warm = [r["seconds"] for r in perf["rows"] if r["batch_size"] == batch and r["cache_hit"]]
        lines.append(f"| {batch} | {statistics.median(cold):.4f} | {statistics.median(warm):.4f} |")
    baseline_profile = perf["baseline"]
    if baseline_profile["status"] == "complete":
        lines.extend(["", f"Preserved `{baseline_profile['commit']}` baseline encode median seconds on the same fixed sample: {statistics.median(r['seconds'] for r in baseline_profile['rows']):.4f}.",
                      "Compare encode-only time to encode-only time; cache-hit measurements include artifact verification. Batch results use a fixed development sample, not the full event workload."])
    lines.extend(["", "## Review provenance and limitations", "", f"Classification review summary: `{reviews['classification']}`.",
                  f"Location review summary: `{reviews['location']}`.", f"Location policy comparison: `{location_changes}`.",
                  "", "Reviews are agent-assisted diagnostics, not independent human annotations or emergency-response expert validation. County centroids remain approximate; boundary lookup is not implemented.",
                  "Human usability sessions, LinuxONE validation, and a human screen recording are not established by the automated synthetic walkthrough.",
                  "Category labels and assistance-category recall are not validated operational urgency. No dispatch or fairness outcome claim follows from these results.",
                  "", "See RESEARCH_EVIDENCE.json for aggregate machine-readable evidence and RESEARCH_SPLIT_MANIFEST.json for workload reconstruction metadata."])
    fileio._atomic_write_text(config.RESEARCH_PUBLIC_RESULTS, "\n".join(lines) + "\n")
    _write_case_study(evidence)
    return str(config.RESEARCH_PUBLIC_RESULTS)


def _write_case_study(evidence):
    quality = evidence["quality"]["reserved"]
    count = sum(r["n"] for r in quality["per_event"].values())
    selected = evidence["selected_configuration"]
    missing = quality["per_class"]["missing_or_found_people"]
    requests = quality["per_class"]["requests_or_urgent_needs"]
    text = ["# SurgeDesk research case study", "", "## Problem and inherited system", "",
            "The original multi-author prototype already tagged, grouped, ranked, and briefed disaster reports with local models. Its historical demonstration did not establish multi-event generalization, and shape-only embedding reuse could accept stale text. Original contributors retain credit for that implementation.",
            "", "## Work on the solo improvement branch", "",
            "Added split-preserving source manifests, leakage exclusions, isolated experiment runs, train-only TF-IDF fitting, model selection gates, held-out evaluation, and computed reports. Added conservative location evidence, content-based cache validation, completion receipts, recovery checks, offline CI, and a synthetic evidence/decision demonstration.",
            "", "## Measured outcome", "",
            f"On {len(quality['per_event'])} reserved events containing {count:,} reports, mean event macro-F1 increased from {quality['baseline']:.4f} to {quality['selected']:.4f} ({quality['delta'] * 100:+.2f} percentage points). The selected representation was {selected['kind']} with C={selected['C']} and class weighting={selected['class_weight']}.",
            f"Urgent-request recall changed from {requests['baseline']['recall']:.4f} to {requests['selected']['recall']:.4f}; missing-person recall regressed from {missing['baseline']['recall']:.4f} to {missing['selected']['recall']:.4f}. Macro-F1 improvement therefore does not imply every assistance-related outcome improved.",
            "", "## Tradeoffs and limits", "",
            "TF-IDF reduced category-classification feature costs in this benchmark, but incident grouping still requires embeddings. Cache timings are distinct from cold inference. Batch profiling uses a fixed development sample; laptop measurements do not establish LinuxONE performance. All results use humanitarian categories rather than validated dispatch urgency.",
            "Agent-assisted development audits are explicitly labeled and are not independent human annotations. County centroids remain approximate. The automated walkthrough records fixture API/terminal actions, not human usability or a screen recording. See RESEARCH_RESULTS.md and RESEARCH_EVIDENCE.json for full metrics and measurement boundaries.",
            "", "## Contribution evidence and resume wording", "",
            "Use `git log a4-integration..surgedesk-month-improvements` to inspect the actual commits and authors. Link authored changes when publishing; no hosted PR or independent usability result is claimed here.",
            "", "Suggested wording, contingent on accurately representing personal ownership:", "",
            f"- Built a local, reproducible disaster-report classification benchmark across {len(quality['per_event'])} held-out events ({count:,} reports); selected TF-IDF/logistic regression over a MiniLM baseline, improving mean event macro-F1 from {quality['baseline']:.3f} to {quality['selected']:.3f} while documenting category-level regressions.",
            "- Implemented content-addressed embedding validation, whole-stage recovery, cross-endian array checks, and an offline regression workflow; packaged a synthetic evidence-review demo with traceable decision logs.",
            "", "These bullets cover the improvement branch, not sole authorship of the original system. Do not add emergency-response adoption, expert validation, LinuxONE speedups, or participant counts without new evidence."]
    fileio._atomic_write_text(config.RESEARCH_CASE_STUDY, "\n".join(text) + "\n")
    backlog = ["# Evidence-based research backlog", "", "Computed from development review summaries. Counts are agent-assisted issue categories, not independent error-rate estimates.",
               "", "| Review stream | Observed issue | Reviewed count |", "| --- | --- | ---: |"]
    for stream, summary in evidence["reviews"].items():
        if summary:
            backlog.extend(f"| {stream} | {r['category']} | {r['count']} |" for r in summary["prioritized_errors"])
    backlog.extend(["", "Priority order for a future research cycle:", "",
                    "1. Examine overlapping labels, off-topic lexical triggers, and the observed missing-person recall regression. Keep original gold labels separate from review judgments.",
                    "2. Independently audit conservative location behavior, especially unqualified real towns and county-centroid approximations. Boundary lookup remains conditional on suitable local data and existing dependencies.",
                    "3. Validate batch/cache measurements on the authorized LinuxONE host and a representative full workload before promoting a throughput claim.",
                    "4. Conduct consenting human usability sessions and record an actual UI walkthrough when the browser surface is available.",
                    "", "The reserved results are now inspected. Do not tune against them and continue calling them untouched; a further model-selection cycle needs a new reserved benchmark."])
    fileio._atomic_write_text(config.RESEARCH_BACKLOG, "\n".join(backlog) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-dev", "selected-dev", "baseline-final", "selected-final", "selection", "profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("classification-review", "location-review", "proposed-locations", "sweep"):
        parser.add_argument("--" + name, type=Path)
    args = vars(parser.parse_args())
    args["manifest"] = fileio.read_json(args["manifest"])
    print(publish(**args))


if __name__ == "__main__":
    main()

"""Small explicit development-only regularization/class-weight experiments."""
from . import config, fileio, research_data
from .provenance import capture
from .textutil import mask_privacy


def run(manifest, run_id, kind):
    from .research import _new_run, train, evaluate
    research_data.validate_manifest(manifest)
    directory = _new_run(run_id)
    journal = {"kind": kind, "partition": "dev", "manifest_fingerprint": manifest["fingerprint"],
               "provenance": capture(), "attempts": [], "status": "running"}
    for c in config.RESEARCH_C_VALUES:
        for weight in config.RESEARCH_CLASS_WEIGHTS:
            name = run_id + "-c" + str(c).replace(".", "p") + "-" + (weight or "none")
            attempt = {"C": c, "class_weight": weight, "status": "running"}
            journal["attempts"].append(attempt)
            fileio.write_json(directory / config.RESEARCH_SWEEP_FILE, journal)
            print(f"Experiment: {kind}, C={c}, class_weight={weight}", flush=True)
            try:
                model = train(manifest, name, kind, c, weight)
                result = evaluate(manifest, model, name + "-dev")
                values = fileio.read_json(result / config.RESEARCH_RESULT_FILE)
                attempt.update(status="complete", model=str(model), evaluation=str(result),
                               mean_event_macro_f1=values["mean_event_macro_f1"],
                               assistance_recall=values["pooled"]["assistance_recall"], timing=values["timing"])
            except (OSError, ValueError, ImportError) as exc:
                attempt.update(status="failed", reason=mask_privacy(str(exc)))
            fileio.write_json(directory / config.RESEARCH_SWEEP_FILE, journal)
    completed = [a for a in journal["attempts"] if a["status"] == "complete"]
    journal["status"] = "complete" if len(completed) == len(journal["attempts"]) else "partial" if completed else "failed"
    fileio.write_json(directory / config.RESEARCH_SWEEP_FILE, journal)
    if not completed:
        raise ValueError("Every development experiment failed; inspect sweep.json")
    return str(directory)

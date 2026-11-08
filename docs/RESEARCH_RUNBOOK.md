# Solo research workflow

The month plan is [MONTH_PLAN.md](../MONTH_PLAN.md). Work stays on
`surgedesk-month-improvements`; `a4-integration` at `6d20725` is the historical
baseline. Existing contributors retain credit for the original implementation.
New experiment outputs are isolated under the gitignored `outputs/research/`.
Do not merge or tag a complete research release while required evaluation is pending.

## Local setup and synthetic verification

Use an existing compatible Python environment or a laptop virtual environment.
Python 3.12 mirrors the original container target; the current development
machine can also exercise fixtures on its installed Python. Exact resolved
versions and hardware are captured per experiment. The fixture dependency file
pins the directly validated laptop packages and notebook tooling; transitive
versions are captured per run. It uses the project's existing allowance; no new ML
framework is required. Do not install packages on LinuxONE.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-research.txt
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python -m surgedesk.selftest
python -m surgedesk.research smoke --run-id fixture-check
python -m surgedesk.research_notebooks
```

On Linux/macOS activate with `source .venv/bin/activate` and set the three thread
variables with `export`. Set them before starting Python, especially if another
module has already initialized BLAS. Project settings use `config.N_THREADS`;
compare runs only when their effective hardware, versions, thread environment,
and workload agree. No models or datasets are downloaded by these commands.
Dependency installation requires network access; fixture execution does not.

Run IDs are unique, safe directory names. Existing runs are never overwritten.
The smoke command creates a frozen synthetic manifest, TF-IDF classifier,
two evaluations, reproducibility comparison, masked review samples, and a
standalone `demo.html`. These are fixture checks, not model-quality claims.
For a second smoke run, choose another ID.

Open `notebooks/06_research_demo.ipynb` for Accept/Override/Dismiss logging.
Every execution creates a new isolated synthetic demo. Open
`notebooks/07_research_results.ipynb` for read-only saved results. Both notebooks
use the existing environment bootstrap and require Restart Kernel / Run All.
The notebook check command executes both with the current interpreter, writes
executed copies under `outputs/notebook_checks/`, and confines Jupyter caches
and kernel registration to that output directory.

## Preserve data provenance and freeze partitions

Obtain HumAID and compatible MiniLM files separately under their source terms.
For laptop model experiments, `requirements-research-models.txt` adds the
validated local-model package versions. Use the appropriate CPU PyTorch build
for your platform; IBM's preinstalled environment remains unchanged.
Runtime code makes no network calls. Use the documented `data/raw/humaid/` and
`models/minilm/` layout, or set `SURGEDESK_ROOT` before importing SurgeDesk.
`--data` can point the research manifest at another local HumAID directory.

The user approved the [QCRI author event mirror](https://huggingface.co/datasets/QCRI/HumAID-events/blob/main/README.md)
for the new benchmark. It is not asserted identical to the old Set 1 archive;
historical results remain separate. Explicit preparation commands are:

```text
python scripts/prepare_research_assets.py
python scripts/prepare_research_assets.py --gazetteer-only
```

Preparation is the only network step. Repository revisions are pinned in config.
Raw source JSON, converted TSVs, and per-file checksums remain local and
gitignored. Conversion preserves report IDs and labels and replaces embedded
tabs/newlines with spaces for the existing unquoted TSV reader. Existing assets
use canonical CRLF line endings so the same TSV checksums can be reproduced on
Windows and Linux. Existing assets
that differ from the pinned source are preserved and cause an error. Model
files are fetched from the pinned MiniLM repository without inference API calls.
The Gazetteer comes from the configured Census national archive year. Consult
the respective source terms; this project does not redistribute raw reports.

```text
python -m surgedesk.research inventory --data data/raw/humaid
python -m surgedesk.research manifest --data data/raw/humaid --train train_event_name --dev development_event_name --test unseen_event_name --historical hurricane_harvey_2017 --rationale "Explain disaster-type/region coverage and previous exposure" --output outputs/research/splits-draft.json
python -m surgedesk.research review-duplicates --manifest outputs/research/splits-draft.json --output outputs/research/duplicate-review.json
```

Replace event names with actual inventory results. Events are disjoint at the
event level; original train/dev/test membership is retained within each event.
An event assigned to training may use its original splits because the benchmark
is explicitly event-held-out, not a claim about official within-event splits.
Never assign `config.DEMO_EVENT` to training or untouched evaluation. Harvey has
already been inspected and belongs in development or historical reporting.

The draft preserves per-source checksums, original split, report ID, row keys,
invalid/empty/repeated-ID counts, and an exclusion ledger. Exact duplicate IDs
or normalized masked text keep a reserved/dev row before a training row.
Near-duplicate detection uses token-set Jaccard and a prefix index; it cannot
detect every semantic paraphrase. Review masked pairs, then edit the draft
manifest's `near_duplicates` entries to `distinct`, `exclude_left`, or
`exclude_right`, with a substantive `note`. Do not invent review decisions.

```text
python -m surgedesk.research freeze --manifest outputs/research/splits-draft.json --output outputs/research/splits-frozen.json
```

Freezing rejects unresolved candidates. Source changes or edits to frozen
partition metadata invalidate the manifest. Its local `root` may be relocated
without changing identity when all relative filenames and source checksums
remain identical. Absent unseen events are recorded as a limitation.

## Train, evaluate, repeat, and review

```text
python -m surgedesk.research train --manifest outputs/research/splits-frozen.json --kind minilm --run-id minilm-baseline
python -m surgedesk.research train --manifest outputs/research/splits-frozen.json --kind tfidf --run-id tfidf-baseline
python -m surgedesk.research evaluate --manifest outputs/research/splits-frozen.json --model outputs/research/minilm-baseline --partition dev --run-id minilm-dev-cold --cache-mode cold
python -m surgedesk.research evaluate --manifest outputs/research/splits-frozen.json --model outputs/research/minilm-baseline --partition dev --run-id minilm-dev-warm --cache-mode warm
python -m surgedesk.research compare outputs/research/minilm-dev-cold outputs/research/minilm-dev-warm
python -m surgedesk.research evaluate --manifest outputs/research/splits-frozen.json --model outputs/research/tfidf-baseline --partition dev --run-id tfidf-dev
python -m surgedesk.research review --manifest outputs/research/splits-frozen.json --evaluation outputs/research/tfidf-dev --output outputs/research/classification-review.json
python -m surgedesk.research review-locations --manifest outputs/research/splits-frozen.json --output outputs/research/location-review.json
python -m surgedesk.research review-summary outputs/research/classification-review.json
```

Training fits TF-IDF only on training text; MiniLM uses local pretrained weights.
Both classifiers are logistic regression. Text passed to either representation
is privacy-masked. Gold labels remain targets, never features. With development
evidence, compare `--C 0.1`, `--C 1`, and `--C 10`, and `--class-weight none` versus
`balanced`. Keep each attempt in a distinct run. Do not claim an unexecuted grid
or unsupported preprocessing experiment. Missing weights fail explicitly.
`python -m surgedesk.research sweep --manifest outputs/research/splits-frozen.json --kind minilm --run-id minilm-tuning`
runs the configured regularization/class-weight grid on development data only,
recording completed and failed attempts. Choose the representation using the
initial development comparison before running its grid.

Each evaluation exports predictions, fixed-taxonomy confusion matrices,
per-category precision/recall/F1/support, per-event and pooled metrics,
configuration, model checksum, commit plus dirty-source hashes, versions,
timing, and sampled process RSS. Absent gold-category recall is null. Macro-F1
uses every contract category, with zero F1 for absent categories; this differs
from some historical metrics using only observed categories. Compare like with
like. Assistance-category recall is category agreement, not dispatch urgency.

Sampling records its seed and source keys. Review artifacts remain private.
Follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). `review-summary` counts only
rows with a reviewer and a recognized outcome; unreviewed rows are not evidence.

## Performance, integration, and recovery

```text
python -m surgedesk.research profile --manifest outputs/research/splits-frozen.json --run-id embedding-profile
```

Profiling uses a seeded fixed development sample capped by
`config.RESEARCH_PROFILE_LIMIT`, saving its row keys. It compares configured batch sizes and repeated application-cache cold
and warm runs, writes incremental measurements, and exports cumulative
`cProfile` statistics. A cold run bypasses the embedding artifact cache but does
not clear OS caches. RSS is a sampled process peak, excludes children, and may
reflect allocator reuse; use fresh processes for stronger memory comparisons.
Missing `psutil` remains unavailable, not zero. Model load, encoding, prediction,
and processing-through-prediction timings have explicit boundaries.
The preserved baseline embedding module is read from the recorded Git commit
and run on the same sample, weights, threads, and default batch. Compare its
encode-only seconds with current `embedding_seconds`; do not compare encode-only
time with cache verification plus model-loading time. The sampled profile is
separate from full-event classifier evaluation throughput.

Embedding caches fingerprint ordered IDs and masked text, model/tokenizer
contents, and relevant configuration/code. Completion metadata and checksums
are published after atomic data writes. Corruption, missing manifests, reordered
IDs, or relevant changes prevent reuse. Whole-stage pipeline receipts resume
validated stages; interrupted stages are recomputed. Partial-batch resume is not
implemented. Big-endian float32 embedding storage is preserved.

The historical `pipeline`/shell runners retain their signatures and historical
merged-event training behavior. Use the research runner for new event-held-out
classification evidence. Do not run the legacy training runner against a
reserved research benchmark and call those events untouched. Newly produced
managed embeddings are validated before tagging; imported legacy arrays without
manifests remain compatibility inputs, not proof of freshness.

Research classifiers can be used by Stage 1 without replacing the historical
model: set `SURGEDESK_CLASSIFIER` to their `classifier.joblib` before starting
Python. TF-IDF changes category inference; grouping still needs MiniLM embeddings.
Location sidecars distinguish approximate centroid assignments, explicit county
names, ambiguity, and unknown. `SURGEDESK_LOCATION_POLICY=conservative` opts into
withholding conflicting county assignments; legacy behavior remains the default
until the location audit justifies a change. No county polygons are bundled.

## Freeze final selection and evaluate once

```text
python -m surgedesk.research select --manifest outputs/research/splits-frozen.json --baseline outputs/research/minilm-baseline --candidates outputs/research/minilm-dev-cold outputs/research/tfidf-dev --output outputs/research/final-selection.json
python -m surgedesk.research evaluate --manifest outputs/research/splits-frozen.json --model outputs/research/minilm-baseline --partition test --selection outputs/research/final-selection.json --run-id baseline-final
```

Evaluate the selected model with the same selection file and reserved events in
a second new run. Only baseline/selected model hashes are authorized. Selection
uses highest development mean event macro-F1, breaking exact ties by prediction
plus representation time excluding encoder loading. All candidate runs must use identical row keys and gold labels. The first
reserved evaluation records the selection checksum and prevents selecting again
on that manifest. These local safeguards document the workflow; they cannot
prove a human never viewed the raw data independently.

Report per-event metrics, pooled category results, equal-weight mean event
macro-F1, sample sizes, regressions, and unavailable uncertainty intervals.
Keep LinuxONE measurements distinct. Do not claim an improvement from synthetic
fixtures or proxy ranking targets. Use the usability task sheet and release
checklist in [MONTH_STATUS.md](MONTH_STATUS.md); public release must exclude
restricted inputs and private review files.

## Reproduce the recorded benchmark and inspect the release artifacts

The computed comparison is in [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md), with
aggregate JSON in `RESEARCH_EVIDENCE.json` and a text-free public split
specification in `RESEARCH_SPLIT_MANIFEST.json`. Reconstruct the same row
workloads after preparing the pinned source:

```text
python -m surgedesk.research reconstruct-manifest --specification docs/RESEARCH_SPLIT_MANIFEST.json --data data/raw/humaid --output outputs/research/reproduced-splits.json
```

Reconstruction verifies source checksums, exclusion keys, and ordered workload
fingerprints. Its private manifest fingerprint can differ because private review
notes are not redistributed. Use the fixed configurations in the evidence JSON
when reproducing the results. These reserved results are now inspected and
published in the local report; a reproduction is not a new untouched study.

For a local recorded demonstration:

```text
python -m surgedesk.research walkthrough --run-id my-walkthrough
```

Open the resulting `walkthrough.html` or use its Asciinema-compatible
`walkthrough.cast`. This records actual automated fixture API/terminal actions,
including validated review decisions and evidence inspection. It is explicitly
not a human usability session or screen recording. The synthetic cards are
scripted fixture scenarios, not trained-model predictions. Browser visual
review remains pending when the computer-use tool cannot verify the local URL.

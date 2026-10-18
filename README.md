# SurgeDesk

SurgeDesk is a research prototype that turns historical disaster reports into a ranked list of incidents for a coordinator to inspect. Local models tag reports, group related messages, and rank incidents. An experimental IBM Granite stage attempts evidence-based extraction and briefings, with explicitly labeled template fallbacks when generation or checks fail. A human accepts, overrides, or dismisses each recommendation; SurgeDesk never dispatches help.

**IBM Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions.**

## Architecture

| Stage | Job | Implementation owner |
| --- | --- | --- |
| Tag | Local MiniLM embeddings in PyTorch, a scikit-learn humanitarian category classifier, text place matching, and vulnerable-person terms | Agent 1 |
| Group | Online similarity, place, and time grouping into incidents | Agent 2 |
| Rank | XGBoost assistance-category model, explicit current-need priority bands, feature contributions, and SVI adjustment within bands | Agent 2 |
| Brief | Locally converted IBM Granite extracts quoted fields and drafts sentences; grounding checks reject unsupported output and provide a template fallback | Agent 3 |
| Review | Historical replay, evidence cards, fairness slider, and an append-only validated human decision log | Agent 4 |

Granite receives representative report text for extraction and briefing; it does not calculate or change the incident ranking. The interface separately explains the ranking model and priority policy. Gold labels are evaluation/training targets only, and the configured demo event is excluded from training. Runtime model and data access is local; preparation scripts perform the downloads.

## Running on IBM LinuxONE

The supplied Dockerfile extends IBM's s390x JupyterLab container. Person 3 uses `scripts/convert_granite.sh` to convert the original IBM Granite weights ourselves into big-endian GGUF for LinuxONE, with a separate little-endian file for laptops. Model files are mounted at runtime. Agent 3 alone owns model loading, using one Granite instance per process.

IBM's environment provides zDNN/NNPA libraries. Hardware capability and the presence of a library do not establish an accelerated execution path: use the environment report, Granite load logs, and `benchmark.py` outputs as evidence. Device comparisons are explicitly skipped unless Agent 1 supplies `embed.compare_devices`.

**Validated workflow:** The real-data embedding, classifier, tagging, assistance-ranking, and evaluation stages have completed on LinuxONE. Historical replay, dashboard interaction, and decision logging have been checked by the human operator. The latest bounded Granite review produced no grounded fields and used templates throughout; two extraction calls failed JSON parsing after their bounded retries. This is a functioning prototype, not a validated emergency-prioritization system. See the current generated [LinuxONE results](docs/RESULTS.md). Formal current runtime/acceleration comparisons have not been established.

## Quick start: synthetic fixture demo

This route runs without real HumAID data, MiniLM weights, or a Granite model. The fixtures are synthetic; their scores and timings are not production results. Run commands from the repository root.

```bash
git clone --branch a4-integration https://github.com/saikarthikbattula/surge-desk.git
cd surge-desk
```

On a development laptop, use a Python virtual environment. Python 3.12 is recommended to mirror the validated container. The development requirements include the optional Granite backend, which may require a platform C/C++ build toolchain. A big-endian LinuxONE GGUF must not be loaded on a little-endian laptop; use a compatible LE model or the fixture/fake path.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
bash run_all.sh --stub
jupyter lab
```

`run_all.sh --stub` runs the environment check, self-test, and synthetic pipeline. Optional model-dependent tests may skip when local model files are absent. For the self-test alone, use `python -m surgedesk.selftest`.

Open `notebooks/04_demo.ipynb`, set `STUB = True`, then choose **Restart Kernel and Run All Cells**. It shows synthetic cards and skips real-data replay. If interactive widgets are unavailable, the notebook also renders static HTML cards. Test decisions append to `outputs/_stub/fixture_event/decisions.jsonl`.

## Run an existing LinuxONE installation

The validated deployment used IBM's running `notebook` container with its preinstalled packages. It did not require installing Python packages on the VM or replacing the container. The Dockerfile is available as an optional deployment definition; the observed workflow is the existing IBM container, not a claimed new custom-image validation.

From the VM host terminal, enter the existing container:

```bash
docker exec -it notebook bash
```

Inside that container:

```bash
source /opt/venv/bin/activate
cd /home/ibm-user/shared/surgedesk
python -m surgedesk.envcheck
```

These are the paths used in the validated environment. If your mount differs, use your actual repository location and set `SURGEDESK_ROOT` before importing SurgeDesk. The older planning runbook uses `/home/jovyan/...`; that is not the mount used by this installation.

If datasets, saved models, and `outputs/hurricane_harvey_2017/` already exist, open `notebooks/04_demo.ipynb` in the existing JupyterLab session, set `STUB = False`, and Restart Kernel / Run All. The notebook runs historical replay, displays stored ranking cards, and offers Accept / Override / Dismiss with notes. Lambda reorders stored scores without loading Granite. Template badges and unknown fields reflect unavailable or rejected model output.

Notebook 05 reads saved evaluation and benchmark artifacts. It performs no training, model inference, or benchmark runs. Missing results remain explicitly unavailable. Open `notebooks/05_eval_benchmark.ipynb` and Restart Kernel / Run All to inspect the current files. Notebook 03 is an optional, separate, potentially slow Granite review; it is not required to display an existing dashboard.

## Real-data setup and pipeline reproduction

**A fresh clone contains code and synthetic fixtures, not the real data, generated outputs, or model weights.** Obtain local data/model artifacts separately under their source terms. The expected layout is:

```text
data/raw/humaid/                 # Event train/dev/test TSVs; nested directories are supported
data/raw/gazetteer/              # Census national place and county Gazetteer text files
data/raw/svi/svi_counties.csv     # County SVI values
models/minilm/                  # Local tokenizer/config and model.safetensors
models/granite/                 # Optional compatible Granite GGUF
models/classifier.joblib        # Produced by the classifier stage
models/ranker.json              # Produced by the ranker stage
models/ranker_meta.json          # Produced beside the ranker
outputs/hurricane_harvey_2017/   # Runtime artifacts, created by the pipeline
data/labels/                    # Optional private reference-label CSVs
```

The preparation scripts `scripts/download_data.sh` and `scripts/download_minilm.py` retrieve HumAID/Gazetteer and MiniLM respectively; county SVI is supplied separately as described by the data script. Preparation needs network access. Project inference code uses local files only. A supplied private data pack can populate the same paths without downloading again. Preserve its checksum manifest if available.

Use the existing Python environment in IBM's container; do not run laptop dependency installation commands there. Confirm paths with `python -m surgedesk.envcheck` before running stages.

The verified model artifact for LinuxONE is:

```text
granite-3.3-2b-instruct-be-fixed.q8_0.gguf
SHA-256: acc4c60e677f954b804ad95cb3c721061a985d199f4f2b86679f6932ad9c759d
```

For an existing copy, select it and check its hash:

```bash
export SURGEDESK_GRANITE="$PWD/models/granite/granite-3.3-2b-instruct-be-fixed.q8_0.gguf"
sha256sum "$SURGEDESK_GRANITE"
export SURGEDESK_TOP_N=3
export SURGEDESK_SNAPSHOT=2017-08-29T22:30:00Z
```

The original conversion helper is `scripts/convert_granite.sh`. Its original-named BE artifact was not the working artifact used by the latest run; do not assume an arbitrary conversion reproduces the verified fixed file. Obtain that verified artifact or validate a compatible conversion separately. Large models are not distributed through this repository.

To reproduce the non-Granite pipeline, run these commands **sequentially**, allowing each to finish:

```bash
python -m surgedesk.pipeline --stage data
python -m surgedesk.pipeline --stage embed
python -m surgedesk.pipeline --stage classifier
python -m surgedesk.pipeline --stage tag
python -m surgedesk.pipeline --stage ranker
python -m surgedesk.pipeline --stage rank
SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval
```

Embedding can take substantial time on the small CPU allocation. Do not restart completed stages unnecessarily: embeddings, tags, ranked outputs, and briefings have caches. Classifier and ranker stages retrain when invoked. Back up runtime artifacts before a forced rebuild. Changes to masked embedding inputs require the full refresh described below; changed ranking feature semantics require ranker retraining even when the feature order is unchanged.

The evaluation command above uses the default human reference-label mode; absent labels/predictions remain unavailable. To reproduce the recorded provisional AI-reference mode, provide the private draft label files and use `SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval`. That command does not launch Granite.

An optional bounded CLI briefing run is:

```bash
SURGEDESK_TOP_N=3 python -m surgedesk.pipeline --stage brief
```

`bash run_all.sh` is the full runner, including the brief stage; it is not a lightweight way to reopen the dashboard. `FORCE=1 bash run_all.sh` forces the stages and should not be used for routine viewing. `bash run_all.sh --from rank` resumes the runner at ranking and continues through briefing/evaluation.

Formal benchmarks are separate: `bash run_benchmark.sh`, on an idle VM with other kernels shut down. The runner sets the benchmark thread count and invokes uncached Granite measurements, so it can be slow. Do not use notebook probe timings or event-time surfacing statistics as formal compute-latency measurements. Missing benchmark files are not evidence of measured acceleration or throughput.

Terminal environment exports do not update an already running Jupyter server or kernel. Set required overrides in the notebook's first controls cell, before importing `surgedesk.config`, and restart the kernel. Verify the selected model, event, and snapshot before deliberately generating new briefings.

Notebook 03 caps the review at three incidents even when `SURGEDESK_TOP_N` is larger. It defaults to `SURGEDESK_TOP_N=3`, `RUN_SPEED_PROBE=False`, `RUN_EXTRACTION_EVAL=False`, and cached briefings reused. A smaller configured count is respected. It loads Granite only for an uncached call and uses the fixed BE model if present and no model override was supplied. With no grounded extraction fields, the pipeline skips sentence generation and uses the labeled template. Set `FORCE_BRIEFINGS=True` only to test new prompts deliberately. For the same smaller CLI workload use `SURGEDESK_TOP_N=3 python -m surgedesk.pipeline --stage brief`; the full CLI default in the frozen config remains unchanged. Notebook probes are development checks, not formal benchmark results. Larger briefing batches must be requested through the CLI; the notebook review stays capped. Extraction-evaluation batches remain an explicit optional choice.

Cached review, dashboard display, and evaluation apply the current evidence checks without model calls. Historical references (relative to the configured demo event year), speculative counts, and unspecified rising tolls cannot establish a current affected-person count. A quoted responder or family reference may still identify who was affected. Locations remain unknown when the reports do not establish a current place; the event name is not a location substitute. List numbering is removed before sentence checking, and comma-separated counts are checked as complete numbers. These rules are conservative filters, not independent verification of the reports. After updating these checks, run notebook 03 with `FORCE_BRIEFINGS=False` to review saved results first; only regenerate deliberately to test the new prompts.

Evaluation also rechecks stored single-report extraction evidence against current source text, retaining the original generation timings. `evaluation_provenance.json` records content hashes of the evaluated inputs, labels, summary, and checking code; notebook 05 warns when saved results are stale or lack provenance. After new briefings or labels arrive, refresh with `SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval` for the provisional AI review. Do this after generation finishes. No Granite calls are needed. Formal Granite benchmark summaries identify the model file, thread/context/token settings, and uncached timing convention: completion tokens divided by total call time includes prompt processing, while model load is recorded separately. Cached notebook timings are not new benchmark measurements.

The same evaluation run exports `incident_review.csv` and `incident_review.json` beside the summary. These diagnose the unchanged top ranking at the demo snapshot: masked member reports, classifier predictions, scores/reasons, checked briefing evidence, and rejection reasons. `representative` marks reports that the stored briefing used; `not_briefed` is distinct from a template fallback. The export does not rerank incidents, call Granite, or supply independent quality labels. Review these sources when broad casualty-news categories outrank concrete requests; proxy ranking scores alone do not establish current dispatch urgency.

The CLI matches the existing runners:

```bash
python -m surgedesk.pipeline --stage data
python -m surgedesk.pipeline --stage rank --event hurricane_harvey_2017 --force
python -m surgedesk.pipeline --stub
bash run_all.sh --from rank
```

Set `SURGEDESK_EVENT`, `SURGEDESK_ROOT`, `SURGEDESK_GRANITE`, and `SURGEDESK_TOP_N` as documented in `surgedesk/config.py`. `run_benchmark.sh` sets the benchmark thread count. No VM dependency installation or remote model inference is required by project code.

## Results and annotation

### Assistance ranking revision

Real events default to `assistance_v1`. The training/evaluation category proxy now includes `requests_or_urgent_needs` and `missing_or_found_people`; injury/death news alone is not positive. Three additional text-feature shares describe current-assistance/unresolved-danger cues, historical/speculative context, and casualty-only updates. These use only masked source text available at the snapshot, never gold labels or Granite. Current means at the report/replay time, not today. They are conservative heuristics, not verified current emergencies.

XGBoost is trained only on non-demo events. Its probability orders incidents within an explicit priority policy: current-assistance cues from the configured incident time window occupy the upper score band, and other context occupies the lower band. Old requests cannot keep a continuously updated incident in the assistance band indefinitely. All incidents remain visible. The score is a **priority score, not a calibrated probability of urgency**. Fairness adjusts within these bands, so an SVI boost cannot promote a casualty-only update over current-need cues. Feature explanations include their values: a positive SHAP contribution for a zero category share does not establish that activity occurred.

Existing fixture schemas and targets stay unchanged. `SURGEDESK_RANKING_OBJECTIVE=legacy` explicitly selects the old behavior for real events and requires a matching model; never mix objectives in one run. The shared contracts accept both complete schemas, and stale production rankings/models fail with a rebuild message. Before overwriting a legacy model, training preserves it in `models/legacy_ranker/`. Evaluation compares the same incidents and orderings against both category targets in `ranking_objective_comparison.csv`; `LegacySurgeDesk` also appears when the archived model is available. Its scores are recomputed on the current groups. Neither category target is independent dispatch-urgency ground truth. Compare methods within one target, not scores across changed targets.

For an existing installation, first back up the event output directory and model files, then run:

```bash
python -m surgedesk.pipeline --stage ranker
python -m surgedesk.pipeline --stage rank --force
SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval
```

Embeddings, classifier weights, and tagging can be reused for this ranking-only revision. Inspect the new `incident_review.csv` before another Granite run. New top incidents may be unbriefed; this is distinct from failed extraction. `rank_model_scores.jsonl` keeps base-model, priority-policy, and available legacy-model scores. `ranking_provenance.json` records the objective and input/model signature so changed inputs cannot silently reuse ranked output.

All reported measurements must come from [`evaluate.py`](surgedesk/evaluate.py) or [`benchmark.py`](surgedesk/benchmark.py). See the generated [results](docs/RESULTS.md) and the [written report](docs/REPORT.md). The report explains the measured workflow and its limitations; RESULTS.md remains the source of evaluation values. Development results are labelled **Development run, NOT LinuxONE**; unavailable measurements remain unavailable. Do not replace these with estimated claims.

After tagging, `make_label_sheet()` writes `data/labels/to_label.csv`; `make_purity_sheet()` writes `data/labels/purity_to_label.csv`. They preserve existing sheets unless explicitly forced. Humans save completed sheets as `hand_labels.csv` and `purity_labels.csv` in the same directory. Preserve `report_id` as plain text when exporting from a spreadsheet. Evaluation rejects scientific notation IDs, skips unfilled labels, and requests uncovered extraction predictions from Agent 3 when available.

For a provisional AI reference review, put `ai_extraction_labels_DRAFT.csv` and `ai_purity_labels_DRAFT.csv` in `data/labels/` and run `SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval`. This evaluates stored predictions only and never starts Granite. Extraction stays unavailable until real predictions exist. Results and notebook 05 identify AI reference labels explicitly; they are not independent human ground truth. Grouping excludes uncertain, blank and unreviewed rows and reports included/excluded counts. Reviewed all-null AI extraction rows are retained as negative examples. Keep `label_provenance.json` with `summary.json` when sharing results. To return to human evaluation, omit `SURGEDESK_LABEL_SOURCE` (or set it to `human`) and provide the human-reviewed files. Keep generation disabled for read-only evaluation; a deliberate extraction batch is a separate operation.

Ranking metrics average eligible snapshots; urgency is a proxy. Time to surface is measured from first report to first top-k appearance; never-surfaced incidents remain censored. Reports, samples, evidence, and review notes displayed by Agent 4 pass through the shared privacy masker. Outputs and labels are ignored by Git.

### Contact masking and existing artifacts

The shared masker redacts handles, URLs, and common North American phone formats, including optional country prefixes and extensions. It is not full anonymization: personal names, private street addresses, and unsupported international/obfuscated phone formats may remain. Review artifacts privately before sharing; Git ignore rules do not make an exported file anonymous.

Evaluation writes `privacy_audit.json`. Notebook 05 also scans current label sheets, cached briefings/predictions, decision/model logs, review exports, and saved notebook outputs without modifying them. Run the same read-only scan from the repo root:

```bash
python -c "from surgedesk.evaluate import audit_privacy_artifacts; import json; print(json.dumps(audit_privacy_artifacts(), indent=2))"
```

The audit returns artifact names and counts, never the matching private text. `no_known_contact_matches` means only that the supported contact patterns did not match; it is not permission to publish source data or private addresses. Raw reports are deliberately excluded from the audit because they remain restricted source data. Inference through a local model is not an upload, but its inputs still pass through masking.

Updating code does not scrub previously downloaded exports, labeled sheets, cached predictions, decisions, model logs, or notebook outputs. Keep those private. Back up human annotations before any deliberate edit; do not force-regenerate completed labels. Rerun evaluation to replace the review export, recheck cached briefings, and clear old notebook outputs before a fresh review. Recheck old predictions through evaluation and review any files flagged by the audit. No automatic scrub deletes or rewrites historical decisions or human work. New evaluations invalidate earlier provenance when the masker changes.

A masking change also changes MiniLM inputs and text deduplication. For consistent fresh offline/live measurements, back up the full artifact chain, then run these stages **one at a time** on LinuxONE as Person 1:

```bash
python -m surgedesk.pipeline --stage embed --force
python -m surgedesk.pipeline --stage classifier
python -m surgedesk.pipeline --stage tag --force
python -m surgedesk.pipeline --stage ranker
python -m surgedesk.pipeline --stage rank --force
SURGEDESK_SNAPSHOT=2017-08-29T22:30:00Z SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval
```

These commands reuse the raw reports and local model files; they do not redownload data or run Granite. The explicit review snapshot overrides a saved selection and must exist in the current incidents. Use it consistently for before/after review rather than choosing a favorable result. Notebook 05 retains the saved snapshot provenance when opened without an override, and detects an explicitly different override as stale.

Inspect the new masked review and both-target comparison before a bounded Granite run. Only after the revised cards and privacy checks are approved, use notebook 03's three-incident controls, optional speed/extraction batches off, and the verified fixed BE model. A deliberate prompt-quality refresh may require force regeneration for those three incidents; reset force afterward. Rerun evaluation after briefings change. Run the formal benchmark separately on an idle VM with current artifacts. Do not claim new quality or performance until Person 1 returns those measurements.

When updating place matching, back up existing outputs, labels, and the ranker before running `--stage tag --force`, `--stage ranker`, and `--stage rank --force` in that order. Embeddings and classifier weights can be reused for a place-only fix. Regrouping can reuse incident IDs with different members: the UI hides briefings with mismatched membership cache keys, and purity evaluation excludes groups whose labeled member sets no longer match, reporting `purity_rows_stale`. Review or regenerate purity sheets after regrouping. Cached briefings are rechecked against the current rejection rules without new model calls when the brief stage runs; prompt changes require `--stage brief --force` to generate new text. Use the fixed GGUF via `SURGEDESK_GRANITE` in both terminal and notebook environments. Recompute evaluation after regenerating downstream outputs; previous geographic/fairness results do not validate the changed implementation. Literal quote checks and conservative context rules do not establish full semantic correctness, and county assignment remains a nearest-centroid approximation.

## Data sources and licenses

**Raw data is not redistributed.** The committed fixtures are synthetic. Models, real reports, labels, and generated runtime outputs remain outside version control.

| Source | Attribution and terms |
| --- | --- |
| HumAID Set 1 | Alam, Qazi, Imran, and Ofli, *HumAID: Human-Annotated Disaster Incidents Data from Twitter*, ICWSM 2021. [Dataset](https://crisisnlp.qcri.org/humaid_dataset) and [QCRI terms](https://crisisnlp.qcri.org/terms-of-use.html): humanitarian research use, confidential contents, citation and deletion requirements. Follow the source's tweet-sharing terms. |
| CDC/ATSDR Social Vulnerability Index | [Official data and documentation](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html). US government data; retain source attribution and the downloaded release's documentation, year, and use terms. |
| Census Gazetteer | [Official Gazetteer files](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html). US government geographic reference data; retain Census attribution and the release metadata. |
| MiniLM | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0 per its model card. Used through local Transformers, with manual mean pooling. |
| IBM Granite | [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct), Apache-2.0 per its model card. Conversion is performed with our preparation script; original model attribution remains applicable. |

## Limitations

These limitations describe the current prototype. Measured values and reference-label provenance are recorded in [RESULTS.md](docs/RESULTS.md); planned annotations are not counted as completed validation.

**Data**
- HumAID is tweets from 2016 to 2019, not 911 calls or texts; English only.
- HumAID is a sampled, de-duplicated research set (Harvey: ~9K tweets over 8 days), so incidents are smaller and more fragmented than a live feed would produce. The replay compresses time (1x = 12x real time).
- Urgency is a proxy built from human humanitarian labels, not real dispatch priority or outcomes.
- Independent team human validation is incomplete. Draft AI extraction/grouping references are provisional and do not establish independent human ground truth. Grouping labels with stale memberships are excluded; current purity is unavailable.
- Only HumAID Set 1 is used.

**Location**
- Most tweets aren't geotagged; place matching from text misses reports and can confuse same-name places. Place-to-county uses the nearest county centroid, which is approximate for large cities.
- SVI is county-level and from one year; neighborhood vulnerability is invisible.

**Models**
- Classifier and ranker are trained on other disasters; performance shifts with disaster type and region.
- Assistance text cues can miss slang, negation, time context, and indirect needs. The explicit priority policy and changed category target require independent operational review; proxy-metric gains do not establish better dispatch decisions.
- Grouping thresholds were tuned by spot check, not a labeled benchmark; wrong merges and splits happen.
- The latest bounded Granite review produced no grounded fields and used templates for every reviewed incident. Saved call logs show parsing failures in part of that sample. This does not establish useful model-written briefings or overall extraction accuracy. The model's confidence values are self-reported, not calibrated.
- Grounding proves a claim traces to a report, not that the report is true. Misinformation passes through.

**Fairness**
- Without ground truth on who most needed help, we can't prove the adjusted ranking is fairer. We show a tradeoff.
- Social media under-represents people without phones, internet, or power.

**System**
- The observed environment is a small LinuxONE CPU allocation; Granite is batched for selected incidents and can be slow. Hardware flags and installed libraries do not establish a measured acceleration benefit.
- Not integrated with any dispatch system; not tested with real emergency staff; no authentication or high availability.


## Team and roles

**Six Eyes**

| Member | Role | Implementation owner |
| --- | --- | --- |
| Karthik (Person 1) | Integration, UI, evaluation, infrastructure, and LinuxONE runner | Agent 4 |
| Ishan (Person 2) | Data preparation, embedding, tagging, and location matching | Agent 1 |
| Yuvan (Person 3) | Granite conversion, extraction, grounding, and briefings | Agent 3 |
| Krish (Person 4) | Grouping, ranking, fairness, and project presentation | Agent 2 |

Built during the IBM Datathon 2026.

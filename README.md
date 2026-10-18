# SurgeDesk

SurgeDesk turns a stream of disaster reports into a ranked list of incidents with quoted evidence a coordinator can inspect. Local models tag reports, group related messages, rank incidents, and draft grounded briefings. A human accepts, overrides, or dismisses each recommendation; SurgeDesk never dispatches help.

**IBM Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions.**

## Architecture

| Stage | Job | Implementation owner |
| --- | --- | --- |
| Tag | Local MiniLM embeddings in PyTorch, a scikit-learn humanitarian category classifier, text place matching, and vulnerable-person terms | Agent 1 |
| Group | Online similarity, place, and time grouping into incidents | Agent 2 |
| Rank | XGBoost assistance-category model, explicit current-need priority bands, feature contributions, and SVI adjustment within bands | Agent 2 |
| Brief | Locally converted IBM Granite extracts quoted fields and drafts sentences; grounding checks reject unsupported output and provide a template fallback | Agent 3 |
| Review | Historical replay, evidence cards, fairness slider, and an append-only validated human decision log | Agent 4 |

Granite explains the stored ranking. Gold labels are evaluation/training targets only, and the configured demo event is excluded from training. Runtime model and data access is local; preparation scripts perform the downloads.

## Running on IBM LinuxONE

The supplied Dockerfile extends IBM's s390x JupyterLab container. Person 3 uses `scripts/convert_granite.sh` to convert the original IBM Granite weights ourselves into big-endian GGUF for LinuxONE, with a separate little-endian file for laptops. Model files are mounted at runtime. Agent 3 alone owns model loading, using one Granite instance per process.

IBM's environment provides zDNN/NNPA libraries. Hardware capability and the presence of a library do not establish an accelerated execution path: use the environment report, Granite load logs, and `benchmark.py` outputs as evidence. Device comparisons are explicitly skipped unless Agent 1 supplies `embed.compare_devices`.

**Current checkout:** All four agents' stages are integrated. Person 1 has exercised the real data/model workflow on LinuxONE, including the fixed BE Granite model and guarded fallback briefings. The revised assistance ranking needs a new LinuxONE training/ranking/evaluation run before its quality or performance can be claimed. Only Person 1 runs code on the VM.

## Quick start

Local fixture check, from the repository root:

```bash
python -m surgedesk.selftest
bash run_all.sh --stub
```

Use the Python interpreter with the dependencies from `requirements-dev.txt`. Open `notebooks/04_demo.ipynb`, set `STUB = True`, then Restart Kernel and Run All. Stub scores are synthetic and must never be used as model performance claims. Nonzero fairness adjustments require Agent 2's module; lambda-zero stored cards remain available before it arrives.

For deployment, Person 1 follows the complete [VM runbook](handoffs/person1_vm_runner.md). Preparation and execution commands, in their documented environments:

```bash
bash scripts/download_data.sh           # VM shell; also follow its SVI instructions
python scripts/download_minilm.py       # container terminal
bash scripts/convert_granite.sh          # Person 3's laptop; then transfer the big-endian model

docker build -t surgedesk:latest .       # VM shell
# Start the mounted container as described in the VM runbook.
bash run_all.sh                         # container terminal, repo root
bash run_benchmark.sh                   # container terminal, other kernels shut down
```

Then open `notebooks/04_demo.ipynb` with `STUB = False`. `notebooks/05_eval_benchmark.ipynb` reads saved evaluation and benchmark files and shows missing results without starting heavy work. Both notebooks begin with the frozen environment bootstrap.

Notebook 03 defaults to a short review: `SURGEDESK_TOP_N=3` unless explicitly set otherwise, `RUN_SPEED_PROBE=False`, `RUN_EXTRACTION_EVAL=False`, and cached briefings reused. It loads Granite only for an uncached call and uses the fixed BE model if present and no model override was supplied. With no grounded extraction fields, the pipeline skips sentence generation and uses the labeled template. Set `FORCE_BRIEFINGS=True` only to test new prompts deliberately. For the same smaller CLI workload use `SURGEDESK_TOP_N=3 python -m surgedesk.pipeline --stage brief`; the full CLI default in the frozen config remains unchanged. Notebook probes are development checks, not formal benchmark results. Larger incident counts and extraction-evaluation batches remain explicit choices.

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

All pitch measurements must come from [`evaluate.py`](surgedesk/evaluate.py) or [`benchmark.py`](surgedesk/benchmark.py). See the generated [results](docs/RESULTS.md) and [report](docs/REPORT.md). Development results are labelled **Development run, NOT LinuxONE**; unavailable measurements remain unavailable. Do not replace these with estimated claims.

After tagging, `make_label_sheet()` writes `data/labels/to_label.csv`; `make_purity_sheet()` writes `data/labels/purity_to_label.csv`. They preserve existing sheets unless explicitly forced. Humans save completed sheets as `hand_labels.csv` and `purity_labels.csv` in the same directory. Preserve `report_id` as plain text when exporting from a spreadsheet. Evaluation rejects scientific notation IDs, skips unfilled labels, and requests uncovered extraction predictions from Agent 3 when available.

For a provisional AI reference review, put `ai_extraction_labels_DRAFT.csv` and `ai_purity_labels_DRAFT.csv` in `data/labels/` and run `SURGEDESK_LABEL_SOURCE=ai SURGEDESK_GENERATE_EXTRACTIONS=0 python -m surgedesk.pipeline --stage eval`. This evaluates stored predictions only and never starts Granite. Extraction stays unavailable until real predictions exist. Results and notebook 05 identify AI reference labels explicitly; they are not independent human ground truth. Grouping excludes uncertain, blank and unreviewed rows and reports included/excluded counts. Reviewed all-null AI extraction rows are retained as negative examples. Keep `label_provenance.json` with `summary.json` when sharing results. To return to human evaluation, omit `SURGEDESK_LABEL_SOURCE` (or set it to `human`) and provide the human-reviewed files. Keep generation disabled for read-only evaluation; a deliberate extraction batch is a separate operation.

Ranking metrics average eligible snapshots; urgency is a proxy. Time to surface is measured from first report to first top-k appearance; never-surfaced incidents remain censored. Reports, samples, evidence, and review notes displayed by Agent 4 pass through the shared privacy masker. Outputs and labels are ignored by Git.

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

The following list is reproduced from `PLAN.md` section 9. Its study size and hardware describe the plan; completed annotation counts and measured machine information are reported only in generated results.

**Data**
- HumAID is tweets from 2016 to 2019, not 911 calls or texts; English only.
- HumAID is a sampled, de-duplicated research set (Harvey: ~9K tweets over 8 days), so incidents are smaller and more fragmented than a live feed would produce. The replay compresses time (1x = 12x real time).
- Urgency is a proxy built from human humanitarian labels, not real dispatch priority or outcomes.
- 100 hand-labeled reports from 4 labelers; no inter-rater agreement measured.
- Only HumAID Set 1 is used.

**Location**
- Most tweets aren't geotagged; place matching from text misses reports and can confuse same-name places. Place-to-county uses the nearest county centroid, which is approximate for large cities.
- SVI is county-level and from one year; neighborhood vulnerability is invisible.

**Models**
- Classifier and ranker are trained on other disasters; performance shifts with disaster type and region.
- Assistance text cues can miss slang, negation, time context, and indirect needs. The explicit priority policy and changed category target require independent operational review; proxy-metric gains do not establish better dispatch decisions.
- Grouping thresholds were tuned by spot check, not a labeled benchmark; wrong merges and splits happen.
- Granite 2B is small; it misses fields and misreads slang. Its confidence values are self-reported, not calibrated.
- Grounding proves a claim traces to a report, not that the report is true. Misinformation passes through.

**Fairness**
- Without ground truth on who most needed help, we can't prove the adjusted ranking is fairer. We show a tradeoff.
- Social media under-represents people without phones, internet, or power.

**System**
- 3 vCPU / 8 GB VM; Granite briefings are batched for the top incidents, not real time for every report.
- Not integrated with any dispatch system; not tested with real emergency staff; no authentication or high availability.


## Team and roles

Person 1: integration/UI/evaluation, infrastructure, and LinuxONE runner. Person 2: data and tagging. Person 3: Granite conversion and grounded briefings. Person 4: grouping/ranking and the pitch/story. Public names and final role wording are pending Person 4.

Built during the IBM Datathon 2026.

# SurgeDesk: Solo 30-Day Improvement Plan

## Goal and working rules

Turn the existing prototype into a reproducible local research system with trustworthy multi-event evaluation, documented classifier experiments, safer caching, and a polished synthetic demonstration. Improvements are hypotheses until measured; a negative experiment is still a useful result.

- Work exclusively on `surgedesk-month-improvements`, branched from `a4-integration` at `6d20725`. Preserve `a4-integration` as the historical baseline. Keep all month-long code, tests, documentation, and experiment infrastructure on the improvement branch; merge only after final evaluation and release checks pass.
- This is a solo project. The user authorized replacing the old four-agent ownership and team-agreement rules with solo ownership. Document that change in `AGENTS.md` before implementation. Preserve attribution for earlier teammates' work; claim only personally authored changes in the new case study and resume evidence.
- Use relative Day 1 through Day 30. Weekly capacity was intentionally left unspecified. Complete required outcomes in dependency order; stretch work never displaces evaluation, correctness, or release documentation.
- Keep runtime models and data local, use only permitted existing dependencies, and make no paid API calls. Keep configuration in `surgedesk/config.py`, preserve public stage signatures and shared record contracts, and use `fileio` for handoff artifacts.
- Never use `gold_label` as an input or train on `config.DEMO_EVENT`. Mask displayed and model-facing text with `mask_privacy`; never publish raw source reports, private annotations, model files, or credentials.
- Compute access is uncertain. Laptop reproduction is the required release target. LinuxONE validation is conditional on access and execution by the authorized human operator; the agent does not run on the VM. Report laptop and LinuxONE measurements separately.

## Week 1: Establish a trustworthy baseline

### Days 1-2: Preserve and partition

- Record the baseline commit, working-tree state, dependency versions, seeds, hardware, model identifiers, and checksums of available data and models. Preserve existing outputs; allocate a distinct directory per new run instead of overwriting historical artifacts.
- Inventory local HumAID events and original train/dev/test files. The current ingestion path merges these files: preserve source file, original split, and report ID in a sidecar manifest before using data for new experiments. Do not change `Report` to carry this metadata.
- Select disjoint training, development, and reserved evaluation events after inventory, using disaster type and region coverage where possible. Record explicit membership and selection rationale before experiments. Never use the configured demo event for training. Harvey's previously inspected results are historical evidence, not untouched evaluation.
- Check repeated IDs, normalized duplicate text, and near-duplicate reports across partitions. Manually review near-duplicate candidates using a fixed procedure recorded in the manifest; resolve leakage and log exclusions before freezing membership. If insufficient unseen events exist, explicitly limit the generalization claim rather than inventing a reserved benchmark.

### Days 3-7: Diagnose and make evaluation repeatable

- Add one evaluation command accepting a manifest, selected events, partition, configuration/model, and run identifier. Export JSON/CSV plus a readable summary with sample counts, configuration, model/data identity, commit, hardware, and reasons for unavailable measurements.
- Report category precision, recall, F1, support, accuracy, macro-F1, and a confusion matrix. Keep humanitarian category agreement separate from ranking's assistance/urgency proxies and operational emergency priority.
- Review up to 50 development classification errors and 50 location examples. Record sampling method, source IDs, uncertainty, and categories such as short text, indirect requests, negation, overlapping classes, ambiguous names, state abbreviations, and multiple locations.
- Measure embedding time, total processing time, throughput, and peak process memory when available. Separate cold application-cache runs from valid cached runs and record model-load time; do not imply the operating-system cache was cleared.
- Repeat the baseline twice: predictions and quality metrics must match for identical inputs/configuration; report runtime variation separately. Missing observations remain unavailable, never zero.
- Write a prioritized backlog using observed frequency, likely impact, and implementation cost.

**Exit gate:** preserved baseline, frozen manifest and leakage ledger, repeatable evaluation command, baseline report, error/location audit, and prioritized experiment backlog.

## Week 2: Compare classifiers and improve location handling

- Compare MiniLM embeddings plus logistic regression against TF-IDF plus logistic regression on identical partitions and masked text. Fit vectorizers and classifiers only on training data; frozen local pretrained embeddings are not permission to fit other transforms on evaluation text.
- Tune the strongest approach using development data only: regularization, class weighting, and at most one text-processing change justified by the audit. Record every attempted configuration, including unsuccessful experiments.
- Select the configuration with the highest development macro-F1; break exact ties using lower measured inference time on the same workload. Report assistance-category recall and resource tradeoffs alongside the selection. Retain the baseline if evidence does not support a replacement.
- Report per-category improvements and regressions, training/inference runtime, and peak memory. Define the assistance-category set before comparisons and preserve it in experiment metadata.
- Fix demonstrated location failures involving explicit state qualifiers, same-name places, and multiple mentions. Preserve unknown when evidence cannot support a county. Keep detected place evidence separate from county assignment confidence.
- Add assignment method, uncertainty, and evidence in a sidecar, preserving shared record formats. Label centroid assignments as approximate. Existing fixtures remain compatibility checks; add focused synthetic cases for newly supported behavior.
- Write an annotation guide with uncertain options and provenance. Keep reviewed development examples separate from reserved evaluation examples. Solo annotations support internal analysis; do not report inter-rater agreement or expert emergency-response validation.
- Stretch only: assess county-boundary lookup if suitable local boundary data can be processed with existing dependencies. Do not assign a tweet to a county polygon from a place centroid as though it supplied the report's coordinates.

**Exit gate:** reproducible experiment comparison, recorded selection decision, tested audit-supported location changes, annotation guide, and private review provenance.

## Week 3: Make processing reliable and measurable

- Profile the fixed baseline workload before optimizing. Compare embedding batches of 16, 32, and 64 within available memory. Keep hardware, thread count, input order, model, and workload fixed for comparisons.
- Replace shape-only embedding reuse with a fingerprint covering ordered report IDs, masked input text, model/tokenizer content checksums, and embedding configuration. Validate shape, dtype, finite values, and completion metadata before reuse.
- Validate tagging and downstream artifacts against relevant input, model, and configuration fingerprints. Reject stale, incomplete, or incompatible results; do not silently display them as current.
- Publish artifacts atomically with completion manifests written last. Resume from validated completed stages; recompute an interrupted stage. Partial-batch embedding recovery is outside required scope.
- Preserve report-to-vector alignment after filtering, reordering, or deduplication. Retain existing cross-endian embedding storage and verify byte-order round trips.
- Integrate stage progress, elapsed time, actionable errors, and quality/performance report links. Ensure review decisions can be traced to the source incident and run/model provenance without breaking decision contracts.
- Add offline fixture CI and targeted regression checks. Test invalidation after same-length text changes, ID reorder, model/configuration changes, and corruption; inject interruption and verify that completed valid work is reused.
- Report application-cache cold and warm measurements separately on the same workload. Record repeated runs and variability; never promise a speedup before measuring it.
- Stretch only: investigate grouping with reviewed development examples covering both false merges and false splits. Keep thresholds unchanged unless evidence supports a change; singleton rate alone is not a success metric.

**Exit gate:** profiling report, measured optimization results, demonstrated valid cache reuse/invalidation, interrupted-stage recovery, and an integrated offline regression workflow.

## Week 4: Evaluate, demonstrate, and release

### Days 22-24: Freeze and evaluate

- Freeze the selected configuration before opening reserved results. Run the preserved baseline and selected configuration on the same held-out events and hardware.
- Report per-event quality, equal-weight mean event macro-F1, pooled per-category metrics, sample counts, missing categories, and regressions. Keep historical metrics separate when their partitions or targets differ.
- Include uncertainty estimates where practical, describing the resampling unit and method; otherwise state that they were not computed. Do not retune using reserved results and continue describing them as untouched.

### Days 25-28: Demonstrate and document

- Package a clearly labeled synthetic demonstration with simple local setup, incident-member inspection, ranking explanations, location uncertainty, evidence links, and Accept/Override/Dismiss decision logging.
- If consenting testers are available, give each the same tasks: find an incident's reports, explain its ranking, identify location uncertainty, record a review decision, and locate supporting evidence. Record completion, time, mistakes, and feedback; fix the clearest blockers. Report an author-only walkthrough as such if no testers participate. Participant counts are not adoption counts.
- Publish architecture notes, reproducible benchmark instructions, experiment history, data-access requirements, known limitations, and a short recorded walkthrough. Keep notebooks thin, with the existing bootstrap pattern.

### Days 29-30: Release and evidence

- Run required checks, inspect the release contents for private artifacts, and tag the release only after acceptance gates pass. Merge into the integration branch only after final evaluation and release checks, preserving baseline provenance.
- Write a technical case study covering problem, personally authored approach, results, regressions, and tradeoffs. Link changes to commits/PRs and distinguish inherited implementation from new work.
- Draft resume bullets only from measured outcomes and personally owned contributions. Document unsuccessful experiments honestly.

**Exit gate:** final comparison or explicit data/compute limitation, reproducible synthetic demo, passing required checks, release documentation, recorded walkthrough, and attributable contribution evidence.

## Verification and fallback policy

- Required tests cover partition exclusions and leakage handling, training-only TF-IDF fitting, metric calculations, unavailable observations, location ambiguity, stale caches, reordered reports, corrupted/incomplete artifacts, recovery, and endian round trips.
- Require `python -m surgedesk.selftest`, the offline fixture CI workflow, and Restart Kernel/Run All for the synthetic demo and read-only evaluation notebook. Model-dependent skips must state which evidence remains unverified.
- Real-data/model evaluation requires local artifacts. If access remains unavailable, finish the infrastructure and synthetic release, explicitly mark real-model results pending, and make no measured model-improvement claims. This fallback does not count as completing the multi-event evaluation objective.
- Priority under time pressure: trustworthy evaluation and provenance; cache/alignment correctness; the two classifier baselines; synthetic release; audit-supported location fixes. Defer optional tuning, boundary lookup, grouping changes, and additional usability work before dropping required correctness checks.
- Granite improvements, ranker redesign, public hosting, acceleration claims, dispatch integration, and emergency-response validation are outside required scope.

This document is a plan. Its experiment counts, sample caps, batch sizes, and days are planning choices, not measured results. No implementation or evaluation milestone is complete solely because it appears here.

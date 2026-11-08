# SurgeDesk research case study

## Problem and inherited system

The original multi-author prototype already tagged, grouped, ranked, and briefed disaster reports with local models. Its historical demonstration did not establish multi-event generalization, and shape-only embedding reuse could accept stale text. Original contributors retain credit for that implementation.

## Work on the solo improvement branch

Added split-preserving source manifests, leakage exclusions, isolated experiment runs, train-only TF-IDF fitting, model selection gates, held-out evaluation, and computed reports. Added conservative location evidence, content-based cache validation, completion receipts, recovery checks, offline CI, and a synthetic evidence/decision demonstration.

## Measured outcome

On 3 reserved events containing 18,970 reports, mean event macro-F1 increased from 0.5224 to 0.5537 (+3.13 percentage points). The selected representation was tfidf with C=1.0 and class weighting=balanced.
Urgent-request recall changed from 0.5852 to 0.7162; missing-person recall regressed from 0.5251 to 0.3743. Macro-F1 improvement therefore does not imply every assistance-related outcome improved.

## Tradeoffs and limits

TF-IDF reduced category-classification feature costs in this benchmark, but incident grouping still requires embeddings. Cache timings are distinct from cold inference. Batch profiling uses a fixed development sample; laptop measurements do not establish LinuxONE performance. All results use humanitarian categories rather than validated dispatch urgency.
Agent-assisted development audits are explicitly labeled and are not independent human annotations. County centroids remain approximate. The automated walkthrough records fixture API/terminal actions, not human usability or a screen recording. See RESEARCH_RESULTS.md and RESEARCH_EVIDENCE.json for full metrics and measurement boundaries.

## Contribution evidence and resume wording

Use `git log a4-integration..surgedesk-month-improvements` to inspect the actual commits and authors. Link authored changes when publishing; no hosted PR or independent usability result is claimed here.

Suggested wording, contingent on accurately representing personal ownership:

- Built a local, reproducible disaster-report classification benchmark across 3 held-out events (18,970 reports); selected TF-IDF/logistic regression over a MiniLM baseline, improving mean event macro-F1 from 0.522 to 0.554 while documenting category-level regressions.
- Implemented content-addressed embedding validation, whole-stage recovery, cross-endian array checks, and an offline regression workflow; packaged a synthetic evidence-review demo with traceable decision logs.

These bullets cover the improvement branch, not sole authorship of the original system. Do not add emergency-response adoption, expert validation, LinuxONE speedups, or participant counts without new evidence.

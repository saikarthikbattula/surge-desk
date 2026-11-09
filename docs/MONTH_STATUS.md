# Month implementation and outstanding evidence

This is an implementation checklist, not a declaration that the month-long
research outcomes have been achieved. Historical results remain in `RESULTS.md`.

| Requirement | Implementation/evidence | Outstanding acceptance |
| --- | --- | --- |
| Isolated branch and preserved baseline | `surgedesk-month-improvements`, baseline `6d20725`; separate research outputs | Local release `surgedesk-month-v1`; integration branch remains preserved |
| Split provenance and leakage audit | Pinned author-mirror inventory, original splits, checksums, exclusions, freeze validation, text-free public specification; workload reconstruction verified | Duplicate review was agent-assisted; independent human audit is not claimed |
| Repeatable evaluation | Real MiniLM and TF-IDF evaluations repeated with identical predictions and quality metrics; JSON/CSV/Markdown exports and provenance | Additional platforms can reproduce using recorded versions and source checksums |
| Classification experiments | Both real local baselines and the development tuning grid completed; selection frozen before reserved evaluation | No further tuning on inspected reserved results |
| Error/location audit | Seeded masked samples, agent-assisted diagnostic reviews, annotation guide, computed summaries and backlog | Independent human annotation remains unavailable |
| Location uncertainty | Audit-supported conservative opt-in, common-word/qualifier/conflict regression cases, evidence sidecars; legacy default retained | Human audit of coverage tradeoffs; county boundaries remain stretch |
| Cache and recovery | Content fingerprints, completion checksums, atomic publication, preserved previous outputs, whole-stage receipts, endian tests; real seeded batch/cache profiling | LinuxONE and full-workload optimization validation remain unverified; no full-pipeline speedup claim |
| Offline regression checks | Fixture workflow in `.github/workflows/offline.yml`; full local checks and standalone clean-checkout self-test, smoke, and notebooks passed | Hosted CI run after publication; no remote CI result claimed locally |
| Demo and read-only results | Scripted synthetic fixture cards, notebooks 06/07, and a recorded automated terminal/API walkthrough with validated decisions | Desktop visual inspection passed in Chrome Guest; no human usability sessions were conducted |
| Final multi-event evaluation | Reserved runs completed under the frozen selection; per-event/category comparisons, regressions, and descriptive event-bootstrap interval generated in `RESEARCH_RESULTS.md` | This reserved benchmark is now inspected; do not call subsequent tuning untouched |
| Release and resume evidence | Computed case study/resume suggestions, experiment history, public split specification, runbook, backlog, and local implementation commit `ed6c48b` | Local release `surgedesk-month-v1`; hosted publication and PR are not claimed |

## Usability task sheet

Recruit consenting participants only if available. Use pseudonymous IDs and
record consent privately. Give everyone these tasks in the same order:

1. Find the member reports associated with a selected incident.
2. Explain why that incident received its ranking using displayed reasons.
3. Identify uncertain location information.
4. Accept, override, or dismiss a recommendation and inspect its logged record.
5. Locate quoted evidence supporting a displayed statement.

The user confirmed that no consenting testers are available for this release.
No human usability sessions were conducted, and no participant results or
adoption counts are claimed. The task sheet remains available for future use.

For each future task record completion, elapsed seconds, errors, and feedback. Do not
invent participant results. An author-only walkthrough is labeled as such;
participants are not ongoing product users. Save private observations beneath
the research outputs directory. Fix blockers supported by observed behavior.

## Recording and case-study outline

Record local setup, the synthetic banner, incident members, ranking reasons,
location uncertainty, quoted evidence, and one decision log entry. Explain
that template briefings and simulated scores are not real model performance.
An automated terminal/API recording is available through
`python -m surgedesk.research walkthrough --run-id NAME`. It is explicitly not a
screen recording or usability session. Automated desktop visual inspection
passed; human usability review and a UI screen recording remain unverified.

The case study should identify the inherited architecture, personally authored
changes, baseline and final event manifest, executed experiments (including
negative results), computed quality/performance comparisons, regressions, and
remaining limitations. Link each contribution to its actual commit/PR. Use only
the computed comparisons in `RESEARCH_RESULTS.md`; do not extend them to
unmeasured platforms or full-pipeline speedups.

## Release gate

- All required local self-tests and synthetic smoke checks pass.
- Demo and read-only results notebooks execute from a restarted kernel.
- Source/model versions and any skipped checks are explicitly recorded.
- Required real-data evaluation is complete, or the release is explicitly an
  infrastructure/synthetic milestone with the research objective still pending.
- Review public files for restricted report text, private annotations, model
  weights, tokens, and credentials; only synthetic examples are publishable here.
- Human walkthrough/usability results are described accurately.
- Tag/merge only the intended milestone after its actual checks pass.

## Local release disposition

The required laptop implementation, real multi-event comparison, regression
checks, clean-checkout smoke/notebook checks, synthetic demonstration, recorded
automated walkthrough, and contribution records are complete. The local release
is `surgedesk-month-v1` on `surgedesk-month-improvements`. It preserves the
historical integration branch and does not publish private data, weights, or
annotations. See `RESEARCH_VALIDATION.md` and its JSON record for check scope.

Hosted CI/publication, independent annotations, human usability sessions, and
LinuxONE validation are not claimed. County-boundary matching and grouping
threshold changes remain documented stretch work. None of these limitations
supports a production emergency-response claim.

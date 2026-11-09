# Internal development annotation guide

These annotations support internal research. A solo review is neither
independent inter-rater agreement nor expert emergency-response validation.
Keep exported reports and annotations under gitignored `outputs/research/`.
All visible text must remain privacy-masked. IDs must remain strings.

## Selection and provenance

Use the seeded development exports from `surgedesk.research`. Never substitute
reserved evaluation reports. Preserve report keys, event, model checksum,
manifest fingerprint, sampling method, and seed. Fill `reviewer`, `outcome`
(`correct`, `incorrect`, or `uncertain`), `error_categories`, and `note`.
Set `review_kind` to `human` or `agent_assisted`; never present agent-assisted
diagnosis as independent human annotation.
Do not overwrite machine predictions or dataset labels when recording review.
For classification, `outcome` assesses the prediction in context; a plausible
alternative to a noisy dataset label should be explained rather than silently
changing gold truth.

## Classification

Use the contract taxonomy. Review the whole masked report, then choose error
categories as needed: `short_text`, `indirect_request`, `negation`,
`overlapping_classes`, `ambiguous_context`, `label_disagreement`, or `other`.
Leave uncertain when the report lacks enough context.

Synthetic examples:

- "We need drinking water at the shelter" expresses a request.
- "Water deliveries have arrived; no more supplies needed" should not be
  interpreted as a current request solely because it mentions supplies.
- "Please help" has insufficient context for many operational decisions even
  if a humanitarian category can be assigned.

Dataset category labels do not independently establish operational urgency.

## Location

Assess place detection separately from county assignment. Record qualifiers,
conflicting mentions, and whether the report actually locates an incident.
Use categories `same_name`, `state_qualifier`, `multiple_places`,
`centroid_approximation`, `no_location_evidence`, or `other`.

- "Houston, MS" must not become Houston, TX because the event is in Texas.
- "Houston and Mission TX need supplies" mentions distinct county candidates;
  do not choose the first as the report's certain location.
- "Harris County" names a county; "Houston" plus a nearest centroid only
  supplies an approximate assignment, not a precise report coordinate.
- "Mission accomplished" is not sufficient evidence for Mission, TX.

## Duplicate/grouping review

For cross-partition duplicate candidates, choose `distinct`, `exclude_left`, or
`exclude_right` with a reason in the draft manifest. For optional grouping work,
review both apparently split duplicates and possibly unrelated merged reports.
Same topic, storm, city, or hashtag alone is insufficient to establish one
incident. Preserve uncertain cases and distinct incidents when evidence is weak.
Never optimize singleton rate without evaluating false merges and false splits.

## Review summary

`python -m surgedesk.research review-summary PATH` computes outcome and error
counts from completed review rows. Rank implementation issues by those counts,
likely impact, and effort; include counterexamples and uncertain cases.
Sampling classification errors intentionally overrepresents failures and cannot
estimate overall accuracy. Location sampling is uniform over development rows.
No agreement statistic should be reported without independent annotations.

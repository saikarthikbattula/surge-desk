# Evidence-based research backlog

Computed from development review summaries. Counts are agent-assisted issue categories, not independent error-rate estimates.

| Review stream | Observed issue | Reviewed count |
| --- | --- | ---: |
| classification | overlapping_classes | 18 |
| classification | off_topic_lexical | 10 |
| classification | short_text | 5 |
| classification | hypothetical_request | 5 |
| classification | label_disagreement | 5 |
| classification | ambiguous_context | 3 |
| classification | historical_context | 2 |
| classification | indirect_request | 2 |
| location | common_word | 5 |
| location | out_of_region | 2 |
| location | gazetteer_coverage | 2 |
| location | ambiguous_context | 1 |
| location | generic_state_city | 1 |
| location | personal_name | 1 |
| location | storm_place_ambiguity | 1 |

Priority order for a future research cycle:

1. Examine overlapping labels, off-topic lexical triggers, and the observed missing-person recall regression. Keep original gold labels separate from review judgments.
2. Independently audit conservative location behavior, especially unqualified real towns and county-centroid approximations. Boundary lookup remains conditional on suitable local data and existing dependencies.
3. Validate batch/cache measurements on the authorized LinuxONE host and a representative full workload before promoting a throughput claim.
4. Conduct consenting human usability sessions and record an actual UI walkthrough when the browser surface is available.

The reserved results are now inspected. Do not tune against them and continue calling them untouched; a further model-selection cycle needs a new reserved benchmark.

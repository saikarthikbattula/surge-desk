# SurgeDesk research comparison

Generated from verified local experiment artifacts; not historical Set 1 results.

Source: `QCRI/HumAID-events` revision `2e7ea23332006f075068b90d401b178d847b447a`. Raw reports and review notes are not redistributed.

Private workload manifest: `a5a6d7c670f5da6f8c42fb5e3d0c930bc37613becdca4e538413f7594095a1c6`.

## Held-out event classification

Macro-F1 is the unweighted mean over the fixed contract taxonomy, including zero F1 for absent classes. The aggregate is an equal-weight mean over events.

| Event | Reports | Baseline MiniLM LR | Selected model | Delta |
| --- | ---: | ---: | ---: | ---: |
| california_wildfires_2018 | 7376 | 0.5350 | 0.5628 | +0.0278 |
| hurricane_irma_2017 | 9399 | 0.5091 | 0.5442 | +0.0351 |
| kaikoura_earthquake_2016 | 2195 | 0.5232 | 0.5541 | +0.0309 |

Mean event macro-F1: 0.5224 → 0.5537; delta +0.0313.
Paired event-bootstrap delta interval: [+0.0278, +0.0351]. Paired event resampling, 2000 draws, seed 42; few events, descriptive only.

Selected configuration: tfidf logistic regression, C=1.0, class weight=balanced, seed=42. Full settings are in RESEARCH_EVIDENCE.json. Selection used development data before reserved results were opened.

## Per-category held-out tradeoffs

| Category | Support | Baseline F1 | Selected F1 | Baseline recall | Selected recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| caution_and_advice | 1245 | 0.6089 | 0.6018 | 0.6723 | 0.6032 |
| displaced_people_and_evacuations | 1210 | 0.7547 | 0.8376 | 0.8050 | 0.8165 |
| dont_know_cant_judge | 0 | 0.0000 | 0.0000 | unavailable | unavailable |
| infrastructure_and_utility_damage | 2615 | 0.7701 | 0.8228 | 0.7966 | 0.8424 |
| injured_or_dead_people | 2945 | 0.8677 | 0.8985 | 0.8238 | 0.9019 |
| missing_or_found_people | 179 | 0.5151 | 0.5154 | 0.5251 | 0.3743 |
| not_humanitarian | 2157 | 0.5535 | 0.5521 | 0.7218 | 0.6018 |
| other_relevant_information | 3707 | 0.4128 | 0.5089 | 0.3369 | 0.4893 |
| requests_or_urgent_needs | 229 | 0.3018 | 0.3914 | 0.5852 | 0.7162 |
| rescue_volunteering_or_donation_effort | 3212 | 0.7167 | 0.7409 | 0.6560 | 0.7024 |
| sympathy_and_support | 1471 | 0.7279 | 0.7524 | 0.6920 | 0.7158 |

Confusion matrices: [baseline](research_figures/baseline_confusion.png), [selected](research_figures/selected_confusion.png).

## Development experiment history

| C | Class weight | Status | Mean event macro-F1 |
| ---: | --- | --- | ---: |
| 0.1 | none | complete | 0.30834170487739027 |
| 0.1 | balanced | complete | 0.46901175128513195 |
| 1.0 | none | complete | 0.4168426658045852 |
| 1.0 | balanced | complete | 0.4860641298019153 |
| 10.0 | none | complete | 0.4412067348969041 |
| 10.0 | balanced | complete | 0.4761794686454432 |

The final configuration was selected from development results. The reserved results were not used to choose additional configurations.

## Runtime and memory

These are laptop measurements, not LinuxONE results. Process RSS is sampled and excludes children. OS caches were not cleared.

Baseline final processing-through-prediction seconds: 165.436; selected: 35.208.
Sampled peak RSS bytes: baseline 944816128; selected 313815040.
The classifier comparison includes each representation's own feature costs; grouping still requires embeddings when using TF-IDF classification.

| Profile batch | Cold encoding median seconds | Warm artifact median seconds |
| ---: | ---: | ---: |
| 16 | 3.9154 | 0.1065 |
| 32 | 4.4428 | 0.1060 |
| 64 | 4.4204 | 0.1013 |

Preserved `6d20725` baseline encode median seconds on the same fixed sample: 4.4142.
Compare encode-only time to encode-only time; cache-hit measurements include artifact verification. Batch results use a fixed development sample, not the full event workload.

## Review provenance and limitations

Classification review summary: `{'sampled': 50, 'reviewed': 50, 'review_kinds': {'agent_assisted': 50}, 'outcomes': {'incorrect': 45, 'uncertain': 5}, 'prioritized_errors': [{'category': 'overlapping_classes', 'count': 18}, {'category': 'off_topic_lexical', 'count': 10}, {'category': 'short_text', 'count': 5}, {'category': 'hypothetical_request', 'count': 5}, {'category': 'label_disagreement', 'count': 5}, {'category': 'ambiguous_context', 'count': 3}, {'category': 'historical_context', 'count': 2}, {'category': 'indirect_request', 'count': 2}], 'agreement': {'status': 'unavailable', 'reason': 'Solo review is not inter-rater agreement'}, 'limitation': 'Internal development review, not expert emergency-response validation'}`.
Location review summary: `{'sampled': 50, 'reviewed': 50, 'review_kinds': {'agent_assisted': 50}, 'outcomes': {'correct': 37, 'uncertain': 5, 'incorrect': 8}, 'prioritized_errors': [{'category': 'common_word', 'count': 5}, {'category': 'out_of_region', 'count': 2}, {'category': 'gazetteer_coverage', 'count': 2}, {'category': 'ambiguous_context', 'count': 1}, {'category': 'generic_state_city', 'count': 1}, {'category': 'personal_name', 'count': 1}, {'category': 'storm_place_ambiguity', 'count': 1}], 'agreement': {'status': 'unavailable', 'reason': 'Solo review is not inter-rater agreement'}, 'limitation': 'Internal development review, not expert emergency-response validation'}`.
Location policy comparison: `{'sampled': 50, 'changed': 8, 'changed_previously_flagged': 8, 'limitation': 'Agent-assisted development audit; not independent county ground truth or a generalization estimate'}`.

Reviews are agent-assisted diagnostics, not independent human annotations or emergency-response expert validation. County centroids remain approximate; boundary lookup is not implemented.
Human usability sessions, LinuxONE validation, and a human screen recording are not established by the automated synthetic walkthrough.
Category labels and assistance-category recall are not validated operational urgency. No dispatch or fairness outcome claim follows from these results.

See RESEARCH_EVIDENCE.json for aggregate machine-readable evidence and RESEARCH_SPLIT_MANIFEST.json for workload reconstruction metadata.

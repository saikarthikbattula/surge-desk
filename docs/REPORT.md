# SurgeDesk: Written Report

**IBM Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions**

**Team identity:** registered team name and member names have not been supplied; implementation roles are recorded below.

## Problem, beneficiaries, and purpose

Disaster coordinators must separate specific requests from repeated headlines, appeals, offers, and incomplete location descriptions. SurgeDesk is a research prototype for reviewing that information, with local emergency-management coordinators as intended users and disaster-affected residents as potential beneficiaries.

For scale, the National Hurricane Center's [Hurricane Harvey report](https://www.nhc.noaa.gov/data/tcr/AL092017_Harvey.pdf) estimated that approximately **40,000 flood victims** were evacuated to or sheltered across Texas and Louisiana. The figure is also recorded in an [official Senate resolution citing NHC](https://www.cassidy.senate.gov/newsroom/press-releases/cassidy-kennedy-cruz-cornyn-introduce-resolution-recognizing-the-first-anniversary-of-hurricane-harvey/). This is historical population context, not a claim that SurgeDesk served these people or improved their outcomes.

The intended value is a reviewable incident queue: inspect member reports, understand the model and policy contributions, adjust the vulnerability weighting, and record a human decision. SurgeDesk does not dispatch assistance.

## Technical design and LinuxONE implementation

| Stage | Implementation | Role |
|---|---|---|
| Tag | Local MiniLM in PyTorch, scikit-learn classifier, Census place/county lookup, vulnerable-person terms | Embed and categorize reports and attach supported geography |
| Group | Online text similarity with location and time constraints | Collect related reports into incidents |
| Rank | XGBoost, feature contributions, current-assistance priority bands, CDC county SVI adjustment | Order incidents for review and expose the scoring policy |
| Brief | IBM Granite 3.3 2B Instruct, team-converted fixed big-endian Q8_0 GGUF, llama.cpp | Attempt literal evidence extraction and short briefings; use labeled templates on failure |
| Review | Jupyter, ipywidgets, historical replay, Accept/Override/Dismiss, validated append-only decisions | Keep the coordinator in control |

PyTorch handles embeddings and XGBoost handles ranking. Granite does not determine the ranking; its selected-incident work is separated from the streaming tagging/ranking path. This is an architectural choice, not a claim of measured real-time latency.

The human operator ran the real workflow in IBM's existing Docker JupyterLab container on **s390x LinuxONE with 3 vCPU and approximately 8 GB RAM**. The replay, real-data cards, lambda changes, and decision logging were exercised. A custom Dockerfile is supplied, but the tested deployment was the existing IBM container. NNPA/zDNN availability was observed; accelerated model execution or hardware speedup was not demonstrated. Models and inference data are local, with no external model API calls at runtime.

## Data, relevance definition, and evaluation

HumAID Set 1 supplies historical English disaster posts and human category labels. Hurricane Harvey is the held-out demo event and is excluded from model training. Census Gazetteer names support place matching; county SVI provides the optional vulnerability adjustment. Real data, models, and private labels are excluded from Git; the committed fixtures are synthetic.

The current assistance target marks an incident relevant when a member report's human label is requests/urgent needs or missing/found people. Injury/death news alone is not positive under this target. These categories are **proxies**, not independent confirmation of a specific unresolved emergency. Gold labels are training/evaluation targets, never inference features. Priority bands use text cues and model outputs; scores are not calibrated emergency probabilities.

The evaluation used **9,112 Harvey reports**, of which **7,539** were tagged actionable, forming **6,779 incidents**. Classifier macro-F1 was **0.588** on Harvey. Ranker development AUC was **0.934** on training-event development data; this is separate from held-out ranking quality.

## Measured results

The following values are rounded from the committed, VM-generated [RESULTS.md](RESULTS.md), produced by `surgedesk.evaluate`.

| Method | Precision@10 | Recall@10 | NDCG@10 | Median minutes to surface | Share surfaced |
|---|---:|---:|---:|---:|---:|
| SurgeDesk | 33.3% | 36.9% | 0.474 | 21.9 | 31.7% |
| FIFO | 10.7% | 19.9% | 0.208 | 358.6 | 10.1% |
| Volume | 13.3% | 20.7% | 0.245 | 95.5 | 6.5% |
| Keyword | 25.3% | 34.0% | 0.412 | 290.3 | 27.3% |

Precision, recall, and NDCG are averaged over eligible snapshots with positive relevance and k capped by incident count. Time-to-surface is historical event time, not inference latency or emergency response time. Its median includes surfaced incidents only; share surfaced includes those never reaching top-k. SurgeDesk leads the listed baselines on the current assistance-category ranking target, while still missing or misprioritizing reports. The archived legacy model performs better on the separate older broad target; the full same-incident comparison is in RESULTS.md.

At the selected snapshot, changing lambda from **0** to **0.25** changed high-SVI representation in top-k from **20%** to **50%**, while category-proxy precision changed from **50%** to **70%**. This is a snapshot-specific policy tradeoff, not proof of fairer outcomes. Context incidents remain below current-assistance incidents under the band policy.

The latest **3-incident Granite review grounded no fields and used templates for every incident**. No briefing sentences were generated, so sentence acceptance is unavailable rather than zero percent. Saved call logs identify JSON parsing failures in part of the sample; useful Granite extraction has not been demonstrated. Extraction agreement and current grouping purity remain unavailable. Draft AI references are provisional, human validation is incomplete, and stale grouping labels are excluded. Formal current stream-latency and Granite-runtime benchmarks are not reported.

## Responsible AI and limitations

- **Human authority:** all recommendations require review. Decisions are logged; no dispatch integration or emergency-staff validation exists.
- **Traceability:** extracted fields require literal source evidence and generated sentences face conservative checks. Quote support does not prove a report true, and the guards can reject valid paraphrases. Templates remain explicitly labeled.
- **Ranking errors:** the review still includes political commentary phrased as requests, volunteer/medicine offers, and conditional rescue directions among highly ranked items. Category precision does not measure dispatch usefulness.
- **Location and grouping:** geography often remains unknown; the Gazetteer cannot resolve street/ZIP-only text. Place-to-county assignment uses approximate same-state centroids. Singleton share is **96.8%**; fragmentation and unavailable purity limit confidence in incident grouping.
- **Privacy:** handles, URLs, and common North American phone formats are masked. Names, addresses, unusual contacts, and older saved artifacts can remain sensitive. Raw reports and private label sheets must not be published.
- **Coverage and fairness:** historical sampled social media is not a complete emergency feed; people without connectivity are underrepresented. County SVI does not measure individual vulnerability, and no outcome-based fairness claim is supported.

Further work should distinguish specific needs from offers/commentary, validate geography and grouping with independent reviewers, diagnose bounded JSON-generation failures, and measure runtime under a controlled workload. These remain research tasks rather than completed benefits.

## Team and reproducibility

| Human driver | Contribution | Coding-agent responsibility |
|---|---|---|
| Person 1 | Infrastructure, LinuxONE execution, integration review, UI, evaluation | Agent 4: integration/UI/evaluation/benchmark |
| Person 2 | Data and MiniLM preparation, tagging/location review | Agent 1: data, embedding, tagging, geography |
| Person 3 | Granite conversion and model-quality work | Agent 3: LLM adapter, extraction, grounding, briefings |
| Person 4 | Grouping/ranking, story and presentation coordination | Agent 2: grouping, features, ranking, fairness, baselines |

The team used coding agents for implementation and humans for coordination, review, and LinuxONE runs. See [README.md](../README.md) for fixture and real-data run instructions, artifact requirements, model checksum, and cache behavior. Only the verified evaluation and demonstrated workflow are claimed here; template outputs are not presented as successful model-written briefings.

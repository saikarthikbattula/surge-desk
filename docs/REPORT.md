# SurgeDesk: Written Report

IBM Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions · Team: [TEAM NAME]

> Template owned by Person 4. Replace every [BRACKET]. Every number must come from `docs/RESULTS.md` (generated on LinuxONE). Target length: about 2 pages.

## 1. Problem and beneficiaries
[2 to 3 sentences on the disaster report flood at small county emergency offices.]
[Sourced statistic, with citation: Author/Agency, title, year, URL.]

## 2. What SurgeDesk does
SurgeDesk turns a stream of disaster reports into a short, ranked list of incidents, each with quoted evidence a coordinator can verify. It recommends; a human decides, and every decision is logged.

## 3. How it works
| Stage | What it does | Technology |
|---|---|---|
| 1. Tag | Embeds and classifies every report; matches places to counties; flags vulnerable people | PyTorch (MiniLM embeddings), scikit-learn classifier, Census Gazetteer |
| 2. Group | Merges reports about the same incident in real time | Online clustering (text similarity + place + time window) |
| 3. Rank | Scores every open incident and explains the score | XGBoost + built-in SHAP contributions, CDC SVI |
| 4. Brief | Extracts facts with exact quotes and writes a short briefing; rejects unsupported sentences | IBM Granite 3.3 2B, converted by us to big-endian GGUF, llama.cpp |
| 5. Review | Cards, fairness slider, accept/override/dismiss with a decision log | Jupyter + ipywidgets |

**Why the design fits real time:** cheap models handle every report; Granite only reads the top incidents.

## 4. Running on IBM LinuxONE
- Our own Docker image built on IBM's s390x JupyterLab image; data and models mounted at runtime.
- Granite 3.3 2B converted by us to a big-endian Q8_0 GGUF for LinuxONE.
- Telum / NNPA: [what we measured, or "zDNN-enabled llama.cpp; no comparison measured"].
- Privacy: reports never leave the machine; no external APIs at runtime.

## 5. Data
- HumAID Set 1 (QCRI): [N] human-labeled tweets from [K] disasters, 2016 to 2019. Demo/test event: [EVENT], never used for training.
- CDC/ATSDR Social Vulnerability Index ([YEAR], county level); US Census Gazetteer ([YEAR]).
- Urgency is a documented proxy: an incident is urgent if any of its reports carries a human label of requests/urgent needs, injured/dead, or missing/found.
- [100] reports hand-labeled by our team to evaluate Granite's extraction.

## 6. Results (held-out event, measured on LinuxONE)
| Metric | SurgeDesk | FIFO | Volume | Keyword |
|---|---|---|---|---|
| Median minutes to surface urgent incidents | [ ] | [ ] | [ ] | [ ] |
| Share of urgent incidents surfaced in top 10 | [ ] | [ ] | [ ] | [ ] |
| Precision@10 (mean over snapshots) | [ ] | [ ] | [ ] | [ ] |

- Classifier macro-F1 on the held-out event: [ ]
- Granite extraction match rate by field: [ ]; JSON valid rate: [ ]; fields grounded: [ ]; briefing sentences accepted: [ ]
- Grouping: [N] incidents from [M] actionable reports; purity on 20 checked incidents: [ ]
- Stream latency (p95) at 1x / 5x / 20x: [ ]; Granite seconds per incident: [ ]
- Fairness: at λ = [ ], high-SVI share of the top 10 moves from [ ] to [ ], precision@10 from [ ] to [ ]

## 7. Responsible AI
- Human in the loop: SurgeDesk never dispatches; every decision is logged.
- Grounding: every displayed fact is an exact quote; unsupported briefing sentences are rejected and shown as rejected.
- Fairness: the SVI slider makes the vulnerability tradeoff explicit; we don't claim it proves fairness.
- Privacy: handles and URLs masked in all displays; raw data never committed.

## 8. Limitations
[Copy the top items from PLAN.md section 9, plus 2 concrete failure cases from Person 3.]

## 9. Team
| Person | Role | Agent driven |
|---|---|---|
| [Name] | Infra lead, LinuxONE runner | Agent 4: integration, UI, evaluation |
| [Name] | Data | Agent 1: data + tagging |
| [Name] | Model (Granite) | Agent 3: briefings |
| [Name] | Story, video, report | Agent 2: grouping + ranking |

All code was written during the datathon by our team and our coding agents.

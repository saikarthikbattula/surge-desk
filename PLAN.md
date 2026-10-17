# SurgeDesk: Master Plan

IBM Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions · Runs on IBM LinuxONE

**Deadline: 9:00 AM CT Sun Oct 18 (we submit by 8:15 AM). Submission: 3-minute video + GitHub repo link + slide deck + written report.**

This file is the big picture. Each person and each Codex agent has a complete handoff in `handoffs/`; those are the specs to build from.

---

## 0. TL;DR
During a hurricane, small county emergency offices get hundreds of messy, duplicate reports in a few hours. SurgeDesk turns that stream into a short, ranked list of incidents, each with quoted evidence a human can verify. It recommends; a human decides.

Cheap models (PyTorch embeddings, our classifier, our XGBoost ranker) process every report in real time. Our own Granite model (converted by us to big-endian for LinuxONE) only reads the top incidents and writes grounded briefings. Everything runs on the LinuxONE VM inside our own container, and no data leaves the machine.

---

## 1. Decisions (settled Oct 17)
| # | Question | Answer |
|---|---|---|
| Q1 | Deadline | 9:00 AM Central, Sun Oct 18. Internal deadline 8:15 AM |
| Q2 | Video limit | 3 minutes (target 2:50) |
| Q3 | Own model | We convert IBM Granite 3.3 2B Instruct ourselves to a big-endian GGUF. No demo-lab model, no community pre-converted file |
| Q4 | Who runs on the VM | Everyone builds with Codex CLI on laptops. **Person 1 is the only one who runs the pipeline on LinuxONE** |
| Q5 | VM internet | Yes: code via `git pull`, data downloaded directly on the VM |
| Q6 | NNPA flag | Present. Telum acceleration is in scope; claim it only with a measurement or a clear runtime log line |
| Q7 | Demo event | `hurricane_harvey_2017` by default; Person 2 confirms it's in HumAID Set 1 (fallbacks in `handoffs/person2_data.md`) |
| Data | Core dataset | **HumAID Set 1** (~48K human-labeled tweets, 2016 to 2019). The Kaggle "Disaster Tweets" set can't drive SurgeDesk (binary label, no timestamps, mixed events, profile locations); optional check only |
| Build | Foundation | Shared contracts, config, I/O, fixtures, self-test, stubs, scripts, and `run_all.sh` are **already written and tested** in this repo, so all four agents start at the same time |

---

## 2. Rules
From the organizers and the rubric:
1. 100% of solution code written during the hackathon, by us or our agents. IBM's demo notebooks are reference only.
2. Our own models: our classifier, our ranker, our converted Granite.
3. At least 2 open-source AI frameworks: PyTorch, XGBoost, scikit-learn, llama.cpp.
4. Containerized deployment with Docker on LinuxONE.
5. Live demo recorded on LinuxONE, under 3 minutes.
6. Clearly tied to Theme 1.

Team rules: no data leaves the VM at runtime; every number in the pitch was measured by us on the VM; synthetic data is always labeled; the repo never contains HumAID text, models, tokens, or keys. Agent rules are in `AGENTS.md`.

---

## 3. The product
| | |
|---|---|
| **Problem** | Disaster report floods: duplicate, unlocated, messy. Someone must read everything, merge duplicates, and decide what gets help first |
| **Persona** | An emergency management coordinator at a small county office during a hurricane |
| **Beneficiary number** | Person 4 finds and cites a real statistic (`handoffs/person4_story.md` section 2) |
| **Value** | Hundreds of reports in, a short ranked list out, each with evidence; urgent incidents surface sooner than first-come-first-served |
| **Hard boundary** | Never dispatches. Ranks and explains; a human accepts, overrides, or dismisses, and every decision is logged |

---

## 4. Architecture
```
 HumAID tweets (replayed as a stream)
   │
   ▼  Stage 1 TAG (Agent 1)       every report, seconds
   │  MiniLM embeddings (PyTorch) → our classifier (sklearn) → places → county FIPS → vulnerable terms
   ▼  Stage 2 GROUP (Agent 2)     every report, seconds
   │  online clustering: similar text + same place + 6-hour window → incidents
   ▼  Stage 3 RANK (Agent 2)      every open incident, every 30 min of event time
   │  snapshot features + CDC SVI → our XGBoost ranker → score + SHAP reasons → fairness λ
   ▼  Stage 4 BRIEF (Agent 3)     top N incidents only, cached
   │  our Granite (llama.cpp, big-endian) → fields with exact quotes → grounding checks → briefing or template
   ▼  Stage 5 REVIEW (Agent 4)    Jupyter + ipywidgets
      live replay, incident cards, λ slider, Accept / Override / Dismiss → decision log
```
Granite output is display-only; it never changes the ranking.

---

## 5. Environment (known)
LinuxONE Community Cloud VM `AI_LAB`: 3 vCPU, 8 GB RAM, 50 GB disk, Ubuntu 24.04, **s390x big-endian**, NNPA flag present, internet available. IBM's JupyterLab image (`registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`) preinstalls numpy 2.0, pandas 3, scikit-learn 1.9, xgboost 3.4, torch 2.11 (CPU) + torch_nnpa, transformers 5.19, llama-cpp-python 0.3.36 (built with zDNN/NNPA + OpenBLAS), ipywidgets, matplotlib, psutil. Not installed: shap, sentence-transformers, gradio, pytest (we don't need any of them).

---

## 6. Repo map and ownership
```
AGENTS.md  PLAN.md  README.md  Dockerfile  run_all.sh  run_benchmark.sh  requirements-*.txt   (foundation; README finalized by Agent 4)
surgedesk/
  config.py contracts.py fileio.py textutil.py envcheck.py selftest.py testing.py stubs.py   (foundation, frozen)
  data_humaid.py geo.py embed.py tag.py replay.py                                           (Agent 1)
  group.py features.py rank.py fairness.py baselines.py                                     (Agent 2)
  llm.py grounding.py brief.py                                                              (Agent 3)
  pipeline.py ui.py evaluate.py benchmark.py                                                (Agent 4)
notebooks/ 00_env_check (foundation) 01_tag (A1) 02_group_rank (A2) 03_brief (A3) 04_demo + 05_eval_benchmark (A4)
tests/     test_foundation.py + test_agent1_*.py ... test_agent4_*.py
scripts/   make_fixtures.py download_data.sh download_minilm.py convert_granite.sh             (foundation)
fixtures/  synthetic events, gazetteer, SVI, expected outputs (the executable spec)          (foundation)
handoffs/  agent1_tag.md agent2_group_rank.md agent3_brief.md agent4_integration.md
           person1_vm_runner.md person2_data.md person3_granite.md person4_story.md
docs/      REPORT.md (Person 4)  RESULTS.md (generated by evaluate.py)
data/ models/ outputs/   (gitignored; live on the VM and on laptops only)
```

**How code becomes results:** each agent works on its branch on a laptop against fixtures and real data → its human posts "Ready to run" → Person 1 merges, `git pull`s on the VM, runs the stage or notebook inside the container, posts the output → final run: `bash run_all.sh` then `bash run_benchmark.sh` in our container. Details: `handoffs/person1_vm_runner.md`.

---

## 7. Who does what
| Person | Drives | Owns | Handoff |
|---|---|---|---|
| Person 1 | Agent 4 | Repo, VM, all LinuxONE runs, container, benchmark, screenshots, screen recording | `handoffs/person1_vm_runner.md` |
| Person 2 | Agent 1 | Downloads, demo event, laptop data pack, labeling sessions, purity check | `handoffs/person2_data.md` |
| Person 3 | Agent 3 | Granite conversion and transfer, VM load test, output quality, failure cases | `handoffs/person3_granite.md` |
| Person 4 | Agent 2 | Beneficiary statistic, slides, video script and voiceover, `docs/REPORT.md`, submission | `handoffs/person4_story.md` |
Everyone labels 25 reports (5:30 to 7:00 PM).

---

## 8. Timeline (Central)
| Time | Team milestone |
|---|---|
| Sat 1:00 PM | Repo pushed (starter repo as delivered); four agents started; VM container up; downloads and Granite conversion started |
| 1:30 PM | `00_env_check` run on the VM and posted |
| 4:30 PM | **Checkpoint 1**: modules for stages 1 to 3 pass fixture tests; `bash run_all.sh --stub` + stub demo cards on the VM |
| 5:00 PM | Real data through Stage 1 on the VM; label sheet exported |
| 5:30 to 7:00 PM | Hand labeling (everyone) |
| 6:00 PM | Our Granite loads on the VM; speed measured |
| 8:30 PM | **Checkpoint 2**: real ranked list for the demo event; dev AUC, precision@10 vs baselines; `TOP_N_BRIEF` set |
| 11:00 PM | Our container built and running |
| 12:30 AM | **Checkpoint 3**: full demo (live replay + dashboard + briefings) inside our container |
| 12:30 to 3:00 AM | Final run, evaluation, benchmark, failure cases, README |
| 3:00 AM | **Feature freeze**; image rebuilt from final `main` |
| 3:00 to 6:00 AM | Persons 2 and 3 sleep; Persons 1 and 4 finish slides/script, record 5:00 to 6:00 |
| 6:00 to 7:30 AM | Persons 1 and 4 sleep; Persons 2 and 3 edit the video, check numbers, clean the repo |
| 7:30 to 8:15 AM | Everyone: final watch; **submit by 8:15** |

**Slip triggers:** Checkpoint 2 after 10:00 PM → cut items 1 to 4 below now. Checkpoint 3 after 2:00 AM → also cut item 5. Nothing recorded by 6:30 AM → record as-is.

**Cut order** (from the top; never the bottom):
1. Optional Kaggle check (task 1.9); 20x and max benchmark runs (keep 1x and 5x)
2. Stretch NNPA embeddings comparison (keep the llama.cpp zDNN log screenshot)
3. Purity check (show spot-check examples instead)
4. λ sweep (keep λ=0 vs one λ)
5. Granite briefing call (keep extraction fields + template briefing)
6. **Never cut:** ranked list with evidence, honest evaluation vs baselines, container on LinuxONE, recorded demo

---

## 9. Limitations (say these honestly)
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
- Grouping thresholds were tuned by spot check, not a labeled benchmark; wrong merges and splits happen.
- Granite 2B is small; it misses fields and misreads slang. Its confidence values are self-reported, not calibrated.
- Grounding proves a claim traces to a report, not that the report is true. Misinformation passes through.

**Fairness**
- Without ground truth on who most needed help, we can't prove the adjusted ranking is fairer. We show a tradeoff.
- Social media under-represents people without phones, internet, or power.

**System**
- 3 vCPU / 8 GB VM; Granite briefings are batched for the top incidents, not real time for every report.
- Not integrated with any dispatch system; not tested with real emergency staff; no authentication or high availability.

---

## 10. Risks and fallbacks
| Risk | Early sign | Fallback |
|---|---|---|
| Harvey not in Set 1 | download listing | Next US hurricane in Set 1 (Person 2) |
| Granite file won't load | "file version ... extremely large" or unknown-architecture error | Re-convert with `--bigendian` / with the llama.cpp commit vendored by llama-cpp-python 0.3.36 (`scripts/convert_granite.sh`) |
| Granite too slow | > 30 s per call at 6 PM | Lower `TOP_N_BRIEF`; extraction + template only |
| Granite JSON invalid | JSON valid rate < 80% | Grammar-constrained decoding (built into `llm.py` spec); prompt tuning |
| 8 GB RAM exhausted | kernel dies silently | One Granite process; shut down kernels; stop IBM's container once ours runs |
| Low place coverage | < 20% of actionable reports have a county | Grouping falls back to text + time (built in); report coverage honestly |
| Ranker doesn't beat Volume | Checkpoint 2 | Report honestly; lead with time-to-surface and evidence quality |
| Container build fails | 11 PM | Run in IBM's container (still Docker on LinuxONE) and say so |
| Agents break each other's code | selftest fails after merge | File ownership + fixture tests; merge only when selftest passes |
| Person 1 is a bottleneck | queue piles up | Priority order in `person1_vm_runner.md`; Person 2 runs downloads |
| Exhaustion | mistakes after 2 AM | Freeze at 3 AM; sleep shifts |
| Live demo glitch | during recording | Backup take |

---

## 11. Rubric map
| Category | Item | Where |
|---|---|---|
| Technical Design | Workloads on L1CC (screenshots/video) | `person1_vm_runner.md` sections 8 and 9 |
| | 2+ open-source AI frameworks | PyTorch, XGBoost, scikit-learn, llama.cpp |
| | Containerized deployment | `Dockerfile`, `person1_vm_runner.md` section 6 |
| | Security or Telum acceleration | Data never leaves the box; zDNN/NNPA log + benchmark |
| Novelty | Code from scratch | `AGENTS.md` rule 1 |
| | Not a reference notebook | Online grouping, grounding checker, fairness slider, decision log, time-to-surface benchmark |
| | Hybrid AI used creatively | Five stages, each model with a distinct job |
| | Structured model + Granite | XGBoost ranks; SHAP reasons feed Granite's briefing |
| Pitch | 3-minute video | `person4_story.md` section 4 |
| | Problem, persona, value | Section 3 |
| | Live demo on LinuxONE | Video 0:25 to 1:45 |
| | Team roles | Slide 8, video end |
| Community Value | Quantified beneficiary | `person4_story.md` section 2 |
| | Tied to a theme | Theme 1 |
| | AI for Good + ethical reasoning | Human in the loop, quotes as evidence, fairness tradeoff, privacy masking |

---

## 12. Submission checklist
See `handoffs/person4_story.md` section 6 and `handoffs/person1_vm_runner.md` section 10.

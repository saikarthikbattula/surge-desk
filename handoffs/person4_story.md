# Person 4 handoff: Story, video, report, submission

You drive **Agent 2** (`handoffs/agent2_group_rank.md`). You own the beneficiary statistic, the slide deck, the 3-minute video script and voiceover, the written report (`docs/REPORT.md`), the team roles slide, and the final submission.

Submission = **3-minute video + GitHub repo link + slide deck + written report**, due **9:00 AM CT Sun Oct 18** (we submit by **8:15 AM**).

## 1. Your timeline (Central)
| Time | Do |
|---|---|
| 1:00 PM | Clone the repo; start Agent 2 on branch `a2-rank` with: "Read AGENTS.md and handoffs/agent2_group_rank.md, then implement it." |
| 1:15 to 3:00 PM | Beneficiary statistic (section 2), while Agent 2 builds |
| 3:00 to 4:30 PM | Review Agent 2's tests; slide skeleton (section 3) |
| 5:30 to 7:00 PM | Label your 25 rows (Person 2's sheet) |
| 8:30 PM | Checkpoint 2: Agent 2's numbers posted; review the spot check (Agent 2 notebook cell 8) |
| 9:00 PM to 3:00 AM | Script draft (section 4), report draft (section 5), slides filled as numbers land |
| 3:00 to 5:00 AM | With Person 1: final slides, final script, rehearse the click path |
| 5:00 to 6:00 AM | Record the demo on the VM (Person 1 drives the screen) and your voiceover |
| 6:00 to 7:30 AM | Sleep (Persons 2 and 3 edit the video) |
| 7:30 to 8:15 AM | Final watch, submit (section 6) |

## 2. Beneficiary statistic (1:15 to 3:00 PM)
The rubric wants "a specific, quantified real-world beneficiary population." We need **one or two real numbers with a source you can link**, never an estimate.
Good targets (search these, then confirm on the primary source):
- How many US county or local emergency management offices have very small staffs (surveys by NACo, IAEM, or GAO reports on local emergency management capacity).
- How many calls/reports a county or city received during Hurricane Harvey (City of Houston, Harris County, or official after-action reports).
- How many US counties had a federal disaster declaration in a recent year (FEMA declarations data).
Rules: prefer government or peer-reviewed sources; write down the exact quote, page, and URL; if you can't find a primary source, don't use the number. Put the citation on the slide and in `docs/REPORT.md`.

## 3. Slide deck (8 slides)
1. **Title**: SurgeDesk, Theme 1 (Real-Time AI for Critical Decisions), team name.
2. **Problem**: the report flood during disasters + the sourced statistic.
3. **Persona and value**: the county coordinator; "hundreds of reports in, a short ranked list out, each with evidence."
4. **Architecture**: the 5 stages (Tag → Group → Rank → Brief → Review) with each model's job; "runs on IBM LinuxONE in our own container; Granite converted by us to big-endian."
5. **Live demo**: a still from the recording (the video carries the real demo).
6. **Results** (from `docs/RESULTS.md` only): time-to-surface vs FIFO/Volume/Keyword; precision@10; extraction match rate; stream latency at 1x/5x/20x; Granite seconds per incident; NNPA note if measured. Caption: "Measured on IBM LinuxONE (s390x), 3 vCPU, 8 GB."
7. **Ethics and limits**: human decides; every fact traced to a quote; fairness slider is a transparent tradeoff; top 3 limitations (`PLAN.md` section 9) + 1 failure case from Person 3.
8. **Team and roles**: who owned what; how four Codex agents split the build (Agent 1 tagging, Agent 2 ranking, Agent 3 Granite, Agent 4 integration) and how humans reviewed and ran everything on LinuxONE.

## 4. Video: 3:00 hard limit, target 2:50
Record the screen silently on the VM (Person 1 drives), then record voiceover separately and lay it over. About 380 spoken words total. Replace every [BRACKET] with a real, measured number from `docs/RESULTS.md`, or cut the sentence.

| Time | On screen | Voiceover draft |
|---|---|---|
| 0:00 to 0:15 | Title, then the statistic slide | "When a hurricane hits, a small county emergency office can get hundreds of reports in a few hours: calls, texts, posts. Most are duplicates, many have no clear location. [ONE SOURCED STATISTIC.] Someone has to read all of it." |
| 0:15 to 0:25 | Persona slide | "Meet the coordinator on shift. SurgeDesk turns that flood into a short ranked list of incidents, each with the evidence behind it. It recommends. She decides." |
| 0:25 to 0:30 | Terminal: `uname -m` → s390x, `docker ps` → surgedesk | "This is SurgeDesk running on IBM LinuxONE, inside our own container." |
| 0:30 to 0:50 | `live_replay` at 5x | "We're replaying real Hurricane Harvey tweets at high speed. Every report is embedded, classified, matched to a place, and merged with reports about the same incident, and every open incident is re-ranked as reports arrive, in about [P95 LATENCY] per batch." |
| 0:50 to 1:15 | Dashboard, top card expanded | "Here's the top incident. Granite extracted each fact with the exact quote it came from, then wrote this briefing. One sentence wasn't supported by the reports, so our grounding check rejected it, and shows why. These bars explain the ranking: [TOP REASONS FROM THE CARD]." |
| 1:15 to 1:30 | λ slider moving | "Connected neighborhoods post more, so they can crowd out vulnerable ones. This slider weights the CDC Social Vulnerability Index and shows exactly how the list changes. It's a choice we make visible, not a hidden fix." |
| 1:30 to 1:45 | Override click, logged line | "She disagrees with a ranking, overrides it, and the decision is logged." |
| 1:45 to 2:20 | Architecture slide, then results slide | "Each model has one job. PyTorch embeddings and our classifier tag every report. XGBoost ranks incidents and explains itself. Granite, which we converted ourselves to run big-endian on LinuxONE, reads only the top incidents, so the real-time path stays fast. [NNPA SENTENCE ONLY IF MEASURED.] On the held-out Harvey data, SurgeDesk surfaced urgent incidents [X] minutes sooner than a first-come-first-served queue, with precision at ten of [Y] versus [Z]." |
| 2:20 to 2:40 | Ethics and limits slide | "SurgeDesk never dispatches anything; a human makes every call. Every fact shown traces to a quote. Our limits: we trained on 2016 to 2019 tweets, urgency is a proxy built from human labels, and locations from text are imperfect." |
| 2:40 to 2:50 | Team slide | "We're [NAMES]. [ONE LINE ON ROLES], with four coding agents splitting the build. SurgeDesk: from the flood of reports to the people who need help first." |

Rules: label any sped-up footage on screen ("5x speed"); never claim real-time for the Granite step (it's batched for the top incidents); mask handles/URLs (the UI already does). Export at 1080p; check the final length is under 3:00.

## 5. Written report: `docs/REPORT.md`
A template is already in the repo. Fill every section; numbers come only from `docs/RESULTS.md`; keep it to about 2 pages. Commit it by 7:30 AM.

## 6. Submission checklist (7:30 to 8:15 AM)
- [ ] Video under 3:00, shows s390x + our container, sped-up parts labeled
- [ ] Slide deck exported (PDF + original)
- [ ] `docs/REPORT.md` complete and committed; README links to it and to `docs/RESULTS.md`
- [ ] Every number on slides/video/report appears in `docs/RESULTS.md`
- [ ] Beneficiary statistic cited with a link
- [ ] Theme 1 named explicitly
- [ ] Repo shared as the organizers require (Person 1), tagged `submission`
- [ ] Submitted by 8:15 AM; confirmation screenshot saved

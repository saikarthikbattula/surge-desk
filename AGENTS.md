# AGENTS.md: SurgeDesk

You are one of four Codex agents building SurgeDesk for the IBM Datathon 2026 (deadline 9:00 AM CT Sun Oct 18; feature freeze 3:00 AM). **Your human tells you your agent number. Read this file, then your handoff file, top to bottom, before writing code.** `PLAN.md` has the big picture.

| Agent | Role | Your handoff (the full spec) | Human driver |
|---|---|---|---|
| Agent 1 | Data + Stage 1 (Tag) | `handoffs/agent1_tag.md` | Person 2 |
| Agent 2 | Stages 2-3 (Group, Rank, Fairness, Baselines) | `handoffs/agent2_group_rank.md` | Person 4 |
| Agent 3 | Stage 4 (Granite briefings) | `handoffs/agent3_brief.md` | Person 3 |
| Agent 4 | Integration, UI, evaluation, benchmark, README | `handoffs/agent4_integration.md` | Person 1 |

## The foundation is already written. Do not edit it.
`surgedesk/config.py, contracts.py, fileio.py, textutil.py, envcheck.py, selftest.py, testing.py, stubs.py`, `scripts/`, `fixtures/`, `run_all.sh`, `run_benchmark.sh`, `tests/test_foundation.py`. They pass `python -m surgedesk.selftest`. If you believe one must change, stop and tell your human (only Agent 4 may change `config.py`, and only after the team agrees).

## Hard rules
1. **Write all code yourself, now.** Never copy code from IBM's demo notebooks or their `loan_demo` package. Reading them for settings is fine.
2. **No new packages.** Project code may only import what's preinstalled in IBM's LinuxONE container: numpy, pandas, scikit-learn, xgboost, torch, transformers, tokenizers, huggingface_hub, llama_cpp (llama-cpp-python), joblib, matplotlib, ipywidgets, psutil, plus the standard library. Never `pip install` on the VM. Laptops may install the same packages from `requirements-dev.txt` to mirror the VM.
3. **No network calls in project code.** Models and data are local files from `config`.
4. **No hardcoded paths or constants.** Everything comes from `surgedesk/config.py`.
5. **Follow the data contracts** in `surgedesk/contracts.py` exactly. Read/write handoff files only with `surgedesk/fileio.py`.
6. **Never use `gold_label` as a model input.** It is only a training target and an evaluation label.
7. **Never train on `config.DEMO_EVENT`.** Assert it in code.
8. **Only edit files your handoff says you own.** Need something from another agent's file? Stop and tell your human.
9. **Implement the exact function signatures in your handoff.** Other agents are coding against them right now.
10. **Notebooks are thin:** bootstrap cell from `notebooks/00_env_check.ipynb` first, then call functions from `surgedesk/`. Restart Kernel and Run All must pass.
11. **Tests:** write the tests listed in your handoff in `tests/test_agent<N>_*.py`; they must run fast and offline; `python -m surgedesk.selftest` must pass before you say you're done.
12. **Never report or write a number you didn't compute in code.** No performance claims in comments or docs.
13. **Respect the hardware:** 3 vCPU, 8 GB RAM, s390x big-endian. Use `config.N_THREADS`. Only Agent 3's code loads Granite, once per process.
14. **Privacy:** pass any text you display or send to Granite through `textutil.mask_privacy`. Never commit raw tweets.
15. **Develop against `fixtures/`** until real upstream outputs exist. Your handoff says exactly which fixture files to compare against.

## Definition of done (every task)
- The acceptance tests in your handoff pass, and `python -m surgedesk.selftest` exits 0.
- Your notebook passes Restart Kernel and Run All on your human's laptop.
- Your human has posted "Ready to run: branch, notebook, expected outputs" and **Person 1 ran it on the LinuxONE VM** and posted the output. You never run anything on the VM yourself.

## When unsure
Stop and ask your human. Don't invent requirements, change contracts, or add features that aren't in your handoff.

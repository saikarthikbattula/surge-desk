# SurgeDesk

Disaster report triage on IBM LinuxONE. IBM Datathon 2026, Theme 1: Real-Time AI for Critical Decisions.

> Starter README. Agent 4 turns this into the final judge-facing README (`handoffs/agent4_integration.md` section 9). Results live in `docs/RESULTS.md` (generated) and the write-up in `docs/REPORT.md`.

During a hurricane, small county emergency offices get hundreds of messy, duplicate reports in a few hours. SurgeDesk turns that stream into a short, ranked list of incidents, each with quoted evidence a coordinator can verify. It recommends; a human decides.

## For the team: start here
1. Read `PLAN.md` (big picture), then your own handoff in `handoffs/` (`person1` to `person4`).
2. Start your Codex agent with: *"You are Agent N. Read AGENTS.md and handoffs/<your file>.md, then implement it."*
3. Laptop setup (Python 3.12):
   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt          # llama-cpp-python only needed on the Agent 3 laptop
   python -m surgedesk.selftest                  # must print SELFTEST PASSED
   ```
4. Real data on your laptop: extract Person 2's data pack at the repo root (`tar -xzf surgedesk_datapack.tar.gz`).

## Running on LinuxONE (Person 1 only)
```bash
bash scripts/download_data.sh                    # VM shell: HumAID Set 1 + Census Gazetteer (+ SVI instructions)
python scripts/download_minilm.py                # container terminal
# Granite: converted on a laptop with scripts/convert_granite.sh, then scp'd to models/granite/
docker build -t surgedesk:latest .               # our container on top of IBM's s390x JupyterLab image
bash run_all.sh                                  # container terminal: every stage + evaluation
bash run_benchmark.sh                            # container terminal, other kernels shut down
```
Then open `notebooks/04_demo.ipynb`. Full runbook: `handoffs/person1_vm_runner.md`.

## Repository layout
See `PLAN.md` section 6.

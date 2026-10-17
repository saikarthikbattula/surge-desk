# Person 1 handoff: Infra lead and the only LinuxONE runner

You drive **Agent 4** (`handoffs/agent4_integration.md`) and you are the **only person who runs anything on the LinuxONE VM** (Person 2 may run the light download script; nobody else starts kernels there). Every number in the pitch comes from a run you did.

VM: `linux1@148.100.112.180`, flavor AI_LAB (3 vCPU, 8 GB RAM, 50 GB disk), Ubuntu 24.04, s390x.

## 1. Your timeline (Central)
| Time | Do |
|---|---|
| 1:00 PM | Create the private GitHub repo, push the starter repo, add teammates. Sections 2 and 3 (VM + IBM container + clone). Start Agent 4 |
| 1:30 PM | Run `00_env_check.ipynb`, post the output to the team chat. Person 2 starts downloads |
| 2:00 PM | Confirm contracts are frozen (they're in the starter repo). Everyone's agents are building |
| 4:30 PM | **Checkpoint 1** (section 5.1) |
| 5:00 PM | Run Agent 1's stages; export the label sheet (5.2) |
| 6:00 PM | Granite load test with Person 3 (5.3) |
| 8:30 PM | **Checkpoint 2** (5.4) |
| 11:00 PM | Build and start our container (section 6) |
| 12:30 AM | **Checkpoint 3** (5.5) |
| 12:30 to 3:00 AM | Final full run, eval, benchmark (section 7); screenshots (section 8) |
| 3:00 AM | Feature freeze. Rebuild the image from final `main` (section 6, `docker build` + recreate the container). Slides/script with Person 4 |
| 5:00 to 6:00 AM | Record the demo on the VM with Person 4 (section 9) |
| 6:00 to 7:30 AM | Sleep |
| 7:30 to 8:15 AM | Repo cleanup and sharing (section 10); submit with the team |

## 2. GitHub repo (1:00 PM)
1. Create a **private** repo (e.g. `surgedesk`). Push the starter repo exactly as delivered.
2. Add your three teammates as collaborators.
3. Create branches `a1-tag`, `a2-rank`, `a3-brief`, `a4-integration` from `main`.
4. Protect yourself from merge pain: agents only edit the files they own (see `AGENTS.md`). You merge each branch into `main` when its owner says it's ready and `python -m surgedesk.selftest` passes on the merged result.
5. For the VM, create a **fine-grained personal access token** with read-only access to this one repo (or a deploy key). Never commit it.

## 3. VM and IBM's Jupyter container (1:00 PM)
```bash
ssh -i <key>.pem linux1@148.100.112.180
uname -m                        # must print s390x
docker ps                       # is IBM's container already running? If yes, skip to "clone"
# Install docker only if missing (from IBM's lab instructions):
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER; newgrp docker; sudo systemctl start docker
# IBM's JupyterLab image (PyTorch, XGBoost, llama-cpp-python with zDNN/NNPA preinstalled)
mkdir -p ~/shared && chmod a+w ~/shared
docker pull registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
export TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))"); echo "$TOKEN"   # save it in your password manager
docker run -d --name notebook -p 38888:8888 -v /home/linux1/shared:/home/jovyan/shared \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest jupyter lab --ServerApp.token="$TOKEN"
sudo iptables -I INPUT -p tcp --dport 38888 -j ACCEPT
# clone (git is usually present; if not: sudo apt-get install -y git)
cd ~/shared && git clone https://<TOKEN_FOR_GITHUB>@github.com/<you>/surgedesk.git surgedesk
sudo chown -R 1000:1000 ~/shared/surgedesk      # the container user (uid 1000) must be able to write
```
Open `http://148.100.112.180:38888` with the token. In JupyterLab, open `shared/surgedesk/notebooks/00_env_check.ipynb`, Run All, screenshot, post to the team. It must show `arch: s390x`, the NNPA flag `True`, and zDNN libs.

**Where commands run:** "VM shell" = your ssh session. "Container terminal" = JupyterLab → File → New → Terminal (inside the container; Python packages live here). Pipeline commands always run in a container terminal from `/home/jovyan/shared/surgedesk`.

## 4. The run queue (all day)
- Teammates post: **"Ready to run: branch `<b>`, notebook `<nb>`, expected outputs `<files>`."**
- You: merge (or check out the branch), `git pull` on the VM (VM shell: `cd ~/shared/surgedesk && git pull`), then in a container terminal run what was asked. Post back the printed output (numbers or the error, verbatim). Your reply is the "confirmed on the VM" step.
- Priority: checkpoint runs > Granite speed test > anything blocking another agent > the rest. Batch Granite work so the model loads once.
- Only one kernel may hold Granite. Before heavy runs: Kernel → Shut Down All Kernels. If a kernel dies silently, it's RAM: shut down other kernels and rerun.
- Keep `SURGEDESK_THREADS=2` for normal runs (default); 3 only for the benchmark.

## 5. Checkpoint commands (container terminal, `cd /home/jovyan/shared/surgedesk`)
### 5.1 Checkpoint 1 (4:30 PM)
```bash
git -C /home/jovyan/shared/surgedesk log -1 --oneline       # confirm you're on the merged main
bash run_all.sh --stub
```
Then open `notebooks/04_demo.ipynb`, set `STUB = True`, Run All: stub cards must render. Post a screenshot.

### 5.2 Data and Agent 1 stages (after Person 2's downloads, ~5:00 PM)
```bash
python scripts/download_minilm.py                           # once; Person 2 may have done it
python -m surgedesk.pipeline --stage data                   # prints events; confirms the demo event exists
python -m surgedesk.pipeline --stage embed                  # longest step (minutes); post reports/sec
python -m surgedesk.pipeline --stage classifier
python -m surgedesk.pipeline --stage tag                    # post macro-F1 and place coverage
python -c "from surgedesk.evaluate import make_label_sheet; print(make_label_sheet())"
```
Download `data/labels/to_label.csv` from JupyterLab (right-click → Download) and give it to Person 2 for the labeling sheet. If the demo event isn't in the data, tell Agent 4 to change `DEMO_EVENT` in `config.py` (Person 2 picks the replacement), pull, and continue.

### 5.3 Granite load test (~6:00 PM, with Person 3)
After Person 3 copies `granite-3.3-2b-instruct-be.q8_0.gguf` into `~/shared/surgedesk/models/granite/`: open `notebooks/03_brief.ipynb`, run cells 1 to 4 only. Screenshot the verbose load log (look for zDNN/NNPA lines) and post the speed table.

### 5.4 Checkpoint 2 (8:30 PM)
```bash
python -m surgedesk.pipeline --stage ranker
python -m surgedesk.pipeline --stage rank
python -m surgedesk.pipeline --stage eval                   # ranking + time-to-surface work even before labels
```
Post: incidents, singleton share, dev AUC, demo snapshot time, precision@10 for SurgeDesk vs FIFO/Volume/Keyword. Agree `TOP_N_BRIEF` with Person 3 from the Granite speed (target: brief stage under ~20 minutes); set it by exporting `SURGEDESK_TOP_N=<n>` in the container terminal and in the `docker run` command (section 6).

### 5.5 Checkpoint 3 (12:30 AM)
Full demo works **inside our container** (section 6): `04_demo.ipynb` with `STUB = False`, `live_replay` runs, dashboard renders, an Override click logs a decision.

## 6. Build and run our container (11:00 PM)
VM shell:
```bash
cd ~/shared/surgedesk && git pull
docker build -t surgedesk:latest .                          # base image is already pulled, so this is quick
docker run -d --name surgedesk -p 38889:8888 \
  -v /home/linux1/shared:/home/jovyan/shared \
  -e SURGEDESK_TOP_N=${SURGEDESK_TOP_N:-10} \
  surgedesk:latest jupyter lab --ServerApp.token="$TOKEN"
sudo iptables -I INPUT -p tcp --dport 38889 -j ACCEPT
docker ps                                                    # screenshot this
```
Open `http://148.100.112.180:38889`. Once it works, stop IBM's container to free RAM: `docker stop notebook`.
If the build fails, check the base image's Python (`docker run --rm <base> which python`) and tell Agent 4. Fallback: keep running in IBM's container and say so honestly in the README.

## 7. Final run, evaluation, benchmark (12:30 to 3:00 AM)
Container terminal in **our** container. Run from the mounted repo so the latest `git pull` is what runs (the code copied into the image is the deployment copy; rebuild the image once at feature freeze so it matches the final code):
```bash
cd /home/jovyan/shared/surgedesk && bash run_all.sh          # cached stages are skipped; brief + eval run
# after hand labels are in data/labels/hand_labels.csv and purity_labels.csv:
python -m surgedesk.pipeline --stage eval
# shut down ALL other kernels first:
bash run_benchmark.sh
```
Check `docs/RESULTS.md` starts with "Measured on IBM LinuxONE (s390x)". Commit the executed `05_eval_benchmark.ipynb`, `04_demo.ipynb` (with outputs), and `docs/RESULTS.md`. Never commit `data/`, `models/`, or `outputs/`.

## 8. Screenshots for Technical Design (save to a shared folder)
1. `00_env_check.ipynb` output (s390x, NNPA flag, zDNN libs).
2. `docker ps` showing `surgedesk:latest`.
3. Granite verbose load log (zDNN/NNPA lines) and speed table.
4. `run_all.sh` finishing with stage timings.
5. `04_demo.ipynb` dashboard with a card expanded; `live_replay` mid-run.
6. Benchmark charts and the ranking/time-to-surface tables from `05_eval_benchmark.ipynb`.

## 9. Recording the demo (5:00 to 6:00 AM, with Person 4)
Follow the script in `handoffs/person4_story.md` section 4. Screen-record the browser + an ssh terminal. Order: terminal `uname -m` and `docker ps` (5 s) → `live_replay(speed=5.0)` (20 s) → dashboard top card expanded (25 s) → λ slider (15 s) → Override click + logged line (15 s). Record it twice (main + backup take). Record silently; Person 4 records the voiceover separately.

## 10. Repo cleanup and sharing (7:30 AM)
- `git status` clean; `git log` shows no `data/`, `models/`, `outputs/`, `*.gguf`, tokens, or keys (`git log --stat | grep -E "data/|\.gguf|\.pem"` returns nothing).
- README links to `docs/RESULTS.md` and `docs/REPORT.md`.
- Share the repo the way the organizers require (make it public, or add the judges). Making it public is safe because raw data is never committed.
- Tag the submission: `git tag submission && git push --tags`.

## 11. Troubleshooting
| Symptom | Fix |
|---|---|
| Browser can't reach 38888/38889 | `sudo iptables -I INPUT -p tcp --dport <port> -j ACCEPT`; check `docker ps` |
| `docker: permission denied` | `newgrp docker` or reconnect ssh |
| Jupyter can't write to `shared/` | VM shell: `sudo chown -R 1000:1000 ~/shared/surgedesk` |
| Kernel dies with no error | RAM. Shut down all other kernels; `free -h`; only one Granite process |
| "GGUF file version ... extremely large" | Granite file is little-endian. Person 3 re-converts with `--bigendian` |
| Granite unknown architecture/metadata error | Person 3 re-converts with the llama.cpp commit vendored by llama-cpp-python 0.3.36 (see `scripts/convert_granite.sh`) |
| A stage says "not implemented yet (owner: Agent N)" | That agent's branch isn't merged yet |
| VM unresponsive | LinuxONE portal → stop the instance, then start it; `docker start notebook` / `docker start surgedesk` |

# Person 2 handoff: Data

You drive **Agent 1** (`handoffs/agent1_tag.md`). You own getting the data onto the VM and onto teammates' laptops, choosing the demo event, and running the labeling sessions. You may run the download script on the VM yourself (it's light); everything else on the VM goes through Person 1's run queue.

## 1. Your timeline (Central)
| Time | Do |
|---|---|
| 1:00 PM | Clone the repo, start Agent 1 on branch `a1-tag` with: "Read AGENTS.md and handoffs/agent1_tag.md, then implement it." |
| 1:15 PM | Run the VM downloads (section 2). Do the SVI manual download (section 2.3) |
| 1:45 PM | Pick the demo event (section 3). Post it to the team |
| 2:00 PM | Build the laptop data pack and share it (section 4) |
| 2:00 to 4:30 PM | Review Agent 1's work against its handoff; run `python -m surgedesk.selftest -k agent1` on your laptop |
| 4:30 PM | Checkpoint 1 |
| ~5:00 PM | Person 1 exports `to_label.csv`. Set up the labeling sheet (section 5) |
| 5:30 to 7:00 PM | Everyone labels 25 rows. You export `hand_labels.csv` to the VM by 7:15 |
| 8:30 PM | Checkpoint 2. Then the purity sheet (section 6) |
| 3:00 to 6:00 AM | Sleep |
| 6:00 to 7:30 AM | Edit the video with Person 3 (`handoffs/person4_story.md` section 4) |

## 2. Downloads on the VM (1:15 PM)
### 2.1 Run the script (VM shell)
```bash
ssh -i <key>.pem linux1@148.100.112.180
cd ~/shared/surgedesk && git pull
bash scripts/download_data.sh
```
It downloads HumAID Set 1 and the Census Gazetteer, and lists every HumAID event it finds. Copy that event list into the team chat.

### 2.2 MiniLM (container terminal, after Person 1 has the container up)
JupyterLab → File → New → Terminal:
```bash
cd /home/jovyan/shared/surgedesk && python scripts/download_minilm.py
```
It must end with "Load test OK, hidden size: 384".

### 2.3 CDC SVI (manual, on your laptop)
1. Open the CDC/ATSDR Social Vulnerability Index "Data & Documentation Download" page.
2. Select Year **2022** (or latest), Geography **United States**, Geography Type **Counties**, File Type **CSV**. Download.
3. Rename to `svi_counties.csv`, then: `scp -i <key>.pem svi_counties.csv linux1@148.100.112.180:~/shared/surgedesk/data/raw/svi/`
4. Check it has columns `STCNTY` (or `FIPS`) and `RPL_THEMES`: `head -1 ~/shared/surgedesk/data/raw/svi/svi_counties.csv`.

### 2.4 Optional: Kaggle Disaster Tweets
Only for Agent 1's optional task 1.9. Unzip `nlp-getting-started.zip` and copy `train.csv` to `~/shared/surgedesk/data/raw/kaggle_disaster_tweets/`.

## 3. Choose the demo event (1:45 PM)
- First choice: **`hurricane_harvey_2017`** (Texas, 9,164 labeled tweets, Aug 25 to Sep 1, 2017). It's the default in `config.py`.
- If Harvey is **not** in Set 1, pick the first available of: `hurricane_florence_2018`, `hurricane_irma_2017`, `hurricane_dorian_2019`, `hurricane_matthew_2016` (US hurricanes, so place matching and SVI work). Tell Person 1 → Agent 4 changes `DEMO_EVENT` in `config.py` → everyone pulls.
- The demo event is never used for training. All other Set 1 events train the models.

## 4. Laptop data pack (2:00 PM)
Teammates need the same data locally so their agents can develop on real files. From the VM (VM shell):
```bash
cd ~/shared/surgedesk
tar -czf ~/surgedesk_datapack.tar.gz data/raw/humaid data/raw/gazetteer data/raw/svi models/minilm
```
On your laptop: `scp -i <key>.pem linux1@148.100.112.180:~/surgedesk_datapack.tar.gz .` and share it in a **private** team folder (Google Drive), not in git. Each teammate extracts it at their repo root: `tar -xzf surgedesk_datapack.tar.gz`. (HumAID text must not go into the repo or anywhere public.)

## 5. Hand labeling (5:00 to 7:15 PM)
1. Person 1 sends you `data/labels/to_label.csv` (100 rows, masked text, `labeler` pre-filled P1 to P4).
2. Import it into a private Google Sheet. Freeze the header row. Paste the guide below into a second tab.
3. Each person labels their 25 rows (about 30 to 40 minutes). Don't look at model outputs while labeling.
4. When all 100 are done: File → Download → CSV. Rename to `hand_labels.csv` and upload to the VM at `~/shared/surgedesk/data/labels/hand_labels.csv` (JupyterLab upload button or `scp`). Keep the exact column names: `report_id, text, hazard, location, people_affected, vulnerable, need, labeler, notes`. Make sure `report_id` stays a plain number (format the column as plain text so Sheets doesn't turn it into 9.01746E+17).

**Labeling guide (copy into the sheet)**
- **hazard**: the physical threat in the text (flooding, fire, wind damage, collapse, outage). Copy the words used. Empty if none.
- **location**: the most specific place named (street, neighborhood, city). Copy as written. Empty if none.
- **people_affected**: who or how many are affected ("family of 5", "my neighbors"). Empty if none.
- **vulnerable**: any elderly, disabled, medical, pregnant, or child mention, copied as written. Empty if none.
- **need**: what is being asked for (rescue, water, medicine, shelter, information). Empty if none.
- Don't infer. If it isn't in the text, leave it empty. Use `notes` for anything unclear.

## 6. Purity check (after Checkpoint 2, ~9:00 PM, 20 minutes)
Person 1 runs `python -c "from surgedesk.evaluate import make_purity_sheet; print(make_purity_sheet())"` and sends you `purity_to_label.csv`. For each row, fill `same_incident` with `y` if the report belongs with the others in its incident, `n` if not, plus your name in `labeler`. Save as `purity_labels.csv` and upload to `~/shared/surgedesk/data/labels/`.

## 7. Reviewing Agent 1 (all afternoon)
Check against `handoffs/agent1_tag.md` section 13. After Person 1's VM run, post: number of events, reports per event, embedding reports/sec, demo-event macro-F1, place coverage of actionable reports. If place coverage is below ~20%, flag it to Person 4 (it's a limitation for the pitch, and Agent 2's grouping relies more on text similarity).

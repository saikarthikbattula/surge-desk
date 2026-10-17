# Agent 2 handoff: Stages 2 and 3 (Group, Rank, Fairness, Baselines)

Human driver: **Person 4**. Read `AGENTS.md` first, then this file top to bottom. If something is missing, stop and ask your human instead of guessing.

## 1. Your job in one paragraph
Merge tagged reports into incidents in real time (online clustering), compute snapshot features, train an XGBoost ranker on other disasters, score and explain every incident at every snapshot of the demo event, apply the fairness adjustment, and provide the three baseline orderings that evaluation compares against. Your `incidents.jsonl` is what the UI shows and what Agent 3 briefs.

## 2. What already exists (do not edit)
| File | What it gives you |
|---|---|
| `surgedesk/config.py` | `WINDOW_HOURS, SIM_PLACE, SIM_NOPLACE, MAX_CANDIDATES, SNAPSHOT_MIN, LAMBDAS, HIGH_SVI, TOP_K_EVAL, RANKER_PATH, RANKER_META_PATH, DEMO_EVENT, SEED, N_THREADS, EMBED_DIM`, `event_file()`, `train_events()` |
| `surgedesk/contracts.py` | `Report, Tagged, Incident, FEATURE_KEYS, FEATURE_PLAIN, URGENT_LABELS, URGENT_KEYWORDS, ContractError` |
| `surgedesk/stubs.py` | **The exact definitions you reuse**: `reference_features(reports, tags, now, svi)`, `reference_incident_location(tags)`, `reference_urgent_proxy(reports)`, `snapshot_times(start, end)`, `find_terms(text, terms)`. Call these; don't reimplement them |
| `scripts/make_fixtures.py` | `reference_group()` and `build_incidents()`: executable reference for grouping and snapshot rows. Your output must match the fixtures exactly |
| `fixtures/outputs/fixture_event/` and `fixture_train_event/` | `reports.jsonl, tagged.jsonl, embeddings.npy, groups_expected.jsonl, incidents.jsonl` (incidents use a FIXTURE score, not a model; compare membership/features/order rules, not scores) |
| `surgedesk/fileio.py`, `surgedesk/textutil.py` | `read_jsonl, write_jsonl, load_embeddings, write_json`, `parse_iso, to_iso, minutes_between` |

You depend on Agent 1 only through files (`reports.jsonl`, `tagged.jsonl`, `embeddings.npy`) and one lookup function (SVI). Every function below takes those as parameters so your tests run on fixtures without Agent 1's code.

## 3. Files you own (create these)
```
surgedesk/group.py
surgedesk/features.py
surgedesk/rank.py
surgedesk/fairness.py
surgedesk/baselines.py
notebooks/02_group_rank.ipynb
tests/test_agent2_group_rank.py
```

## 4. Grouping algorithm (exact)
Process reports in file order (already sorted by timestamp, report_id). Skip reports whose tag has `is_actionable == False` (they never join an incident).
For each actionable report `r` with tag `t` and embedding `e` (L2-normalized):
1. **Candidates** = incidents whose `last_seen` is within `WINDOW_HOURS` before `r`'s time (`(t_r - last_seen) <= WINDOW_HOURS * 3600` seconds), sorted by `last_seen` descending, first `MAX_CANDIDATES`.
2. For each candidate, compute its current location with `reference_incident_location(candidate.tags)` → `(iplace, icounty)`.
   - `rep_has` = report has any place or county; `inc_has` = incident has place or county.
   - `place_match` = (`t.county_fips` and `icounty` and they're equal) or (`t.places` and `iplace` and `t.places[0].lower() == iplace.lower()`).
   - If `rep_has and inc_has and not place_match` → skip this candidate (different places never merge).
   - Threshold = `SIM_PLACE` if `place_match` else `SIM_NOPLACE`.
   - Similarity = dot(`e`, normalized sum of the candidate's member embeddings).
3. Choose the eligible candidate with the highest similarity ≥ its threshold. Strictly greater wins, so on exact ties the earlier candidate in the sorted list (most recent `last_seen`) is kept.
4. If none: create a new incident `"<event>-inc-<k>"`, k = 1, 2, ... in creation order.
5. Add the report: append to members, add `e` to the float64 sum vector, set `last_seen` = report time.

Expected on fixtures: `fixture_event` → 10 incidents (sizes 4, 3, 3, then seven singletons); `fixture_train_event` → 8 incidents. Exact memberships are in `groups_expected.jsonl`.

**Real-data note.** HumAID is a *sampled and de-duplicated* set (Harvey: ~9K tweets over 8 days), so you will see many singletons and fewer big clusters than in a live feed. That's expected. Report the numbers honestly; do not lower thresholds just to make bigger clusters. You may tune `SIM_PLACE`/`SIM_NOPLACE` only via the 20-incident spot check in section 9, and only by telling Person 1 to change `config.py` (you don't own it).

## 5. Snapshots and features (exact)
- **Snapshot grid** for an event: `snapshot_times(first report time, last report time)` from `stubs.py` (every `SNAPSHOT_MIN` minutes, floored start, last tick ≥ end).
- An incident is **open at snapshot s** if it has ≥ 1 member with timestamp ≤ s and `s - (latest member ≤ s) <= WINDOW_HOURS`.
- For each open incident at s: members ≤ s → `reference_features(reports, tags, s, svi)` where `svi = svi_lookup(county)` and `(place, county) = reference_incident_location(tags_until_s)`.
- `urgent_proxy` = `reference_urgent_proxy(reports_until_s)` (None if no gold labels exist, e.g. live data).
- Gold labels are used **only** for `urgent_proxy`. Never as features. Add `assert "gold" not in " ".join(FEATURE_KEYS)` and never pass `gold_label` into any feature code.

## 6. Exact interfaces

### 6.1 `surgedesk/group.py`
```python
SVILookup = Callable[[str | None], float | None]

@dataclass
class IncidentState:
    incident_id: str
    event: str
    reports: list[Report]          # time order
    tags: list[Tagged]
    sum_vec: numpy.ndarray         # float64 [384]
    last_seen: datetime
    @property
    def first_seen(self) -> datetime: ...
    @property
    def centroid(self) -> numpy.ndarray: ...   # sum_vec / norm, float32
    def members_until(self, now: datetime) -> tuple[list[Report], list[Tagged]]: ...
    def location(self, now: datetime | None = None) -> tuple[str | None, str | None]:
        # reference_incident_location over tags (until now if given)

class GroupState:
    def __init__(self, event: str): ...
        # self.incidents: dict[str, IncidentState] (creation order); self.assignments: dict[report_id, incident_id]
    def add(self, report: Report, tag: Tagged, emb: numpy.ndarray) -> str | None:
        """Section 4. Returns the incident_id, or None for non-actionable reports."""
    def open_incidents(self, now: datetime) -> list[IncidentState]:
        """Section 5 'open' rule, in creation order."""

def load_stage1(event: str, src_dir: Path | None = None) -> tuple[list[Report], list[Tagged], numpy.ndarray]:
    """Read reports.jsonl, tagged.jsonl, embeddings.npy from src_dir (default config.event_dir(event)).
    Assert same length, same report_id order, tagged[i].emb_row == i, embeddings rows == len(reports)."""

def group_event(event: str, src_dir: Path | None = None, out_dir: Path | None = None) -> GroupState:
    """load_stage1 + add every report in order. Write <out_dir>/groups.jsonl
    ({incident_id, report_ids} per incident, creation order) and <out_dir>/group_stats.json:
    {n_reports, n_actionable, n_incidents, singleton_share, mean_size, max_size, size_histogram {size: count}}.
    out_dir defaults to config.event_dir(event). Print the stats."""
```

### 6.2 `surgedesk/features.py`
```python
def snapshot_features(inc: IncidentState, now: datetime, svi_lookup: SVILookup) -> dict[str, float] | None:
    """None if the incident has no members <= now; else reference_features(...) as in section 5."""

def build_snapshots(event: str, state: GroupState, reports: list[Report], svi_lookup: SVILookup,
                    out_dir: Path | None = None) -> pandas.DataFrame:
    """One row per (snapshot, open incident). Columns, in order:
    incident_id, snapshot_time (ISO), place, county_fips, svi, first_seen, last_seen (members <= s, ISO),
    report_ids (JSON string list), *FEATURE_KEYS, urgent_proxy.
    Rows sorted by snapshot_time then incident creation order. Save <out_dir>/snapshots.csv (index=False).
    `reports` gives the event's time span for the grid."""
```

### 6.3 `surgedesk/rank.py`
```python
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", random_state=config.SEED)
# n_jobs=config.N_THREADS and scale_pos_weight=neg/pos are set at fit time.

def default_svi_lookup() -> SVILookup:
    """Lazy: from surgedesk.geo import SVI; return SVI().get. If geo isn't importable yet, return lambda f: None
    and print a warning (keeps your work unblocked before Agent 1 merges)."""

def train_ranker(train_events: list[str], svi_lookup: SVILookup | None = None,
                 outputs_root: Path = config.OUTPUTS, ranker_path: Path = config.RANKER_PATH,
                 allow_fixture: bool = False) -> dict:
    """assert config.DEMO_EVENT not in train_events. For each event: group_event(src from outputs_root/<event>),
    build_snapshots, keep rows with urgent_proxy not None. X = rows[FEATURE_KEYS], y = urgent_proxy.
    Dev metrics: hold out 20% of incident_ids (sklearn GroupShuffleSplit, groups = event + incident_id,
    random_state=SEED); fit, compute dev ROC-AUC and average precision (None if a split has one class);
    then refit on all rows. Save the booster JSON with model.save_model(ranker_path) and write
    <ranker_path>.parent / "ranker_meta.json": {feature_keys, train_events, n_rows, pos_rate, dev_auc,
    dev_ap, params, trained_at}. Return the meta dict."""

class Ranker:
    def __init__(self, path: Path = config.RANKER_PATH): ...
        # xgboost.Booster().load_model(path); read ranker_meta.json; assert meta feature_keys == FEATURE_KEYS
    def score(self, X: pandas.DataFrame) -> numpy.ndarray:
        # booster.predict(DMatrix(X[FEATURE_KEYS].astype(float).values, feature_names=FEATURE_KEYS)); clip to [0, 1]
    def explain(self, X: pandas.DataFrame, k: int = 5) -> list[list[list]]:
        # pred_contribs=True, drop the last column (bias); per row: top k [feature, round(contrib, 6)] by |contrib| desc

def rows_to_incidents(event: str, rows: pandas.DataFrame, scores, explanations) -> list[Incident]:
    """One snapshot's rows -> Incident records with lam=0.0, score_adj=score, rank=rank_adj by
    score desc, ties by first_seen then incident_id. Scores rounded to 6 decimals. Validate each."""

def rank_now(state: GroupState, now: datetime, ranker: Ranker, svi_lookup: SVILookup,
             lam: float = 0.0) -> list[Incident]:
    """Live API (demo + benchmark): features for state.open_incidents(now), score, explain,
    rows_to_incidents, then fairness.apply(lam). urgent_proxy from reference_urgent_proxy (None for live data).
    Returns sorted by rank_adj."""

def rank_event(event: str = config.DEMO_EVENT, force: bool = False, svi_lookup: SVILookup | None = None,
               src_dir: Path | None = None, out_dir: Path | None = None,
               ranker: Ranker | None = None) -> list[Incident]:
    """group_event + build_snapshots + score/explain every snapshot -> <out_dir>/incidents.jsonl
    (all snapshots, lam=0.0). Skip if the file exists and not force. Return all incidents."""

def pick_demo_snapshot(incidents: list[Incident]) -> str:
    """os.environ['SURGEDESK_SNAPSHOT'] if set; else the snapshot_time with the largest
    sum of features['reports_last_60m'] across its incidents; ties -> earliest."""
```

### 6.4 `surgedesk/fairness.py`
```python
def adjust(score: float, svi: float | None, lam: float) -> float:
    """round(min(1.0, score * (1 + lam * svi)), 6) if svi is not None, else round(score, 6)."""

def apply(incidents: list[Incident], lam: float) -> list[Incident]:
    """For ONE snapshot. Returns new Incident objects (dataclasses.replace) with score_adj = adjust(...),
    lam = lam, rank_adj = 1.. by score_adj desc, ties by rank. Input order/objects unchanged."""

def fairness_table(incidents: list[Incident], lams: list[float] = config.LAMBDAS,
                   k: int = config.TOP_K_EVAL) -> pandas.DataFrame:
    """For ONE snapshot. Columns: lam, n_incidents, precision_at_k (mean urgent_proxy of top k by rank_adj;
    NaN if urgent_proxy is None), high_svi_share_top_k (share of top k with svi >= HIGH_SVI; unknown svi
    counts as not high), unknown_svi_share_top_k, jaccard_vs_lam0 (top-k incident_id sets)."""
```

### 6.5 `surgedesk/baselines.py`
```python
def order_fifo(incidents: list[Incident]) -> list[str]:       # first_seen asc, then incident_id
def order_volume(incidents: list[Incident]) -> list[str]:     # n_reports desc, then first_seen, incident_id
def order_keyword(incidents: list[Incident], texts_by_id: dict[str, str]) -> list[str]:
    # score = sum over members of len(find_terms(text, URGENT_KEYWORDS)); desc, then first_seen, incident_id
def order_surgedesk(incidents: list[Incident]) -> list[str]:  # rank_adj asc
```
All take the incidents of ONE snapshot and return incident_ids in order.

## 7. Pipeline calls you must support
```python
rank.train_ranker(config.train_events(events))            # stage "ranker"
rank.rank_event(config.DEMO_EVENT, force=force)            # stage "rank"
rank.pick_demo_snapshot(incidents)                         # stage "brief" (Agent 4) and the UI
fairness.apply(incidents_at_snapshot, lam)                 # UI slider, brief stage, evaluation
baselines.order_*(...)                                     # evaluation (Agent 4)
GroupState / rank_now                                      # live replay + benchmark (Agent 4)
```

## 8. Tests you must write: `tests/test_agent2_group_rank.py`
Use `from surgedesk.testing import skip, load_fixture, fixture_dir`.
1. `test_grouping_matches_fixture`: for both fixture events, `group_event(e, src_dir=fixture_dir(e), out_dir=<temp>)` memberships equal `groups_expected.jsonl` exactly (same incident ids, same report_ids in order).
2. `test_snapshots_match_fixture`: build snapshots for `fixture_event` with `svi_lookup` built from the fixture incidents (`{i.county_fips: i.svi}`); the set of `(incident_id, snapshot_time)` pairs equals the fixture `incidents.jsonl` pairs, and every feature equals the fixture value within 1e-6.
3. `test_open_rule`: an incident whose last member is more than `WINDOW_HOURS` before a snapshot is not open (construct a 2-report case).
4. `test_different_places_never_merge`: two reports with identical embeddings but different county_fips form two incidents.
5. `test_fairness_adjust_and_apply`: `adjust(0.5, 0.8, 1.0) == 0.9`, `adjust(0.9, 1.0, 1.0) == 1.0`, `adjust(0.5, None, 1.0) == 0.5`; `apply` on the fixture's last snapshot with lam=1.0 gives ranks 1..n with no gaps and doesn't mutate the input.
6. `test_baselines_fixture`: on the fixture's last snapshot, `order_fifo` starts with `fixture_event-inc-1`; every order is a permutation of the snapshot's incident ids.
7. `test_train_ranker_fixture`: skip if xgboost isn't installed; `train_ranker(["fixture_train_event"], svi_lookup=lambda f: None, outputs_root=config.FIXTURES / "outputs", ranker_path=<temp>/ranker.json, allow_fixture=True)` runs; `Ranker(<temp>/ranker.json)` scores fixture snapshot rows into [0, 1] and `explain` returns ≤ 5 valid pairs.
8. `test_rank_event_fixture`: with the temp ranker from test 7, `rank_event("fixture_event", force=True, svi_lookup=..., src_dir=fixture_dir(), out_dir=<temp>, ranker=...)` writes a valid `incidents.jsonl` (validate_file) and ranks within each snapshot are 1..n.
9. `test_no_gold_in_features`: monkeypatch every fixture report's `gold_label` to None; features are identical to before (only `urgent_proxy` changes to None).

Run: `python -m surgedesk.selftest -k agent2`.

## 9. Notebook `notebooks/02_group_rank.ipynb` (thin)
1. Title + purpose. 2. Bootstrap cell (same as `00_env_check.ipynb`).
3. `events = json.load(open(config.OUTPUTS / "events.json"))`; `meta = rank.train_ranker(config.train_events(events))` → display dev AUC/AP.
4. `incidents = rank.rank_event()` → display `group_stats.json` for the demo event.
5. `snap = rank.pick_demo_snapshot(incidents)`; table of the top 10 at `snap` (rank, place, county, svi, n_reports, score, top 3 reasons in plain English via `FEATURE_PLAIN`).
6. Baselines at `snap`: four columns side by side (SurgeDesk, FIFO, Volume, Keyword), top 10 each.
7. `fairness.fairness_table(...)` at `snap`.
8. **Spot check** (for threshold tuning): print 20 random multi-report incidents from a train event with their member texts (masked). Your human judges them; if most are wrong merges or obvious misses, propose new thresholds to Person 1.
Restart Kernel and Run All must pass.

## 10. Performance budget on the VM
- Grouping Harvey (~9K reports): under 2 minutes. Keep candidate lists short (`MAX_CANDIDATES`) and vectorize similarity with numpy (stack candidate centroids, one matrix-vector product).
- Snapshots: Harvey has ~380 snapshots; cache per-incident member timestamps so `members_until` is a bisect, not a scan.
- Ranker training on all train events: under 5 minutes. `rank_now` for ~300 open incidents: under 1 second.

## 11. Timeline (Central time)
| By | Milestone |
|---|---|
| 3:30 PM | group.py + features.py + fairness.py + baselines.py; tests 1 to 6 and 9 pass on fixtures |
| 4:30 PM | Checkpoint 1 (stub run on VM); rank.py + tests 7 and 8 pass on your laptop |
| 8:00 PM | After Agent 1's tag stage runs on the VM, Person 1 runs your stages `ranker` and `rank` |
| 8:30 PM | Checkpoint 2: real ranked list for the demo event, dev AUC, precision@10 vs baselines posted |
| 12:30 AM | Fairness + baselines final; spot check done |

## 12. Definition of done
- [ ] Section 6 interfaces implemented exactly; section 8 tests pass (`python -m surgedesk.selftest`).
- [ ] `02_group_rank.ipynb` passes Restart and Run All on your laptop with real data.
- [ ] Your human posted "Ready to run: branch `a2-rank`, notebook `notebooks/02_group_rank.ipynb`, expected outputs `models/ranker.json`, `outputs/<demo>/incidents.jsonl, snapshots.csv, groups.jsonl, group_stats.json`" and Person 1 confirmed it on the VM.
- [ ] You reported the VM numbers: incidents, singleton share, dev AUC/AP, demo snapshot time, top 10 at that snapshot.

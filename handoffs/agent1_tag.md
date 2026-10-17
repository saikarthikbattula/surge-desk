# Agent 1 handoff: Data + Stage 1 (Tag)

Human driver: **Person 2**. Read `AGENTS.md` first, then this file top to bottom. Everything you need is here; if something is missing, stop and ask your human instead of guessing.

## 1. Your job in one paragraph
Turn the raw HumAID disaster tweets into clean, time-ordered `Report` records, embed them with MiniLM (PyTorch), train a category classifier, and tag every report with a predicted category, matched places and county, and vulnerable-person terms. You also provide the live replay generator and a batch tagging API that the real-time demo and the benchmark use. Your outputs feed Agent 2 (grouping and ranking) directly.

## 2. What already exists (do not edit)
| File | What it gives you |
|---|---|
| `surgedesk/config.py` | All paths and constants. Use `config.HUMAID_DIR`, `config.GAZETTEER_DIR`, `config.SVI_PATH`, `config.MINILM_DIR`, `config.CLASSIFIER_PATH`, `config.event_file(event, name)`, `config.states_for_event(event)`, `config.train_events(events)`, `config.N_THREADS`, `config.EMBED_*`, `config.REPLAY_COMPRESSION`, `config.BATCH_SECONDS`, `config.DEMO_EVENT` |
| `surgedesk/contracts.py` | `Report`, `Tagged`, `CATEGORIES`, `NON_ACTIONABLE`, `VULN_TERMS`, `normalize_label()`, `is_actionable()`, `ContractError` |
| `surgedesk/fileio.py` | `read_jsonl`, `write_jsonl`, `save_embeddings`, `load_embeddings`, `write_json` |
| `surgedesk/textutil.py` | `tweet_id_to_datetime`, `to_iso`, `parse_iso`, `mask_privacy` |
| `surgedesk/stubs.py` | `find_terms(text, terms)` (use it for vulnerable terms so results match the fixtures exactly) |
| `scripts/make_fixtures.py` | Executable reference for place matching: `build_place_index()` and `match_places()`. Your `geo.py` must produce **identical** results on the fixtures |
| `fixtures/humaid/` | Two fake events in the real HumAID file layout (one test file uses Title Case labels on purpose) |
| `fixtures/gazetteer/`, `fixtures/svi/` | Tiny Census Gazetteer and SVI files with the real column layouts, including traps (Houston, MS; "Mission"; an SVI FIPS stored without its leading zero; a -999 value) |
| `fixtures/outputs/<event>/` | Expected `reports.jsonl` and `tagged.jsonl` for both fixture events |

## 3. Files you own (create these)
```
surgedesk/data_humaid.py
surgedesk/geo.py
surgedesk/embed.py
surgedesk/tag.py
surgedesk/replay.py
notebooks/01_tag.ipynb
tests/test_agent1_tag.py
```
Optional, only if ahead: `surgedesk/kaggle_check.py` (section 10).

## 4. Data you'll read (real format)
- **HumAID Set 1** under `data/raw/humaid/` (any nesting). Event files are named `<event>_train.tsv`, `<event>_dev.tsv`, `<event>_test.tsv`, tab-separated, header `tweet_id  tweet_text  class_label`. Example event names: `hurricane_harvey_2017`, `kerala_floods_2018`. Other `.tsv` files that don't end in `_train/_dev/_test` must be ignored. Labels may be snake_case or Title Case; always pass them through `normalize_label()`.
- Hurricane Harvey: 9,164 labeled tweets, collected 2017-08-25 to 2017-09-01 (use this to sanity-check timestamps).
- **Census Gazetteer**: `*_Gaz_place_national.txt` (columns `USPS GEOID ANSICODE NAME LSAD FUNCSTAT ALAND AWATER ALAND_SQMI AWATER_SQMI INTPTLAT INTPTLONG`) and `*_Gaz_counties_national.txt` (`USPS GEOID ANSICODE NAME ALAND AWATER ALAND_SQMI AWATER_SQMI INTPTLAT INTPTLONG`). Tab-separated. **The last header has trailing spaces**: strip every column name. Read everything as `str`.
- **CDC SVI county CSV** at `config.SVI_PATH`: columns include `ST, STATE, ST_ABBR, STCNTY, COUNTY, FIPS, LOCATION, RPL_THEMES`. FIPS may be stored as an integer (leading zero lost). `-999` means missing.
- **MiniLM** files in `config.MINILM_DIR` (downloaded by `scripts/download_minilm.py`).

## 5. Exact interfaces

### 5.1 `surgedesk/data_humaid.py`
```python
def discover_event_files(humaid_dir: Path = config.HUMAID_DIR) -> dict[str, list[Path]]:
    """Recursively find <event>_{train,dev,test}.tsv. Returns {event: [paths sorted by name]}, events sorted.
    Ignores any other .tsv. Raises FileNotFoundError (with the directory listed) if nothing is found."""

def load_event(event: str, files: list[Path], source: str = "humaid") -> list[Report]:
    """Read and merge the event's TSVs into Reports.
    - pandas.read_csv(f, sep="\\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, on_bad_lines="warn")
    - column names lower-cased and stripped; id column = first of ["tweet_id", "id"], text = ["tweet_text", "text"],
      label = ["class_label", "label"]; raise ContractError listing the columns found if any is missing
    - drop rows whose id is not all digits or whose text is empty after strip
    - gold_label = normalize_label(label)
    - deduplicate by report_id (keep first occurrence in file order train, dev, test)
    - timestamp = to_iso(tweet_id_to_datetime(report_id)); is_synthetic = (source == "synthetic")
    - return sorted by (timestamp, report_id)"""

def build_reports(events: list[str] | None = None, humaid_dir: Path = config.HUMAID_DIR,
                  force: bool = False, source: str = "humaid") -> pandas.DataFrame:
    """For every discovered event (or the given subset): write outputs/<event>/reports.jsonl
    (skip if it exists and force is False). Also write outputs/events.json (sorted list of event names)
    and outputs/events_summary.csv with columns: event, n_reports, first, last, states, label_counts (JSON).
    Print the summary table. Warn (don't fail) if any timestamp is outside 2010-2025.
    Returns the summary DataFrame."""
```

### 5.2 `surgedesk/geo.py`
```python
COMMON_WORD_PLACES: set[str] = {"mission", "victoria", "humble", "liberty", "hope", "rose", "star",
                                "union", "center", "friendship"}

class Gazetteer:
    def __init__(self, gaz_dir: Path = config.GAZETTEER_DIR): ...
        # loads the first *_Gaz_place_national.txt and *_Gaz_counties_national.txt found; FileNotFoundError if missing
    def place_index(self, states: list[str]) -> dict[str, tuple[str, str]]:
        """lowercase key -> (display name, 5-digit county FIPS). Algorithm = scripts/make_fixtures.py build_place_index:
        - places in `states`: display = NAME minus its trailing LSAD word(s) (" city and borough", " city", " town",
          " village", " CDP", " borough", " municipality", " (balance)"); county = nearest county centroid
          (haversine on INTPTLAT/INTPTLONG) within the SAME state; duplicate keys keep the larger ALAND
        - counties in `states`: key = NAME lowercased (e.g. "harris county", "calcasieu parish"), display = NAME,
          FIPS = GEOID; counties override places with the same key"""

class PlaceMatcher:
    def __init__(self, event: str, gazetteer: Gazetteer | None = None): ...
        # states = config.states_for_event(event); if [] the matcher always returns ([], None)
    def match(self, text: str) -> tuple[list[str], str | None]:
        """Algorithm = scripts/make_fixtures.py match_places:
        1. hashtags: '#HoustonFlood' -> ' Houston Flood ' (split CamelCase), then lowercase
        2. tokens = re.findall(r"[a-z0-9']+", text)
        3. scan left to right; at each position try n-grams of 4, 3, 2, 1 tokens; first key found wins;
           single-token keys in COMMON_WORD_PLACES count only if the next token (or next two) is a state
           abbreviation or full state name of the event's states (e.g. "victoria tx", "mission texas")
        4. after a hit, skip past it; keep display names deduplicated in order of first appearance
        Returns (places, county FIPS of places[0] or None)."""

class SVI:
    def __init__(self, path: Path = config.SVI_PATH): ...
        # FIPS column = "STCNTY" if present else "FIPS"; str, zfill(5). If values are longer than 5 digits
        # (tract file downloaded by mistake), county = first 5 digits and RPL_THEMES = mean over its tracts,
        # ignoring -999. RPL_THEMES < 0 -> None. If the file is missing: print one warning and return None always.
    def get(self, fips: str | None) -> float | None: ...
    @property
    def available(self) -> bool: ...
```

### 5.3 `surgedesk/embed.py`
```python
class Embedder:
    def __init__(self, model_dir: Path = config.MINILM_DIR, threads: int = config.N_THREADS, device: str = "cpu"): ...
        # transformers AutoTokenizer/AutoModel from the local dir (local_files_only=True), model.eval(),
        # torch.set_num_threads(threads). device "nnpa" only if stretch task 1.8 succeeds, else NotImplementedError.
    def encode(self, texts: list[str], batch_size: int = config.EMBED_BATCH) -> numpy.ndarray:
        """Input texts are passed through mask_privacy() first. Tokenize (truncation, max_length=config.EMBED_MAX_TOKENS,
        padding), torch.inference_mode(), masked mean pooling of last_hidden_state over attention_mask,
        L2-normalize. Returns float32 [len(texts), 384]. Empty input -> shape (0, 384)."""

def embed_event(event: str, force: bool = False, embedder: Embedder | None = None) -> numpy.ndarray:
    """Embed outputs/<event>/reports.jsonl in file order -> outputs/<event>/embeddings.npy (save_embeddings).
    Skip and load if the file exists with the right row count and force is False.
    Write outputs/<event>/embed_timing.json: {n, seconds, per_sec, threads, device}. Print per_sec."""
```

### 5.4 `surgedesk/tag.py`
```python
def find_vuln_terms(text: str) -> list[str]:
    return stubs.find_terms(text, VULN_TERMS)      # exactly this

def train_classifier(train_events: list[str], allow_fixture: bool = False,
                     classifier_path: Path = config.CLASSIFIER_PATH,
                     outputs_root: Path = config.OUTPUTS) -> dict:
    """assert config.DEMO_EVENT not in train_events; fixture events only if allow_fixture.
    Reads <outputs_root>/<event>/reports.jsonl and embeddings.npy (tests pass config.FIXTURES / "outputs").
    X = embeddings of all reports in train_events with a gold label; y = gold_label.
    Hold out 10% (sklearn train_test_split, stratify=y if every class has >= 2 rows, random_state=config.SEED)
    to compute dev macro-F1, then refit on ALL rows. If the stratified split raises ValueError, split without
    stratify; if there are fewer than 200 labeled rows (fixtures), skip the holdout and set dev_macro_f1 = None.
    Model:
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)
    Save with joblib to classifier_path: {"model", "classes", "train_events", "trained_at", "dev_macro_f1"}.
    Also write classifier_meta.json next to it (same minus "model"). Return the metrics dict."""

class Tagger:
    def __init__(self, event: str, classifier_path: Path = config.CLASSIFIER_PATH,
                 embedder: Embedder | None = None, matcher: PlaceMatcher | None = None): ...
        # Embedder is created lazily only if tag_batch is called without embeddings
    def tag_batch(self, reports: list[Report], embeddings: numpy.ndarray | None = None,
                  start_row: int = 0) -> tuple[list[Tagged], numpy.ndarray]:
        """Live and batch API. If embeddings is None, encode report texts. pred = argmax predict_proba,
        pred_conf = round(max prob, 6). emb_row = start_row + i. places/county from PlaceMatcher,
        vuln_terms from find_vuln_terms(text), is_actionable from contracts.is_actionable.
        Returns (tagged in input order, the embeddings used)."""

def tag_event(event: str, force: bool = False) -> list[Tagged]:
    """Requires reports.jsonl and embeddings.npy. Writes outputs/<event>/tagged.jsonl in the same order
    (tagged[i].report_id == reports[i].report_id, emb_row == i). Also writes outputs/<event>/tag_stats.json:
    {n, actionable_share, place_coverage_actionable (share of actionable with county_fips),
    vuln_share, category_counts}. Skip if tagged.jsonl exists with matching length and not force."""

def evaluate_classifier(event: str = config.DEMO_EVENT) -> dict:
    """Compare pred_category with gold_label on the event (never trained on). Write
    outputs/<event>/classifier_eval.json: {n, accuracy, macro_f1, per_class_f1 {label: f1},
    labels (order used), confusion (list of lists, rows = gold, cols = pred)}. Return it."""
```

### 5.5 `surgedesk/replay.py`
```python
def busiest_window_start(reports: list[Report], hours: float = 1.0) -> datetime:
    """Timestamp t (one of the report timestamps) maximizing the count of reports in [t, t + hours).
    Ties: earliest t. Reports are time-ordered."""

def iter_batches(reports: list[Report], speed: float | str, start: datetime | None = None,
                 duration_s: float | None = None, batch_seconds: float = config.BATCH_SECONDS,
                 clock=time.monotonic, sleep=time.sleep) -> Iterator[tuple[float, list[Report]]]:
    """Replay reports with timestamp >= start (default: first report) as micro-batches.
    Event time maps to wall time as  wall_offset = (t - start).total_seconds() / (config.REPLAY_COMPRESSION * speed).
    Every batch_seconds of wall time, yield (scheduled_wall_time, reports whose wall_offset falls in that slot);
    empty slots are skipped (not yielded). Sleep until each slot's end before yielding it.
    speed == "max": never sleep; each batch covers batch_seconds * config.REPLAY_COMPRESSION * 20 seconds of event
    time; scheduled_wall_time = clock() at yield. Stop after duration_s of wall time (speed modes) or after
    the same event-time span a 20x run would cover in duration_s (max mode), or when reports run out.
    clock/sleep are injectable so tests can use a fake clock. Every report is yielded at most once, in order."""
```

## 6. Pipeline calls you must support
Agent 4's `pipeline.py` will call, in this order (so these signatures must not change):
```python
data_humaid.build_reports(force=force)                         # stage "data"
embed.embed_event(e, force=force)  for e in events             # stage "embed"
tag.train_classifier(config.train_events(events))              # stage "classifier"
tag.tag_event(e, force=force)  for e in events                 # stage "tag"
tag.evaluate_classifier(config.DEMO_EVENT)                     # stage "tag"
```
and the live demo / benchmark use `Tagger(event).tag_batch(batch)` with `replay.iter_batches(...)`.

## 7. Tests you must write: `tests/test_agent1_tag.py`
Use `from surgedesk.testing import skip, load_fixture, fixture_dir`. Tests must be fast and offline.
1. `test_load_event_matches_fixture`: for both fixture events, `load_event(e, discover_event_files(config.FIXTURES / "humaid")[e], source="synthetic")` → `[r.to_dict() for r in ...]` equals the fixture `reports.jsonl` dicts exactly (Title Case labels included).
2. `test_discover_ignores_other_tsv`: create a temp dir with `x_train.tsv` and `labeled_5_set1.tsv`; only event `x` is found.
3. `test_place_matcher_matches_fixture`: `Gazetteer(config.FIXTURES / "gazetteer")`; for both fixture events, `PlaceMatcher(e, gaz).match(r.text)` equals `(t.places, t.county_fips)` from fixture `tagged.jsonl` for every report.
4. `test_place_matcher_traps`: for event "fixture_event": "Houston" maps to 48201 (not Houston, MS), "mission accomplished" → no place, "Mission TX flooding" → ["Mission"] / 48215, "#KatyFlood" → ["Katy"].
5. `test_svi_fixture`: `SVI(config.FIXTURES / "svi" / "svi_counties.csv")`: get("01001") == 0.39, get("48201") == 0.84, get("48039") is None, get("99999") is None, get(None) is None.
6. `test_vuln_terms_match_fixture`: `find_vuln_terms(text)` equals fixture `vuln_terms` for every fixture report.
7. `test_iter_batches_fake_clock`: with a fake clock/sleep, speeds 1, 20 and "max" yield every fixture report exactly once, in order, and `busiest_window_start` returns a fixture timestamp.
8. `test_embedder_shape`: `skip("MiniLM not downloaded")` if `config.MINILM_DIR` has no config.json; else encode 3 texts → shape (3, 384), float32, norms ≈ 1, and `encode([])` → (0, 384).
9. `test_tagger_fixture`: `train_classifier(["fixture_train_event"], allow_fixture=True, classifier_path=<temp file>, outputs_root=config.FIXTURES / "outputs")`, then `Tagger("fixture_event", classifier_path=tmp, matcher=PlaceMatcher("fixture_event", gaz))` tags fixture reports using fixture embeddings: every row validates, places/county/vuln/actionable consistency hold (category accuracy is not asserted). Skip if scikit-learn is missing.

Run: `python -m surgedesk.selftest -k agent1` → all pass (or skip only for missing local model files).

## 8. Notebook `notebooks/01_tag.ipynb` (thin)
Cells, in order:
1. Markdown title + one-line purpose.
2. Bootstrap cell, exactly as in `notebooks/00_env_check.ipynb` (adds repo root to `sys.path`, runs `envcheck.run()`).
3. `summary = data_humaid.build_reports()` → display summary.
4. `events = json.load(open(config.OUTPUTS / "events.json"))`; loop `embed.embed_event(e)` printing reports/sec.
5. `metrics = tag.train_classifier(config.train_events(events))` → display.
6. loop `tag.tag_event(e)`; display `tag_stats.json` for the demo event.
7. `ev = tag.evaluate_classifier()` → table of per-class F1 + a matplotlib confusion-matrix heatmap.
8. 10 sample tagged demo reports: masked text, pred_category, places, county, vuln terms.
9. Markdown: "Numbers above were produced on: <arch from envcheck>".
Restart Kernel and Run All must pass with no manual edits.

## 9. Performance budget on the VM (3 vCPU, 8 GB)
- Loading all Set 1 TSVs: under 1 minute.
- Embedding ~48K tweets: expect roughly 5 to 20 minutes on CPU. Batch, use `inference_mode`, cache to disk, never re-embed if the file exists. Print reports/sec so Person 1 can post it.
- Classifier training: under 2 minutes.
- `tag_batch` on 50 reports with precomputed embeddings: well under 1 second; with live embedding: a few seconds at most.
- Peak RAM under 3 GB (embeddings for 48K × 384 float32 ≈ 74 MB).

## 10. Stretch and optional tasks (only after sections 5 to 8 are done and merged)
**1.8 NNPA embeddings (timebox 45 minutes).** `torch_nnpa` is preinstalled in IBM's container. Inspect it on the VM via Person 1 (`python -m pip show -f torch_nnpa`, read its README/module docstrings) to learn the device name and usage. Do not guess the API. If it works: `Embedder(device="nnpa")`, and `embed.compare_devices(texts, n=512) -> dict` returning `{cpu_per_sec, nnpa_per_sec, max_abs_diff, min_cosine}` (cosine between CPU and NNPA vectors must be > 0.99). If it doesn't work in 45 minutes, write a short note in `docs/NNPA_NOTES.md` explaining what you tried and stop.

**1.9 Kaggle figurative check (only if ahead).** `surgedesk/kaggle_check.py: run(path=config.KAGGLE_DIR / "train.csv") -> dict`: embed Kaggle texts, run the classifier, report agreement between `pred_category == "not_humanitarian"` and `target == 0`, plus 10 example disagreements (masked). Never use Kaggle data for training. Write `outputs/kaggle_check.json`.

## 11. Rules recap (from AGENTS.md)
No new packages, no network calls, no hardcoded paths, never use `gold_label` as a model input, never train on `config.DEMO_EVENT`, mask privacy in anything displayed, notebooks thin, Restart and Run All must pass, never report a number you didn't compute.

## 12. Timeline (Central time)
| By | Milestone |
|---|---|
| 4:00 PM | 5.1, 5.2, 5.5 done; tests 1 to 7 pass on fixtures |
| 4:30 PM | Checkpoint 1: merged; Person 1 runs `bash run_all.sh --stub` on the VM |
| 6:00 PM | 5.3 and 5.4 done; tests 8 and 9 pass on your laptop with real MiniLM |
| 7:00 PM | Person 1 has run stages data, embed, classifier, tag on the VM; you've reviewed `classifier_eval.json` and `tag_stats.json` |
| 8:30 PM | Checkpoint 2 (with Agent 2's real ranking) |
| after | Stretch 1.8, then optional 1.9 |

## 13. Definition of done
- [ ] All files in section 3 exist with the exact interfaces in section 5.
- [ ] `python -m surgedesk.selftest` passes on your laptop.
- [ ] `01_tag.ipynb` passes Restart and Run All on your laptop with real data.
- [ ] Your human posted "Ready to run: branch `a1-tag`, notebook `notebooks/01_tag.ipynb`, expected outputs `outputs/<event>/{reports,tagged}.jsonl, embeddings.npy, models/classifier.joblib`" and Person 1 confirmed it on the VM.
- [ ] You told your human the real numbers from the VM run: event count, report counts, embedding reports/sec, demo-event macro-F1, place coverage.

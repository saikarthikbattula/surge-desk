# Agent 3 handoff: Stage 4 (Granite briefings)

Human driver: **Person 3**. Read `AGENTS.md` first, then this file top to bottom. If something is missing, stop and ask your human instead of guessing.

## 1. Your job in one paragraph
Load our own Granite 3.3 2B model (converted by Person 3) through llama.cpp, and for the top incidents of the demo snapshot: pick 1 or 2 representative reports, have Granite extract five fields with exact quoted evidence, reject anything that isn't literally in the reports, then have Granite write a 2 to 3 sentence briefing where every sentence is checked against the facts and rejected if unsupported. Fall back to a fixed template when nothing survives. You also produce Granite extractions on the hand-labeled reports for evaluation. Granite never decides ranking; it only explains.

## 2. What already exists (do not edit)
| File | What it gives you |
|---|---|
| `surgedesk/config.py` | `GRANITE_PATH` (env `SURGEDESK_GRANITE` overrides), `GRANITE_CTX, EXTRACT_MAX_TOKENS, BRIEF_MAX_TOKENS, TOP_N_BRIEF, N_THREADS, SEED, LLM_LOG_FILE, BRIEFINGS_FILE` |
| `surgedesk/contracts.py` | `Briefing, Incident, Report, Tagged, BRIEF_FIELDS, SENTENCE_TAGS, EXTRACTION_SCHEMA, FEATURE_PLAIN, empty_fields(), zero_timing()` |
| `surgedesk/stubs.py` | `template_briefing_text(incident, fields)`: the official template. Your `template_briefing` must return exactly this |
| `surgedesk/textutil.py` | `mask_privacy, normalize_for_match, content_words` |
| `scripts/make_fixtures.py` | `select_representatives()`: executable reference for section 4 |
| `fixtures/outputs/fixture_event/` | `briefings.jsonl` shows correct records, including one rejected sentence and template fallbacks |

## 3. Files you own (create these)
```
surgedesk/llm.py
surgedesk/grounding.py
surgedesk/brief.py
notebooks/03_brief.ipynb
tests/test_agent3_brief.py
```

## 4. Representative selection (exact)
For an incident at a snapshot: centroid = normalized sum of its members' embeddings (rows from `Tagged.emb_row`). `first` = member with the highest dot product with the centroid (ties: earliest in `report_ids`). `second` = member with the most `vuln_terms`, ties broken by longest text, then earliest. Return `[first]` if they're the same report, else `[first, second]`. Must equal the fixture `rep_report_ids` for every fixture briefing.

## 5. Prompts (use exactly; Granite only ever sees masked text)
```python
EXTRACT_SYSTEM = (
    "You extract facts from disaster reports for an emergency coordinator. Use ONLY the report text. "
    "For each field, copy the shortest exact phrase from a report as evidence. If a report does not state "
    "a field, set value and evidence to null and confidence to 0. Never guess. Return JSON only."
)
EXTRACT_USER = (
    "Fields:\n"
    "- hazard: the physical threat (flooding, fire, wind damage, collapse, outage)\n"
    "- location: the most specific place named\n"
    "- people_affected: who or how many people are affected\n"
    "- vulnerable: any elderly, disabled, medical, pregnant, or child mention\n"
    "- need: what is being asked for\n\n"
    "Reports:\n{reports}\n\n"
    "Return JSON with keys hazard, location, people_affected, vulnerable, need. "
    "Each is an object with value, evidence, confidence (0 to 1)."
)   # {reports} = lines "[1] <masked text>", "[2] <masked text>"

BRIEF_SYSTEM = (
    "You write short briefings for an emergency coordinator. Use ONLY the facts given. Write 2 or 3 sentences. "
    "End every sentence with tags in square brackets naming the facts it uses, chosen from: "
    "hazard, location, people_affected, vulnerable, need, reason. Do not add any fact that is not given."
)
BRIEF_USER = (
    "Facts:\nhazard: {hazard}\nlocation: {location}\npeople_affected: {people_affected}\n"
    "vulnerable: {vulnerable}\nneed: {need}\nnumber of reports: {n_reports}\n"
    "reasons it ranked high: {reasons}\n\nWrite the briefing."
)   # unknown fields shown as "unknown"; reasons = up to 3 FEATURE_PLAIN names of positive shap_top entries joined by "; "
```

## 6. Exact interfaces

### 6.1 `surgedesk/llm.py`
```python
class GraniteLLM:
    def __init__(self, model_path: Path = config.GRANITE_PATH, n_threads: int = config.N_THREADS,
                 n_ctx: int = config.GRANITE_CTX, verbose: bool = False, log_path: Path | None = None): ...
        # import llama_cpp lazily; FileNotFoundError with a message pointing to handoffs/person3_granite.md
        # if the file is missing. Llama(model_path=str(...), n_ctx=n_ctx, n_threads=n_threads,
        # n_threads_batch=n_threads, seed=config.SEED, verbose=verbose). Record self.load_s and self.model_file (name only).
    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int) -> tuple[dict | None, dict]:
        """temperature=0.0. Attempt 1: create_chat_completion(..., response_format={"type": "json_object", "schema": schema}).
        If that raises TypeError/ValueError (unsupported), attempt 2: grammar=LlamaGrammar.from_json_schema(json.dumps(schema)).
        Parse the content with json.loads; on failure retry once with "Return only valid JSON." appended to user.
        Returns (parsed or None, timing) with timing = {seconds, prompt_tokens, completion_tokens, ok (bool), attempts}.
        Append one line per call to log_path (if set): {kind: "json", timing, error or null}."""
    def chat_text(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        """temperature=0.0, same timing dict, same logging."""

def get_llm(**kwargs) -> GraniteLLM:   # module-level singleton; loads once per process
def unload() -> None:                  # drop the singleton and gc.collect() (frees ~3 GB RAM)
```

### 6.2 `surgedesk/grounding.py`
```python
def evidence_ok(evidence: str | None, sources: list[str]) -> bool:
    """True iff evidence is a str with >= 2 non-space chars and normalize_for_match(evidence) is a substring of
    normalize_for_match(s) for some source s. Sources are the MASKED texts Granite saw."""

def ground_fields(raw: dict | None, sources: list[str]) -> dict[str, dict]:
    """For every BRIEF_FIELDS name: take raw[name] (missing/non-dict -> treated as null). confidence = float clamped
    to [0, 1] (non-numeric -> 0.0). If evidence_ok(evidence): {"value": evidence.strip(), "evidence": evidence.strip(),
    "confidence": c, "grounded": True}  (value is REPLACED by the quote so every shown value is verifiable).
    Else: {"value": None, "evidence": None, "confidence": 0.0, "grounded": False}. Key order = BRIEF_FIELDS."""

def split_tagged_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Parse 'Sentence one. [location, need] Sentence two [reason].' into
    [("Sentence one.", ["location", "need"]), ("Sentence two.", ["reason"])].
    Use re.finditer(r"(.+?)\\s*\\[([^\\[\\]]*)\\]\\s*\\.?", text, re.S); sentence stripped and given a final period if it
    has none; tags lower-cased, stripped, split on commas, empties dropped. Leftover text after the last match
    with >= 3 words becomes one more sentence with tags []."""

def numbers_allowed(incident: Incident, fields: dict) -> set[str]:
    """{str(int(features['n_reports'])), str(int(features['reports_last_60m']))} plus every digit run
    in grounded field values."""

def check_sentence(sentence: str, tags: list[str], fields: dict, reasons: list[str],
                   numbers: set[str]) -> tuple[bool, str | None]:
    """In this order:
    - no tags -> (False, "no tags")
    - a tag not in SENTENCE_TAGS -> (False, f"unknown tag '{tag}'")
    - for each field tag: value None -> (False, f"tagged field '{tag}' is unknown");
      cw = content_words(value); if cw and share of cw present in content_words(sentence) < 0.5 ->
      (False, f"tagged field '{tag}' not supported by the sentence")
    - tag 'reason': supported iff some reason phrase has >= 0.5 of its content words in the sentence, OR the
      sentence contains a number token in `numbers`; else (False, "reason not supported")
    - any digit run in the sentence not in `numbers` -> (False, f"unsupported number '{n}'")
    - else (True, None)"""
```

### 6.3 `surgedesk/brief.py`
```python
def select_representatives(incident: Incident, reports_by_id: dict[str, Report],
                           tags_by_id: dict[str, Tagged], embeddings: numpy.ndarray) -> list[Report]: ...   # section 4

def extract_fields(llm: GraniteLLM, reports: list[Report]) -> tuple[dict, dict]:
    """Masked texts -> EXTRACT prompts -> llm.chat_json(..., EXTRACTION_SCHEMA, config.EXTRACT_MAX_TOKENS)
    -> ground_fields(parsed, masked texts). Returns (fields, timing)."""

def write_briefing(llm: GraniteLLM, incident: Incident, fields: dict) -> tuple[list[dict], dict]:
    """BRIEF prompts -> chat_text(..., config.BRIEF_MAX_TOKENS) -> split_tagged_sentences -> check_sentence each.
    Returns (sentences as {text, tags, accepted, reject_reason}, timing)."""

def template_briefing(incident: Incident, fields: dict) -> str:
    return stubs.template_briefing_text(incident, fields)      # exactly this

def cache_key(incident: Incident, model_file: str) -> str:
    """sha1('|'.join([incident_id] + sorted(report_ids) + [model_file])).hexdigest()"""

def brief_incidents(event: str, incidents: list[Incident], top_n: int = config.TOP_N_BRIEF,
                    force: bool = False, llm: GraniteLLM | None = None,
                    src_dir: Path | None = None, out_dir: Path | None = None) -> list[Briefing]:
    """`incidents` = ONE snapshot, already fairness-applied; brief the top_n by rank_adj.
    Load reports/tagged/embeddings from src_dir (default event dir). Existing <out_dir>/briefings.jsonl entries
    with the same cache_key are reused unless force. For each new incident: representatives -> extract_fields ->
    write_briefing; briefing = accepted sentences joined by spaces, or template if none (used_fallback=True).
    Rewrite briefings.jsonl after EVERY incident (atomic write_jsonl of old + new) so progress survives a crash.
    If the Granite file is missing: print a loud warning and produce template briefings with
    model_file="template", fields=empty_fields(), sentences=[], used_fallback=True (pipeline keeps going).
    Calls are logged to <out_dir>/llm_calls.jsonl. Returns the briefings for these incidents."""

def extract_for_eval(event: str, report_ids: list[str], force: bool = False, llm: GraniteLLM | None = None,
                     src_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Run extract_fields on each single report (one report per call) for the hand-label evaluation.
    Write <out_dir>/extraction_preds.jsonl: {report_id, fields, timing} per line (resume-safe: skip ids already
    present unless force). Return the path."""
```

## 7. Pipeline calls you must support
```python
# stage "brief" (Agent 4 calls):
snap = rank.pick_demo_snapshot(incidents)
at_snap = fairness.apply([i for i in incidents if i.snapshot_time == snap], lam=0.0)
brief.brief_incidents(config.DEMO_EVENT, at_snap)
# stage "eval" (Agent 4 calls, only if data/labels/hand_labels.csv exists):
brief.extract_for_eval(config.DEMO_EVENT, report_ids_from_hand_labels)
# benchmark (Agent 4): GraniteLLM(...), extract_fields, write_briefing timing
```

## 8. Tests you must write: `tests/test_agent3_brief.py`
No test may load Granite unless its name ends in `_granite` (those run only with `python -m surgedesk.selftest --granite`).
1. `test_representatives_match_fixture`: for every fixture briefing, `select_representatives` returns ids equal to `rep_report_ids`.
2. `test_evidence_ok`: true for an exact (case/whitespace-different) substring of a masked source; false for a paraphrase, for "", for None, for a 1-char string.
3. `test_ground_fields_rejects_fabrication`: raw with one real quote and one invented quote → real field grounded with value == evidence; invented field all-null, grounded False; missing fields null; output validates inside a `Briefing`.
4. `test_split_tagged_sentences`: the examples in the docstring, plus a sentence without tags.
5. `test_check_sentence_fixture`: for the fixture briefing of `fixture_event-inc-1`, re-checking each fixture sentence with `check_sentence` reproduces its `accepted` and `reject_reason` exactly (use `numbers_allowed` and the FEATURE_PLAIN names of positive shap_top entries as reasons).
6. `test_check_sentence_rules`: unknown tag rejected; unsupported number ("12 people") rejected when 12 isn't allowed; a reason sentence with an allowed number accepted.
7. `test_template_matches_stub`: `template_briefing` == `stubs.template_briefing_text` for every fixture incident at the last snapshot with `empty_fields()`.
8. `test_brief_incidents_without_model`: with `config.GRANITE_PATH` pointing to a non-existent file (pass an `llm=None` and monkeypatch the path), `brief_incidents("fixture_event", <last-snapshot fixture incidents>, top_n=3, src_dir=fixture_dir(), out_dir=<temp>)` writes 3 valid template briefings.
9. `test_extract_real_granite_granite` (runs only with `--granite`): load Granite from `SURGEDESK_GRANITE`, extract fields for the first fixture briefing's representatives; all returned fields validate; at least one field is grounded; prints seconds per call.

## 9. Notebook `notebooks/03_brief.ipynb` (thin)
1. Title + purpose. 2. Bootstrap cell.
3. Load Granite with `verbose=True` **in its own cell** (its stderr shows the llama.cpp backends; Person 1 screenshots any zDNN/NNPA lines). Print `load_s`, file name, file size.
4. Speed test: `extract_fields` on 5 demo-event reports one at a time → table of seconds and completion tokens; print mean seconds per call and tokens/sec; print process RSS via `psutil` (preinstalled).
5. `incidents = read_jsonl(config.event_file(DEMO, config.INCIDENTS_FILE), Incident)`; `snap = rank.pick_demo_snapshot(incidents)`; `at_snap = fairness.apply(...)`; `briefings = brief.brief_incidents(DEMO, at_snap)`.
6. Display each briefing: rank, place, fields with evidence quotes, accepted sentences, rejected sentences with reasons, `used_fallback`.
7. Summary counts: fields grounded / total, sentences accepted / total, fallback count, mean seconds per incident.
8. If `data/labels/hand_labels.csv` exists: `brief.extract_for_eval(DEMO, ids)`; else a markdown note that it runs after labeling.
9. `llm.unload()` at the end.
Restart Kernel and Run All must pass.

## 10. Development setup on your laptop
Person 3 gives you `granite-3.3-2b-instruct-le.q8_0.gguf` (little-endian). Put it in `models/granite/` and run with `SURGEDESK_GRANITE=models/granite/granite-3.3-2b-instruct-le.q8_0.gguf`. `pip install llama-cpp-python==0.3.36` on your laptop only. On the VM, Person 1 uses the big-endian file at the default `config.GRANITE_PATH`.
Until the file arrives, build and test everything with a fake LLM object that has the same `chat_json`/`chat_text` methods (put it in your test file), so you're never blocked.

## 11. Budget on the VM
Granite 2B Q8_0 on 3 vCPU: unknown until measured (expect seconds to tens of seconds per call). Only one process may hold Granite (~3 GB RAM). `TOP_N_BRIEF` (default 10, env `SURGEDESK_TOP_N`) is set at Checkpoint 2 from the measured speed so the brief stage finishes in under ~20 minutes. Keep `max_tokens` as configured; shorter outputs are faster.

## 12. Timeline (Central time)
| By | Milestone |
|---|---|
| 4:00 PM | grounding.py + brief.py with a fake LLM; tests 1 to 8 pass |
| 4:30 PM | Checkpoint 1 merge |
| 5:30 PM | llm.py works on your laptop with the LE file; test 9 passes with `--granite` |
| 6:00 PM | Person 1 runs `03_brief.ipynb` cells 2 to 4 on the VM with the BE file: load OK, speed posted |
| 8:30 PM | Checkpoint 2: set TOP_N_BRIEF from measured speed |
| 12:30 AM | Briefings for the demo snapshot done on the VM; prompts tuned if acceptance is poor (prompts only, never weaken the checks) |
| 2:00 AM | `extract_for_eval` done on the hand-labeled reports |

## 13. Definition of done
- [ ] Section 6 interfaces implemented exactly; section 8 tests pass, including `--granite` on your laptop.
- [ ] `03_brief.ipynb` passes Restart and Run All on your laptop.
- [ ] Your human posted "Ready to run: branch `a3-brief`, notebook `notebooks/03_brief.ipynb`, expected outputs `outputs/<demo>/briefings.jsonl, llm_calls.jsonl, extraction_preds.jsonl`" and Person 1 confirmed it on the VM.
- [ ] You reported the VM numbers: load seconds, seconds per call, tokens/sec, RSS, fields grounded share, sentences accepted share, fallbacks.

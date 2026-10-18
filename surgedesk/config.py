"""Central configuration for SurgeDesk.

Every path, threshold, and constant lives here. No other module may hardcode a
path. Environment variables override the defaults so the same code runs on a
laptop, in IBM's Jupyter container, and in our own container.

Owner: foundation (frozen). Change only with the whole team's agreement.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SURGEDESK_ROOT", PACKAGE_DIR.parent)).resolve()

DATA = ROOT / "data"
RAW = DATA / "raw"
HUMAID_DIR = RAW / "humaid"                      # contains <event>/<event>_{train,dev,test}.tsv
GAZETTEER_DIR = RAW / "gazetteer"                # Census *_Gaz_place_national.txt + *_Gaz_counties_national.txt
SVI_PATH = RAW / "svi" / "svi_counties.csv"      # CDC/ATSDR SVI, United States, county level, CSV
KAGGLE_DIR = RAW / "kaggle_disaster_tweets"      # optional task 1.9 only
LABELS_DIR = DATA / "labels"                     # to_label.csv, hand_labels.csv, purity_*.csv

MODELS = ROOT / "models"
MINILM_DIR = MODELS / "minilm"                   # sentence-transformers/all-MiniLM-L6-v2 files
CLASSIFIER_PATH = MODELS / "classifier.joblib"
RANKER_PATH = MODELS / "ranker.json"
RANKER_META_PATH = MODELS / "ranker_meta.json"
GRANITE_PATH = Path(
    os.environ.get(
        "SURGEDESK_GRANITE",
        str(MODELS / "granite" / "granite-3.3-2b-instruct-be.q8_0.gguf"),
    )
)

OUTPUTS = ROOT / "outputs"
STUB_OUTPUTS = OUTPUTS / "_stub"                 # pipeline --stub writes here, never mixed with real outputs
FIXTURES = ROOT / "fixtures"
DOCS = ROOT / "docs"

# Per-event output file names (all inside outputs/<event>/)
REPORTS_FILE = "reports.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
TAGGED_FILE = "tagged.jsonl"
SNAPSHOTS_FILE = "snapshots.csv"
INCIDENTS_FILE = "incidents.jsonl"
BRIEFINGS_FILE = "briefings.jsonl"
DECISIONS_FILE = "decisions.jsonl"
LLM_LOG_FILE = "llm_calls.jsonl"
SUMMARY_FILE = "summary.json"

# --------------------------------------------------------------------------- run settings
DEMO_EVENT = os.environ.get("SURGEDESK_EVENT", "hurricane_harvey_2017")
FIXTURE_EVENT = "fixture_event"
FIXTURE_TRAIN_EVENT = "fixture_train_event"
N_THREADS = int(os.environ.get("SURGEDESK_THREADS", "2"))   # 3 only for benchmark runs
SEED = 42

# --------------------------------------------------------------------------- stage 1: tag
EMBED_DIM = 384
EMBED_BATCH = 64
EMBED_MAX_TOKENS = 128

# --------------------------------------------------------------------------- stage 2: group
WINDOW_HOURS = 6.0          # an incident accepts new reports only if last_seen is within this window
SIM_PLACE = 0.75            # cosine threshold when report and incident share a place/county
SIM_NOPLACE = 0.85          # cosine threshold when either side has no place
MAX_CANDIDATES = 200        # only compare against the most recently updated incidents

# --------------------------------------------------------------------------- stage 3: rank
SNAPSHOT_MIN = 30           # snapshot every N minutes of event time
LAMBDAS = [0.0, 0.25, 0.5, 1.0]
HIGH_SVI = 0.75             # "high vulnerability" county threshold for fairness metrics
TOP_K_EVAL = 10
RANKING_OBJECTIVE = os.environ.get("SURGEDESK_RANKING_OBJECTIVE", "assistance_v1")
if RANKING_OBJECTIVE not in {"assistance_v1", "legacy"}:
    raise ValueError("SURGEDESK_RANKING_OBJECTIVE must be assistance_v1 or legacy")
RANK_PROVENANCE_FILE = "ranking_provenance.json"
RANK_SCORES_FILE = "rank_model_scores.jsonl"
LEGACY_RANKER_DIR = MODELS / "legacy_ranker"
# Explicit priority bands, not probabilities. Context remains below current
# needs even after the dashboard's maximum configured fairness adjustment.
ASSISTANCE_SCORE_MIN = 0.6
CONTEXT_SCORE_MAX = 0.25
CONTEXT_ADJUSTED_MAX = 0.5
ASSISTANCE_CACHE_SIZE = 25000
# Named groups describe cue policy only; they add no Incident/schema fields.
# Resources without a request predicate are context, as are broad help appeals.
ASSISTANCE_RESOURCE = (
    r"(?:rescue|boats?|food|(?:drinking )?water|shelter|medicines?|medications?|"
    r"oxygen|insulin|nebulizers?|inhalers?|ambulances?|medical (?:care|attention|help)|"
    r"supplies|volunteers|evacuation|transport(?:ation)?|fuel|gas|gasoline|"
    r"clothes|clothing|bedding|blankets|diapers|personal[- ]care goods|hygiene (?:kits|products))"
)
ASSISTANCE_PATTERNS = (
    r"(?P<context>\b(?:if|unless|whenever)\s+(?:you|anyone|someone|we|they|people|residents|families)\b)",
    r"(?P<context>\b(?:how|ways)\s+to\s+help\b|\b(?:fundraiser|fundraising|medical fund)\b)",
    r"(?P<specific>\b(?:sos|mayday)\b|^\s*#?help(?:\s+(?:now|please|us|me))?\s*$|"
    r"\bneeds?\s+to\s+be\s+(?:rescued|evacuated)\b|\b(?:awaiting|waiting for)\s+(?:help|rescue|evacuation)\b|"
    r"\b(?:trapped|stranded|stuck)\b|\b(?:not|never)\s+(?:yet\s+)?rescued(?:\s+yet)?\b|"
    r"\b(?:missing|searching for)\s+(?:\w+\s+){0,3}(?:person|people|child|children|man|woman|boy|girl|teen|neighbor|family|mother|father|son|daughter)\b|"
    r"\b(?:child|person|man|woman|teen|girl|boy|neighbor|mother|father|son|daughter)\s+(?:is|remains)\s+missing\b)",
    r"(?P<specific>\b(?:i|we|my\s+(?:family|neighbor|friend|mother|father|child|son|daughter)|"
    r"our\s+(?:family|neighbors|children)|(?:a|an|one|the)\s+(?:(?:elderly|older|injured|disabled)\s+)?"
    r"(?:person|resident|woman|man|child|family|neighbor))\s+"
    r"(?:(?:still|urgently|desperately|currently)\s+){0,2}(?:need|needs|require|requires)\s+(?:help|assistance|aid)\b|"
    r"\b(?:please\s+)?(?:help|rescue)\s+(?:me|us|my\s+(?:neighbor|family|friend|child)|our\s+(?:family|children))\b|"
    r"\b(?:can|could)\s+(?:anyone|someone|you)\s+help\s+(?:me|us|my\s+\w+|our\s+\w+)\b)",
    rf"(?P<resource>\b(?:need|needs|needing|request|requesting|requests|require|requires|asking for|looking for)\s+"
    rf"(?:\w+[\s-]+){{0,4}}{ASSISTANCE_RESOURCE}\b|"
    rf"\b(?:please\s+)?(?:send|bring|deliver|donate)\s+(?:\w+\s+){{0,2}}{ASSISTANCE_RESOURCE}\b|"
    rf"\b(?:anyone|someone)\s+(?:have|got|with)\s+(?:\w+\s+){{0,2}}{ASSISTANCE_RESOURCE}\b|"
    rf"\b(?:without|out of|running out of)\s+{ASSISTANCE_RESOURCE}\b|"
    rf"\b{ASSISTANCE_RESOURCE}\s+(?:is |are )?(?:needed|required|urgently needed)\b)",
)
ASSISTANCE_NEGATION = (
    r"\b(?:nobody|no one)\s+(?:is |remains )?(?:trapped|stranded|stuck|missing)\b|"
    r"\b(?:no longer|not|never|don't|doesn't|do not|does not|didn't|did not)\s+"
    r"(?:(?:really|currently|still|actually|urgently)\s+){0,2}(?:need|needs|trapped|stranded|stuck|missing|require|help|rescue)\b|"
    r"(?<!not )(?<!never )(?<!yet )(?<!n't )\b(?:was rescued|were rescued|is rescued|are rescued|been rescued|already rescued|"
    r"rescued (?:and|after|from|by|today|safely)|safe now|already evacuated|"
    r"no (?:help|rescue|supplies) needed|found (?:alive|safe|unharmed))\b|"
    r"\bmissing (?:child|person|man|woman|teen|girl|boy)\s+(?:was |is |has been )?found\b"
)
ASSISTANCE_SPECULATION = r"\b(?:might|would|forecast\w*|predict\w*|expected|projected)\b|\b(?:could|may)\s+(?:be|leave|need|require|become|cause|affect|displace|kill|injure|reach|rise|approach)\b"
ASSISTANCE_HISTORY = r"\b(?:historical|historically|last year|years? ago|previous (?:storm|hurricane|flood)|back in)\b"
CASUALTY_PATTERN = r"\b(?:dead|death|deaths|died|killed|casualties|death toll)\b"


def ranking_objective(event: str) -> str:
    """Frozen fixtures keep their original schema and broad relevance target."""
    return "legacy" if event.startswith("fixture") else RANKING_OBJECTIVE

# --------------------------------------------------------------------------- stage 4: brief
TOP_N_BRIEF = int(os.environ.get("SURGEDESK_TOP_N", "10"))
GRANITE_CTX = 2048
EXTRACT_MAX_TOKENS = 200
BRIEF_MAX_TOKENS = 120

# --------------------------------------------------------------------------- replay + benchmark
REPLAY_COMPRESSION = 12.0   # at speed 1x, one hour of event time plays in 5 minutes
BATCH_SECONDS = 5.0         # micro-batch size in wall-clock seconds
BENCH_RUN_SECONDS = 180.0   # wall-clock length of each benchmark run
BENCH_SPEEDS = [1.0, 5.0, 20.0, "max"]

# --------------------------------------------------------------------------- evaluation
LABEL_SAMPLE_N = 100
LABEL_SAMPLE_ACTIONABLE = 60
PURITY_SAMPLE_N = 20

# --------------------------------------------------------------------------- event -> US states
# Matched by keyword inside the HumAID event folder name, so exact folder spelling doesn't matter.
# Events with no match (non-US) get no place matching and no SVI.
EVENT_STATE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("harvey", ["TX", "LA"]),
    ("irma", ["FL", "GA", "SC"]),
    ("matthew", ["FL", "GA", "SC", "NC"]),
    ("florence", ["NC", "SC", "VA"]),
    ("dorian", ["FL", "GA", "SC", "NC"]),
    ("maryland", ["MD"]),
    ("midwest", ["NE", "IA", "MO", "KS", "SD"]),
    ("california", ["CA"]),
    ("fixture", ["TX"]),
]


def states_for_event(event: str) -> list[str]:
    """US state abbreviations relevant to an event, or [] for non-US events."""
    name = event.lower()
    for keyword, states in EVENT_STATE_KEYWORDS:
        if keyword in name:
            return list(states)
    return []


def event_dir(event: str, stub: bool = False) -> Path:
    """outputs/<event>/ (or outputs/_stub/<event>/), created if missing."""
    base = STUB_OUTPUTS if stub else OUTPUTS
    d = base / event
    d.mkdir(parents=True, exist_ok=True)
    return d


def event_file(event: str, name: str, stub: bool = False) -> Path:
    return event_dir(event, stub=stub) / name


def train_events(all_events: list[str]) -> list[str]:
    """Every event except the demo event and fixture events. Models never train on the demo event."""
    return sorted(
        e for e in all_events
        if e != DEMO_EVENT and not e.startswith("fixture")
    )

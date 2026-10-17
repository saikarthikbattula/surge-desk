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

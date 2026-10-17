"""Small shared helpers: time conversion, privacy masking, text normalization.

Owner: foundation (frozen). Every agent imports from here instead of rewriting these.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

TWITTER_EPOCH_MS = 1288834974657  # snowflake epoch (Nov 4, 2010)

_HANDLE = re.compile(r"@\w{1,30}")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9']+")

STOPWORDS = frozenset(
    """a an the and or but if of to in on at for from by with without into onto over under
    is are was were be been being am do does did have has had this that these those it its
    i me my we our you your he she they them their his her there here as so than then too
    very just not no yes can could will would should may might must up down out off about
    near all any some each more most other such only own same s t don now rt amp""".split()
)


# --------------------------------------------------------------------------- time
def tweet_id_to_datetime(tweet_id: str | int) -> datetime:
    """Creation time encoded in a Twitter snowflake ID (IDs after Nov 2010), UTC."""
    ms = (int(tweet_id) >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def datetime_to_tweet_id(dt: datetime, sequence: int = 0) -> str:
    """Inverse of tweet_id_to_datetime (used to build realistic fixture IDs)."""
    ms = int(dt.timestamp() * 1000) - TWITTER_EPOCH_MS
    return str((ms << 22) + sequence)


def to_iso(dt: datetime) -> str:
    """UTC ISO 8601 string with a trailing Z, second precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    """Parse an ISO string produced by to_iso (or any ISO 8601 with offset). Always UTC-aware."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def minutes_between(a: datetime, b: datetime) -> float:
    """b - a in minutes."""
    return (b - a).total_seconds() / 60.0


# --------------------------------------------------------------------------- text
def mask_privacy(text: str) -> str:
    """Replace @handles with @user and URLs with <url>. Use for anything displayed or recorded."""
    text = _URL.sub("<url>", text)
    text = _HANDLE.sub("@user", text)
    return text


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Lowercase, collapse whitespace, unify quotes. Used for evidence substring checks."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    return normalize_ws(t)


def dedupe_key(text: str) -> str:
    """Key for 'same text' checks: masked, lowercase, whitespace collapsed."""
    return normalize_for_match(mask_privacy(text))


def content_words(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords, in order."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]

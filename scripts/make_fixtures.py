"""Generate every file under fixtures/ (synthetic data only).

    python scripts/make_fixtures.py

The fixtures double as executable specs: the place matching, grouping, snapshot,
and representative-selection code below follows the handoff specs exactly, so
agent tests can compare their real implementations against these outputs.

All text here is invented (is_synthetic = true). Owner: foundation.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surgedesk import config  # noqa: E402
from surgedesk.contracts import (  # noqa: E402
    BRIEF_FIELDS, FEATURE_KEYS, VULN_TERMS, Briefing, Decision, Incident, Report, Tagged,
    is_actionable, zero_timing,
)
from surgedesk.fileio import save_embeddings, write_jsonl  # noqa: E402
from surgedesk.stubs import (  # noqa: E402
    find_terms, reference_features, reference_incident_location, reference_urgent_proxy,
    snapshot_times, template_briefing_text,
)
from surgedesk.textutil import datetime_to_tweet_id, parse_iso, to_iso  # noqa: E402

FX = ROOT / "fixtures"
UTC = timezone.utc

# =========================================================================== fixture gazetteer + SVI
# (name, USPS, LSAD suffix in NAME, lat, lon, ALAND)
PLACES = [
    ("Houston", "TX", "city", 29.7863, -95.3889, 1658000000),
    ("Katy", "TX", "city", 29.7905, -95.8353, 49000000),
    ("Rockport", "TX", "city", 28.0293, -97.0544, 40000000),
    ("Port Arthur", "TX", "city", 29.8849, -93.9399, 200000000),
    ("Beaumont", "TX", "city", 30.0860, -94.1018, 220000000),
    ("Dickinson", "TX", "city", 29.4548, -95.0589, 25000000),
    ("Victoria", "TX", "city", 28.8053, -96.9853, 94000000),
    ("Mission", "TX", "city", 26.2159, -98.3253, 85000000),
    ("Humble", "TX", "city", 29.9988, -95.2622, 26000000),
    ("Lake Charles", "LA", "city", 30.2266, -93.2174, 170000000),
    ("Houston", "MS", "city", 33.8987, -89.0034, 20000000),   # same name, other state: must NOT match for TX events
]
# (GEOID, USPS, NAME, lat, lon)
COUNTIES = [
    ("48201", "TX", "Harris County", 29.8578, -95.3936),
    ("48157", "TX", "Fort Bend County", 29.5274, -95.7709),
    ("48007", "TX", "Aransas County", 28.1050, -96.9998),
    ("48245", "TX", "Jefferson County", 29.8536, -94.1490),
    ("48167", "TX", "Galveston County", 29.3958, -94.9645),
    ("48469", "TX", "Victoria County", 28.7963, -96.9715),
    ("48215", "TX", "Hidalgo County", 26.3965, -98.1810),
    ("22019", "LA", "Calcasieu Parish", 30.2290, -93.3575),
    ("28017", "MS", "Chickasaw County", 33.9206, -88.9479),
]
# (STCNTY/FIPS as the real CSV stores it, i.e. possibly without leading zero, RPL_THEMES)
SVI_ROWS = [
    ("48201", 0.84), ("48157", 0.42), ("48007", 0.71), ("48245", 0.86), ("48167", 0.47),
    ("48469", 0.66), ("48215", 0.97), ("22019", 0.58), ("28017", 0.80),
    ("1001", 0.39),     # Autauga County AL stored as int: loader must zfill(5) -> 01001
    ("48039", -999),    # missing value -> None
]

COMMON_WORD_PLACES = {"mission", "victoria", "humble", "liberty", "hope", "rose", "star", "union", "center", "friendship"}
STATE_NAMES = {"TX": "texas", "LA": "louisiana", "MS": "mississippi", "FL": "florida", "GA": "georgia",
               "SC": "south carolina", "NC": "north carolina", "VA": "virginia", "MD": "maryland", "CA": "california",
               "NE": "nebraska", "IA": "iowa", "MO": "missouri", "KS": "kansas", "SD": "south dakota"}
LSAD_SUFFIXES = [" city and borough", " city", " town", " village", " CDP", " borough", " municipality", " (balance)"]


def write_gazetteer(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "2023_Gaz_place_national.txt", "w", newline="") as f:
        cols = ["USPS", "GEOID", "ANSICODE", "NAME", "LSAD", "FUNCSTAT", "ALAND", "AWATER",
                "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG                                                                                                               "]
        f.write("\t".join(cols) + "\n")
        for i, (name, st, lsad, lat, lon, aland) in enumerate(PLACES):
            f.write("\t".join([st, f"99{i:05d}", "0", f"{name} {lsad}", "25", "A", str(aland), "0",
                               f"{aland / 2.59e6:.3f}", "0", f"{lat:.6f}", f"{lon:.6f}"]) + "\n")
    with open(d / "2023_Gaz_counties_national.txt", "w", newline="") as f:
        cols = ["USPS", "GEOID", "ANSICODE", "NAME", "ALAND", "AWATER", "ALAND_SQMI", "AWATER_SQMI",
                "INTPTLAT", "INTPTLONG                                                                                                               "]
        f.write("\t".join(cols) + "\n")
        for geoid, st, name, lat, lon in COUNTIES:
            f.write("\t".join([st, geoid, "0", name, "1", "0", "1", "0", f"{lat:.6f}", f"{lon:.6f}"]) + "\n")


def write_svi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ST", "STATE", "ST_ABBR", "STCNTY", "COUNTY", "FIPS", "LOCATION", "RPL_THEMES"])
        for fips, rpl in SVI_ROWS:
            w.writerow([fips[:-3] or "0", "X", "X", fips, "X County", fips, "X County, X", rpl])


# =========================================================================== reference place matching
def _clean_place(name: str) -> str:
    for suf in LSAD_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_place_index(states: list[str]) -> dict[str, tuple[str, str]]:
    """lowercase key -> (display name, county FIPS). Spec: handoffs/agent1_tag.md section 4."""
    counties = [c for c in COUNTIES if c[1] in states]
    index: dict[str, tuple[str, str, int]] = {}
    for name, st, lsad, lat, lon, aland in PLACES:
        if st not in states:
            continue
        clean = _clean_place(f"{name} {lsad}")
        same_state = [c for c in counties if c[1] == st]
        if not same_state:
            continue
        fips = min(same_state, key=lambda c: _haversine(lat, lon, c[3], c[4]))[0]
        key = clean.lower()
        if key not in index or aland > index[key][2]:
            index[key] = (clean, fips, aland)
    for geoid, st, name, lat, lon in counties:
        index[name.lower()] = (name, geoid, 10**15)
    return {k: (v[0], v[1]) for k, v in index.items()}


def _split_hashtags(text: str) -> str:
    def repl(m):
        tag = m.group(1)
        return " " + re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tag) + " "
    return re.sub(r"#(\w+)", repl, text)


def match_places(text: str, index: dict[str, tuple[str, str]], states: list[str]) -> tuple[list[str], str | None]:
    tokens = re.findall(r"[a-z0-9']+", _split_hashtags(text).lower())
    qualifiers = {s.lower() for s in states} | {STATE_NAMES[s] for s in states if s in STATE_NAMES}
    found: list[str] = []
    fips_list: list[str] = []
    i = 0
    while i < len(tokens):
        hit = None
        for n in (4, 3, 2, 1):
            if i + n > len(tokens):
                continue
            key = " ".join(tokens[i: i + n])
            if key in index:
                if n == 1 and key in COMMON_WORD_PLACES:
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
                    nxt2 = " ".join(tokens[i + 1: i + 3])
                    if nxt not in qualifiers and nxt2 not in qualifiers:
                        continue
                hit = (key, n)
                break
        if hit:
            disp, fips = index[hit[0]]
            if disp not in found:
                found.append(disp)
                fips_list.append(fips)
            i += hit[1]
        else:
            i += 1
    return found, (fips_list[0] if fips_list else None)


# =========================================================================== synthetic reports
FIXTURE_EVENT_ROWS = [
    # (minutes after 10:00 UTC Aug 27 2017, text, gold label, cluster)
    (0, "Water rising fast on Braeswood in Meyerland Houston, my grandma is trapped on the second floor, please send a boat", "requests_or_urgent_needs", "A"),
    (5, "Rockport roofs torn off, power lines down across downtown Rockport", "infrastructure_and_utility_damage", "B"),
    (20, "Elderly neighbor stuck in attic near Meyerland Houston, water up to the roof, need rescue now", "requests_or_urgent_needs", "A"),
    (30, "Beaumont has lost its water supply, hospitals need drinking water", "requests_or_urgent_needs", "C"),
    (40, "Praying for everyone in Texas tonight", "sympathy_and_support", "S1"),
    (45, "Houston Meyerland area flooding badly, people on rooftops waiting for rescue", "requests_or_urgent_needs", "A"),
    (50, "Huge damage in Rockport, the high school roof collapsed", "infrastructure_and_utility_damage", "B"),
    (55, "Please help, water is coming into the house and we have a newborn", "requests_or_urgent_needs", "N1"),
    (65, "Two people injured when a tree fell on a car in Port Arthur", "injured_or_dead_people", "I1"),
    (70, "Anyone have a boat near Meyerland Houston? Family of 5 stranded", "requests_or_urgent_needs", "A"),
    (80, "This storm coverage is wild, the news anchors look exhausted lol", "not_humanitarian", "S2"),
    (90, "Beaumont water pumps failed, no running water in the whole city", "infrastructure_and_utility_damage", "C"),
    (95, "My dad uses oxygen and our power is out in Dickinson, where can we go?", "requests_or_urgent_needs", "D1"),
    (100, "Avoid driving through flooded roads in Katy, turn around don't drown", "caution_and_advice", "K1"),
    (105, "Shelter needs blankets and diapers, drop off tonight", "rescue_volunteering_or_donation_effort", "N2"),
    (110, "Families evacuated from Port Arthur arriving at the convention center shelter", "displaced_people_and_evacuations", "P1"),
    (120, "Power is out everywhere in Rockport and the water tower is damaged", "infrastructure_and_utility_damage", "B"),
    (125, "Thoughts with Houston, stay strong", "sympathy_and_support", "S3"),
    (150, "Dialysis patients in Beaumont need to be moved, no water at the clinic", "requests_or_urgent_needs", "C"),
    (170, "Just donated to the Red Cross, you should too @friend http://t.co/abc123", "rescue_volunteering_or_donation_effort", "N3"),
]

TRAIN_TEMPLATES = [
    # (cluster, place, label, list of texts)
    ("TA", "Houston", "requests_or_urgent_needs", [
        "Flooding near Kingwood Houston, elderly couple trapped, need rescue",
        "Kingwood Houston family stranded on roof, please send help",
        "Need a boat in Kingwood Houston, kids trapped upstairs",
        "Water rising in Kingwood Houston, my grandpa is stranded",
        "SOS Kingwood Houston, people trapped in cars on the feeder road",
    ]),
    ("TB", "Beaumont", "infrastructure_and_utility_damage", [
        "Beaumont roads closed, bridge damage on I-10",
        "Power out across Beaumont, substations flooded",
        "Beaumont I-10 bridge damaged, traffic stopped",
        "Major power outage in Beaumont after flooding",
        "Beaumont water main broke, roads washed out",
    ]),
    ("TC", "Katy", "rescue_volunteering_or_donation_effort", [
        "Shelter at Katy high school needs volunteers tonight",
        "Katy shelter asking for donations of water and towels",
        "Volunteers needed at the Katy shelter, bring cots",
        "Donate supplies to the Katy high school shelter",
        "Katy volunteers organizing a boat rescue team",
    ]),
    ("TD", "Port Arthur", "displaced_people_and_evacuations", [
        "Mandatory evacuations ordered in Port Arthur",
        "Port Arthur residents evacuating to Beaumont shelters",
        "Port Arthur evacuation buses leaving downtown",
        "Families displaced in Port Arthur after levee overtopped",
    ]),
    ("TE", "Rockport", "injured_or_dead_people", [
        "Injured residents taken to hospital in Rockport",
        "Rockport officials confirm one dead after storm",
        "Several injured in Rockport when building collapsed",
        "Rockport hospital treating storm injuries",
    ]),
    ("TF", "Victoria TX", "caution_and_advice", [
        "Victoria TX boil water notice in effect",
        "Victoria TX residents told to stay off flooded roads",
        "Curfew tonight in Victoria TX for safety",
        "Victoria TX warns of downed power lines",
    ]),
    ("TG", None, "other_relevant_information", [
        "Rainfall totals now over 40 inches in places",
        "The storm is moving slowly east tonight",
        "National Guard units arriving to assist",
        "FEMA says aid applications open online",
        "Flood gauges show record river levels",
    ]),
    ("TH", "Dickinson", "missing_or_found_people", [
        "Missing: 72 year old man last seen in Dickinson",
        "Dickinson police searching for missing teen",
    ]),
    ("TS", None, "sympathy_and_support", [
        "Sending love to everyone affected",
        "Praying for the Gulf Coast",
        "Heartbreaking images tonight, stay safe",
        "Thinking of all the first responders",
        "Stay strong Texas",
    ]),
    ("TN", None, "not_humanitarian", [
        "Football game postponed, ugh",
        "My cat hates thunder lol",
        "Watching the weather channel all day",
        "Anyone else bored indoors",
        "New phone who dis",
    ]),
    ("TK", None, "dont_know_cant_judge", [
        "wow",
        "this is it",
    ]),
]


def _vec(rng: np.random.Generator, base: np.ndarray | None) -> np.ndarray:
    if base is None:
        v = rng.standard_normal(config.EMBED_DIM)
    else:
        v = base + 0.015 * rng.standard_normal(config.EMBED_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def build_event(event: str, rows: list[tuple[datetime, str, str, str]], seed: int):
    """rows: (timestamp, text, gold, cluster). Returns reports, tagged, embeddings."""
    rng = np.random.default_rng(seed)
    states = config.states_for_event(event)
    index = build_place_index(states)
    rows = sorted(rows, key=lambda r: r[0])
    bases: dict[str, np.ndarray] = {}
    reports, tagged, vecs = [], [], []
    for i, (ts, text, gold, cluster) in enumerate(rows):
        rid = datetime_to_tweet_id(ts, sequence=i)
        reports.append(Report(report_id=rid, event=event, text=text, timestamp=to_iso(ts),
                              source="synthetic", gold_label=gold, is_synthetic=True))
        conf = round(0.6 + 0.35 * (int(hashlib.sha1(text.encode()).hexdigest()[:4], 16) / 65535), 6)
        places, fips = match_places(text, index, states)
        tagged.append(Tagged(report_id=rid, emb_row=i, pred_category=gold, pred_conf=conf, places=places,
                             county_fips=fips, vuln_terms=find_terms(text, VULN_TERMS), is_actionable=is_actionable(gold)))
        if cluster not in bases:
            bases[cluster] = rng.standard_normal(config.EMBED_DIM)
            bases[cluster] /= np.linalg.norm(bases[cluster])
        vecs.append(_vec(rng, bases[cluster]))
    return reports, tagged, np.vstack(vecs)


# =========================================================================== reference grouping
def reference_group(event, reports, tagged, emb):
    """Spec: handoffs/agent2_group_rank.md section 4. Returns {incident_id: [row indices]} in creation order."""
    incidents: dict[str, dict] = {}
    for i, (r, t) in enumerate(zip(reports, tagged)):
        if not t.is_actionable:
            continue
        ts = parse_iso(r.timestamp)
        open_inc = [(iid, inc) for iid, inc in incidents.items()
                    if (ts - inc["last_seen"]).total_seconds() <= config.WINDOW_HOURS * 3600]
        open_inc.sort(key=lambda x: x[1]["last_seen"], reverse=True)
        open_inc = open_inc[: config.MAX_CANDIDATES]
        best, best_sim = None, -2.0
        for iid, inc in open_inc:
            iplace, icounty = reference_incident_location([tagged[j] for j in inc["rows"]])
            rep_has = bool(t.places or t.county_fips)
            inc_has = bool(iplace or icounty)
            place_match = bool(
                (t.county_fips and icounty and t.county_fips == icounty)
                or (t.places and iplace and t.places[0].lower() == iplace.lower())
            )
            if rep_has and inc_has and not place_match:
                continue
            thr = config.SIM_PLACE if place_match else config.SIM_NOPLACE
            c = inc["sum"] / np.linalg.norm(inc["sum"])
            sim = float(np.dot(emb[i], c))
            if sim >= thr and sim > best_sim:
                best, best_sim = iid, sim
        if best is None:
            best = f"{event}-inc-{len(incidents) + 1}"
            incidents[best] = {"rows": [], "sum": np.zeros(config.EMBED_DIM, dtype=np.float64), "last_seen": ts}
        inc = incidents[best]
        inc["rows"].append(i)
        inc["sum"] += emb[i]
        inc["last_seen"] = ts
    return {iid: inc["rows"] for iid, inc in incidents.items()}


def fixture_score(f: dict) -> tuple[float, list[list]]:
    """FIXTURE SCORE, NOT A MODEL. Deterministic stand-in so the UI and tests have numbers."""
    contrib = {
        "share_requests_or_urgent_needs": 1.2 * f["share_requests_or_urgent_needs"],
        "share_injured_or_dead_people": 1.0 * f["share_injured_or_dead_people"],
        "n_reports": 0.35 * math.log1p(f["n_reports"]),
        "vuln_share": 0.8 * f["vuln_share"],
        "growth_ratio": 0.2 * min(f["growth_ratio"], 3.0),
        "minutes_since_last": -0.004 * f["minutes_since_last"],
    }
    z = -1.5 + sum(contrib.values())
    score = round(1 / (1 + math.exp(-z)), 6)
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return score, [[k, round(v, 6)] for k, v in top]


def build_incidents(event, reports, tagged, groups, svi_map):
    start, end = parse_iso(reports[0].timestamp), parse_iso(reports[-1].timestamp)
    out = []
    for snap in snapshot_times(start, end):
        rows = []
        for iid, idx in groups.items():
            members = [j for j in idx if parse_iso(reports[j].timestamp) <= snap]
            if not members:
                continue
            last = parse_iso(reports[members[-1]].timestamp)
            if (snap - last).total_seconds() > config.WINDOW_HOURS * 3600:
                continue
            reps = [reports[j] for j in members]
            tags = [tagged[j] for j in members]
            place, county = reference_incident_location(tags)
            svi = svi_map.get(county) if county else None
            feats = reference_features(reps, tags, snap, svi)
            score, shap = fixture_score(feats)
            rows.append(dict(iid=iid, reps=reps, place=place, county=county, svi=svi, feats=feats, score=score, shap=shap))
        rows.sort(key=lambda x: (-x["score"], x["reps"][0].timestamp, x["iid"]))
        for rank, x in enumerate(rows, 1):
            out.append(Incident(
                incident_id=x["iid"], event=event, snapshot_time=to_iso(snap),
                report_ids=[r.report_id for r in x["reps"]], first_seen=x["reps"][0].timestamp,
                last_seen=x["reps"][-1].timestamp, place=x["place"], county_fips=x["county"], svi=x["svi"],
                features=x["feats"], score=x["score"], score_adj=x["score"], lam=0.0, rank=rank, rank_adj=rank,
                shap_top=x["shap"], urgent_proxy=reference_urgent_proxy(x["reps"]),
            ))
    return out


def select_representatives(inc: Incident, by_id, tag_by_id, emb) -> list[str]:
    """Spec: handoffs/agent3_brief.md section 4."""
    rows = [tag_by_id[r].emb_row for r in inc.report_ids]
    c = emb[rows].sum(axis=0)
    c = c / np.linalg.norm(c)
    sims = [float(np.dot(emb[row], c)) for row in rows]
    first = inc.report_ids[int(np.argmax(sims))]
    second = max(inc.report_ids, key=lambda r: (len(tag_by_id[r].vuln_terms), len(by_id[r].text),
                                                 -inc.report_ids.index(r)))
    return [first] if second == first else [first, second]


AUTHORED = {
    # cluster -> field -> candidate phrases (first one found in the representative texts wins)
    "A": {"hazard": ["Water rising fast", "water up to the roof", "flooding badly"],
          "location": ["Meyerland Houston", "Braeswood in Meyerland Houston"],
          "people_affected": ["Family of 5", "my grandma", "Elderly neighbor", "people on rooftops"],
          "vulnerable": ["my grandma is trapped", "Elderly neighbor stuck in attic", "grandma"],
          "need": ["please send a boat", "need rescue now", "waiting for rescue", "Anyone have a boat"]},
    "C": {"hazard": ["lost its water supply", "water pumps failed", "no water at the clinic"],
          "location": ["Beaumont"],
          "people_affected": ["hospitals", "Dialysis patients"],
          "vulnerable": ["Dialysis patients"],
          "need": ["need drinking water", "need to be moved"]},
    "D1": {"hazard": ["our power is out"], "location": ["Dickinson"], "people_affected": ["My dad"],
           "vulnerable": ["My dad uses oxygen"], "need": ["where can we go"]},
    "N1": {"hazard": ["water is coming into the house"], "location": [], "people_affected": [],
           "vulnerable": ["we have a newborn"], "need": ["Please help"]},
    "B": {"hazard": ["roofs torn off", "roof collapsed", "water tower is damaged"],
          "location": ["Rockport", "downtown Rockport"],
          "people_affected": [], "vulnerable": [], "need": []},
}


def _fields_for(cluster: str, rep_texts: list[str]) -> dict:
    spec = AUTHORED.get(cluster, {})
    joined = [t.lower() for t in rep_texts]
    fields = {}
    for f in BRIEF_FIELDS:
        val = None
        for phrase in spec.get(f, []):
            for t_low, t in zip(joined, rep_texts):
                k = t_low.find(phrase.lower())
                if k >= 0:
                    val = t[k: k + len(phrase)]
                    break
            if val:
                break
        fields[f] = {"value": val, "evidence": val, "confidence": 0.9 if val else 0.0, "grounded": val is not None}
    return fields


def build_briefings(event, incidents, reports, tagged, emb, cluster_of_row):
    by_id = {r.report_id: r for r in reports}
    tag_by_id = {t.report_id: t for t in tagged}
    last_snap = max(i.snapshot_time for i in incidents)
    top = sorted([i for i in incidents if i.snapshot_time == last_snap], key=lambda i: i.rank)[:3]
    out = []
    for inc in top:
        cluster = cluster_of_row[tag_by_id[inc.report_ids[0]].emb_row]
        reps = select_representatives(inc, by_id, tag_by_id, emb)
        fields = _fields_for(cluster, [by_id[r].text for r in reps])
        sentences = []
        if cluster == "A":
            sentences = [
                {"text": f"Residents near {fields['location']['value']} report {fields['hazard']['value'].lower()} with people trapped.",
                 "tags": ["location", "hazard"], "accepted": True, "reject_reason": None},
                {"text": "Several houses have completely collapsed.", "tags": ["hazard"], "accepted": False,
                 "reject_reason": "tagged field 'hazard' not supported by the sentence"},
                {"text": f"Ranked high because of {inc.features['n_reports']:.0f} reports and urgent requests for help.",
                 "tags": ["reason"], "accepted": True, "reject_reason": None},
            ]
        accepted = [s["text"] for s in sentences if s["accepted"]]
        used_fallback = not accepted
        text = " ".join(accepted) if accepted else template_briefing_text(inc, fields)
        key = hashlib.sha1(("|".join([inc.incident_id] + sorted(inc.report_ids) + ["fixture"])).encode()).hexdigest()
        out.append(Briefing(
            incident_id=inc.incident_id, event=event, snapshot_time=inc.snapshot_time, cache_key=key,
            rep_report_ids=reps, fields=fields, sentences=sentences, briefing=text, used_fallback=used_fallback,
            model_file="fixture", timing=zero_timing(),
        ))
    return out


def write_humaid_tsvs(event: str, reports: list[Report], d: Path, title_case_test: bool) -> None:
    """Event folder with <event>_{train,dev,test}.tsv like the real HumAID release."""
    ed = d / event
    ed.mkdir(parents=True, exist_ok=True)
    n = len(reports)
    splits = {"train": reports[: int(n * 0.7)], "dev": reports[int(n * 0.7): int(n * 0.8)], "test": reports[int(n * 0.8):]}
    for split, rows in splits.items():
        with open(ed / f"{event}_{split}.tsv", "w", newline="", encoding="utf-8") as f:
            f.write("tweet_id\ttweet_text\tclass_label\n")
            for r in rows:
                label = r.gold_label
                if title_case_test and split == "test":
                    label = label.replace("_", " ").capitalize()  # e.g. "Requests or urgent needs"
                f.write(f"{r.report_id}\t{r.text}\t{label}\n")


def main() -> None:
    write_gazetteer(FX / "gazetteer")
    write_svi(FX / "svi" / "svi_counties.csv")
    svi_map = {f.zfill(5): (None if v == -999 else v) for f, v in SVI_ROWS}

    base = datetime(2017, 8, 27, 10, 0, tzinfo=UTC)
    fx_rows = [(base + timedelta(minutes=m), t, g, c) for m, t, g, c in FIXTURE_EVENT_ROWS]
    train_rows = []
    tbase = datetime(2017, 8, 28, 8, 0, tzinfo=UTC)
    k = 0
    for cluster, _place, label, texts in TRAIN_TEMPLATES:
        for j, text in enumerate(texts):
            train_rows.append((tbase + timedelta(minutes=17 * k + 3 * j), text, label, cluster))
            k += 1

    for event, rows, seed in [(config.FIXTURE_EVENT, fx_rows, 7), (config.FIXTURE_TRAIN_EVENT, train_rows, 11)]:
        reports, tagged, emb = build_event(event, rows, seed)
        cluster_of_row = [c for (_, _, _, c) in sorted(rows, key=lambda r: r[0])]
        groups = reference_group(event, reports, tagged, emb)
        incidents = build_incidents(event, reports, tagged, groups, svi_map)
        out = FX / "outputs" / event
        write_jsonl(out / "reports.jsonl", reports)
        write_jsonl(out / "tagged.jsonl", tagged)
        save_embeddings(out / "embeddings.npy", emb)
        write_jsonl(out / "incidents.jsonl", incidents)
        write_jsonl(out / "groups_expected.jsonl", [{"incident_id": i, "report_ids": [reports[j].report_id for j in idx]} for i, idx in groups.items()])
        write_humaid_tsvs(event, reports, FX / "humaid", title_case_test=(event == config.FIXTURE_EVENT))
        if event == config.FIXTURE_EVENT:
            briefs = build_briefings(event, incidents, reports, tagged, emb, cluster_of_row)
            write_jsonl(out / "briefings.jsonl", briefs)
            top = briefs[0]
            write_jsonl(out / "decisions.jsonl", [Decision(
                incident_id=top.incident_id, event=event, snapshot_time=top.snapshot_time, action="accept",
                new_rank=None, note="fixture decision", reviewer="fixture", timestamp=to_iso(base + timedelta(hours=4)), lam=0.0)])
        print(f"{event}: {len(reports)} reports, {len(groups)} incidents, {len(incidents)} incident-snapshot rows")
    _ = FEATURE_KEYS


if __name__ == "__main__":
    main()

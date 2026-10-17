"""Local Gazetteer matching and county SVI lookup (Agent 1)."""
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from . import config

COMMON_WORD_PLACES: set[str] = {"mission", "victoria", "humble", "liberty", "hope", "rose", "star",
                                "union", "center", "friendship"}
# Text parsing vocabulary, as specified by the fixture reference.
_STATE_NAMES = dict(zip(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split(),
    ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
     "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
     "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
     "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
     "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
     "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
     "virginia", "washington", "west virginia", "wisconsin", "wyoming", "district of columbia"]))
_SUFFIXES = (" city and borough", " city", " town", " village", " CDP", " borough", " municipality", " (balance)")


def _distance(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * math.asin(math.sqrt(min(1.0, a)))


class Gazetteer:
    def __init__(self, gaz_dir: Path = config.GAZETTEER_DIR):
        tables = []
        for pattern in ("*_Gaz_place_national.txt", "*_Gaz_counties_national.txt"):
            paths = sorted(Path(gaz_dir).rglob(pattern))
            if not paths:
                raise FileNotFoundError(f"Missing {pattern} in {gaz_dir}")
            frame = pd.read_csv(paths[0], sep="\t", dtype=str, keep_default_na=False)
            frame.columns = frame.columns.str.strip()
            tables.append([{k: v.strip() for k, v in row.items()} for row in frame.to_dict("records")])
        self.places, self.counties = tables
        self._indexes = {}

    def place_index(self, states: list[str]) -> dict[str, tuple[str, str]]:
        cache_key = tuple(states)
        if cache_key in self._indexes:
            return dict(self._indexes[cache_key])
        counties = [c for c in self.counties if c["USPS"] in states]
        by_state = {st: [c for c in counties if c["USPS"] == st] for st in states}
        index = {}
        for place in self.places:
            candidates = by_state.get(place["USPS"], [])
            if not candidates:
                continue
            display = place["NAME"]
            for suffix in _SUFFIXES:
                if display.endswith(suffix):
                    display = display[:-len(suffix)]
                    break
            nearest = min(candidates, key=lambda c: _distance(float(place["INTPTLAT"]), float(place["INTPTLONG"]),
                                                            float(c["INTPTLAT"]), float(c["INTPTLONG"])))
            key, area = display.lower(), int(place["ALAND"])
            if key not in index or area > index[key][2]:
                index[key] = (display, nearest["GEOID"].zfill(5), area)
        for county in counties:
            index[county["NAME"].lower()] = (county["NAME"], county["GEOID"].zfill(5), 0)
        result = {k: (v[0], v[1]) for k, v in index.items()}
        self._indexes[cache_key] = result
        return dict(result)


class PlaceMatcher:
    def __init__(self, event: str, gazetteer: Gazetteer | None = None):
        self.states = config.states_for_event(event)
        self._gazetteer = (gazetteer or Gazetteer()) if self.states else None
        self.index = self._gazetteer.place_index(self.states) if self._gazetteer else {}
        self._state_indexes = {}
        self.qualifiers = {s.lower() for s in self.states} | {_STATE_NAMES[s] for s in self.states if s in _STATE_NAMES}
        self._qualifier_states = {name: state for state in self.states
                                  for name in (state.lower(), _STATE_NAMES.get(state, state.lower()))}
        storm = re.fullmatch(r"(?:hurricane|cyclone|typhoon|tropical_storm)_(.+)_\d{4}", event)
        self._storm_name = storm[1].replace("_", " ") if storm else None

    def match(self, text: str) -> tuple[list[str], str | None]:
        def expand(match):
            return " " + re.sub(r"(?<=[a-z])(?=[A-Z])", " ", match[1]) + " "
        tokens = re.findall(r"[a-z0-9']+", re.sub(r"#(\w+)", expand, text).lower())
        found, counties = [], []
        i = 0
        while i < len(tokens):
            for n in (4, 3, 2, 1):
                if i + n > len(tokens):
                    continue
                key = " ".join(tokens[i:i + n])
                if key not in self.index:
                    continue
                before = tokens[max(0, i - 2):i]
                # A named storm is not the Gazetteer place with the same name.
                if before and (before[-1] in {"hurricane", "cyclone", "typhoon"}
                               or before == ["tropical", "storm"]):
                    continue
                state = next((self._qualifier_states[value] for width in (3, 2, 1)
                              if (value := " ".join(tokens[i + n:i + n + width])) in self._qualifier_states), None)
                if key == self._storm_name and state is None:
                    continue
                if n == 1 and key in COMMON_WORD_PLACES:
                    if state is None:
                        continue
                index = self.index
                if state is not None:
                    if state not in self._state_indexes:
                        self._state_indexes[state] = self._gazetteer.place_index([state])
                    index = self._state_indexes[state]
                    if key not in index:
                        continue
                display, fips = index[key]
                if display not in found:
                    found.append(display)
                    counties.append(fips)
                i += n
                break
            else:
                i += 1
        return found, counties[0] if counties else None


class SVI:
    def __init__(self, path: Path = config.SVI_PATH):
        self._values = {}
        self._available = Path(path).is_file()
        if not self._available:
            print(f"Warning: SVI file missing: {path}; county vulnerability unavailable")
            return
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.columns = frame.columns.str.strip()
        fips_column = "STCNTY" if "STCNTY" in frame else "FIPS"
        buckets = {}
        for code, raw in frame[[fips_column, "RPL_THEMES"]].itertuples(index=False, name=None):
            code = code.strip()
            if not code.isdigit():
                continue
            code = code.zfill(11)[:5] if len(code) > 5 else code.zfill(5)
            values = buckets.setdefault(code, [])
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0:
                values.append(value)
        self._values = {code: sum(values) / len(values) if values else None for code, values in buckets.items()}

    def get(self, fips: str | None) -> float | None:
        return self._values.get(str(fips).zfill(5)) if fips is not None else None

    @property
    def available(self) -> bool:
        return self._available

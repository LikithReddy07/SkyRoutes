#!/usr/bin/env python3
"""
Version 2 of scrap_flights_parallel.py. Same output, but it reuses weekdays
it already knows instead of looking them all up again on every run.

Scrapes every airport and its outbound routes from flightsfrom.com into
data/airline_routes.json, using the JSON API the site's map uses
(apis.flightsfrom.com/map). The output keeps the same schema as before, so
the dashboard doesn't need changes.

The API sits behind Cloudflare, which blocks bursts of requests, so requests
are throttled across all workers. Each airport's routes take 1 request, and
working out which weekdays they fly takes 7 more (one per day), which is
where most of the time goes. So each airport records the week its weekdays
were checked for (days_week), and later runs reuse them for airports whose
routes haven't changed, as long as that week is in the current airline
schedule season. Timetables switch on the last Sundays of March and October,
so the first run after each switch looks every airport up again (about 8
hours). So does the first run on output from version 1, which doesn't record
the week. Use --refresh-days to look everything up at any time.

The existing airline_routes.json is only replaced after a complete run. If
anything fails, results go to airline_routes.partial.json instead, and the
next run reuses the weekdays it had already looked up.

Usage: python data/scrap_flights_parallel_v2.py [--refresh-days]
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import lxml.html
from curl_cffi import CurlError, requests
from geopy.distance import geodesic

REQUESTS_PER_SECOND = 2  # Shared by all workers. Raise it and you'll spend more time blocked.
MAX_WORKERS = 4  # Enough to keep up with REQUESTS_PER_SECOND. More won't make it faster.
FETCH_DAYS = True  # Work out which weekdays each route runs. Up to 7 extra requests per airport.
BLOCK_COOLDOWN = 60  # Seconds to pause everything after a 403/429. Doubles on each block in a row.
MAX_BLOCKS_IN_A_ROW = 4  # Give up if still blocked after 1 + 2 + 4 + 8 minutes of cooldown.
MAX_ATTEMPTS = 5  # For network and server errors. Waits roughly 5s, 10s, 20s, 40s between attempts.
MAX_CONSECUTIVE_FAILURES = 10  # This many failed airports in a row means something's wrong. Stop.

API_BASE = "https://apis.flightsfrom.com/map"
AIRPORT_LIST_URL = "https://www.flightsfrom.com/airports"
HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.flightsfrom.com",
    "Referer": "https://www.flightsfrom.com/",
}
SERVER_ERRORS = {500, 502, 503, 504}
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
# The API doesn't expose these anymore, so they're carried over from the previous output.
CARRIED_OVER_FIELDS = ["continent", "elevation", "icao", "timezone"]
IATA_RE = re.compile(r"^[A-Z0-9]{3}$")
AIRLINE_CODE_RE = re.compile(r"^[A-Z0-9]{2,3}$")

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "airline_routes.json"
PARTIAL_PATH = DATA_DIR / "airline_routes.partial.json"


class FetchError(Exception):
    """A request that failed for good."""


_abort = threading.Event()  # Set to stop all workers early (blocked, too many failures, or Ctrl+C).


class Throttle:
    """
    Shared by all worker threads. Spaces requests evenly, and when the site
    starts blocking, pauses everyone with a cooldown that doubles each time
    the block is still there afterwards.
    """

    def __init__(self, per_second):
        self._interval = 1 / per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._paused_until = 0.0
        self._blocks_in_a_row = 0
        self.cooldowns = 0  # For the end-of-run summary.
        self.paused_seconds = 0.0

    def wait_turn(self):
        with self._lock:
            slot = max(time.monotonic(), self._next_slot, self._paused_until)
            self._next_slot = slot + self._interval
        _abort.wait(max(0.0, slot - time.monotonic()))  # Returns early if the run is stopped.

    def blocked(self, reason):
        with self._lock:
            now = time.monotonic()
            if now < self._paused_until:
                return  # Another request already hit this block and started a cooldown.
            self._blocks_in_a_row += 1
            giving_up = self._blocks_in_a_row > MAX_BLOCKS_IN_A_ROW
            if not giving_up:
                cooldown = BLOCK_COOLDOWN * 2 ** (self._blocks_in_a_row - 1)
                self._paused_until = now + cooldown
                self.cooldowns += 1
                self.paused_seconds += cooldown
        if giving_up:
            print(f"! Still blocked ({reason}) after {MAX_BLOCKS_IN_A_ROW} cooldowns. Stopping.")
            _abort.set()
        else:
            print(f"! Blocked by the site ({reason}), pausing all requests for {format_duration(cooldown)}")

    def succeeded(self):
        with self._lock:
            self._blocks_in_a_row = 0


throttle = Throttle(REQUESTS_PER_SECOND)
_local = threading.local()
_sessions = []
_sessions_lock = threading.Lock()


def get_session():
    """One curl_cffi session per worker thread, so connections get reused."""
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session(impersonate="chrome", headers=HEADERS, timeout=30)
        _local.session = session
        with _sessions_lock:
            _sessions.append(session)
    return session


def close_sessions():
    with _sessions_lock:
        for session in _sessions:
            session.close()
        _sessions.clear()


def format_duration(seconds):
    minutes = int(seconds // 60)
    if minutes < 1:
        return f"{int(seconds)}s"
    return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"


def plural(count, word):
    return f"{count} {word}" + ("" if count == 1 else "s")


def fetch(url, params=None, as_json=True):
    """
    GET a URL through the throttle. Waits out blocks (403/429), and retries
    network and server errors up to MAX_ATTEMPTS times.
    Returns parsed JSON (or text if as_json=False), or None if the URL is a 404/410.
    """
    errors = 0
    while True:
        throttle.wait_turn()
        if _abort.is_set():
            raise FetchError("run stopped")
        try:
            response = get_session().get(url, params=params)
        except CurlError as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            status = response.status_code
            if status == 200:
                try:
                    data = json.loads(response.text) if as_json else response.text
                except json.JSONDecodeError:
                    # Cloudflare serves its challenge page instead of the data.
                    throttle.blocked("got a challenge page instead of JSON")
                    continue
                throttle.succeeded()
                return data
            if status in (404, 410):
                throttle.succeeded()
                return None
            if status in (403, 429):
                throttle.blocked(f"HTTP {status}")
                continue
            if status not in SERVER_ERRORS:
                raise FetchError(f"HTTP {status} for {url}")
            reason = f"HTTP {status}"
        errors += 1
        if errors >= MAX_ATTEMPTS:
            raise FetchError(f"{reason} for {url}, gave up after {errors} attempts")
        delay = 5 * 2 ** (errors - 1) + random.uniform(0, 2)
        print(f"! {reason} for {url}, retrying in {delay:.0f}s")
        _abort.wait(delay)


def load_airline_names():
    """Airline code -> name, e.g. "DL" -> "Delta". This endpoint returns an HTML fragment."""
    html = fetch(f"{API_BASE}/airline/all.json", as_json=False)
    if not html or not html.strip():
        raise FetchError("airline list was empty")
    names = {}
    root = lxml.html.fragment_fromstring(html, create_parent="div")
    for option in root.xpath(".//*[@d-al]"):
        code = option.get("d-al", "").strip()
        if AIRLINE_CODE_RE.match(code):
            label = option.text_content().strip()
            names[code] = re.sub(r"\s*\(" + re.escape(code) + r"\)$", "", label) or code
    if not names:
        raise FetchError("no airlines found in the airline list (the format may have changed)")
    return names


def load_airport_list():
    """
    IATA -> {name, city_name, country, country_code, latitude, longitude} from
    the site's airport list. The routes API doesn't include airport names.
    """
    data = fetch(AIRPORT_LIST_URL)
    try:
        listed = data["response"]["airports"]
    except (TypeError, KeyError) as e:
        raise FetchError(f"unexpected airport list format ({e!r})") from e
    return {
        a["IATA"]: a for a in listed
        if isinstance(a, dict) and IATA_RE.match(str(a.get("IATA", "")))
    }


def load_map_airport_ids():
    """IDs of every airport on the site's map. The routes API is keyed by these, not IATA codes."""
    data = fetch(f"{API_BASE}/airport/all.json")
    if not isinstance(data, dict):
        raise FetchError("map airport list was empty")
    ids = list(dict.fromkeys(i for i in data.get("i", []) if isinstance(i, int)))
    if not ids:
        raise FetchError("map airport list was empty")
    return ids


def load_previous_output(path):
    """An earlier run's output, for the fields the API no longer provides and for reusing weekdays."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"! {path.name} isn't valid JSON, ignoring it")
        return {}
    return data if isinstance(data, dict) else {}


def last_sunday(year, month):
    last_day = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() + 1) % 7)


def schedule_season(day):
    """
    The airline schedule season a date falls in. Airlines switch timetables
    on the last Sunday of March (summer) and the last Sunday of October (winter).
    """
    if day < last_sunday(day.year, 3):
        return f"winter {day.year - 1}/{day.year % 100:02d}"
    if day < last_sunday(day.year, 10):
        return f"summer {day.year}"
    return f"winter {day.year}/{(day.year + 1) % 100:02d}"


def pick_week(today):
    """
    The 7 days to check weekdays for. Starts tomorrow, unless a new schedule
    season starts during that week: then it starts on the switch day instead,
    so all 7 days follow the same timetable.
    """
    start = today + timedelta(days=1)
    for year in (start.year, start.year + 1):
        for switch in (last_sunday(year, 3), last_sunday(year, 10)):
            if start < switch <= start + timedelta(days=6):
                start = switch
    return [start + timedelta(days=n) for n in range(7)]


def route_signature(routes):
    """What has to stay the same for an airport's weekdays from an earlier run to still apply."""
    return frozenset(
        (
            route["iata"],
            tuple(sorted(carrier["iata"] for carrier in route["carriers"])),
            route.get("flights_per_day"),
            route.get("flights_per_week"),
        )
        for route in routes
    )


def valid_days(days):
    return isinstance(days, dict) and all(isinstance(days.get(day), bool) for day in WEEKDAYS)


def build_days_index(*outputs):
    """
    IATA -> [(week, route signature, weekdays per destination), ...] for every
    airport in earlier outputs that recorded which week its weekdays were checked for.
    """
    index = {}
    for output in outputs:
        for iata, airport in output.items():
            if not isinstance(airport, dict) or not airport.get("days_week"):
                continue
            try:
                week = date.fromisoformat(airport["days_week"])
                routes = airport["routes"]
                days = {route["iata"]: route["days"] for route in routes}
                signature = route_signature(routes)
            except (KeyError, TypeError, ValueError):
                continue  # Malformed entry, skip it.
            if days and all(valid_days(d) for d in days.values()):
                index.setdefault(iata, []).append((week, signature, days))
    return index


def reusable_days(iata, routes, days_index, season):
    """
    (week, weekdays per destination) from the most recent earlier check of
    this airport that still applies: same schedule season and same routes.
    None if there isn't one.
    """
    signature = route_signature(routes)
    matches = [
        (week, days) for week, checked_signature, days in days_index.get(iata, [])
        if checked_signature == signature and schedule_season(week) == season
    ]
    return max(matches, key=lambda match: match[0], default=None)


def parse_airport_entry(entries, coords, idx):
    """Basic info for the airport at position idx in a routes response, or None if malformed."""
    entry = entries[idx]
    if not (isinstance(entry, list) and len(entry) >= 4 and IATA_RE.match(str(entry[0]))):
        return None
    try:
        lat, lon = float(coords[2 * idx]), float(coords[2 * idx + 1])
    except (IndexError, TypeError, ValueError):
        lat = lon = None
    return {
        "iata": entry[0],
        "city_name": entry[1],
        "country_code": entry[2],
        "country": entry[3],
        "latitude": lat,
        "longitude": lon,
    }


def parse_frequency(freq):
    """
    The API encodes frequency as [4, 0, "11-23"] (flights per day) or [4, 1, 3]
    (flights per week). Other codes are notes about seasonal, new or ending routes.
    Returns (flights_per_day, flights_per_week) in the same format as before.
    """
    if isinstance(freq, list) and len(freq) >= 3 and freq[0] == 4:
        if freq[1] == 0:
            per_day = str(freq[2])
            return f"{per_day} flight" + ("" if per_day == "1" else "s"), None
        if freq[1] == 1:
            return None, freq[2]
    return None, None


def fetch_operating_days(airport_id, week):
    """Destination IATA -> {"mon": bool, ...}, by asking the API which routes run on each date."""
    days = {}
    for day in week:
        iso = day.isoformat()
        data = fetch(f"{API_BASE}/airport/f{airport_id}.json", params={"dateFrom": iso, "dateTo": iso})
        if not isinstance(data, dict):
            continue
        entries = data.get("a", [])
        for idx in data.get("d", []):
            iata = entries[idx][0]
            days.setdefault(iata, dict.fromkeys(WEEKDAYS, False))[WEEKDAYS[day.weekday()]] = True
    return days


@dataclass
class ScrapedAirport:
    origin: dict  # Basic info: iata, city_name, country, country_code, latitude, longitude.
    routes: list = field(default_factory=list)  # In the output schema.
    destinations: list = field(default_factory=list)  # Basic info plus the API id of each destination.
    days_week: str | None = None  # First day of the week the weekdays were checked for.
    days_reused: bool = False  # Whether the weekdays came from an earlier run.


def scrape_airport(airport_id, airline_names, week, days_index):
    """
    Fetch one airport's outbound routes, and which weekdays they fly unless
    the ones from an earlier run still apply. Returns None if the API has
    nothing for this id.
    """
    data = fetch(f"{API_BASE}/airport/{airport_id}.json")
    if not isinstance(data, dict) or not data.get("o"):
        return None

    entries, coords, ids = data["a"], data["c"], data["i"]
    origin = parse_airport_entry(entries, coords, data["o"][0])
    if origin is None:
        return None

    result = ScrapedAirport(origin)
    for idx in data.get("d", []):
        dest = parse_airport_entry(entries, coords, idx)
        entry = entries[idx]
        if dest is None or len(entry) < 7 or not isinstance(entry[4], int) or not isinstance(entry[6], list):
            print(f"! Skipping malformed route from {origin['iata']}: {entry!r:.120}")
            continue
        try:
            km = int(geodesic((origin["latitude"], origin["longitude"]), (dest["latitude"], dest["longitude"])).km)
        except (TypeError, ValueError):
            km = 0
        flights_per_day, flights_per_week = parse_frequency(entry[7] if len(entry) > 7 else None)
        result.routes.append({
            "carriers": [
                {"iata": code, "name": airline_names.get(code, code)}
                for code in entry[6] if isinstance(code, str) and AIRLINE_CODE_RE.match(code)
            ],
            "km": km,
            "min": entry[4],
            "iata": dest["iata"],
            "flights_per_day": flights_per_day,
            "flights_per_week": flights_per_week,
        })
        if isinstance(ids[idx], int):
            result.destinations.append({**dest, "id": ids[idx]})

    if week and result.routes:  # Airports without routes have no weekdays to look up.
        reuse = reusable_days(origin["iata"], result.routes, days_index, schedule_season(week[0]))
        if reuse:
            checked_week, days = reuse
            result.days_reused = True
        else:
            checked_week, days = week[0], fetch_operating_days(airport_id, week)
        result.days_week = checked_week.isoformat()
        for route in result.routes:
            route["days"] = dict(days.get(route["iata"], dict.fromkeys(WEEKDAYS, False)))
    return result


def build_airport(info, routes, airport_list, previous, days_week=None):
    """
    One airport record in the output schema. For each field, prefers the site's
    airport list, then the routes API, then the previous output.
    """
    iata = info["iata"]
    sources = [s for s in (airport_list.get(iata), info, previous.get(iata)) if isinstance(s, dict)]

    def pick(key):
        return next((s[key] for s in sources if s.get(key) not in (None, "")), None)

    city = pick("city_name") or iata
    country = pick("country")
    latitude, longitude = pick("latitude"), pick("longitude")
    airport = {
        "iata": iata,
        "name": pick("name") or city,
        "city_name": city,
        "country": country,
        "country_code": pick("country_code"),
        "display_name": f"{city} ({iata}), {country}" if country else f"{city} ({iata})",
        # Stored as strings, like before. Some source values have stray commas.
        "latitude": str(latitude).strip(" ,") if latitude is not None else None,
        "longitude": str(longitude).strip(" ,") if longitude is not None else None,
        "routes": routes,
        # The week the routes' weekdays were checked for, so later runs know if they can reuse them.
        "days_week": days_week,
    }
    old = previous.get(iata) if isinstance(previous.get(iata), dict) else {}
    for key in CARRIED_OVER_FIELDS:
        airport[key] = old.get(key)
    return airport


def write_json(path, data):
    """Write via a temp file and rename, so a crash mid-write can't leave a truncated file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, sort_keys=True, separators=(",", ": "))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scrape airline routes from flightsfrom.com into data/airline_routes.json."
    )
    parser.add_argument(
        "--refresh-days", action="store_true",
        help="look up every airport's weekdays again instead of reusing earlier ones",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    try:
        print("Fetching airline names...")
        airline_names = load_airline_names()
        print("Fetching airport list...")
        airport_list = load_airport_list()
        airport_ids = load_map_airport_ids()
    except FetchError as e:
        print(f"Couldn't load the starting data, nothing was changed: {e}")
        close_sessions()
        return 1
    previous = load_previous_output(OUTPUT_PATH)
    week = pick_week(date.today()) if FETCH_DAYS else None
    days_index = {}
    if week:
        season = schedule_season(week[0])
        print(f"Weekdays: checking {week[0]:%a %d %b} to {week[-1]:%a %d %b %Y} ({season}).")
        if args.refresh_days:
            print("Looking up every airport's weekdays again (--refresh-days).")
        else:
            # A failed run's partial results count too, so a rerun picks up where it stopped.
            days_index = build_days_index(previous, load_previous_output(PARTIAL_PATH))
            known = sum(
                1 for checks in days_index.values()
                if any(schedule_season(checked_week) == season for checked_week, _, _ in checks)
            )
            if known:
                print(f"Weekdays from earlier runs this season: {plural(known, 'airport')}. "
                      "They're reused wherever the routes haven't changed.")
            else:
                print("No weekdays from earlier runs this season, so they'll be looked up for every airport.")
    print(f"Scraping {len(airport_ids)} airports at {REQUESTS_PER_SECOND} requests/second...")

    airports = {}
    destinations_seen = {}  # IATA -> basic info, for airports we only know as a destination
    failed = []
    seen_ids = set(airport_ids)
    processed = 0
    consecutive_failures = 0
    days_looked_up = days_reused = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        pending = {executor.submit(scrape_airport, i, airline_names, week, days_index): i for i in airport_ids}
        while pending and not _abort.is_set():
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                airport_id = pending.pop(future)
                processed += 1
                try:
                    result = future.result()
                except (FetchError, KeyError, IndexError, TypeError, ValueError) as e:
                    if _abort.is_set():
                        continue  # Stopped mid-airport, not a real failure.
                    failed.append(airport_id)
                    consecutive_failures += 1
                    print(f"! Giving up on airport id {airport_id}: {e!r}")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"! {consecutive_failures} airports failed in a row. Stopping.")
                        _abort.set()
                    continue
                consecutive_failures = 0
                if result is None:
                    continue
                iata = result.origin["iata"]
                airports[iata] = build_airport(result.origin, result.routes, airport_list, previous, result.days_week)
                if result.days_reused:
                    days_reused += 1
                elif result.days_week:
                    days_looked_up += 1
                elapsed = time.monotonic() - started
                left = (len(seen_ids) - processed) * elapsed / processed
                note = ", weekdays reused" if result.days_reused else ""
                print(f"[{processed}/{len(seen_ids)}] {iata}: {len(result.routes)} routes{note}, "
                      f"~{format_duration(left)} left")
                for dest in result.destinations:
                    destinations_seen.setdefault(dest["iata"], dest)
                    if dest["id"] not in seen_ids:
                        seen_ids.add(dest["id"])
                        new_future = executor.submit(scrape_airport, dest["id"], airline_names, week, days_index)
                        pending[new_future] = dest["id"]
    except KeyboardInterrupt:
        print("\nInterrupted, stopping...")
        _abort.set()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        close_sessions()

    # Airports without outbound routes still get an entry, as before. The
    # dashboard needs one for every destination, or it hides routes to it.
    for iata in set(airport_list) | set(destinations_seen):
        if iata not in airports:
            airports[iata] = build_airport(destinations_seen.get(iata, {"iata": iata}), [], airport_list, previous)

    took = format_duration(time.monotonic() - started)
    route_count = sum(len(a["routes"]) for a in airports.values())
    stats = []
    if week:
        stats.append(f"Weekdays looked up for {plural(days_looked_up, 'airport')}, reused for {days_reused}.")
    if throttle.cooldowns:
        stats.append(f"Paused {plural(throttle.cooldowns, 'time')} by blocks, "
                     f"{format_duration(throttle.paused_seconds)} in total.")
    if failed or _abort.is_set():
        write_json(PARTIAL_PATH, airports)
        print(f"\nIncomplete run after {took}: {processed}/{len(seen_ids)} airports processed, "
              f"{len(failed)} failed. Left {OUTPUT_PATH.name} untouched and saved what we got "
              f"to {PARTIAL_PATH.name}.")
        if failed:
            print(f"Failed airport ids: {sorted(failed)}")
        if week:
            stats.append("Running it again reuses the weekdays looked up so far.")
        if stats:
            print(" ".join(stats))
        return 1
    write_json(OUTPUT_PATH, airports)
    PARTIAL_PATH.unlink(missing_ok=True)  # Anything useful in it is in the new output now.
    print(f"\nDone in {took}: {len(airports)} airports, {route_count} routes saved to {OUTPUT_PATH}")
    if stats:
        print(" ".join(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

import streamlit as st
import itertools
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Set the page configuration for a wider layout
st.set_page_config(layout="wide")

# Highest Max Connections offered. A single destination can already have over
# 100,000 routes with 3 connections, and 4 or more would run into the millions.
MAX_CONNECTIONS = 3
# A search that finds more routes than this stops and asks for a narrower one.
MAX_ROUTES = 250_000
# Most rows shown in the results table. The full count is still reported.
MAX_TABLE_ROWS = 100_000
DAY_LABELS = [
    ('mon', 'Mon'), ('tue', 'Tue'), ('wed', 'Wed'), ('thu', 'Thu'),
    ('fri', 'Fri'), ('sat', 'Sat'), ('sun', 'Sun'),
]

# --- Data Loading ---
class RouteIndex(NamedTuple):
    """Lookup tables built once from airline_routes.json for the route search."""
    carriers: dict        # origin -> {destination: frozenset of airline names}
    km: dict              # origin -> {destination: flight distance in km}
    inbound: dict         # destination -> set of airports with a flight to it
    first_leg_info: dict  # (origin, destination) -> (days, flights/week, flights/day) as display text
    country: dict         # airport -> country
    display_name: dict    # airport -> display name, e.g. "Bengaluru (BLR), India"
    airport_name: dict    # airport -> airport name, e.g. "Kempegowda International Airport"

@st.cache_resource(show_spinner="Loading route data…")
def load_route_index(file_path):
    """Reads the routes JSON and builds the lookup tables, once per file.
    The result is shared by every session, so it must never be modified."""
    with open(file_path, 'r', encoding='utf-8') as f:
        airport_data = json.load(f)
    # Add a check to ensure each entry is a dictionary and has the 'display_name' key
    airports = {
        iata: info for iata, info in airport_data.items()
        if isinstance(info, dict) and 'display_name' in info
    }
    carriers, km, inbound, first_leg_info = {}, {}, {}, {}
    for origin, info in airports.items():
        for route in info.get('routes', []):
            dest = route.get('iata')
            names = frozenset(c['name'] for c in route.get('carriers', []) if 'name' in c)
            # Flights to unknown airports or without a named airline can't be used.
            if dest not in airports or dest == origin or not names:
                continue
            # A destination listed twice keeps the first entry's details, with the airlines merged.
            dests = carriers.setdefault(origin, {})
            dests[dest] = dests.get(dest, frozenset()) | names
            inbound.setdefault(dest, set()).add(origin)
            if (origin, dest) in first_leg_info:
                continue
            km.setdefault(origin, {})[dest] = route.get('km') or 0
            days = route.get('days') or {}
            days_text = ''
            if days:
                days_text = ', '.join(label for key, label in DAY_LABELS if days.get(key)) or '—'
            first_leg_info[(origin, dest)] = (
                days_text,
                str(route['flights_per_week']) if route.get('flights_per_week') else '',
                str(route['flights_per_day']) if route.get('flights_per_day') else '',
            )
    return RouteIndex(
        carriers=carriers,
        km=km,
        inbound=inbound,
        first_leg_info=first_leg_info,
        country={iata: info.get('country') for iata, info in airports.items()},
        display_name={iata: info['display_name'] for iata, info in airports.items()},
        airport_name={iata: info.get('name', '') for iata, info in airports.items()},
    )

# --- Route Search ---
class Route(NamedTuple):
    """One way to fly from the origin to a destination. A flight can have several
    airlines, and the results table shows one row per airline combination."""
    airports: tuple       # origin, connection airports..., destination
    leg_carriers: tuple   # per flight: frozenset of airlines flying it (excluded ones removed)
    km: int               # total distance flown
    via_countries: tuple  # countries of the connection airports, in order

    @property
    def connections(self):
        return len(self.airports) - 2

def usable_routes(index, excluded):
    """index.carriers with the excluded airlines removed. A flight with no other
    airline is dropped, so no route can use it."""
    if not excluded:
        return index.carriers
    usable = {}
    for origin, dests in index.carriers.items():
        kept = {dest: names - excluded for dest, names in dests.items() if not names <= excluded}
        if kept:
            usable[origin] = kept
    return usable

def find_routes(usable, inbound, origin, targets, max_legs, max_routes, progress_callback=None):
    """Every route from `origin` with at most `max_legs` flights that never visits
    an airport twice and ends at one of `targets` (anywhere if targets is None).

    Returns a list of (airports, leg_carriers) pairs, or None as soon as there are
    more than `max_routes`. progress_callback, if given, is called with the
    fraction (0-1) of the search done so far.
    """
    legs_to_target = None
    if targets is not None:
        # Fewest flights from each airport to a target, so branches that can't
        # get there within max_legs are skipped.
        legs_to_target = dict.fromkeys(targets, 0)
        queue = deque(targets)
        while queue:
            airport = queue.popleft()
            steps = legs_to_target[airport] + 1
            if steps >= max_legs:
                continue
            for previous in inbound.get(airport, ()):
                if previous not in legs_to_target and airport in usable.get(previous, {}):
                    legs_to_target[previous] = steps
                    queue.append(previous)

    found = []
    first_flights = list(usable.get(origin, {}).items())
    for done, (first_stop, first_leg) in enumerate(first_flights):
        if progress_callback:
            progress_callback(done / len(first_flights))
        if legs_to_target is not None and legs_to_target.get(first_stop, max_legs) > max_legs - 1:
            continue
        stack = [((origin, first_stop), (first_leg,))]
        while stack:
            airports, legs = stack.pop()
            if targets is None or airports[-1] in targets:
                found.append((airports, legs))
                if len(found) > max_routes:
                    return None
            flights_left = max_legs - len(legs) - 1  # after taking the next flight
            if flights_left < 0:
                continue
            for next_stop, next_leg in usable.get(airports[-1], {}).items():
                if next_stop in airports:
                    continue
                if legs_to_target is not None and legs_to_target.get(next_stop, max_legs) > flights_left:
                    continue
                stack.append((airports + (next_stop,), legs + (next_leg,)))
    return found

def make_route(index, airports, legs):
    """Builds a Route from find_routes output."""
    return Route(
        airports=airports,
        leg_carriers=legs,
        km=sum(index.km[a][b] for a, b in zip(airports, airports[1:])),
        via_countries=tuple(index.country.get(a) for a in airports[1:-1]),
    )

def reachable_destinations(index, usable, origin, max_legs, carriers, via_sequence,
                           avoid_last_in_dest=False, avoid_reentry=False):
    """Airports reachable from `origin` in at most `max_legs` flights where at
    least one flight can use one of `carriers` (if any are picked), the
    connections pass the via countries in order, and the destination-country
    toggles don't hide the route (see hidden_by_country_toggles). Excluded
    airlines are already removed from `usable`.

    This walks airport by airport instead of listing every route, so it's fast
    for any destination, but it doesn't stop a route from passing through the
    same airport twice. With 2+ connections a destination can therefore, rarely,
    be listed even though no real route to it passes the filters. The results
    table only ever shows real routes.
    """
    country = index.country
    home = country.get(origin)
    via_count = len(via_sequence)
    # For avoid_reentry, each walk also carries a summary of where its connections
    # have been: (countries of earlier connections other than the current
    # airport's, whether the current airport's country was left and re-entered).
    # From a connection, a destination in one of those countries would be hidden.
    # When several walks reach the same state, only what they all share is kept,
    # because a destination is listed if any one of them reaches it unhidden.
    no_history = (frozenset(), False)
    start = (origin, not carriers, 0)
    best = {start: no_history}
    frontier = {start: no_history}
    reachable = set()
    for _ in range(max_legs):
        next_frontier = {}
        for (airport, carrier_ok, via_done), (left, reentered) in frontier.items():
            is_connection = airport != origin
            here = country.get(airport)
            # Flying on from here makes this airport a connection (unless it's the origin).
            if is_connection and via_done < via_count and here in via_sequence[via_done]:
                via_done += 1
            for dest, leg in usable.get(airport, {}).items():
                if dest == origin:
                    continue
                there = country.get(dest)
                ok = carrier_ok or not carriers.isdisjoint(leg)
                if ok and via_done == via_count and dest not in reachable:
                    # Domestic routes (destination in the departure country) are never hidden.
                    hidden = is_connection and there is not None and there != home and (
                        (avoid_last_in_dest and there == here)
                        or (avoid_reentry and (there in left or (there == here and reentered)))
                    )
                    if not hidden:
                        reachable.add(dest)
                state = (dest, ok, via_done)
                if avoid_reentry and is_connection:
                    history = ((left | {here}) - {there}, there in left or (there == here and reentered))
                else:
                    history = no_history
                previous = best.get(state)
                if previous is None:
                    best[state] = next_frontier[state] = history
                elif avoid_reentry:
                    shared = (previous[0] & history[0], previous[1] and history[1])
                    if shared != previous:
                        # Walk on from here again, with fewer flights left but fewer
                        # hidden countries.
                        best[state] = next_frontier[state] = shared
        frontier = next_frontier
    return reachable

# --- Helper Functions ---
def matches_via_sequence(via_countries, via_sequence):
    """Checks that via_countries contains, in order, a country from each set in
    via_sequence. Other countries can come in between."""
    stops = iter(via_countries)
    # any() stops at the first match, so each set is looked for after the previous match.
    return all(any(country in allowed for country in stops) for allowed in via_sequence)

def route_matches(route, carriers=None, via_sequence=None):
    """True if at least one flight can use one of `carriers` and the connections
    pass the via countries in order. An empty (or None) filter matches every route.
    The destination and excluded-airline filters are applied by the search itself.
    via_sequence is a list of country sets, one per picked via slot, in order."""
    if carriers and all(carriers.isdisjoint(leg) for leg in route.leg_carriers):
        return False
    if via_sequence and not matches_via_sequence(route.via_countries, via_sequence):
        return False
    return True

def hidden_by_country_toggles(route, country, avoid_last_in_dest, avoid_reentry):
    """True if one of the destination-country toggles hides this route:
    - avoid_last_in_dest: its last connection is in the destination's country,
      like ICN → DEL → BOM.
    - avoid_reentry: it connects in the destination's country, then somewhere
      outside it, before arriving, like ICN → DEL → KTM → BOM.
    Domestic routes (destination in the departure country) are never hidden."""
    via = route.via_countries
    dest_country = country.get(route.airports[-1])
    if not via or dest_country is None or dest_country == country.get(route.airports[0]):
        return False
    if avoid_last_in_dest and via[-1] == dest_country:
        return True
    if avoid_reentry and dest_country in via:
        first = via.index(dest_country)
        return any(c != dest_country for c in via[first + 1:])
    return False

def carrier_combinations(route, carriers=None):
    """The airline for each flight of every results row of a route, in name order.
    With `carriers`, only the combinations that use at least one of them."""
    for combo in itertools.product(*(sorted(leg) for leg in route.leg_carriers)):
        if not carriers or not carriers.isdisjoint(combo):
            yield combo

def row_count(route, carriers=None):
    """How many rows carrier_combinations gives, without listing them."""
    total = math.prod(len(leg) for leg in route.leg_carriers)
    if not carriers:
        return total
    return total - math.prod(len(leg - carriers) for leg in route.leg_carriers)

def airlines_in_rows(routes, carriers=None):
    """Airlines that appear in at least one results row of these routes."""
    found = set()
    for route in routes:
        legs = route.leg_carriers
        if carriers:
            included = [i for i, leg in enumerate(legs) if not carriers.isdisjoint(leg)]
            if not included:
                continue
            if len(included) == 1:
                # Every row of this route uses an included airline on that one flight.
                for i, leg in enumerate(legs):
                    found |= (leg & carriers) if i == included[0] else leg
                continue
        for leg in legs:
            found |= leg
    return found

def via_slot_options(routes, before, after):
    """Countries that can fill a via slot on at least one route, coming after a
    country from each earlier picked slot and before one from each later slot.
    before/after are lists of country sets."""
    options = set()
    if not before and not after:
        for route in routes:
            options.update(route.via_countries)
        options.discard(None)
        return options
    for route in routes:
        via_countries = route.via_countries
        for country in set(via_countries) - options:
            if country and matches_via_sequence(via_countries, before + [{country}] + after):
                options.add(country)
    return options

def path_text(airports, combo):
    """The Path column, e.g. "ICN  —(Korean Air)→  DEL  —(Air India)→  BLR"."""
    return airports[0] + ''.join(
        f"  —({carrier})→  {airport}" for carrier, airport in zip(combo, airports[1:])
    )

def city_sort_key(display_name):
    """Sort key for airport display names like "Bengaluru (BLR), India": the city."""
    return display_name.split('(')[0]

# Each via slot holds a list of countries (any one of them counts as that stop).
VIA_KEYS = ['via_countries_1', 'via_countries_2', 'via_countries_3']
# session_state keys holding the applied (last Run) value of every sidebar control.
CONTROL_KEYS = [
    'departure', 'max_stops', 'country_filter', 'airport_filter',
    'avoid_dest_connection', 'avoid_dest_reentry',
    'carrier_filter', 'excluded_carrier_filter', *VIA_KEYS,
]

def apply_controls():
    """Run button callback: the values picked in the sidebar form become the applied ones."""
    for state_key in CONTROL_KEYS:
        st.session_state[state_key] = st.session_state[f"{state_key}_widget"]

def restored_widget(widget, label, state_key, **kwargs):
    """Renders a form widget showing the applied value from st.session_state[state_key].

    Streamlit resets a widget whenever its options change, and the filter lists
    narrow after every Run, so the applied value is copied back into the widget
    before rendering.
    """
    widget_key = f"{state_key}_widget"
    st.session_state[widget_key] = st.session_state[state_key]
    return widget(label, key=widget_key, **kwargs)

def progress_step(bar, start, end, text):
    """Shows `text` on the progress bar and returns a callback that maps a 0-1
    fraction of this step onto the [start, end] part of the bar. Updates are
    throttled to ~10 per second and never move the bar backwards."""
    bar.progress(start, text=text)
    last = {'value': start, 'time': time.monotonic()}

    def report(fraction):
        value = start + (end - start) * min(max(fraction, 0.0), 1.0)
        now = time.monotonic()
        if value > last['value'] and now - last['time'] >= 0.1:
            last.update(value=value, time=now)
            bar.progress(value, text=f"{text} {value:.0%}")

    return report

def get_routes(index, usable, origin, targets, max_legs, excluded, progress_bar):
    """Every route for the applied departure, connections, destinations and
    excluded airlines (see find_routes), or None if there are too many to list.

    The last result is kept in session_state, so a Run that only changes the
    airline or via filters reuses it instead of searching again.
    """
    search_key = (origin, max_legs, targets, frozenset(excluded))
    cached = st.session_state.get('route_search')
    if cached and cached.get('key') == search_key:
        return cached['routes']

    report = progress_step(progress_bar, 0.0, 0.8, "Finding routes…")
    found = find_routes(usable, index.inbound, origin, targets, max_legs, MAX_ROUTES, report)
    routes = None if found is None else [make_route(index, airports, legs) for airports, legs in found]
    st.session_state['route_search'] = {'key': search_key, 'routes': routes}
    return routes

def show_results(index, final_routes, selected_carriers, progress_bar):
    """Shows the routes as a table: one row per airline combination, fewest
    stops first, then shortest distance."""
    final_routes = sorted(final_routes, key=lambda r: (r.connections, r.km, r.airports))
    total_rows = sum(row_count(route, selected_carriers) for route in final_routes)
    shown_rows = min(total_rows, MAX_TABLE_ROWS)

    report = progress_step(progress_bar, 0.92, 1.0, "Building results table…")
    table_rows = []
    for route_number, route in enumerate(final_routes):
        if route_number % 500 == 0:
            report(len(table_rows) / shown_rows)
        destination = index.display_name[route.airports[-1]]
        # Days and frequency are for the first flight.
        first_leg = index.first_leg_info[route.airports[:2]]
        for combo in carrier_combinations(route, selected_carriers):
            table_rows.append(
                (destination, route.connections, route.km, path_text(route.airports, combo), *first_leg)
            )
            if len(table_rows) == shown_rows:
                break
        if len(table_rows) == shown_rows:
            break

    df = pd.DataFrame(table_rows, columns=[
        "Destination", "Stops", "Distance (km)", "Path", "Days", "Flights/Week", "Flights/Day",
    ])
    # Fill blanks for routes that don't have days data yet
    info_columns = ["Days", "Flights/Week", "Flights/Day"]
    df[info_columns] = df[info_columns].replace("", "—")

    column_config = {
        "Destination": st.column_config.Column("Destination", width="medium"),
        "Stops": st.column_config.Column("Stops", width="small"),
        "Distance (km)": st.column_config.NumberColumn("Distance (km)", width="small", format="localized"),
        "Path": st.column_config.Column("Path", width="large"),
        "Days": st.column_config.Column("Days", width="medium"),
        "Flights/Week": st.column_config.Column("Flights/Week", width="small"),
        "Flights/Day": st.column_config.Column("Flights/Day", width="small"),
    }

    if total_rows > MAX_TABLE_ROWS:
        st.info(
            f"Showing the first {MAX_TABLE_ROWS:,} of {total_rows:,} routes: fewest stops first, "
            "then shortest distance. Add filters to narrow it down."
        )
    else:
        st.caption(f"{total_rows:,} routes, fewest stops first, then shortest distance.")
    st.dataframe(df, width="stretch", hide_index=True, column_config=column_config)
    progress_bar.empty()

# --- Main Application ---
def main():
    """The main function to run the Streamlit dashboard."""
    
    # Build the data path relative to this script, not the current working
    # directory. Assumes a project structure like:
    # /flights/
    # ├── dashboard/dashboard.py
    # └── data/airline_routes.json
    file_path = Path(__file__).resolve().parent.parent / 'data' / 'airline_routes.json'

    try:
        index = load_route_index(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file was not found at {file_path}")
        return
    except json.JSONDecodeError:
        st.error(f"Error: The file at {file_path} is not a valid JSON file.")
        return

    st.title("✈️ Airline Route Explorer")
    st.markdown("Pick a departure airport and filters in the sidebar, then click **Run** to see the matching routes.")

    # --- Sidebar for User Input ---
    st.sidebar.header("Dashboard Controls")

    # Airport display names (e.g. "Bengaluru (BLR), India") mapped to their IATA
    # codes, for the departure and destination airport dropdowns.
    airport_iata_by_name = {name: iata for iata, name in index.display_name.items()}
    airport_options = sorted(airport_iata_by_name, key=city_sort_key)

    # Fall back to the first airport if the default is not found.
    default_airport = "Bengaluru (BLR), India"
    if default_airport not in airport_iata_by_name:
        default_airport = airport_options[0]

    # --- Applied Controls ---
    # The sidebar controls sit in a form, so picking values doesn't run anything.
    # Clicking Run copies them into these keys (apply_controls) and reruns the
    # script; everything below works from these applied values.
    st.session_state.setdefault('departure', default_airport)
    st.session_state.setdefault('max_stops', 0)
    st.session_state.setdefault('country_filter', [])
    st.session_state.setdefault('airport_filter', [])
    st.session_state.setdefault('avoid_dest_connection', False)
    st.session_state.setdefault('avoid_dest_reentry', False)
    st.session_state.setdefault('carrier_filter', [])
    st.session_state.setdefault('excluded_carrier_filter', [])
    for state_key in VIA_KEYS:
        st.session_state.setdefault(state_key, [])
    # Max Connections used to go up to 5.
    st.session_state.max_stops = min(st.session_state.max_stops, MAX_CONNECTIONS)

    # The departure dropdown can be cleared with its ✕, which leaves None.
    departure_iata = airport_iata_by_name.get(st.session_state.departure)
    max_stops = st.session_state.max_stops
    max_legs = max_stops + 1
    selected_countries = set(st.session_state.country_filter)
    selected_airport_names = set(st.session_state.airport_filter)
    selected_airports = {
        airport_iata_by_name[name] for name in selected_airport_names
        if name in airport_iata_by_name
    }
    avoid_last_in_dest = st.session_state.avoid_dest_connection
    avoid_reentry = st.session_state.avoid_dest_reentry
    selected_carriers = set(st.session_state.carrier_filter)
    excluded_carriers = set(st.session_state.excluded_carrier_filter)
    # Countries picked in each via slot, in slot order. Empty slots are skipped.
    via_choices = [set(st.session_state[state_key]) for state_key in VIA_KEYS]
    via_sequence = [countries for countries in via_choices if countries]

    # --- Main Panel Display ---
    has_routes = bool(index.carriers.get(departure_iata))
    if departure_iata:
        st.header(f"Routes from {index.airport_name[departure_iata]} ({departure_iata})")
    progress_bar = st.progress(0.0, text="Finding routes…")

    # --- Route Search ---
    # Excluded airlines are removed from every flight before searching, so a
    # flight that only they fly can't be used at all.
    usable = usable_routes(index, excluded_carriers)

    # Destination lists: airports reachable under the airline, via and
    # destination-country filters, each list narrowed by the other one.
    reachable = (
        reachable_destinations(
            index, usable, departure_iata, max_legs, selected_carriers, via_sequence,
            avoid_last_in_dest=avoid_last_in_dest, avoid_reentry=avoid_reentry,
        )
        if has_routes else set()
    )
    country_options = {
        index.country.get(d) for d in reachable
        if not selected_airports or d in selected_airports
    } - {None}
    airport_name_options = {
        index.display_name[d] for d in reachable
        if not selected_countries or index.country.get(d) in selected_countries
    }

    # Every route with up to max_stops connections to the picked destinations
    # (to anywhere if none are picked).
    targets = None
    if selected_countries or selected_airports:
        targets = frozenset(
            iata for iata in index.display_name
            if (not selected_countries or index.country.get(iata) in selected_countries)
            and (not selected_airports or iata in selected_airports)
        )
    routes = (
        get_routes(index, usable, departure_iata, targets, max_legs, excluded_carriers, progress_bar)
        if has_routes else []
    )
    # The destination-country toggles filter the search results rather than
    # changing the search, so switching them reuses it.
    if routes and (avoid_last_in_dest or avoid_reentry):
        routes = [
            r for r in routes
            if not hidden_by_country_toggles(r, index.country, avoid_last_in_dest, avoid_reentry)
        ]

    # --- Interdependent Filtering ---
    # Every dropdown lists only the choices that still return routes given all the
    # other applied filters, so the lists narrow after each Run.
    progress_bar.progress(0.85, text="Applying filters…")
    carrier_options, exclude_options = set(), set()
    via_options = [set() for _ in VIA_KEYS]
    final_routes = []
    if routes:
        # Each filter's options come from the routes that pass all the *other* filters.
        for route in routes:
            if route_matches(route, via_sequence=via_sequence):
                for leg in route.leg_carriers:
                    carrier_options |= leg
        routes_before_via = [r for r in routes if route_matches(r, carriers=selected_carriers)]
        via_options = [
            via_slot_options(
                routes_before_via,
                before=[c for c in via_choices[:slot] if c],
                after=[c for c in via_choices[slot + 1:] if c],
            )
            for slot in range(len(VIA_KEYS))
        ]
        # All filters applied: the routes for the results table.
        final_routes = [r for r in routes_before_via if route_matches(r, via_sequence=via_sequence)]
        # The exclude list offers the airlines used in these results, other than the
        # included ones. Excluding any other airline wouldn't change the results.
        exclude_options = airlines_in_rows(final_routes, selected_carriers) - selected_carriers

    # --- Sidebar Form ---
    # Current selections always stay in their own list, so a filter is only
    # ever removed by the user, even if it no longer matches any route.
    # The departure dropdown uses index=None, which gives it a ✕ button to
    # clear the selection (like the ✕ on the multi-select chips).
    with st.sidebar.form("controls"):
        restored_widget(
            st.selectbox, "Select Departure Airport:", 'departure',
            options=airport_options, index=None, placeholder="Choose an airport",
        )
        restored_widget(
            st.slider, "Max Connections:", 'max_stops',
            min_value=0,
            max_value=MAX_CONNECTIONS,
            help="Shows every route with up to this many connections. 0 for direct flights only.",
        )
        restored_widget(
            st.multiselect, "Filter by Destination Country:", 'country_filter',
            options=sorted(country_options | selected_countries),
        )
        restored_widget(
            st.multiselect, "Filter by Destination Airport:", 'airport_filter',
            options=sorted(airport_name_options | selected_airport_names, key=city_sort_key),
        )
        restored_widget(
            st.toggle, "Avoid connecting in the destination country", 'avoid_dest_connection',
            help="Hides routes whose last connection is in the destination's country, "
                 "like ICN → DEL → BOM. Domestic routes (destination in the departure "
                 "country) aren't affected.",
        )
        restored_widget(
            st.toggle, "Avoid re-entering the destination country", 'avoid_dest_reentry',
            help="Hides routes that connect in the destination's country, fly out of it "
                 "and come back, like ICN → DEL → KTM → BOM. Only has an effect with 2 or "
                 "more connections. Domestic routes aren't affected.",
        )
        restored_widget(
            st.multiselect, "Filter by Airline:", 'carrier_filter',
            options=sorted(carrier_options | selected_carriers),
            help="Shows routes that use at least one of these airlines on any leg.",
        )
        restored_widget(
            st.multiselect, "Exclude Airlines:", 'excluded_carrier_filter',
            options=sorted(exclude_options | excluded_carriers),
            help="Hides routes that use any of these airlines on any leg.",
        )

        # --- Additional Filters ---
        st.subheader(
            "Sequential Via Stops",
            help="Each via stop can be in any of the countries picked for it. "
                 "Stops are matched in order (1, then 2, then 3), and other stops can come in between.",
        )
        for slot, (state_key, chosen) in enumerate(zip(VIA_KEYS, via_choices)):
            options = sorted(via_options[slot] | chosen)
            restored_widget(
                st.multiselect, f"Via Country {slot + 1}:", state_key,
                options=options,
                # With no options Streamlit shows "No options to select" and disables it.
                placeholder="Any country" if options else None,
            )

        st.form_submit_button("Run", type="primary", on_click=apply_controls, width="stretch")
        st.caption("Pick as many filters as you like, then click Run. The lists narrow to match after each Run.")

    if not departure_iata:
        progress_bar.empty()
        st.info("Pick a departure airport in the sidebar, then click Run.")
        return
    if not has_routes:
        progress_bar.empty()
        st.warning("This airport has no direct routes in the dataset.")
        return
    if routes is None:
        progress_bar.empty()
        st.warning(
            f"More than {MAX_ROUTES:,} routes match, which is too many to list. "
            "Pick a destination airport or country, or use fewer connections, then click Run."
        )
        return

    # Possible when the same airline is picked in both lists before one Run.
    in_both = sorted(selected_carriers & excluded_carriers)
    if in_both:
        one = len(in_both) == 1
        st.warning(
            f"{', '.join(in_both)} {'is' if one else 'are'} in both Filter by Airline and "
            f"Exclude Airlines, so routes using {'it' if one else 'them'} are hidden."
        )

    if not final_routes:
        progress_bar.empty()
        st.info("No routes match the selected filter(s).")
        return

    show_results(index, final_routes, selected_carriers, progress_bar)

if __name__ == "__main__":
    main()

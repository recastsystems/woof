"""Background declarations and exact time selections, independent of transport.

A declaration is not a payload verification. Preparation retains ownership of
field selectors, surface donors, member byte identity and interpolation halos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
import math
import hashlib
import json
from typing import Sequence

SCHEMA = "gpuwm.background-selection.v1"
CATALOG_SCHEMA = "gpuwm.background-capabilities.v1"
DOCUMENT = "docs/background-contract.md"


class BackgroundWindowError(ValueError):
    """A valid source selection cannot cover the requested forcing window."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, not a boolean or string")
    return value


def _hour(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer number of hours")
    return value


def _utc(value, name):
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    # Existing library callers use naive UTC. Aware values are normalized,
    # never stripped before their offset has been applied.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def capability(source: str) -> dict:
    """Project the live source and authority registries, including local routes.

    Read the existing declaration documents, without network or device probes.
    Missing acquisition does not change ``preparable``. A consumer must
    display the input and preparation obligations as well.
    """
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_profile
    from woof import fetch_routes
    from woof.source_cycles import cycle_grid_for
    from woof.runplan import drivability_for

    adapter = get_source_adapter(_text(source, "source"))
    profile = packaged_profile(adapter.packaged_profile) if adapter.packaged_profile else None
    route = (fetch_routes.route_for(adapter.source_id)
             if adapter.source_id in fetch_routes.route_ids() else None)
    grid = cycle_grid_for(adapter.source_id)
    axis = adapter.time_axis or ("forecast_leads" if adapter.max_forecast_hour > 0 else "supplied_times")
    from woof.source_cli import preparation_runners
    from woof.regional_preparation import preparation_chains
    preparable = bool(adapter.runnable and adapter.runner in preparation_runners()
                      and (profile is None or profile["composition_state"] == "composed"))
    obligations = ["Verify the complete atmospheric and surface state through the preparation owner.",
                   "Verify target geometry, vertical coordinate, interpolation halo and boundary coverage."]
    if adapter.member_set or adapter.source_kind.value == "ensemble_members":
        obligations.append("Verify one source member through its native member identity owner; regional member count is a separate setting.")
    if adapter.composition_requirement:
        obligations.append(adapter.composition_requirement)
    if profile is None:
        obligations.append(f"Field and surface requirements are owned by {adapter.field_mapping}; no field census is inferred from seed_fields.")
    acquisition = ("table" if route is not None else "legacy"
                   if adapter.source_id in fetch_routes.LEGACY_ROUTE_SOURCES else "supplied")
    if acquisition == "supplied":
        obligations.append("Use the existing supplied-input preparation route; no automatic acquisition is declared here.")
    # The same owner routes ordinary forecasts and local-input preparation.
    # A second runner-to-family table would strand newly integrated sources.
    drivability = drivability_for(adapter.source_id)
    operation = drivability.get("chain")
    # An analysis archive without a declared publication window can be
    # selected only from an inspected native time inventory, as plan()
    # requires below. A downloadable preparation chain alone does not
    # make initialization without supplied inputs available.  Nor does a
    # row whose valid times are only ever supplied (a generated forecast
    # file, such as a model-table row): plan() binds a ladder only for
    # declared forecast leads or a declared analysis window, so offering
    # "automatic" there sent the local DA door into its refusal ("needs a
    # supplied native time inventory rather than a fabricated forecast
    # ladder") on every model-table row.
    automatic = (operation in preparation_chains() and grid is not None
                 and axis in ("forecast_leads", "analysis_times"))
    if axis == "analysis_times" and grid is None:
        obligations.append("Supply an inspected native time inventory; no analysis publication window is declared.")
    return dict(source=adapter.source_id, label=adapter.display_title, preparable=preparable,
                initialization_modes=(['prepared'] +
                    (['local'] if drivability.get('requires_source_root') else
                     ['automatic'] if automatic else [])) if preparable else [],
                local_preparation_operation=operation,
                requires_source_root=bool(drivability.get("requires_source_root")),
                preparation_routes=list(drivability.get("routes", ())),
                runner=adapter.runner, source_kind=adapter.source_kind.value,
                default_product=adapter.default_product, required_products=list(adapter.required_products),
                time_axis=axis, cycle_grid=None if grid is None else grid.declaration(),
                forcing_interval_seconds=adapter.forcing_interval_seconds,
                member_set=adapter.member_set, selection_owner=adapter.selection_owner,
                coverage=None if adapter.coverage_window is None else _plain(adapter.coverage_window),
                acquisition=acquisition,
                archive_windows=[_plain(row) for row in adapter.archive_windows],
                credentials=[_plain(row) for row in adapter.credentials],
                acquisition_contract=None if route is None else dict(
                    cycle_hours=list(route.cycle_hours), ladders=_plain(route.ladders),
                    # A173: any whole multiple of the ladder's spacing the
                    # preparation takes; the row lists none.
                    cadences="whole_multiples", default_cadence=route.default_cadence,
                    members=_plain(route.members),
                    files=[_plain(row) for row in route.files], axes=_plain(route.axes),
                    compose=[_plain(row) for row in route.compose],
                    donors=[_plain(row) for row in route.donors], prep=_plain(route.prep)),
                record_subset_supported=None if route is None else route.record_subset_supported,
                authority=None if profile is None else _plain(profile),
                field_mapping=adapter.field_mapping, level_mapping=adapter.level_mapping,
                cadence_mapping=adapter.cadence_mapping, obligations=obligations)


def capability_fingerprint(source: str) -> str:
    """Bind the declared preparation and acquisition obligations, not a clock."""
    declaration = capability(source)
    # Display text and an optional byte-range optimization do not change the
    # scientific selection. Authority pins and donor/file roles do.
    for key in ("label", "obligations", "record_subset_supported"):
        declaration.pop(key, None)
    # Which records a file row (record_index) or donor (records) moves by its
    # index is that same byte-range optimization: the decode reads the same
    # records either way.  Named breakage: a background reviewed before a
    # route took its records by .idx was refused as "the source preparation
    # or acquisition contract changed after review".
    contract = declaration.get("acquisition_contract") or {}
    for key, rows in (("record_index", contract.get("files")), ("records", contract.get("donors"))):
        for row in rows or ():
            row.pop(key, None)
    encoded = json.dumps(declaration, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog() -> dict:
    from woof.source_adapters import source_adapters
    rows = []
    for adapter in source_adapters():
        try:
            rows.append(capability(adapter.source_id))
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            rows.append(dict(source=adapter.source_id, preparable=None, error=str(exc)))
    return dict(schema=CATALOG_SCHEMA, document=DOCUMENT, sources=rows,
                rule="Preparation capability is independent of acquisition and byte-range transport.")


def _selection(adapter, route, product, member, provider, cadence, supplied_member=None):
    """Resolve product semantics before consulting transport member tokens."""
    interval = adapter.forcing_interval_seconds
    if cadence is None:
        # Preserve the original regional hourly forcing preference. These
        # entries are compatibility defaults, not the capability roster.
        from woof.da.background import _LEGACY_BACKGROUND_DEFAULTS
        legacy = _LEGACY_BACKGROUND_DEFAULTS.get(adapter.source_id)
        interval = legacy.forcing_interval_seconds if legacy else interval
        if interval is None or type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0 or interval % 3600:
            raise ValueError("The source needs an explicit whole-hour boundary cadence from its input inventory")
        cadence = int(interval / 3600)
    cadence = _hour(cadence, "forcing cadence")
    if adapter.selection_owner == "woof.era5_member":
        from woof.era5_member import validate_selection
        product = "reanalysis" if product is None else _text(product, "product")
        provider = "cds" if provider is None else _text(provider, "provider")
        member = validate_selection(product_type=product, member=member,
                                    provider=provider, cadence=cadence)
        return product, member, provider, cadence
    product = adapter.default_product if product is None else _text(product, "product")
    if product != adapter.default_product:
        raise ValueError(f"{adapter.source_id} prepares product {adapter.default_product!r}, not {product!r}; select a declared product route")
    if provider is not None:
        raise ValueError("provider is not a product selector for this route; acquisition host selection is separate")
    if adapter.source_kind.value == "ensemble_members":
        if member is not None and (not isinstance(member, str) or not member.strip()):
            raise ValueError("source member must be a nonempty member identifier, not a numeric or boolean coercion")
        if route is None or not adapter.member_set or route.members is None:
            if supplied_member is None or (member is not None and member != supplied_member):
                raise ValueError('The supplied input member is missing or differs from the selected member; verify one native member inventory.')
            member = supplied_member
        else:
            from woof.fetch_routes import resolve_member
            member, _ = resolve_member(route, member)
    elif member is not None:
        raise ValueError(f"{adapter.source_id} product {product!r} does not select a source member")
    elif route is not None and route.members is not None:
        raise ValueError("The acquisition member table contradicts the declared product semantics")
    return product, member, None, cadence


def _leads(adapter, route, cycle, start, end, cadence):
    if route is not None:
        from woof.fetch_routes import resolve_leads
        values = resolve_leads(route, cycle, end - start, cadence=cadence, start_hour=start)
    else:
        from woof import fetch
        if adapter.forecast_time_owner is None:
            raise ValueError("No forecast-lead owner is declared for this supplied-input route; use its inspected time inventory")
        owner = getattr(fetch, adapter.forecast_time_owner)
        if adapter.forecast_time_cycle_argument and cadence != 1:
            raise ValueError("The native hourly route requires a one-hour forcing cadence")
        values = owner(end - start, cycle if adapter.forecast_time_cycle_argument else cadence, start)
    expected = tuple(range(start, end + 1, cadence))
    if tuple(values) != expected or not expected or expected[-1] != end:
        raise ValueError("The source ladder does not cover every required boundary frame and the final bracket")
    return expected


@dataclass(frozen=True)
class Selection:
    source: str
    product: str
    member: str | int | None
    provider: str | None
    time_axis: str
    cycle: str | None
    init: str
    end: str
    forcing_interval_seconds: int
    valid_times: tuple[str, ...]
    forecast_leads: tuple[int, ...]
    fetch_leads: tuple[int, ...]
    publication_lag_seconds: float
    publication_basis: str
    acquisition: str
    source_contract_sha256: str

    def record(self) -> dict:
        return dict(schema=SCHEMA, **_plain(asdict(self)))

    def fetch_hints(self) -> dict | None:
        """Exact transport intent; supplied inventories never trigger a fetch."""
        if self.acquisition == "supplied":
            return None
        from woof.source_adapters import get_source_adapter
        cadence = self.forcing_interval_seconds // 3600
        if self.time_axis == "forecast_leads":
            start, end = self.fetch_leads[0], self.fetch_leads[-1]
            result = dict(source=self.source, cycle=self.cycle[:13], hours=end - start)
            if start:
                result["forecast_start_hour"] = start
        else:
            result = dict(source=self.source, cycle=self.init[:13],
                          hours=(len(self.valid_times) - 1) * cadence)
        if not get_source_adapter(self.source).fetch_entire_window:
            result["cadence"] = cadence
        if self.member is not None:
            result["member"] = self.member
        if get_source_adapter(self.source).selection_owner == "woof.era5_member":
            result.update(era5_product=self.product, era5_provider=self.provider, retrieve=True)
        return result


def plan(source: str, *, init: datetime, now: datetime, run_seconds: float,
         product=None, member=None, provider=None, cadence_hours=None,
         cycle: datetime | None = None,
         inventory_times: Sequence[datetime] | None = None, supplied_member=None,
         probe=None) -> Selection:
    """Bind one request once. Publication timing is a declaration, not a probe.

    ``inventory_times`` must come from the source's native inspector. This
    function verifies its temporal coverage, not the payloads it describes.
    """
    from woof import fetch_routes
    from woof.source_adapters import get_source_adapter
    from woof.source_cycles import cycle_grid_for
    adapter = get_source_adapter(_text(source, "source"))
    facts = capability(adapter.source_id)
    if not facts["preparable"]:
        raise ValueError(f"{adapter.source_id} declares no complete preparation route; {adapter.composition_requirement or adapter.status.value}")
    init, now = _utc(init, "init"), _utc(now, "now")
    if init.minute or init.second or init.microsecond:
        raise ValueError("Initialization must be on a whole hour (UTC); no initial-state interpolation is implied")
    duration = _number(run_seconds, "run_seconds")
    end_time = init + timedelta(seconds=duration)
    route = fetch_routes.route_for(adapter.source_id) if adapter.source_id in fetch_routes.route_ids() else None
    product, member, provider, cadence = _selection(adapter, route, product, member, provider, cadence_hours, supplied_member)
    axis = facts["time_axis"]
    grid = cycle_grid_for(adapter.source_id)
    if adapter.selection_owner == "woof.era5_member":
        from woof.era5_member import validate_selection
        validate_selection(product_type=product, member=member, provider=provider,
                           cadence=cadence, cycle=init)
    if inventory_times is not None:
        if isinstance(inventory_times, (str, bytes)) or len(inventory_times) < 2:
            raise ValueError("A supplied inventory needs at least two actual valid times")
        inventory = tuple(_utc(t, "inventory time") for t in inventory_times)
        if tuple(sorted(set(inventory))) != inventory:
            raise ValueError("Supplied valid times must be sorted and unique")
        if init not in inventory:
            raise BackgroundWindowError("Supplied inputs do not contain the requested initial time")
        final = init + timedelta(hours=math.ceil(duration / (3600 * cadence)) * cadence)
        times = tuple(init + timedelta(hours=i * cadence)
                      for i in range(int((final - init).total_seconds() / (3600 * cadence)) + 1))
        if any(t not in inventory for t in times):
            raise BackgroundWindowError("Supplied inputs have a missing boundary frame or final bracket")
        if axis == "forecast_leads" and cycle is None:
            raise ValueError("A supplied forecast inventory also needs its inspected source cycle")
        if axis == "analysis_times" and cycle is not None:
            raise ValueError("An analysis sequence has independent valid times, not a forecast cycle")
        selected_cycle = _utc(cycle, "cycle") if cycle is not None else None
        if selected_cycle is not None and (selected_cycle > init or selected_cycle.minute or selected_cycle.second or selected_cycle.microsecond):
            raise ValueError("The supplied source cycle must be a whole UTC hour at or before initialization")
        leads = (() if axis != "forecast_leads" else tuple(int((t - selected_cycle).total_seconds() / 3600) for t in times))
        return Selection(adapter.source_id, product, member, provider, axis,
                         None if selected_cycle is None else selected_cycle.isoformat(),
                         init.isoformat(), end_time.isoformat(), cadence * 3600,
                         tuple(t.isoformat() for t in times), leads, (), 0.,
                         "supplied native time inventory; payload identity still requires preparation verification", "supplied",
                         capability_fingerprint(adapter.source_id))
    if axis == "analysis_times":
        if cycle is not None:
            raise ValueError("An analysis sequence has valid times, not one source cycle with forecast leads")
        if grid is None:
            raise ValueError("This analysis archive needs a supplied native time inventory; no publication window is declared")
        last = init + timedelta(hours=math.ceil(duration / (3600 * cadence)) * cadence)
        if grid.record_end is not None and last > _utc(grid.record_end, "record_end"):
            raise BackgroundWindowError("The analysis archive ends before the requested boundary bracket; supply a covering inventory or choose another forcing source")
        times = tuple(init + timedelta(hours=i * cadence) for i in range(int((last - init).total_seconds() / (3600 * cadence)) + 1))
        return Selection(adapter.source_id, product, member, provider, axis, None,
                         init.isoformat(), end_time.isoformat(), cadence * 3600,
                         tuple(t.isoformat() for t in times), (), (), grid.delay_hours * 3600,
                         "declared analysis publication delay; not a payload completeness probe", facts["acquisition"],
                         capability_fingerprint(adapter.source_id))
    if axis != "forecast_leads" or grid is None:
        raise ValueError("This route needs a supplied native time inventory rather than a fabricated forecast ladder")
    candidate = _utc(cycle, "cycle") if cycle is not None else grid.snap(min(init, now))
    if candidate.minute or candidate.second or candidate.microsecond or candidate.hour not in grid.hours or candidate > init:
        raise ValueError("Source cycle must be a declared whole UTC cycle at or before initialization")
    first = candidate
    last_error = "no publication candidate"
    from woof.da.background import _LEGACY_BACKGROUND_DEFAULTS
    legacy = _LEGACY_BACKGROUND_DEFAULTS.get(adapter.source_id)
    while (first - candidate).total_seconds() <= grid.search_hours * 3600:
        start = int((init - candidate).total_seconds() / 3600)
        end = math.ceil((start + duration / 3600) / cadence) * cadence
        # When the window's last lead is due, on its row's expected line:
        # this lag is the answer wherever no probe checks the objects.  A
        # cycle running later than that is waited for by the run's fetch
        # until its row calls the lead late (expected time plus a budget
        # no shorter than the row's measured late spread); planning on the
        # latest posting seen instead put every plan behind the cycle that
        # was out (A136 L1 follow-ups).
        lag = legacy.lag_seconds(end) if legacy else grid.delay(candidate, end) * 3600
        try:
            if start % cadence:
                raise ValueError("Requested initialization is absent from the source's selected boundary cadence")
            if probe is None and cycle is None and (now - candidate).total_seconds() < lag:
                raise ValueError("The requested cycle's complete window is not plausibly published yet")
            leads = _leads(adapter, route, candidate, start, end, cadence)
            # Table routes may need f000 invariants. Acquire the declared
            # prefix, separately from the integration window, until their
            # handoff can address invariant-only dependency closures.
            # The prefix opens at the route's first published lead: a
            # retrospective row with no f000 (the GEFSv12 reforecast,
            # first_lead = 3, lane/cap-gefs-archive) has no invariant frame
            # before it, and asking for f000 refused every window.
            fetch_leads = (_leads(adapter, route, candidate, route.first_lead, end, cadence)
                           if route else leads)
            basis = "declared publication timing; not a payload completeness probe"
            if probe is not None:
                from woof.fetch import probe_cycle_window
                publication = probe_cycle_window(adapter.source_id, candidate, fetch_leads,
                    now=now, probe=probe, cadence=cadence, member=member)
                if publication['available'] is False:
                    raise ValueError(f"The required objects for cycle {candidate.isoformat()} leads {fetch_leads} are not all available")
                if publication['available'] is True:
                    basis = (f"actual object availability on {publication['endpoint']} for leads {fetch_leads}; "
                             "native preparation still verifies payload and member identity")
                if publication['probeable'] and publication['available'] is None:
                    # A host that could not be heard said nothing about the objects, so the
                    # declared timing decides, as it does with no probe at all (GS-05).
                    if cycle is None and (now - candidate).total_seconds() < lag:
                        raise ValueError("The requested cycle's objects could not be checked (a host was not heard) "
                                         "and its complete window is not plausibly published yet")
                    basis = ("declared publication timing; the object check could not reach its host, so "
                             "the fetch checks each object as it downloads")
            times = tuple((candidate + timedelta(hours=h)).isoformat() for h in leads)
            return Selection(adapter.source_id, product, member, provider, axis,
                             candidate.isoformat(), init.isoformat(), end_time.isoformat(), cadence * 3600,
                             times, leads, fetch_leads, lag,
                             basis, facts["acquisition"],
                             capability_fingerprint(adapter.source_id))
        except ValueError as exc:
            last_error = str(exc)
        if cycle is not None:
            break
        candidate = grid.snap(candidate - timedelta(hours=1))
    raise BackgroundWindowError(f"No complete {adapter.source_id} boundary window serves {init.isoformat()}: {last_error}. Supply the required frames, choose an available epoch or change the forcing request explicitly.")


def from_record(record: dict) -> Selection:
    """Read an exact, internally consistent selection without a latest lookup.

    This validates structure and temporal arithmetic. Native preparation must
    still verify the contents, member identity and source geometry of the files.
    """
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("The saved background selection has an unknown schema")
    expected = set(Selection.__dataclass_fields__) | {"schema"}
    if set(record) != expected:
        raise ValueError("The saved background selection has missing or unknown fields")
    values = {k: v for k, v in record.items() if k != "schema"}
    pin = values["source_contract_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Saved source contract must carry a lowercase SHA256 digest")
    for name in ("source", "product", "publication_basis"):
        if _text(values[name], name) != values[name]:
            raise ValueError(f"Saved {name} must be normalized")
    if values["provider"] is not None:
        _text(values["provider"], "provider")
    member = values["member"]
    if member is not None and not (type(member) is int and member >= 0
                                   or isinstance(member, str) and bool(member.strip())):
        raise ValueError("Saved member must be an integer member number or a member identifier")
    axis = values["time_axis"]
    if axis not in ("forecast_leads", "analysis_times", "supplied_times"):
        raise ValueError("Saved time_axis is not a declared temporal representation")
    if values["acquisition"] not in ("legacy", "table", "supplied"):
        raise ValueError("Saved acquisition has an unknown representation")
    seconds = _hour(values["forcing_interval_seconds"], "forcing interval seconds")
    if seconds % 3600:
        raise ValueError("Saved boundary cadence must be an exact whole number of hours")
    lag = values["publication_lag_seconds"]
    if type(lag) not in (int, float) or not math.isfinite(lag) or lag < 0:
        raise ValueError("Saved publication lag must be finite and nonnegative")

    def stamp(text, name):
        _text(text, name)
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Saved {name} is not a timestamp") from exc
        if value.tzinfo is None or value.astimezone(timezone.utc).isoformat() != text:
            raise ValueError(f"Saved {name} must be an explicit canonical UTC timestamp")
        return value

    initial, final = stamp(values["init"], "init"), stamp(values["end"], "end")
    if initial.minute or initial.second or initial.microsecond or final <= initial:
        raise ValueError("Saved initialization must be on an exact UTC hour and precede the end")
    for name in ("valid_times", "forecast_leads", "fetch_leads"):
        if not isinstance(values[name], list):
            raise ValueError(f"Saved {name} must be an array")
        values[name] = tuple(values[name])
    times = tuple(stamp(t, "valid time") for t in values["valid_times"])
    needed = math.ceil((final - initial).total_seconds() / seconds)
    expected_times = tuple(initial + timedelta(seconds=i * seconds) for i in range(needed + 1))
    if times != expected_times:
        raise ValueError("Saved valid times do not exactly cover the initial state and final boundary bracket")
    for name in ("forecast_leads", "fetch_leads"):
        leads = values[name]
        if any(type(h) is not int or h < 0 for h in leads) or tuple(sorted(set(leads))) != leads:
            raise ValueError(f"Saved {name} must contain sorted unique nonnegative integer leads")
    cycle = None if values["cycle"] is None else stamp(values["cycle"], "cycle")
    if cycle is not None and (cycle.minute or cycle.second or cycle.microsecond or cycle > initial):
        raise ValueError("Saved source cycle must be an exact UTC hour at or before initialization")
    if axis == "forecast_leads":
        if cycle is None:
            raise ValueError("Saved forecast selection is missing its source cycle")
        expected_leads = tuple(int((t - cycle).total_seconds() / 3600) for t in times)
        if values["forecast_leads"] != expected_leads:
            raise ValueError("Saved forecast leads do not describe the saved valid times")
        fetch = values["fetch_leads"]
        if values["acquisition"] == "supplied":
            if fetch:
                raise ValueError("Supplied inputs must not carry a fetch ladder")
        else:
            first = 0 if values["acquisition"] == "table" else expected_leads[0]
            if fetch != tuple(range(first, expected_leads[-1] + 1, seconds // 3600)):
                raise ValueError("Saved fetch leads do not describe the required acquisition window")
    elif cycle is not None or values["forecast_leads"] or values["fetch_leads"]:
        raise ValueError("Saved analysis or supplied-time selections cannot masquerade as forecast leads")
    return Selection(**values)

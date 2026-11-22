"""ERA5's native model-level atmosphere is a registry ROW, not a route.

The hybrid closure taught both engines to read a GRIB2 Section-4 pv
coordinate ladder, materialize ``p = A + B*ps`` from it, and integrate
geopotential height hydrostatically up the resulting half levels.  That
is engine work and it is done.  What was still missing is the thing a
user actually needs: a NAME.  Until this row existed, initializing from
ERA5's 137 native model levels meant hand-authoring a mapping, a
composition, a donor mapping and a sealed manifest outside the product
and driving ``woof.mapped_direct`` by hand -- engine-proven, not
shipped.

These tests hold the arbitrary acceptance test at the place it is
easiest to break: the difference between "woof can decode ERA5 model
levels" and "``era5-l137`` is selectable like any other source" must be
table data -- three pinned JSON authorities plus one ``_adapter(...)``
row -- and never a runner, a module, or a branch that names this model.
"""

from __future__ import annotations

import pytest

from woof.source_adapters import (AdapterStatus, SourceKind,
                                   get_source_adapter,
                                   packaged_profile_sources, source_adapters)
from woof.source_authorities import packaged_profile, packaged_profile_ids


ROUTE = "era5-l137"
PROFILE = "era5-model-level-l137-grib2-v1"


def test_the_model_level_route_is_a_registered_source():
    """A name a user can type, resolvable through the same door as any
    other source id or alias."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.source_id == ROUTE
    assert adapter.file_family == "GRIB2"
    assert adapter.source_kind is SourceKind.DETERMINISTIC_STATE
    # Reanalysis: every valid time is an analysis, no forecast leads.
    assert adapter.max_forecast_hour == 0
    for alias in ("era5-model-level", "era5-ml"):
        assert get_source_adapter(alias).source_id == ROUTE


def test_the_route_runs_on_a_packaged_profile_and_no_runner_of_its_own():
    """The arbitrary seam: a shipped mapping, not a code path."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.runnable is True
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.packaged_profile == PROFILE
    assert PROFILE in packaged_profile_ids()
    assert packaged_profile_sources()[ROUTE] == PROFILE
    profile = packaged_profile(PROFILE)
    assert profile["source_format"] == "grib2"
    # The land surface ERA5's model-level product does not publish is
    # BORROWED from the same-hour pressure-level analysis, so the profile
    # is a real cross-source composition, never a pending declaration.
    assert profile["composition_state"] != "pending_cross_source"
    assert profile["data_role"] == "physical_analysis_surface_data"


def test_the_row_declares_the_cadence_and_the_coverage():
    """The two facts `woof domain` needs before any JSON is opened.

    ERA5 is a global reanalysis published hourly: the cadence is the
    hourly spacing of its analyses, and the coverage window is ``None``
    because a global product has no corner to refuse.
    """

    adapter = get_source_adapter(ROUTE)
    assert adapter.forcing_interval_seconds == 3600.0
    assert adapter.coverage_window is None
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED


def test_the_row_states_that_its_land_surface_is_borrowed():
    """A user who reads the registry learns the second file is required
    BEFORE paying for an acquisition, not out of a prep refusal."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.composition_requirement
    assert "pressure-level" in adapter.composition_requirement


def test_the_pressure_level_row_and_the_model_level_row_are_distinct():
    """The existing ``era5`` row keeps its certified pressure-level
    identity; the model-level route does not silently take it over."""

    pressure = get_source_adapter("era5")
    model_level = get_source_adapter(ROUTE)
    assert pressure.source_id != model_level.source_id
    assert pressure.status is AdapterStatus.CERTIFIED
    assert pressure.packaged_profile is None
    assert model_level.credentials == pressure.credentials
    assert model_level.credentials, (
        "ERA5 model levels come from the same CDS account as every other "
        "ERA5 product; a row that declares no credential sends a user to "
        "an authentication failure instead of a setup step")


def test_no_module_names_this_route():
    """The row is the whole difference.  If any importable module grows
    an ``era5-l137`` branch, the arbitrary acceptance test has been lost
    and this test is where that shows up."""

    from pathlib import Path

    package = Path(__file__).parents[1] / "woof"
    offenders = []
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ROUTE in text and path.name != "source_adapters.py":
            offenders.append(str(path.relative_to(package)))
    assert offenders == [], (
        f"{ROUTE} reached code outside the registry table: {offenders}")


@pytest.mark.parametrize("row", source_adapters())
def test_every_runnable_row_still_declares_its_cadence(row):
    """The row added here must not be the one that breaks the property
    the wizard depends on."""

    if not row.runnable:
        return
    if row.time_axis == "supplied_times" and row.model_table is None:
        # A caller-supplied route (the generic mapped source, a woof hex
        # export) reads cadence from the frames the caller supplies, and the
        # wizard asks the caller for it (domain_wizard.py's cadence_h fallback
        # when adapter.forcing_interval_seconds is None). A parameterized
        # model table declares its own native cadence before files exist.
        assert row.forcing_interval_seconds is None
        return
    assert row.forcing_interval_seconds is not None
    assert row.forcing_interval_seconds > 0.0


# ------------------------------------------------------------------
# The fetch door's answer for a source whose bytes woof does not broker
# ------------------------------------------------------------------

def test_the_fetch_refusal_does_not_call_a_runnable_row_unrunnable():
    """A refusal must be TRUE before it can be useful.

    ``era5-l137`` is ``runnable=True`` -- the packaged profile decodes
    its bytes and a preparation reaches rc 0 -- but its bytes come from
    a Copernicus CDS request woof does not broker.  The fetch door's
    fallback answered that with "the registry row is not runnable ...
    nothing in this WOOF-ArWen could read the bytes a download produced",
    which is a false statement of cause: this WOOF-ArWen reads them fine, it
    just cannot go and get them.  A reader who believes it stops.
    """

    from woof import fetch_routes

    with pytest.raises(ValueError) as refusal:
        fetch_routes.route_for(ROUTE)
    text = str(refusal.value)
    assert "the registry row is not runnable" not in text
    # It must name the real breakage and a way forward.
    assert "--source-root" in text


def test_the_row_declares_why_its_bytes_are_not_downloadable():
    """Table data, like every other undownloadable source: the reason
    lives in the fetch-route authority beside 20crv3's, not in a
    fallback sentence generated from the wrong field."""

    from woof import fetch_routes

    assert ROUTE in fetch_routes.refusal_ids()
    with pytest.raises(ValueError) as refusal:
        fetch_routes.route_for(ROUTE)
    text = str(refusal.value)
    assert "Copernicus" in text or "CDS" in text


# ------------------------------------------------------------------
# The prepared-forecast stage's per-source tables
# ------------------------------------------------------------------

def test_the_prepared_stage_tables_cover_every_packaged_profile():
    """A new packaged row must not raise KeyError one stage later.

    ``woof/prepared_single_domain_forecast.py`` carries six per-source
    lookups that every composed mapped profile answers IDENTICALLY --
    the mapped input-manifest schema, the mapped direct and hierarchy
    proof schemas, their legacy companions and the mapped adapter id.
    Spelled as literal id lists, they turned "add a registry row" into
    "add a registry row and remember six more places", and the miss
    surfaced as a bare ``KeyError`` in the stage that runs a prepared
    bundle rather than as a refusal.  Derived from the packaged-profile
    table, they cost a new source nothing -- which is what the arbitrary
    acceptance test asks for.
    """

    from woof.prepared_single_domain_forecast import (
        _HIERARCHY_PROOF_SCHEMA, _LEGACY_HIERARCHY_PROOF_SCHEMAS,
        _LEGACY_PROOF_SCHEMAS, _MAPPED_PACKAGED_PROFILE, _PROOF_SCHEMA,
        _SOURCE_ADAPTER, _SOURCE_SCHEMA)

    tables = {
        "_SOURCE_SCHEMA": _SOURCE_SCHEMA,
        "_PROOF_SCHEMA": _PROOF_SCHEMA,
        "_HIERARCHY_PROOF_SCHEMA": _HIERARCHY_PROOF_SCHEMA,
        "_SOURCE_ADAPTER": _SOURCE_ADAPTER,
        "_LEGACY_PROOF_SCHEMAS": _LEGACY_PROOF_SCHEMAS,
        "_LEGACY_HIERARCHY_PROOF_SCHEMAS": _LEGACY_HIERARCHY_PROOF_SCHEMAS,
    }
    for name, table in tables.items():
        missing = sorted(set(_MAPPED_PACKAGED_PROFILE) - set(table))
        assert missing == [], f"{name} does not answer for {missing}"


def test_the_stage_keeps_the_member_route_told_apart_from_the_rest():
    """Deriving the common rows must not flatten the one that differs:
    the 20CRv3 member route writes its OWN input-manifest schema, and
    that difference is how the stage knows to demand a member manifest.
    """

    from woof.prepared_single_domain_forecast import _SOURCE_SCHEMA

    assert _SOURCE_SCHEMA["20crv3"] == "gpuwm-20crv3-grib2-inputs-v1"
    assert _SOURCE_SCHEMA[ROUTE] == "gpuwm-mapped-composition-inputs-v1"
    assert _SOURCE_SCHEMA["20crv3"] != _SOURCE_SCHEMA[ROUTE]


def test_the_forecast_stage_accepts_the_new_source_by_name():
    """`--source era5-l137` must be a name the forecast stage takes.

    ``SUPPORTED_SOURCES`` was a seventh literal id list, and the prep
    door prints a ready-to-run forecast command for any packaged source
    -- so a row missing from it would have printed a command the very
    next stage refuses.  Derived from the packaged-profile table, the
    two cannot disagree.
    """

    from woof.prepared_single_domain_forecast import (
        SUPPORTED_SOURCES, _MAPPED_PACKAGED_PROFILE)

    assert ROUTE in SUPPORTED_SOURCES
    assert set(_MAPPED_PACKAGED_PROFILE) <= SUPPORTED_SOURCES
    # The generic caller-supplied-mapping id is IN since 2026-09-04: a
    # caller-authored mapped bundle runs through the same forecast
    # validation as a packaged one, checked against the bundle's own
    # certificate rather than a packaged profile.
    assert "mapped" in SUPPORTED_SOURCES


# ------------------------------------------------------------------
# The water state: the same donor fields, packed under the same names
# ------------------------------------------------------------------

#: Canonical name in the composition -> the name the direct pressure-level
#: route decodes the same GRIB1 record under (woof/ingest/grib.py), which
#: is the name the water-temperature assembly reads.
WATER_STATE = {
    "sea_surface_temperature": "SST",
    "sea_ice_fraction": "SEAICE",
    "lake_water_temperature": "LAKE_WATER_TEMP",
    "lake_ice_temperature": "LAKE_ICE_TEMP",
    "lake_ice_depth": "LAKE_ICE_DEPTH",
}


def _direct_route_identity(name):
    """``(center, table, parameter, level_type, level)`` the direct route
    decodes ``name`` from."""

    from woof.ingest import grib

    for identity, decoded in grib._NATIVE_LAKE_SPECS.items():
        if decoded == name:
            return identity
    for (parameter, level_type), (_, decoded) in grib._CANONICAL_SPECS.items():
        if decoded == name:
            return (98, 128, parameter, level_type, 0)
    raise AssertionError(f"the direct route decodes no {name}")


def test_the_composition_borrows_the_water_state_the_direct_route_reads():
    """Named breakage: from ONE donor file the model-level route gave
    677 of 701 lake cells a skin temperature 4.4 K rms away from the
    pressure-level route's, land identical, because its composition
    borrowed no lake state.  The rows must name the same records the
    direct route decodes, or the two routes answer differently over
    water from the same bytes."""

    import json

    from woof.source_authorities import (packaged_composition,
                                          packaged_contributing_mappings)

    composition = packaged_composition(PROFILE)
    borrowed = {name for binding in composition["field_sources"].values()
                for name in binding["fields"]}
    assert set(WATER_STATE) <= borrowed
    [donor_path] = packaged_contributing_mappings(PROFILE).values()
    donor = json.loads(donor_path.read_text(encoding="utf-8"))
    for canonical, decoded in WATER_STATE.items():
        [selector] = donor["fields"][canonical]["selectors"]
        assert (selector["center"], selector["table_version"],
                selector["parameter"], selector["level_type"],
                selector["level_value"]) == _direct_route_identity(decoded), (
            canonical)


def test_the_regular_join_packs_the_water_state_under_the_direct_routes_names():
    """The join is the one place a canonical name becomes the name the
    horizontal mapping reads; a borrowed field it has no name for would
    be decoded and then dropped."""

    from dataclasses import replace

    import numpy as np

    import test_mapped_frameset_streaming as fixture
    from woof.mapped_source import mapped_frames_to_regular_snapshots

    frame = fixture._one_frame()
    fields = dict(frame.fields)
    template = fields["skin_temperature"]
    for offset, canonical in enumerate(WATER_STATE):
        fields[canonical] = replace(
            template, name=canonical,
            values=template.values + float(offset + 1))
    snapshot = mapped_frames_to_regular_snapshots(
        (replace(frame, fields=fields),),
        initialize_absent_hydrometeors=True)[0]
    for canonical, decoded in WATER_STATE.items():
        assert np.array_equal(snapshot.fields[decoded],
                              fields[canonical].values), canonical


def test_the_composition_borrows_the_snow_the_direct_route_reads():
    """Named breakage: from one donor file the model-level route started
    every land cell with no snow where the pressure-level route had up
    to 389 kg m-2 (rms 15 kg m-2 over a May domain whose mountains still
    held snow), because its mapping left the donor's snow unbound.

    The donor record is the one the direct route decodes as SNOW_EC,
    metres of water equivalent, which its soil initializer multiplies by
    1000 to reach kg m-2; the row carries the same factor as its unit
    transform, and the join packs the result as SNOW."""

    import json

    from woof.source_authorities import (packaged_composition,
                                          packaged_contributing_mappings)

    composition = packaged_composition(PROFILE)
    borrowed = {name for binding in composition["field_sources"].values()
                for name in binding["fields"]}
    assert "snow_water_equivalent" in borrowed
    [donor_path] = packaged_contributing_mappings(PROFILE).values()
    field = json.loads(donor_path.read_text(encoding="utf-8"))[
        "fields"]["snow_water_equivalent"]
    [selector] = field["selectors"]
    assert (selector["center"], selector["table_version"],
            selector["parameter"], selector["level_type"],
            selector["level_value"]) == _direct_route_identity("SNOW_EC")
    assert field["units"] == {"source": "m", "target": "kg m-2",
                              "scale": 1000.0}

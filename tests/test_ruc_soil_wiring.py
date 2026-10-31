"""Does the nine-level RUC remap actually reach a RUC forecast?

``woof/ingest/ruc_soil.py`` was ``init_soil_depth_3`` + ``init_soil_3_real``
at max_ulp 0 against ``woof/data/ruc/oracle/soil_ingest.csv`` for a week
before it had a single importer anywhere under ``gpuwm/``.  All seven
initializers called ``preprocess_noah_soil`` unconditionally, so a
``sf_surface_physics=3`` request received four Noah layer MIDPOINTS and ran a
complete, plausible forecast on a soil discretization RUC does not use.

This file is the wiring, bound to code rather than to prose.  Three claims:

1. every initializer routes its soil source through the land-surface seam,
   and a RUC config through that seam gets RUC's nine LEVEL depths;
2. a Noah or Noah-MP config through the same seam is **bitwise** what
   ``preprocess_noah_soil`` returns -- every array, every byte; and
3. the seam is fail-closed on the selector, so an unrecognised
   ``sf_surface_physics`` raises instead of quietly taking Noah's arm.

Claim 2 is structural as well as measured: ``preprocess_ruc_soil`` *calls*
``preprocess_noah_soil`` for the surface bookkeeping rather than copying or
parameterising it, and ``woof/ingest/soil.py`` is not edited at all.  The
byte comparison below is the check on that claim, not the reason to believe
it.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from woof.ingest.ruc_soil import (RUC_LEVEL_DEPTHS_M, RucSoilState,
                                   ruc_soil_depths)
from woof.ingest.ruc_soil import (
    preprocess_land_surface_soil as _preprocess_land_surface_soil,
    preprocess_ruc_soil as _preprocess_ruc_soil)
from woof.ingest.soil import (HRRR_SOIL_NODE_DEPTHS_M, NoahSoilState)
from woof.ingest.soil import (
    preprocess_noah_soil as _preprocess_noah_soil)


# These fixtures hand the soil router the raw SST/SKINTEMP pair on purpose:
# what they pin is WRF's OWN per-cell water-skin fallback
# (module_initialize_real.F:2844-2866), which is exactly what
# `water_temperature_policy = "wrf_compat"` names.  The router refuses that
# pair when nobody declares a decision, so the declaration is made once here
# instead of at every call site below, and the tests keep asserting the
# historical numbers they were written for.
def preprocess_land_surface_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_land_surface_soil(fields, **kwargs)


def preprocess_ruc_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_ruc_soil(fields, **kwargs)


def preprocess_noah_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_noah_soil(fields, **kwargs)


_ROOT = Path(__file__).resolve().parents[1]

#: The same seven files ``tests/test_ruc_admission.py`` names.  Listed rather
#: than globbed, because the claim is about these seven call sites.
_INITIALIZERS = (
    "woof/era5_direct.py",
    "woof/gfs_direct.py",
    "woof/hrrr_hierarchy_direct.py",
    "woof/mapped_direct.py",
    "woof/runtime.py",
    "woof/ingest/nest_init.py",
    "woof/ingest/hrrr_physics.py",
)

_NOAH_SCHEMES = (2, 4)
_RUC = 3


def _era5_fields(ny: int = 3, nx: int = 4) -> dict:
    """A synthetic ERA5 layer source: land, open water, and a sea-ice cell."""
    land = np.ones((ny, nx))
    land[:, -1] = 0.0
    xice = np.zeros((ny, nx))
    xice[0, -1] = 1.0
    return {
        "LANDSEA": land,
        "SKINTEMP": np.full((ny, nx), 288.0),
        "SST": np.full((ny, nx), 286.0),
        "XICE": xice,
        "ST000007": np.full((ny, nx), 289.0),
        "ST007028": np.full((ny, nx), 287.5),
        "ST028100": np.full((ny, nx), 285.0),
        "ST100289": np.full((ny, nx), 282.0),
        "SM000007": np.full((ny, nx), 0.12),
        "SM007028": np.full((ny, nx), 0.18),
        "SM028100": np.full((ny, nx), 0.24),
        "SM100289": np.full((ny, nx), 0.30),
        "TMN": np.full((ny, nx), 281.0),
        "SNOW": np.full((ny, nx), 4.0),
    }


def _gfs_fields(ny: int = 3, nx: int = 4) -> dict:
    fields = _era5_fields(ny, nx)
    for name in list(fields):
        if name.startswith(("ST", "SM")):
            del fields[name]
    for name, value in (("GFS_ST000010", 289.0), ("GFS_ST010040", 287.0),
                        ("GFS_ST040100", 285.0), ("GFS_ST100200", 283.0),
                        ("GFS_SM000010", 0.13), ("GFS_SM010040", 0.19),
                        ("GFS_SM040100", 0.25), ("GFS_SM100200", 0.31)):
        fields[name] = np.full((ny, nx), value)
    return fields


def _hrrr_fields(ny: int = 3, nx: int = 4) -> dict:
    fields = _era5_fields(ny, nx)
    for name in list(fields):
        if name.startswith(("ST", "SM")):
            del fields[name]
    nodes = HRRR_SOIL_NODE_DEPTHS_M.size
    fields["SOILT"] = np.stack(
        [np.full((ny, nx), 290.0 - 2.0 * k) for k in range(nodes)])
    fields["SOILW"] = np.stack(
        [np.full((ny, nx), 0.10 + 0.02 * k) for k in range(nodes)])
    return fields


def _soil_type(shape) -> np.ndarray:
    return np.full(shape, 6.0)   # STAS loam


_SOURCES = {"era5": _era5_fields, "gfs": _gfs_fields, "hrrr": _hrrr_fields}


# ---------------------------------------------------------------------------
# 1.  a RUC config reaches RUC's geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", sorted(_SOURCES))
def test_a_ruc_config_initializes_on_ruc_levels_not_noah_layers(source) -> None:
    """The blocker, closed and checked on all three wired source modes."""
    fields = _SOURCES[source]()
    shape = fields["LANDSEA"].shape
    state = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=_soil_type(shape))

    assert isinstance(state, RucSoilState)
    assert not isinstance(state, NoahSoilState)
    zs, dzs = ruc_soil_depths(9)
    np.testing.assert_array_equal(state.level_depths, zs)
    np.testing.assert_array_equal(state.level_thicknesses, dzs)
    np.testing.assert_array_equal(
        state.level_depths,
        np.asarray(RUC_LEVEL_DEPTHS_M[9], dtype=np.float32))

    for name in ("soil_temperature", "soil_moisture", "liquid_moisture"):
        array = getattr(state, name)
        assert array.shape == (9, *shape), (
            f"{name} is {array.shape}; a RUC run needs nine LEVELS, and a "
            "(4, ...) here is exactly the silent Noah initialization this "
            "gate exists to refuse")
        assert array.dtype == np.float32
    assert np.all((state.soil_temperature >= 170.0)
                  & (state.soil_temperature <= 400.0))
    assert np.all((state.soil_moisture >= 0.0) & (state.soil_moisture <= 1.0))

    land = state.landmask >= 0.5
    if source == "hrrr":
        # A node source has no anchors at all (:1958-1968 with
        # ``flag_soil_levels``), so level 0 is the source's own 0 m node.
        # The layer sources below are the ones WRF anchors.
        return
    # The distinguishing check.  RUC's shallowest level is the GROUND SURFACE
    # itself (0.0 m), so on a LAYER source WRF anchors it with the skin
    # temperature (:1899-1907) -- not the 5 cm layer mean Noah's shallowest
    # midpoint carries.  If the seam were still returning Noah's column this
    # could not hold.  ``atol`` is one float32 ULP at 288 K because the
    # anchored endpoint still goes through WRF's ``(x*d)/d`` interpolation;
    # see test_wrfs_own_interpolation_is_not_exact_at_a_coinciding_node.
    ulp = np.spacing(np.float32(400.0))
    np.testing.assert_allclose(
        state.soil_temperature[0][land], state.tsk[land].astype(np.float32),
        rtol=0, atol=ulp)
    # ...and the deepest level is anchored at TMN, WRF's 3 m bottom.
    np.testing.assert_allclose(
        state.soil_temperature[8][land],
        state.deep_soil_temperature[land].astype(np.float32),
        rtol=0, atol=ulp)


def test_wrfs_own_interpolation_is_not_exact_at_a_coinciding_node() -> None:
    """HRRR's nodes ARE RUC's levels -- and WRF still does not copy them.

    ``woof.ingest.soil.HRRR_SOIL_NODE_DEPTHS_M`` is ``RUC_LEVEL_DEPTHS_M[9]``
    in metres, bit for bit, so this source is the one case with an exactly
    known answer and it is what binds the ingest to WRF's
    ``flag_soil_levels`` arm rather than to the layer arm with its invented
    anchors.

    The expected answer is NOT the source node.  ``init_soil_3_real``
    (:1969-1972) evaluates

        (x(k-1)*(lower - target) + x(k)*(target - upper)) / (lower - upper)

    and ``_bracket`` (:1958-1968) takes the FIRST interval containing an
    interior target, so a target that coincides with node ``k`` is taken from
    the interval ABOVE it and the expression collapses to ``(x(k)*d)/d`` --
    two float32 roundings, not a copy.  Measured on this fixture: 20 of 90
    land values differ from their own source node, all by exactly one ULP.

    So the gate is the WRF expression re-derived here, bitwise, rather than
    the identity a reader would expect.  Reproducing the identity instead
    would be a *correction* to WRF applied silently at the ingest.
    """
    np.testing.assert_array_equal(
        HRRR_SOIL_NODE_DEPTHS_M, np.asarray(RUC_LEVEL_DEPTHS_M[9]))

    fields = _hrrr_fields()
    state = preprocess_ruc_soil(
        fields, soil_type=_soil_type(fields["LANDSEA"].shape))
    land = state.landmask >= 0.5

    depths = np.asarray(RUC_LEVEL_DEPTHS_M[9], dtype=np.float32)
    moved = 0
    for name, field in (("soil_temperature", "SOILT"),
                        ("soil_moisture", "SOILW")):
        got = getattr(state, name)[:, land]
        nodes = fields[field][:, land].astype(np.float32)
        want = np.empty_like(got)
        for level in range(depths.size):
            # target == depths[level]; ``_bracket`` returns level-1 for every
            # interior level and 0 for the surface, and in both cases the
            # surviving term is the node at ``level`` itself.
            below = max(level - 1, 0)
            span = np.float32(depths[level + (level == 0)] - depths[below])
            want[level] = (nodes[level] * span) / span
        np.testing.assert_array_equal(got, want)
        moved += int(np.count_nonzero(got != nodes))

    assert moved > 0, (
        "every value came back exactly equal to its source node, so this "
        "test is no longer measuring WRF's (x*d)/d rounding and the claim "
        "in docs/wrf_ruc_runtime_admission.md should be re-derived")
    # ...and never by more than one ULP, which is what says the remap is a
    # rounding of the identity rather than a different interpolation.
    for name, field in (("soil_temperature", "SOILT"),
                        ("soil_moisture", "SOILW")):
        got = getattr(state, name)[:, land]
        nodes = fields[field][:, land].astype(np.float32)
        gap = np.abs(got.view(np.int32) - nodes.view(np.int32))
        assert int(gap.max()) <= 1, f"{name} moved {int(gap.max())} ULP"


def test_every_initializer_routes_through_the_land_surface_seam() -> None:
    """Seven call sites, each named, each checked in its own file.

    The import is checked with an AST walk rather than a text search: a text
    search for ``ruc_soil`` also matches the three places
    ``woof/core/ruc_runtime.py`` names the module inside an error message,
    which are documentation and not wiring.
    """
    for path in _INITIALIZERS:
        source = (_ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        assert "preprocess_land_surface_soil" in imported, (
            f"{path} does not import the land-surface soil seam, so a RUC "
            "config reaching it is initialised on Noah's four layers")
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "preprocess_noah_soil" not in called, (
            f"{path} still calls preprocess_noah_soil directly; that call "
            "takes no land-surface selector and is the failure this seam "
            "replaces")
        assert "preprocess_land_surface_soil" in called, path


def test_the_gfs_front_door_hands_the_seam_its_elevation_pair() -> None:
    """`woof/gfs_direct.py` arms `adjust_soil_temp_new`, structurally.

    The defect this pins: the GFS door called the seam with neither
    ``terrain`` nor ``source_orography``.  ``preprocess_noah_soil``'s
    guard is all-or-none, so passing NEITHER is accepted in silence and
    the lapse is simply never applied -- no warning, no receipt, no
    refusal.  ``SOURCE_OROGRAPHY`` was decoded the whole time: the GFS
    bridge gate refuses any decode whose ``invariant_fields`` is not
    ``"SOURCE_OROGRAPHY,LANDSEA"``.

    Checked on the AST of the actual call rather than by text search, so
    a mention in a comment cannot satisfy it.
    """
    tree = ast.parse((_ROOT / "woof/gfs_direct.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "preprocess_land_surface_soil"
    ]
    assert len(calls) == 1, f"expected one soil seam call, found {len(calls)}"
    passed = {keyword.arg for keyword in calls[0].keywords}
    missing = {"terrain", "source_orography"} - passed
    assert not missing, (
        f"woof/gfs_direct.py calls the soil seam without {sorted(missing)}, "
        "so WRF's adjust_soil_temp_new elevation lapse is silently skipped "
        "for every GFS run")


def test_the_gfs_elevation_lapse_matches_wrf_and_the_untreated_arm_differs():
    """The lapse is -0.0065 K/m, and NOT arming it is visibly different.

    Counts and slopes, not a boolean, and both arms are run: an
    exact-zero delta between two arms is the signature of a treatment
    that never happened, so the untreated arm has to be shown to differ
    rather than assumed to.
    """
    fields = _SOURCES["gfs"]()
    shape = fields["LANDSEA"].shape
    landmask, terrain, source_orography = _seam_inputs(fields)
    soil_type = _soil_type(shape)

    # `landmask` is held CONSTANT across the arms: it selects the
    # decision surface, and letting it vary with the elevation pair
    # would make this an A/B on two treatments at once.  The single
    # variable is the terrain pair -- exactly what the GFS door was
    # failing to pass.
    untreated = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask)
    treated = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography)

    difference = np.asarray(terrain - source_orography, dtype=np.float64)
    delta = (np.asarray(treated.tsk, dtype=np.float64)
             - np.asarray(untreated.tsk, dtype=np.float64))

    # The RATE, not the mask: which cells count as terrestrial is the
    # decision surface's business and is measured by its own test above
    # (the fixture deliberately makes the static landmask disagree with
    # the met source's LANDSEA).  What this pins is that wherever the
    # lapse DID apply, it applied at WRF's -0.0065 K/m, and that it
    # applied somewhere.
    moved = np.nonzero(delta)
    moved_count = int(delta[moved].size)
    assert moved_count, (
        "treated and untreated TSK are identical everywhere, so the "
        "elevation lapse never ran -- an exact-zero delta between two "
        "arms is a treatment that did not happen")
    expected = -0.0065 * difference[moved]
    matched = int(np.count_nonzero(np.isclose(delta[moved], expected, atol=1e-9)))
    assert matched == moved_count, (
        f"{matched}/{moved_count} moved TSK cells carry WRF's -0.0065 K/m "
        f"lapse; got {delta[moved]} against {expected}")

    # And every cell it moved had a real elevation difference to move on.
    assert int(np.count_nonzero(difference[moved] != 0.0)) == moved_count


# ---------------------------------------------------------------------------
# 2.  Noah and Noah-MP are bitwise unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", _NOAH_SCHEMES)
@pytest.mark.parametrize("source", sorted(_SOURCES))
def test_noah_and_noahmp_soil_is_bitwise_what_it_was(scheme, source) -> None:
    """Every array, every byte, against ``preprocess_noah_soil`` itself.

    Not "close", not "same shape": ``tobytes()`` on both, so a one-ULP drift
    in the freezing-curve SH2O partition or in the lake override would fail.
    """
    fields = _SOURCES[source]()
    soil_type = _soil_type(fields["LANDSEA"].shape)

    # The seam's one change for a Noah geometry is the land-state remap of
    # the SOURCE moisture (woof/ingest/land_state.py): a foreign soil
    # model's moisture (ERA5's HTESSEL) is index-remapped onto Noah's
    # hydraulics first; a Noah source (GFS) and HRRR's RUC (the table's
    # ruc -> noah copy pair) are passed as is.  After that it is
    # preprocess_noah_soil itself, byte for byte.
    from woof.ingest.land_state import convert_source_soil_moisture

    converted, receipt = convert_source_soil_moisture(
        fields, sf_surface_physics=scheme, soil_type=soil_type)
    assert (converted is fields) == (source in ("gfs", "hrrr")), receipt
    reference = preprocess_noah_soil(converted, soil_type=soil_type)
    through_seam = preprocess_land_surface_soil(
        fields, sf_surface_physics=scheme, soil_type=soil_type)

    assert type(through_seam) is type(reference) is NoahSoilState
    assert through_seam.land_state == receipt
    for name in NoahSoilState.__dataclass_fields__:
        if name == "land_state":
            continue
        raw_want = getattr(reference, name)
        raw_got = getattr(through_seam, name)
        if isinstance(raw_want, Mapping) or isinstance(raw_got, Mapping):
            # the moisture-floor receipt is a mapping, not an array; byte
            # identity for it is VALUE equality, not object-pointer bytes
            assert raw_got == raw_want, (
                f"{name} moved for sf_surface_physics={scheme} on the "
                f"{source} source")
            continue
        want = np.asarray(raw_want)
        got = np.asarray(raw_got)
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert got.tobytes() == want.tobytes(), (
            f"{name} moved for sf_surface_physics={scheme} on the {source} "
            "source; the RUC ingest was wired in on the promise that Noah's "
            "path is not merely equivalent but the same call")


def test_the_ruc_seam_does_not_reach_into_the_noah_module() -> None:
    """The structural half of the claim above.

    ``woof/ingest/soil.py`` is not edited by the RUC lane at all -- the RUC
    seam only calls it.  If a future change moves Noah logic into
    ``ruc_soil.py`` and parameterises it, this stops holding and the byte
    comparison above becomes a comparison of a function against itself.
    """
    ruc = (_ROOT / "woof" / "ingest" / "ruc_soil.py").read_text(
        encoding="utf-8")
    tree = ast.parse(ruc)
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert "preprocess_noah_soil" not in defined, (
        "ruc_soil.py now defines its own preprocess_noah_soil; the Noah path "
        "must remain the one in woof/ingest/soil.py")
    assert "NoahSoilState" not in defined


# ---------------------------------------------------------------------------
# 3.  fail-closed, and the failing forms
# ---------------------------------------------------------------------------

def test_an_unrecognised_land_surface_selector_is_refused() -> None:
    """Fail-closed on the selector.

    The failure this seam replaces was not a crash.  It was a complete,
    plausible forecast on the wrong soil geometry, so falling through to
    Noah for an unknown scheme would reproduce it for the next one added.
    """
    fields = _era5_fields()
    soil_type = _soil_type(fields["LANDSEA"].shape)
    for scheme in (0, 1, 5, 7, 8):
        with pytest.raises(ValueError, match="has no soil ingest"):
            preprocess_land_surface_soil(
                fields, sf_surface_physics=scheme, soil_type=soil_type)


def test_the_mapped_contract_path_admits_only_the_ruc_identity_source():
    """The mapped arm's RUC target is DECLARED, and it is a table.

    The RUC target is declared (``soil_contract.RUC_TARGET_LEVEL_DEPTHS_M``
    + ``ruc_soil_remap_policy``) rather than fabricated by widening the
    Noah ``target_layers`` comparison.  What it declares is a TABLE of two
    remap policies, one per source geometry, each dispatching into the
    matching arm of WRF's own ``init_soil_3_real``:

    * a node contract runs ``flag_soil_levels``.  For the RUC-family
      ladders that is the identity, and the admitted arrays are
      byte-identical to what the native SOILT/SOILW arm returns.
    * a LAYER contract runs ``flag_soil_layers``, and its admitted arrays
      are byte-identical to what the native ERA5 ST0000xx arm returns.
      This half REPLACES a refusal.  That refusal ("a layer source has no
      oracled layers-to-levels policy") was true of the mapped route and
      false of the module: the layer arm was already ported and already
      measured against ``era5_layers_noadj_nosst`` at 0 ULP.  It is
      retired here by the measurement below, not by assumption.

    Mapped arrays with no contract at all keep their refusal by name.
    """
    from woof.ingest.ruc_soil import _source_soil_profiles
    from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE,
                                            RUC_TARGET_LEVEL_DEPTHS_M)

    # Mapped arrays with no contract: still refused by name.
    fields = _era5_fields()
    fields[MAPPED_SOIL_TEMPERATURE] = np.zeros((4, 3, 4))
    with pytest.raises(ValueError, match="explicit soil_layer_contract"):
        preprocess_ruc_soil(
            fields, soil_type=_soil_type(fields["LANDSEA"].shape))

    # The identity admission: a contract whose declared nodes ARE RUC's
    # nine level depths hands the source's own samples through unchanged,
    # equal to the native arm on the same arrays.
    rng = np.random.default_rng(11)
    temperature = rng.uniform(275.0, 305.0, (9, 3, 4)).astype(np.float32)
    moisture = rng.uniform(0.05, 0.4, (9, 3, 4)).astype(np.float32)
    node_contract = {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {"depth": depth, "selectors": {
                "soil_temperature": {
                    "format": "grib2", "discipline": 2, "category": 0,
                    "parameter": 2, "level_type": 106, "level_value": depth,
                    "second_level_type": 106, "second_level_value": depth},
                "volumetric_soil_moisture": {
                    "format": "grib2", "discipline": 2, "category": 0,
                    "parameter": 192, "center": 7, "subcenter": 0,
                    "master_table_version": 2, "local_table_version": 1,
                    "level_type": 106,
                    "level_value": depth, "second_level_type": 106,
                    "second_level_value": depth}}}
            for depth in RUC_TARGET_LEVEL_DEPTHS_M],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0}],
        "remap": {"kind": "linear_node_samples",
                  "source_value_location": "level_node",
                  "target_value_location": "layer_midpoint"},
        "missing": {"land": "reject",
                    "ocean": {"stage": "after_horizontal_interpolation",
                              "temperature": "skin_temperature",
                              "moisture": 1.0}},
    }
    mapped = _source_soil_profiles(
        {MAPPED_SOIL_TEMPERATURE: temperature,
         MAPPED_SOIL_MOISTURE: moisture}, node_contract)
    native = _source_soil_profiles(
        {"SOILT": temperature, "SOILW": moisture}, None)
    assert mapped[3] == native[3] == "levels"
    assert np.array_equal(mapped[2], native[2])
    assert np.array_equal(mapped[0], native[0])
    assert np.array_equal(mapped[1], native[1])

    # A layer-source contract (ERA5/GFS shape): RUNS, through the same
    # flag_soil_layers arm the native ERA5 field names select, with WRF's
    # own char2int2 integer-centimetre midpoints 3/17/64/194 cm -- the
    # exact level set of the oracle's era5_layers_noadj_nosst rows.
    layer_contract = {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_layers": [
            {"top": top, "bottom": bottom, "selectors": {
                "soil_temperature": {"format": "netcdf", "name": f"stl{i}"},
                "volumetric_soil_moisture": {
                    "format": "netcdf", "name": f"swvl{i}"}}}
            for i, (top, bottom) in enumerate(
                ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89)), 1)],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0}],
        "remap": {
            "kind": "linear_point_samples",
            "source_value_location": "wrf_integer_cm_layer_midpoint",
            "target_value_location": "layer_midpoint",
            "top_anchor": {"depth": 0.0, "temperature": "skin_temperature",
                           "moisture": "repeat_shallowest"},
            "bottom_anchor": {"depth": 3.0,
                              "temperature": "deep_soil_temperature",
                              "moisture": "repeat_deepest"}},
        "missing": {"land": "reject",
                    "ocean": {"stage": "after_horizontal_interpolation",
                              "temperature": "skin_temperature",
                              "moisture": 1.0}},
    }
    mapped_layers = _source_soil_profiles(
        {MAPPED_SOIL_TEMPERATURE: temperature[:4],
         MAPPED_SOIL_MOISTURE: moisture[:4]}, layer_contract)
    native_layers = _source_soil_profiles(_era5_fields(), None)
    assert mapped_layers[3] == native_layers[3] == "layers"
    assert list(mapped_layers[2]) == list(native_layers[2]) == [3, 17, 64, 194]
    assert mapped_layers[4] == native_layers[4] == 100
    assert np.array_equal(mapped_layers[0], temperature[:4])
    assert np.array_equal(mapped_layers[1], moisture[:4])


# ---------------------------------------------------------------------------
# 4.  the landmask/orography soil seam reaches RUC (ERA5/mapped init paths)
# ---------------------------------------------------------------------------

def _seam_inputs(fields):
    """Static-build landmask + terrain/source-orography for the fixture.

    The landmask deliberately DISAGREES with the met-source LANDSEA at
    [1, 0] (static says water, source says land) so the decision-surface
    claim is measured, not assumed; the terrain sits 400 m above the
    declared source orography on one row and matches it elsewhere.
    """
    shape = fields["LANDSEA"].shape
    landmask = np.asarray(fields["LANDSEA"]).copy()
    landmask[1, 0] = 0.0
    source_orography = np.full(shape, 300.0)
    terrain = source_orography.copy()
    terrain[0, :] += 400.0
    return landmask, terrain, source_orography


def test_ruc_era5_init_receives_the_landmask_orography_seam() -> None:
    """The step-1 merge finding, closed: the exact argument list
    ``era5_direct``/``mapped``/``runtime`` pass (fields + landmask +
    terrain + source_orography) initializes RUC soil instead of raising.
    """
    fields = _era5_fields()
    landmask, terrain, source_orography = _seam_inputs(fields)
    state = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC,
        soil_type=_soil_type(fields["LANDSEA"].shape),
        landmask=landmask, terrain=terrain,
        source_orography=source_orography)
    assert isinstance(state, RucSoilState)
    assert state.soil_temperature.shape == (9, *fields["LANDSEA"].shape)
    assert np.all((state.soil_temperature >= 170.0)
                  & (state.soil_temperature <= 400.0))
    assert np.all((state.soil_moisture >= 0.0)
                  & (state.soil_moisture <= 1.0))
    # The decision surface is the STATIC landmask, not the met-source
    # LANDSEA: the disagreeing cell gets WRF's open-water fill.
    assert state.landmask[1, 0] == 0.0
    assert np.all(state.soil_moisture[:, 1, 0] == np.float32(1.0))
    # ...and the lapse actually moved the adjusted row's land columns.
    unadjusted = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC,
        soil_type=_soil_type(fields["LANDSEA"].shape), landmask=landmask)
    land_row = landmask[0] >= 0.5
    # Levels 1..8 (0..1.6 m) interpolate adjusted samples and must all
    # drop; level 9 sits AT the 3 m TMN anchor, which WRF's lapse never
    # touches (the Noah path bakes terrain into TMN at the static build),
    # so its being byte-identical is the deep-anchor semantics, not a gap.
    assert np.all(state.soil_temperature[:8, 0, land_row]
                  < unadjusted.soil_temperature[:8, 0, land_row])
    assert state.soil_temperature[8, 0, land_row].tobytes() == \
        unadjusted.soil_temperature[8, 0, land_row].tobytes()


@pytest.mark.parametrize("source", ("era5", "gfs", "hrrr"))
def test_ruc_lapse_mirrors_the_noah_geometry_semantics_bitwise(source):
    """Seam-adjusted == hand-adjusted inputs, byte for byte, on land.

    ``adjust_soil_temp_new`` is one increment applied to TSK and every
    soil-temperature input level before EITHER init_soil arm, so running
    the seam must equal running no seam on inputs whose SKINTEMP and
    temperature profiles were shifted by the identical float64 delta.
    Land columns only: on water/ice WRF applies no adjustment, checked
    separately below.
    """
    fields = _SOURCES[source]()
    shape = fields["LANDSEA"].shape
    landmask, terrain, source_orography = _seam_inputs(fields)
    soil_type = _soil_type(shape)

    seam = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography)

    delta = -0.0065 * (terrain - source_orography)
    adjusted = {name: np.asarray(value, dtype=np.float64)
                for name, value in fields.items()}
    land = landmask >= 0.5
    increment = np.where(land, delta, 0.0)
    adjusted["SKINTEMP"] = adjusted["SKINTEMP"] + increment
    for name in list(adjusted):
        if name.startswith(("ST", "GFS_ST")) and name != "ST":
            adjusted[name] = adjusted[name] + increment
    if "SOILT" in adjusted:
        adjusted["SOILT"] = adjusted["SOILT"] + increment[None, ...]
    hand = preprocess_land_surface_soil(
        adjusted, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask)

    for name in ("soil_temperature", "soil_moisture", "liquid_moisture"):
        got = getattr(seam, name)[:, land]
        want = getattr(hand, name)[:, land]
        assert got.tobytes() == want.tobytes(), name
    assert seam.tsk[land].tobytes() == hand.tsk[land].tobytes()

    # Water and sea-ice columns receive NO adjustment: identical to the
    # same call with no terrain/orography at all.
    plain = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask)
    not_land = ~land
    for name in ("soil_temperature", "soil_moisture"):
        assert getattr(seam, name)[:, not_land].tobytes() == \
            getattr(plain, name)[:, not_land].tobytes(), name
    assert seam.tsk[not_land].tobytes() == plain.tsk[not_land].tobytes()


def test_ruc_lapse_respects_wrfs_sanity_guards() -> None:
    """A land cell 3000+ m from its source elevation is left alone
    (module_soil_pre.F:993-1073 guard band), exactly as on Noah."""
    fields = _era5_fields()
    shape = fields["LANDSEA"].shape
    landmask = np.asarray(fields["LANDSEA"]).copy()
    source_orography = np.full(shape, 300.0)
    terrain = source_orography.copy()
    terrain[0, 0] += 3500.0          # outside the +-3000 m band
    terrain[2, 0] += 400.0           # inside: must move
    soil_type = _soil_type(shape)
    seam = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography)
    plain = preprocess_land_surface_soil(
        fields, sf_surface_physics=_RUC, soil_type=soil_type,
        landmask=landmask)
    assert seam.soil_temperature[:, 0, 0].tobytes() == \
        plain.soil_temperature[:, 0, 0].tobytes()
    # Levels above the untouched 3 m TMN anchor must all move.
    assert np.all(seam.soil_temperature[:8, 2, 0]
                  != plain.soil_temperature[:8, 2, 0])


def test_ruc_seam_refusals_survive_the_port() -> None:
    """Genuinely inconsistent inputs still refuse, with WRF's reasons."""
    fields = _era5_fields()
    shape = fields["LANDSEA"].shape
    soil_type = _soil_type(shape)
    landmask, terrain, source_orography = _seam_inputs(fields)
    with pytest.raises(ValueError, match="provided together"):
        preprocess_land_surface_soil(
            fields, sf_surface_physics=_RUC, soil_type=soil_type,
            landmask=landmask, terrain=terrain)
    bad_mask = landmask.copy()
    bad_mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="boolean/0/1"):
        preprocess_land_surface_soil(
            fields, sf_surface_physics=_RUC, soil_type=soil_type,
            landmask=bad_mask, terrain=terrain,
            source_orography=source_orography)
    from woof.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE

    mapped = _era5_fields()
    mapped[MAPPED_SOIL_TEMPERATURE] = np.zeros((4, *shape))
    with pytest.raises(ValueError, match="explicit soil_layer_contract"):
        preprocess_land_surface_soil(
            mapped, sf_surface_physics=_RUC, soil_type=soil_type,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography)


def test_the_gate_above_can_see_a_noah_column_wearing_ruc_clothing() -> None:
    """The falsification, in-test, so a passing run is not the only evidence.

    A ``RucSoilState`` built out of Noah's four-layer column padded to nine
    slots satisfies every shape assertion in the first gate.  What it cannot
    satisfy is the level-0-equals-skin-temperature check, which is the one
    that distinguishes a LEVEL discretization from a layer-midpoint one.  If
    that check ever stops discriminating, this test fails first.
    """
    fields = _era5_fields()
    shape = fields["LANDSEA"].shape
    soil_type = _soil_type(shape)
    noah = preprocess_noah_soil(fields, soil_type=soil_type)

    padded = np.concatenate(
        [noah.soil_temperature,
         np.repeat(noah.soil_temperature[-1:], 5, axis=0)]).astype(np.float32)
    assert padded.shape == (9, *shape)

    land = noah.landmask >= 0.5
    assert not np.array_equal(
        padded[0][land], noah.tsk[land].astype(np.float32)), (
        "Noah's shallowest layer midpoint happens to equal TSK on this "
        "fixture, so the distinguishing check in "
        "test_a_ruc_config_initializes_on_ruc_levels_not_noah_layers is not "
        "discriminating and must be replaced")


# ---------------------------------------------------------------------------
# 6.  one depth function, both doors: the validator checks the depths the
#     remap actually interpolates
# ---------------------------------------------------------------------------


def _linear_contract(source_layers, *, source_value_location):
    """A linear_point_samples contract over the declared source layers."""

    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_layers": [
            {"top": top, "bottom": bottom, "selectors": {
                "soil_temperature": {"format": "netcdf", "name": f"stl{i}"},
                "volumetric_soil_moisture": {
                    "format": "netcdf", "name": f"swvl{i}"}}}
            for i, (top, bottom) in enumerate(source_layers, 1)],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0}],
        "remap": {
            "kind": "linear_point_samples",
            "source_value_location": source_value_location,
            "target_value_location": "layer_midpoint",
            "top_anchor": {"depth": 0.0, "temperature": "skin_temperature",
                           "moisture": "repeat_shallowest"},
            "bottom_anchor": {"depth": 3.0,
                              "temperature": "deep_soil_temperature",
                              "moisture": "repeat_deepest"}},
        "missing": {"land": "reject",
                    "ocean": {"stage": "after_horizontal_interpolation",
                              "temperature": "skin_temperature",
                              "moisture": 1.0}},
    }


def test_a_source_that_states_its_values_at_layer_bottoms_is_admitted():
    """A declared sample geometry is a declaration, not a code change."""

    from woof.ingest.soil import _remap_declared_soil
    from woof.ingest.soil_contract import validate_soil_layer_contract

    bottoms = [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 2.0)]
    contract = _linear_contract(bottoms, source_value_location="layer_bottom")
    validated = validate_soil_layer_contract(contract)
    assert validated["remap"]["source_value_location"] == "layer_bottom"

    shape = (2, 2)
    temperature = np.stack([np.full(shape, v)
                            for v in (281.0, 283.0, 287.0, 291.0)])
    moisture = np.stack([np.full(shape, v)
                         for v in (0.10, 0.20, 0.30, 0.40)])
    tsk = np.full(shape, 275.0)
    deep = np.full(shape, 295.0)
    remapped_t, remapped_m = _remap_declared_soil(
        temperature, moisture, validated, tsk=tsk, deep=deep)

    # Hand-computed linear interpolation on the declared BOTTOMS, bracketed
    # by TSK at 0 m and TMN at 3 m, evaluated at the four Noah midpoints.
    depths = [0.0, 0.02, 0.2, 0.8, 2.0, 3.0]
    t_nodes = [275.0, 281.0, 283.0, 287.0, 291.0, 295.0]
    m_nodes = [0.10, 0.10, 0.20, 0.30, 0.40, 0.40]

    def _line(nodes, z):
        for lower in range(len(depths) - 1):
            if depths[lower] <= z <= depths[lower + 1]:
                span = depths[lower + 1] - depths[lower]
                weight = (z - depths[lower]) / span
                return nodes[lower] + weight * (
                    nodes[lower + 1] - nodes[lower])
        raise AssertionError(z)

    for index, midpoint in enumerate((0.05, 0.25, 0.7, 1.5)):
        assert np.allclose(remapped_t[index],
                           _line(t_nodes, midpoint), rtol=0.0, atol=1e-12)
        assert np.allclose(remapped_m[index],
                           _line(m_nodes, midpoint), rtol=0.0, atol=1e-12)


def test_a_location_the_sampler_has_no_rule_for_is_refused_by_name():
    from woof.ingest.soil_contract import validate_soil_layer_contract

    contract = _linear_contract(
        [(0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89)],
        source_value_location="layer_centroid_by_mass")
    with pytest.raises(ValueError) as caught:
        validate_soil_layer_contract(contract)
    message = str(caught.value)
    assert "layer_centroid_by_mass" in message
    assert "wrf_integer_cm_layer_midpoint" in message
    assert "layer_bottom" in message
    assert "layer_top" in message


def test_colliding_integer_centimetre_midpoints_are_refused_at_the_contract():
    """The validator reads the depths the remap interpolates, not the bottoms.

    These four layers have strictly ordered BOTTOMS (0.005, 0.02, 0.6, 2.8)
    and colliding integer-centimetre MIDPOINTS (0, 1, 31, 170 cm, the first
    landing on the 0 m skin anchor).  Validating the bottoms admitted the
    contract and then handed ``_interp_nodes`` a duplicated node.
    """

    from woof.ingest.soil_contract import validate_soil_layer_contract

    layers = [(0.0, 0.005), (0.005, 0.02), (0.02, 0.6), (0.6, 2.8)]
    contract = _linear_contract(
        layers, source_value_location="wrf_integer_cm_layer_midpoint")
    with pytest.raises(ValueError) as caught:
        validate_soil_layer_contract(contract)
    message = str(caught.value)
    assert "strictly ordered" in message
    assert "the top anchor" in message
    assert "source layer 1" in message
    assert "zero" in message


def test_one_depth_function_answers_both_soil_doors():
    """The depths the validator checks are the depths the remap samples."""

    from woof.ingest.soil_contract import linear_source_sample_depths

    assert linear_source_sample_depths(_linear_contract(
        [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 2.0)],
        source_value_location="layer_bottom")) == (
            0.0, 0.02, 0.2, 0.8, 2.0, 3.0)
    # Re-anchored from (0.0, 0.0, 0.02, 0.2, 0.8, 3.0).  A layer_top source
    # whose shallowest layer starts at the surface STATES the 0 m value, so
    # the skin anchor is not a second node at that depth: it is dropped and
    # the source's own surface value is the node.  The old shape was two
    # values at one depth, which the collision refusal rejected -- for
    # every layer_top source there is, since a soil column starts at 0 m.
    assert linear_source_sample_depths(_linear_contract(
        [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 2.0)],
        source_value_location="layer_top")) == (
            0.0, 0.02, 0.2, 0.8, 3.0)
    # The deep end is symmetric: a layer_bottom source whose deepest layer
    # ends at the deep anchor states that value too.
    assert linear_source_sample_depths(_linear_contract(
        [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 3.0)],
        source_value_location="layer_bottom")) == (
            0.0, 0.02, 0.2, 0.8, 3.0)
    assert linear_source_sample_depths(_linear_contract(
        [(0.0, 0.005), (0.005, 0.02), (0.02, 0.6), (0.6, 2.8)],
        source_value_location="wrf_integer_cm_layer_midpoint")) == (
            0.0, 0.0, 0.01, 0.31, 1.7, 3.0)


def test_a_layer_top_source_starting_at_the_surface_remaps_on_its_own_value():
    """The other half of the sampler's table, usable.

    Every layer-form soil source starts its shallowest layer at 0 m, so a
    `layer_top` declaration always put a source sample on the 0 m skin
    anchor and always hit the collision refusal, whose way out ("separate
    the layers, or declare the location their values are really stated
    at") was not available: the values ARE stated at the layer tops.  The
    source states the surface value, so it is the surface node and the
    anchor is not one.
    """

    from woof.ingest.soil import _remap_declared_soil
    from woof.ingest.soil_contract import (
        linear_sample_plan, validate_soil_layer_contract)

    tops = [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 2.0)]
    contract = _linear_contract(tops, source_value_location="layer_top")
    validated = validate_soil_layer_contract(contract)
    plan = linear_sample_plan(validated)
    assert plan.depths == (0.0, 0.02, 0.2, 0.8, 3.0)
    assert plan.top_anchor is False and plan.bottom_anchor is True
    assert plan.labels == (
        "source layer 1", "source layer 2", "source layer 3",
        "source layer 4", "the bottom anchor")

    shape = (2, 2)
    temperature = np.stack([np.full(shape, v)
                            for v in (281.0, 283.0, 287.0, 291.0)])
    moisture = np.stack([np.full(shape, v)
                         for v in (0.10, 0.20, 0.30, 0.40)])
    tsk = np.full(shape, 275.0)
    deep = np.full(shape, 295.0)
    remapped_t, remapped_m = _remap_declared_soil(
        temperature, moisture, validated, tsk=tsk, deep=deep)

    # Hand-computed on the declared TOPS.  TSK is not a node: the source
    # states the 0 m value itself, and 275.0 appears nowhere in the answer.
    depths = [0.0, 0.02, 0.2, 0.8, 3.0]
    t_nodes = [281.0, 283.0, 287.0, 291.0, 295.0]
    m_nodes = [0.10, 0.20, 0.30, 0.40, 0.40]

    def _line(nodes, z):
        for lower in range(len(depths) - 1):
            if depths[lower] <= z <= depths[lower + 1]:
                span = depths[lower + 1] - depths[lower]
                weight = (z - depths[lower]) / span
                return nodes[lower] + weight * (
                    nodes[lower + 1] - nodes[lower])
        raise AssertionError(z)

    for index, midpoint in enumerate((0.05, 0.25, 0.7, 1.5)):
        assert np.allclose(remapped_t[index],
                           _line(t_nodes, midpoint), rtol=0.0, atol=1e-12)
        assert np.allclose(remapped_m[index],
                           _line(m_nodes, midpoint), rtol=0.0, atol=1e-12)


def test_a_derived_sample_that_lands_on_the_anchor_is_still_a_collision():
    """A rounded location collapsing onto 0 m is not a stated surface value.

    The integer-centimetre midpoint of a 0-0.5 cm layer rounds DOWN onto
    the skin anchor.  That is the derivation collapsing, not the source
    saying where its value sits, so it stays refused with the two nodes
    named.
    """

    from woof.ingest.soil_contract import validate_soil_layer_contract

    contract = _linear_contract(
        [(0.0, 0.005), (0.005, 0.02), (0.02, 0.6), (0.6, 2.8)],
        source_value_location="wrf_integer_cm_layer_midpoint")
    with pytest.raises(ValueError) as caught:
        validate_soil_layer_contract(contract)
    assert "the top anchor" in str(caught.value)
    assert "source layer 1" in str(caught.value)

"""The soil-layer count is a property of the selected land-surface scheme.

WRF resolves ``num_soil_layers`` from ``sf_surface_physics`` rather than
honouring the namelist: ``share/module_check_a_mundo.F`` subroutine
``set_physics_rconfigs`` (:3546-3577) overwrites it per scheme and calls
``wrf_error_fatal`` on a scheme with no entry.  Noah and Noah-MP share
``init_soil_depth_2`` (4 layers, fatal at anything else,
``share/module_soil_pre.F:1128-1151``); RUC uses ``init_soil_depth_3``
(:1153-1194), which tabulates zs for 6 and for 9 and is fatal at 4 or 5.

What this file gates, in the order the numbers travel:

1. :func:`woof.config.soil_layer_count` resolves from the SCHEME, and refuses
   a request the scheme does not define -- including the case that used to
   pass silently, a RUC config whose ``num_soil_layers`` was left at the
   ``RunConfig`` default that belongs to Noah.
2. No soil-layer count is written into ``LAND_SURFACE_SOIL_LAYERS``.  Checked
   over the source, because the point of the table is that it holds *no*
   literal -- neither the old 4 nor a new 9 -- and an equality assertion
   against the scheme constants would still pass if someone typed the numbers
   back in.
3. The VRAM preflight sizes soil arrays from the resolved count.  This is a
   correctness bar, not an estimate: this hardware corrupts near capacity and
   has no ECC, so an understated preflight is a silent wrong-answer machine.
   Measured cost of the old behaviour on a 120x100 domain: the four soil
   arrays were sized (4, 100, 120) instead of (9, 100, 120), understating the
   physics category by 960,000 bytes per domain.
4. wrfout writes ``soil_layers_stag`` at the resolved count and the NetCDF
   reads back at that shape.
5. A four-layer configuration is pinned unchanged, including the writer
   default that stands in for "no land-surface scheme selected".

RUC is still refused as *executable* by
``woof.physics_compat.pending_wrf_physics_components``; that is a separate
gate and this file does not touch it.  Every ``RunConfig`` here is therefore
built directly rather than through :func:`woof.config.load_config` -- which
is also the path that made the old silent-4 reachable, since direct
construction runs no validation at all.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from woof.config import (LAND_SURFACE_SOIL_LAYERS,
                          NO_LAND_SURFACE_SOIL_LAYERS, RunConfig,
                          soil_layer_count, validated_soil_layer_count)
from woof.core import noah as noah_scheme
from woof.core import ruc_contract
from woof.io.wrfout import WrfoutWriter

REPO = Path(__file__).resolve().parents[1]

#: Big enough that a per-layer error is a visible number of bytes, small
#: enough to write in a test.
_NX, _NY, _NZ = 120, 100, 50

_BASE = dict(nx=_NX, ny=_NY, nz=_NZ, dx=3000.0, dy=3000.0, ztop=20000.0,
             dt=15.0, run_seconds=3600.0, mp_physics=8,
             sf_sfclay_physics=1, bl_pbl_physics=1)

_SOIL_FIELDS = ("TSLB", "SMOIS", "SH2O")


def _cfg(scheme: int, layers: int | None = None) -> RunConfig:
    extra = {} if layers is None else {"num_soil_layers": layers}
    return RunConfig(**_BASE, sf_surface_physics=scheme, **extra)


def _physics_bytes(shapes) -> int:
    return sum(4 * math.prod(shape) for shape in shapes.values())


# ---------------------------------------------------------------------------
# 1. the resolver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme,layers,expected", [
    (0, None, noah_scheme.NUM_SOIL_LAYERS),   # no LSM: the wrfout axis only
    (2, 4, noah_scheme.NUM_SOIL_LAYERS),      # Noah, init_soil_depth_2
    (4, 4, noah_scheme.NUM_SOIL_LAYERS),      # Noah-MP, same generator
    (3, 9, ruc_contract.NUM_SOIL_LAYERS),     # RUC, woof's validated geometry
    (3, 6, 6),                                # RUC, WRF's other tabulated zs
])
def test_soil_layer_count_resolves_per_scheme(scheme, layers, expected):
    assert soil_layer_count(_cfg(scheme, layers)) == expected


def test_a_code_built_ruc_config_is_refused_not_silently_given_noahs_four():
    """The exact failure this work exists to close.

    ``RunConfig`` leaves ``num_soil_layers`` at 4 -- Noah's number -- and
    direct construction runs no validation.  The old resolver was
    ``int(cfg.num_soil_layers)``, so this config reported four layers under a
    nine-layer scheme and every consumer downstream believed it.
    """
    ruc = _cfg(3)
    assert ruc.num_soil_layers == noah_scheme.NUM_SOIL_LAYERS
    with pytest.raises(ValueError, match="num_soil_layers must be 6 or 9"):
        soil_layer_count(ruc)


@pytest.mark.parametrize("scheme,layers", [
    (2, 9),   # Noah cannot have RUC's geometry
    (4, 9),   # nor can Noah-MP
    (3, 4),   # nor RUC Noah's
    (3, 5),   # init_soil_depth_3 is explicitly fatal at 5
])
def test_soil_layer_count_refuses_a_geometry_the_scheme_does_not_define(
        scheme, layers):
    with pytest.raises(ValueError, match="num_soil_layers must be"):
        soil_layer_count(_cfg(scheme, layers))


def test_soil_layer_count_refuses_a_scheme_with_no_declared_geometry():
    """``module_check_a_mundo.F:3573-3576``'s fail-closed default."""
    with pytest.raises(ValueError,
                       match="no associated number of soil levels"):
        soil_layer_count(_cfg(99, 4))


def test_validate_run_config_and_the_resolver_cannot_disagree():
    """One implementation of the rule, not two.

    ``validate_run_config`` used to repeat the table lookup inline while
    ``soil_layer_count`` did none, so a loaded config and a code-built one
    answered the same question differently.
    """
    from woof.config import validate_run_config

    with pytest.raises(ValueError, match="num_soil_layers must be 6 or 9"):
        validate_run_config(_cfg(3))


# ---------------------------------------------------------------------------
# 2. the table itself carries no count
# ---------------------------------------------------------------------------

def test_the_scheme_table_is_sourced_from_the_schemes():
    assert LAND_SURFACE_SOIL_LAYERS[2] == (noah_scheme.NUM_SOIL_LAYERS,)
    assert LAND_SURFACE_SOIL_LAYERS[4] == (noah_scheme.NUM_SOIL_LAYERS,)
    # WRF's order, unchanged: the schema states what init_soil_depth_3
    # tabulates, not what woof has finished.
    assert (LAND_SURFACE_SOIL_LAYERS[3]
            == tuple(ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS))
    assert NO_LAND_SURFACE_SOIL_LAYERS == LAND_SURFACE_SOIL_LAYERS[0][0]


def test_what_the_scheme_defines_and_what_gpuwm_validated_stay_separate():
    """RUC defines two geometries; woof has validated one of them.

    Conflating those -- by reading the validated count off the schema tuple's
    first element -- would make WRF's own table order essential and would
    silently relabel the other geometry as validated if WRF reordered it.
    """
    assert validated_soil_layer_count(3) == ruc_contract.NUM_SOIL_LAYERS
    assert validated_soil_layer_count(3) in LAND_SURFACE_SOIL_LAYERS[3]
    assert len(LAND_SURFACE_SOIL_LAYERS[3]) > 1
    for single in (0, 2, 4):
        assert (validated_soil_layer_count(single),) \
            == LAND_SURFACE_SOIL_LAYERS[single]
    # The refusal has to name it, or a RUC config that forgot the geometry is
    # told only that it guessed wrong.
    with pytest.raises(ValueError,
                       match=f"validated geometry is "
                             f"{ruc_contract.NUM_SOIL_LAYERS}"):
        soil_layer_count(_cfg(3))


def test_the_scheme_table_holds_no_soil_layer_literal():
    """Not one hardcoded 4 traded for one hardcoded 9.

    Read over the source rather than the value: an equality test against the
    scheme constants passes just as happily if the numbers are typed back in.
    The provider table may carry the ``sf_surface_physics`` selector as a key
    and module/attribute NAMES as values -- no count.
    """
    tree = ast.parse((REPO / "woof" / "config.py").read_text())
    providers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_PROVIDERS"
                for t in node.targets)]
    assert len(providers) == 1, "the provider table is assigned once"
    table = providers[0].value
    assert isinstance(table, ast.Dict)
    assert set(LAND_SURFACE_SOIL_LAYERS) == {
        key.value for key in table.keys}, "every schema scheme has a provider"
    for key, value in zip(table.keys, table.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, int)
        literals = [n.value for n in ast.walk(value)
                    if isinstance(n, ast.Constant)
                    and not isinstance(n.value, (str, type(None)))]
        assert literals == [], (
            f"sf_surface_physics={key.value} states its soil-layer count as "
            f"the literal(s) {literals} instead of naming the module that "
            "owns the scheme")


def test_the_scheme_table_defers_the_forecast_side_read(monkeypatch):
    """Asking about Noah must not import RUC's forecast-side contract.

    This is what lets the RW-WPS preprocessing wheel stage woof/config.py
    without staging woof/core/ruc_contract.py.  If the table ever goes back
    to a plain dict built at import time, that distribution stops importing.
    """
    import sys

    from woof.config import _LandSurfaceSoilLayers

    table = _LandSurfaceSoilLayers()
    forbidden = "woof.core.ruc_contract"
    monkeypatch.setitem(sys.modules, forbidden, None)
    # ``None`` in sys.modules makes any import of it raise, so a stray read
    # cannot pass by finding an already-imported module.
    assert table[2] == (noah_scheme.NUM_SOIL_LAYERS,)
    assert table[0] == (noah_scheme.NUM_SOIL_LAYERS,)
    assert table[4] == (noah_scheme.NUM_SOIL_LAYERS,)
    with pytest.raises(ImportError):
        table[3]


def test_noahs_mapped_ingest_target_cannot_drift_from_noahs_own_count():
    from woof.ingest.soil_contract import (NOAH_LAYER_BOUNDS_M,
                                            NOAH_TARGET_SOIL_LAYERS)

    assert NOAH_TARGET_SOIL_LAYERS == noah_scheme.NUM_SOIL_LAYERS
    assert len(NOAH_LAYER_BOUNDS_M) == noah_scheme.NUM_SOIL_LAYERS


def test_rucs_kernel_depth_tables_match_the_ingest_table():
    """Every geometry ``ruc.cu`` compiles at carries WRF's own depths.

    This replaces a LENGTH check.  Until the RUC_NZS lift, the kernel held one
    table, ``__constant__ real ruc_soil_layer_depth[9]``, and this test
    asserted that its extent was ``ruc_contract.NUM_SOIL_LAYERS``.  That
    premise died with the lift -- the extent is now ``[RUC_NZS]`` and the
    literals are selected by an ``#if`` ladder -- so the guard is rewritten
    rather than deleted, into a strictly STRONGER one: it pins the VALUES of
    every arm, not just the count of one.  A length check was perfectly happy
    with six wrong depths.

    The authority is ``woof.ingest.ruc_soil.ruc_soil_depths``, the single
    transcription of WRF's ``init_soil_depth_3`` that is oracle-matched
    against 4.7.1 ``real.exe``'s ZS/DZS at both geometries.  The device
    cannot import it -- ``__constant__`` needs literals -- so the two
    transcriptions are pinned to each other here.
    """
    from woof.ingest.ruc_soil import ruc_soil_depths

    source = (REPO / "woof" / "core" / "kernels" / "ruc.cu").read_text(
        encoding="utf-8")
    assert "ruc_soil_layer_depth[RUC_NZS]" in source

    # Scoped to the depth table's own sentinel block.  The RUC_NZS ladder
    # higher up the file carries the same `#if RUC_NZS == 9` / `#elif == 6`
    # arms for the derived macros, and an unscoped search finds those first.
    table = re.search(
        r"// >>> RUC_NZS DEPTH TABLE >>>\n(.*?)// <<< RUC_NZS DEPTH TABLE <<<",
        source, re.S)
    assert table is not None, "ruc.cu has no sentinel-delimited depth table"

    for count in ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS:
        arm = re.search(
            rf"#(?:if|elif) RUC_NZS == {count}\n.*?"
            r"ruc_soil_layer_depth\[RUC_NZS\] = \{(.*?)\}",
            table.group(1), re.S)
        assert arm is not None, f"ruc.cu has no depth table at {count} levels"
        literals = np.asarray(
            [float(token) for token in re.findall(r"(-?\d+\.\d+)f",
                                                  arm.group(1))],
            dtype=np.float32)
        expected = np.asarray(ruc_soil_depths(count)[0], dtype=np.float32)
        assert len(literals) == count
        assert np.array_equal(literals.view(np.uint32),
                              expected.view(np.uint32)), (
            f"ruc.cu's {count}-level depths are not WRF's")

    assert len(ruc_contract.RUC_SOIL_LEVELS_M) == ruc_contract.NUM_SOIL_LAYERS


# ---------------------------------------------------------------------------
# 3. the VRAM preflight
# ---------------------------------------------------------------------------

def test_vram_preflight_sizes_soil_from_the_resolved_count():
    from woof.core.preflight import physics_array_shapes

    noah = physics_array_shapes(_cfg(2, 4))
    ruc = physics_array_shapes(_cfg(3, 9))
    soil_slots = ("fields/smois", "fields/tslb", "fields/sh2o",
                  "fields/smcrel")
    for slot in soil_slots:
        assert noah[slot] == (noah_scheme.NUM_SOIL_LAYERS, _NY, _NX)
        assert ruc[slot] == (ruc_contract.NUM_SOIL_LAYERS, _NY, _NX)
    # ``set(noah) == set(ruc)`` held while RUC had no runtime and is now
    # false: admitting the scheme allocated its own state, so the delta is
    # decomposed instead of being asserted away.  Every added slot must be a
    # RUC name -- an addition from somewhere else would show up here as an
    # unaccounted slot rather than being absorbed into a total.
    from woof.core.ruc_runtime import (RUC_DIAGNOSTICS_2D, RUC_STATE_2D,
                                        RUC_STATE_3D,
                                        RUC_FRACTIONAL_SEAICE_FIELDS)
    from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS

    added = set(ruc) - set(noah)
    assert not set(noah) - set(ruc), "RUC dropped a Noah array"
    assert added == {
        f"fields/{name}" for name in (
            *RUC_STATE_2D, *RUC_STATE_3D, *RUC_DIAGNOSTICS_2D,
            *RUC_FRACTIONAL_SEAICE_FIELDS,
            *SURFACE_PRECIPITATION_FIELDS, "gsw",
        )
    }
    for name in RUC_STATE_3D:
        assert ruc[f"fields/{name}"] == (
            ruc_contract.NUM_SOIL_LAYERS, _NY, _NX), name

    extra_layers = ruc_contract.NUM_SOIL_LAYERS - noah_scheme.NUM_SOIL_LAYERS
    deeper_soil = 4 * len(soil_slots) * extra_layers * _NY * _NX
    ruc_2d = 4 * (len(RUC_STATE_2D) + len(RUC_DIAGNOSTICS_2D)) * _NY * _NX
    ruc_3d = 4 * len(RUC_STATE_3D) * ruc_contract.NUM_SOIL_LAYERS * _NY * _NX
    arw_surface_seams = 4 * (
        len(RUC_FRACTIONAL_SEAICE_FIELDS)
        + len(SURFACE_PRECIPITATION_FIELDS) + 1
    ) * _NY * _NX
    assert deeper_soil == 960_000
    assert ruc_2d == 768_000
    assert ruc_3d == 864_000
    assert arw_surface_seams == 1_152_000
    assert (_physics_bytes(ruc) - _physics_bytes(noah)
            == deeper_soil + ruc_2d + ruc_3d + arw_surface_seams
            == 3_744_000)


def test_the_nine_layer_preflight_estimate_is_larger_end_to_end():
    """Through ``estimate_domain``, not just the shape dict.

    A shape table that is right while the itemizer drops the category would
    still understate the number the launch gate compares against.
    """
    from woof.core.preflight import estimate_domain
    from woof.experiment import DomainConfig

    def domain(cfg: RunConfig) -> DomainConfig:
        return DomainConfig(
            grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, parent_time_step_ratio=1,
            history_interval_s=3600.0, run=cfg, time_step=15)

    from woof.core.ruc_runtime import (RUC_DIAGNOSTICS_2D, RUC_STATE_2D,
                                        RUC_STATE_3D,
                                        RUC_FRACTIONAL_SEAICE_FIELDS)
    from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS

    noah = estimate_domain(domain(_cfg(2, 4)))
    ruc = estimate_domain(domain(_cfg(3, 9)))
    extra_layers = ruc_contract.NUM_SOIL_LAYERS - noah_scheme.NUM_SOIL_LAYERS
    # The same decomposition as the shape test above, carried end to end.
    # Before RUC was admitted this was the deeper soil column alone; the two
    # RUC terms are what admitting the scheme added, and they are named so an
    # unaccounted growth cannot hide inside a single total.
    expected_delta = (
        4 * 4 * extra_layers * _NY * _NX
        + 4 * (len(RUC_STATE_2D) + len(RUC_DIAGNOSTICS_2D)) * _NY * _NX
        + 4 * len(RUC_STATE_3D) * ruc_contract.NUM_SOIL_LAYERS * _NY * _NX
        + 4 * (
            len(RUC_FRACTIONAL_SEAICE_FIELDS)
            + len(SURFACE_PRECIPITATION_FIELDS) + 1
        ) * _NY * _NX)
    from woof.core.ruc_memory import ruc_runtime_memory_bytes
    pools = ruc_runtime_memory_bytes(_NY * _NX, ruc_contract.NUM_SOIL_LAYERS)
    expected_delta += sum(pools[name] for name in (
        "driver_workspace", "sfctmp_workspace", "tables"))
    assert (ruc.category_bytes("physics") - noah.category_bytes("physics")
            == expected_delta)
    # Resident, because all of it is persistent physics state: it must land in
    # the number the launch gate compares against, not in a transient peak
    # that a later pass could reuse away.  On this hardware an understated
    # preflight is a correctness failure, not an estimate.
    assert ruc.resident_bytes - noah.resident_bytes == expected_delta
    assert ruc.transient_bytes - noah.transient_bytes == pools["sfctmp_outputs"]


# ---------------------------------------------------------------------------
# 4/5. wrfout writes and reads back at the resolved count
# ---------------------------------------------------------------------------

def _write_soil_frame(path: Path, layers: int, *, ny=6, nx=8, nz=3):
    """One frame carrying only the three soil fields, at ``layers`` rank."""
    frame = {}
    for index, name in enumerate(_SOIL_FIELDS):
        block = np.arange(layers * ny * nx, dtype=np.float32)
        frame[name] = (block + 1000.0 * index).reshape(layers, ny, nx)
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                      soil_layers=layers, field_schema=frame) as writer:
        writer.write_frame("1974-04-03_12:00:00", frame)
    return frame


@pytest.mark.parametrize("scheme,requested", [
    (3, 9),   # RUC: the count this work exists to admit
    (2, 4),   # Noah: pinned unchanged from today
])
def test_wrfout_writes_and_reads_back_the_schemes_soil_rank(
        tmp_path, scheme, requested):
    cfg = _cfg(scheme, requested)
    layers = soil_layer_count(cfg)
    assert layers == requested
    path = tmp_path / f"wrfout_soil_{layers}.nc"
    written = _write_soil_frame(path, layers)
    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions["soil_layers_stag"].size == layers
        for name in _SOIL_FIELDS:
            var = ds.variables[name]
            assert var.dimensions == ("Time", "soil_layers_stag",
                                      "south_north", "west_east")
            assert var.shape == (1, layers, 6, 8)
            # WRF marks the soil axis staggered; a 9-layer file must not lose
            # that just because the length changed.
            assert var.stagger == "Z"
            np.testing.assert_array_equal(var[0], written[name])


def test_a_four_layer_wrfout_is_byte_shaped_exactly_as_today(tmp_path):
    """The 4-layer pin, including the writer's no-scheme default.

    Omitting ``soil_layers`` used to take a literal ``4``; it now takes the
    schema's no-LSM axis.  Both must still be 4, or every frozen idealized
    wrfout header moves.
    """
    path = tmp_path / "wrfout_default_axis.nc"
    with WrfoutWriter(path, nx=8, ny=6, nz=3, dx=1000.0, dy=1000.0) as writer:
        writer.write_frame("1974-04-03_12:00:00", {})
    with netCDF4.Dataset(path) as ds:
        assert ds.dimensions["soil_layers_stag"].size == 4
    assert NO_LAND_SURFACE_SOIL_LAYERS == 4
    assert soil_layer_count(_cfg(0)) == 4
    assert soil_layer_count(_cfg(2, 4)) == 4


def test_wrfout_refuses_a_soil_field_that_is_not_the_declared_rank(tmp_path):
    """Nine layers of TSLB into a four-layer file is a shape error, loudly.

    This is the guard that makes a missed caller survivable rather than a
    silently truncated soil column.
    """
    frame = {"TSLB": np.zeros((9, 6, 8), dtype=np.float32)}
    with pytest.raises(ValueError, match="must have shape"):
        WrfoutWriter(tmp_path / "mismatch.nc", nx=8, ny=6, nz=3, dx=1000.0,
                     dy=1000.0, soil_layers=4, field_schema=frame)


def test_every_wrfout_caller_with_a_config_resolves_the_soil_axis():
    """No production writer may fall back to the no-scheme default.

    ``woof/runtime.py``, ``offline_child_run.py`` and
    ``offline_child_smoke.py`` all constructed a ``WrfoutWriter`` without
    ``soil_layers`` and so took the old literal 4 while holding a cfg that
    could have answered.  Checked over the source: the runtime paths are GPU
    paths, so exercising them here would need a device.
    """
    for module in ("runtime.py", "offline_child_run.py",
                   "offline_child_smoke.py", "io/wrfout.py"):
        source = (REPO / "woof" / module).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name not in ("WrfoutWriter", "AsyncDomainWrfoutWriter"):
                continue
            keywords = {kw.arg for kw in node.keywords}
            assert "soil_layers" in keywords, (
                f"woof/{module}:{node.lineno} constructs {name} without "
                "soil_layers and would take the no-scheme axis for whatever "
                "land-surface scheme the run selected")


#: WRF 4.7.1 ``real.exe`` ``wrfinput_d01`` ZS/DZS, as ``ncdump -v ZS,DZS``
#: prints them, for the summer reference case run at each RUC geometry that
#: ``share/module_check_a_mundo.F:3574-3581`` admits.  The six-level row is the
#: OPERATIONAL geometry -- every ``sf_surface_physics = 3`` namelist in the
#: reference tree sets ``num_soil_layers = 6``, none sets 9.
#:
#: ``DZS(7) = 0.4999999`` at nine levels is not a typo.  It is FP32
#: ``1.30 - 0.80`` in WRF's own expression order, and a derivation that
#: computes in float64 and rounds once returns 0.5 instead.  That single word
#: is the whole reason this table is measured against ``real.exe`` rather than
#: asserted from arithmetic.
WRF_471_REAL_EXE_RUC_GEOMETRY = {
    6: (
        (0.0, 0.05, 0.2, 0.4, 1.6, 3.0),
        (0.025, 0.125, 0.175, 0.7, 1.3, 0.7),
    ),
    9: (
        (0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0),
        (0.005, 0.025, 0.045, 0.13, 0.25, 0.35, 0.4999999, 1.0, 0.7),
    ),
}


def test_ruc_soil_geometry_matches_wrf_real_exe_at_every_admitted_geometry():
    """The forecast-side geometry is a PARAMETER, oracle-matched at each row.

    This is the property the lane exists to create.  Before it,
    ``ruc_soil_geometry()`` took no argument and transcribed WRF's derivation
    a second time over a nine-element constant, so the six-level geometry --
    the one the operational namelists actually run -- was reachable from
    ``woof.ingest.ruc_soil`` and unreachable from the forecast module.
    """
    import numpy as np

    from woof.core.ruc import ruc_soil_geometry
    from woof.core.ruc_contract import WRF_SUPPORTED_NUM_SOIL_LAYERS

    assert (tuple(sorted(WRF_471_REAL_EXE_RUC_GEOMETRY))
            == tuple(sorted(WRF_SUPPORTED_NUM_SOIL_LAYERS)))

    for count, (zs_ref, dzs_ref) in WRF_471_REAL_EXE_RUC_GEOMETRY.items():
        zs, dzs = ruc_soil_geometry(count)
        assert zs.dtype == np.float32 and dzs.dtype == np.float32
        assert len(zs) == count and len(dzs) == count
        # ncdump prints float32 at %.7g; compare at exactly that precision so
        # the assertion is against the printed oracle, not a re-rounding.
        assert [f"{float(v):.7g}" for v in zs] == [f"{v:.7g}" for v in zs_ref]
        assert [f"{float(v):.7g}" for v in dzs] == [f"{v:.7g}" for v in dzs_ref]


def test_ruc_dzs_is_not_a_partition_of_the_soil_column():
    """WRF's DZS overcounts, by a different amount at each geometry.

    ``init_soil_depth_3`` writes ``zs2(2)`` twice and differences the second
    write against ``zs2(1) = 0``, so ``sum(dzs)`` exceeds the 3 m column by
    the first half-interval.  Locked per geometry because code that assumed a
    partition would be wrong by 5 mm at nine levels and 25 mm at six -- a
    five-fold difference, i.e. exactly the kind of thing a single hardcoded
    geometry hides.
    """
    from woof.core.ruc import ruc_soil_geometry

    for count, excess in ((6, 0.025), (9, 0.005)):
        _, dzs = ruc_soil_geometry(count)
        assert abs(float(sum(float(v) for v in dzs)) - (3.0 + excess)) < 1e-6

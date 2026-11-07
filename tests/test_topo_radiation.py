"""WRF slope_rad / topo_shading: the device port against WRF itself.

Two fixtures, both WRF's own Fortran cut verbatim from the sources and
compiled with WRF's gfortran flags:

* ``wrf_topo_radiation_v471`` (tools/wrf_topo_radiation_v471_oracle):
  each stage alone over seeded planes, run twice -- linked against stock
  glibc 2.43 (``glibc``), as wrf.exe runs, and against a correctly rounded
  float libm (``cr``).
* ``wrf_topo_radiation_v461`` (tools/wrf_topo_radiation_v461_oracle):
  WRF v4.6.1's whole chain (slope, shadow, diffuse fraction, surface
  adjustment, restore) over 20 realistic patches labelled by regime,
  32,100 columns, each stage fed by its own side's previous stage; WRF
  linked to glibc 2.43 (``wrf_glibc243``) and to the build box's stock
  glibc 2.39 (``wrf_glibc239``).

The port evaluates SIN, COS, TAN, ASIN, ACOS, ATAN, EXP and ** with WOOF's
own float32 libm words and ATAN2 correctly rounded, so it is held bit for
bit to WRF on the glibc 2.43 libm in both fixtures.  Until 2026-10-07 it
evaluated every transcendental as the double function rounded to float,
which matched the ``cr`` build and missed WRF on every real libm (3,660
values of the v4.6.1 chain on glibc 2.43, 7,492 on glibc 2.39).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = (Path(__file__).parent / "fixtures" / "wrf_topo_radiation_v471"
           / "topo_radiation_v471.npz")
CHAIN_FIXTURE = (Path(__file__).parent / "fixtures" / "wrf_topo_radiation_v461"
                 / "topo_radiation_chain_v461.npz")
#: The side of the v4.7.1 fixture the port is held to: WRF on stock glibc.
REF = "glibc"


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _names(fx, prefix, depth):
    return sorted({"/".join(k.split("/")[:depth]) for k in fx.files
                   if k.startswith(prefix + "/")})


def _ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def test_fixture_covers_every_branch(fx):
    """The cases reach what the port has to get right."""
    slope_names = _names(fx, "slope", 2)
    assert "slope/alps_geogrid" in slope_names
    flat = sum(int((fx[f"{n}/cr/slope"] == 0).sum()) for n in slope_names)
    steep = sum(int((fx[f"{n}/cr/slope"] > 0.3).sum()) for n in slope_names)
    negative_cosa = sum(int((fx[f"{n}/cosa"] < 0).sum())
                        for n in slope_names)
    assert flat and steep and negative_cosa
    shadowed = [int(fx[k].sum()) for k in fx.files
                if k.endswith("/cr/shadowmask")]
    assert sum(1 for s in shadowed if s) >= 10
    raised = [k for k in fx.files if k.endswith("/nested1/cr/ht_loc")]
    assert any((fx[k] != fx[k.split("/sun")[0] + "/ht"]).any()
               for k in raised)
    adjusted = fx["adjust/random/cr/swdown"] != fx["adjust/random/swdown"]
    assert adjusted.any() and (~adjusted).any()


@requires_gpu
def test_slope_geometry_matches_wrf_start_em(fx):
    import cupy as cp

    from woof.core.topo_radiation import slope_geometry

    for name in _names(fx, "slope", 2):
        dx = float(np.float32(1.0) / fx[f"{name}/rdx"])
        slope, azi = slope_geometry(
            fx[f"{name}/ht"], fx[f"{name}/msft"], fx[f"{name}/msft"],
            fx[f"{name}/sina"], fx[f"{name}/cosa"], dx=dx, dy=dx)
        # rdx is what start_em.F reads; the port recomputes it from dx,
        # so the round trip must land on the same single-precision value.
        assert np.float32(1.0) / np.float32(dx) == fx[f"{name}/rdx"]
        np.testing.assert_array_equal(cp.asnumpy(slope),
                                      fx[f"{name}/{REF}/slope"], err_msg=name)
        np.testing.assert_array_equal(cp.asnumpy(azi),
                                      fx[f"{name}/{REF}/slp_azi"], err_msg=name)


@requires_gpu
def test_terrain_shadow_matches_wrf_toposhad(fx):
    import cupy as cp

    from woof.core.topo_radiation import terrain_shadow

    for base in _names(fx, "shadow", 2):
        for sun in range(8):
            xtime, gmt, radt, declin, dx, dy, shadlen = (
                float(v) for v in fx[f"{base}/sun{sun}/scalars"])
            for nested in (0, 1):
                key = f"{base}/sun{sun}/nested{nested}"
                mask, shad = terrain_shadow(
                    fx[f"{base}/ht"], fx[f"{base}/xlat"],
                    fx[f"{base}/xlong"], fx[f"{base}/sina"],
                    fx[f"{base}/cosa"], xtime_minutes=xtime, gmt=gmt,
                    radt_minutes=radt, declin=declin, dx=dx, dy=dy,
                    shadlen=shadlen,
                    parent_shadow=(fx[f"{base}/ht_shad_in"] if nested
                                   else None))
                np.testing.assert_array_equal(
                    cp.asnumpy(mask), fx[f"{key}/{REF}/shadowmask"],
                    err_msg=key)
                np.testing.assert_array_equal(
                    cp.asnumpy(shad), fx[f"{key}/{REF}/ht_shad"], err_msg=key)


@requires_gpu
def test_diffuse_fraction_matches_wrf_radiation_driver(fx):
    import cupy as cp

    from woof.core.topo_radiation import diffuse_fraction

    for ruiz in (0, 1):
        key = f"diffuse/ruiz{ruiz}"
        swddif = cp.asarray(fx[f"{key}/swddif_in"]).copy()
        frac = diffuse_fraction(
            fx[f"{key}/coszen"], fx[f"{key}/swdown"], fx[f"{key}/ht"],
            swddif, solcon=float(fx[f"{key}/solcon"]),
            scheme_splits_direct_beam=not ruiz)
        np.testing.assert_array_equal(cp.asnumpy(swddif),
                                      fx[f"{key}/{REF}/swddif"], err_msg=key)
        np.testing.assert_array_equal(cp.asnumpy(frac),
                                      fx[f"{key}/{REF}/diffuse_frac"],
                                      err_msg=key)


@requires_gpu
def test_surface_adjustment_matches_wrf_topo_rad_adj(fx):
    import cupy as cp

    from woof.core.topo_radiation import (adjust_surface_shortwave,
                                           restore_surface_shortwave)

    key = "adjust/random"
    declin, _solcon = (float(v) for v in fx[f"{key}/scalars"])
    swdown = cp.asarray(fx[f"{key}/swdown"]).copy()
    gsw = cp.asarray(fx[f"{key}/gsw"]).copy()
    save = adjust_surface_shortwave(
        swdown, gsw, xlat=fx[f"{key}/xlat"], coszen=fx[f"{key}/coszen"],
        shadowmask=fx[f"{key}/shadowmask"],
        diffuse_frac=fx[f"{key}/diffuse_frac"], hrang=fx[f"{key}/hrang"],
        slope=fx[f"{key}/slope"], slp_azi=fx[f"{key}/slp_azi"],
        declin=declin)
    np.testing.assert_array_equal(cp.asnumpy(swdown),
                                  fx[f"{key}/{REF}/swdown"])
    np.testing.assert_array_equal(cp.asnumpy(gsw), fx[f"{key}/{REF}/gsw"])
    np.testing.assert_array_equal(cp.asnumpy(save.swnorm),
                                  fx[f"{key}/{REF}/swnorm"])
    day = fx[f"{key}/swdown"] > np.float32(1e-3)
    np.testing.assert_array_equal(cp.asnumpy(save.gswsave)[day],
                                  fx[f"{key}/{REF}/gswsave"][day])
    # After the land surface: the flat fluxes come back exactly, SWNORM
    # holds the slope flux (module_surface_driver.F:4461-4481).
    swnorm = restore_surface_shortwave(swdown, gsw, save)
    np.testing.assert_array_equal(cp.asnumpy(swdown), fx[f"{key}/swdown"])
    np.testing.assert_array_equal(cp.asnumpy(gsw), fx[f"{key}/gsw"])
    np.testing.assert_array_equal(
        cp.asnumpy(swnorm)[day], fx[f"{key}/{REF}/swdown"][day])


def test_glibc_seam_is_the_libm_alone(fx):
    """The fixture tells the two libm evaluations apart, pinned.

    The oracle's two builds share every object but the libm, so any
    difference between them is a libm difference.  The port is held to the
    ``glibc`` side; this pin is what makes that hold mean something: a port
    that went back to the correctly rounded evaluation (the defect fixed
    2026-10-07) would match the ``cr`` side and miss these 333 values, and
    a regenerated fixture that lost the separation would show here.
    """
    diffs = {}
    for key in fx.files:
        if "/cr/" not in key:
            continue
        other = key.replace("/cr/", "/glibc/")
        a, b = fx[key], fx[other]
        if a.dtype == np.float32:
            ulp = _ulp(a, b)
            if ulp.any():
                diffs[key] = (int((ulp > 0).sum()), int(ulp.max()))
        elif (a != b).any():
            diffs[key] = (int((a != b).sum()), None)
    assert diffs == GLIBC_SEAM, diffs


#: What stock glibc 2.43 (gfortran 15.2, test host 2's x86-64, where glibc
#: dispatches its FMA variants of sinf/cosf/expf/powf) gives differently
#: from the correctly rounded build, as ``array: (values that differ,
#: largest difference in ulp)``: 333 of 189,080 values, no shadow mask
#: flipped.  The largest differences are shadow heights at a grazing sun,
#: where tan(topoelev) - tan(asin(csza)) magnifies a one-ulp sine into a
#: few hundred ulp of a height.  glibc's sinf, cosf, expf and powf are not
#: correctly rounded, and they are the same words in glibc 2.39 and 2.43
#: (whole-domain digests, tools/wrf_topo_radiation_v461_oracle/
#: libm_digest.c, 2026-10-07), so the port evaluates them as glibc does.
GLIBC_SEAM: dict = {
    "adjust/random/cr/gsw": (25, 5),
    "adjust/random/cr/swdown": (27, 3),
    "diffuse/ruiz1/cr/diffuse_frac": (5, 16),
    "diffuse/ruiz1/cr/swddif": (5, 15),
    "shadow/alps_geogrid/sun7/nested0/cr/ht_shad": (10, 10),
    "shadow/alps_geogrid/sun7/nested1/cr/ht_shad": (8, 10),
    "shadow/synthetic0/sun1/nested0/cr/ht_shad": (2, 8),
    "shadow/synthetic0/sun1/nested1/cr/ht_shad": (2, 8),
    "shadow/synthetic0/sun7/nested0/cr/ht_shad": (116, 818),
    "shadow/synthetic0/sun7/nested1/cr/ht_shad": (126, 818),
    "shadow/synthetic1/sun2/nested0/cr/ht_shad": (1, 72),
    "shadow/synthetic1/sun2/nested1/cr/ht_shad": (1, 72),
    "slope/alps_geogrid/cr/slope": (4, 1),
    "slope/synthetic0/cr/slope": (1, 1),
}


# ---------------------------------------------------------------------------
# WRF v4.6.1, the whole chain (tools/wrf_topo_radiation_v461_oracle)
# ---------------------------------------------------------------------------

#: The chain fixture's reference side: WRF v4.6.1 linked to glibc 2.43.
CHAIN_REF = "wrf_glibc243"

#: The six inputs where the double atan2 rounded once to float lands on the
#: wrong side of a float midpoint (measured on sm_120 at NVRTC 12.9.86 in
#: two whole-domain sweeps, 2026-10-07), with the correctly rounded result
#: (glibc 2.43 atan2f and the long-double atan2 rounded to float agree):
#: (y bits, x bits, atan2f bits).
ATAN2_HARD = (
    (0x3DD5A208, 0x447A0000, 0x38DAC297),
    (0xBDD5A208, 0x447A0000, 0xB8DAC297),
    (0x516A64F9, 0x447A0000, 0x3FC90FDB),
    (0xD16A64F9, 0x447A0000, 0xBFC90FDB),
    (0x447A0000, 0xC53B8906, 0x40347937),
    (0x447A0000, 0xD0EA64F9, 0x40490FDB),
)


@pytest.fixture(scope="module")
def chain():
    return np.load(CHAIN_FIXTURE)


def _column_oracle():
    import importlib.util
    path = (Path(__file__).resolve().parent.parent / "tools"
            / "wrf_topo_radiation_v461_oracle" / "column_oracle.py")
    spec = importlib.util.spec_from_file_location("_topo_column_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain_case(chain, name):
    return {k.split("/", 1)[1]: chain[k] for k in chain.files
            if k.startswith(name + "/") and k.count("/") == 1}


def test_chain_fixture_covers_the_regimes(chain):
    """The v4.6.1 chain reaches every regime and branch it is quoted for.

    The breakage this prevents: a regenerated fixture that quietly lost a
    regime (no night, no shadow, no nest, no flat ocean) would still pass
    the bitwise test below while proving less than REPORT.md says.
    """
    names = [str(n) for n in chain["_names"]]
    regimes = [str(r) for r in chain["_regimes"]]
    assert {CHAIN_REF, "wrf_glibc239"} <= {str(s) for s in chain["_sides"]}
    columns = {}
    for name, regime in zip(names, regimes):
        columns[regime] = columns.get(regime, 0) + chain[f"{name}/ht"].size
    assert set(columns) == {"convective", "grazing", "stable_night",
                            "diurnal_sweep", "snow_ice", "ocean",
                            "high_terrain", "rotated_grid", "overhead_sun",
                            "edges"}
    assert min(columns.values()) >= 64 and sum(columns.values()) == 32100

    def total(field, test):
        return sum(int(test(chain[f"{n}/{CHAIN_REF}/{field}"]).sum())
                   for n in names if f"{n}/{CHAIN_REF}/{field}" in chain.files)
    assert total("shadowmask", lambda a: a == 1) > 100
    assert total("slope", lambda a: a == 0) > 100          # flat cells
    assert total("diffuse_frac", lambda a: a == 1) > 10    # all diffuse
    assert total("swnorm", lambda a: a <= 1e-3) > 1000     # night
    raised = [n for n in names if f"{n}/{CHAIN_REF}/ht_loc" in chain.files
              and (chain[f"{n}/{CHAIN_REF}/ht_loc"] != chain[f"{n}/ht"]).any()]
    assert raised                                          # a nest edge
    # Snow: the land surface's net flux is a fifth of the downward one.
    assert (chain["polar_plateau_snow/gsw"]
            <= np.float32(0.2) * chain["polar_plateau_snow/swdown"]).all()
    assert (chain["ocean_tropics/ht"] == 0).all()


@requires_gpu
def test_the_whole_chain_matches_wrf461_bit_for_bit(chain):
    """WRF v4.6.1's slope_rad / topo_shading chain, every field, 0 ulp.

    Each stage is fed by WOOF's own previous stage, as the WRF side is fed
    by WRF's, so a one-ulp slope would show again in the adjusted flux.
    """
    oracle = _column_oracle()
    for name in (str(n) for n in chain["_names"]):
        got = oracle.woof_chain(_chain_case(chain, name))
        for field, value in got.items():
            np.testing.assert_array_equal(
                value, chain[f"{name}/{CHAIN_REF}/{field}"],
                err_msg=f"{name}/{field}")


@requires_gpu
def test_the_atan2_word_is_correctly_rounded_where_the_double_is_not():
    """tr_atan2 at the six measured near-midpoint inputs.

    The breakage this prevents: SLP_AZI is atan2(hx, hy) + pi, and the
    double atan2 rounded once to float puts these six on the wrong side of
    a float midpoint, one ulp from WRF's atan2f.
    """
    import cupy as cp

    from woof.core.kernels import module_options, module_source
    probe = r"""
extern "C" __global__ void topo_atan2_probe(const float* y, const float* x,
                                            float* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tr_atan2(y[i], x[i]);
}
"""
    mod = cp.RawModule(code=module_source("topo_radiation") + probe,
                       options=module_options("topo_radiation"))
    bits = np.asarray(ATAN2_HARD, dtype=np.uint32)
    y = cp.asarray(bits[:, 0].view(np.float32))
    x = cp.asarray(bits[:, 1].view(np.float32))
    out = cp.empty(len(bits), dtype=cp.float32)
    mod.get_function("topo_atan2_probe")((1,), (32,),
                                         (y, x, out, np.int32(len(bits))))
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32), bits[:, 2])


def test_the_unit_evaluates_the_libm_words():
    """No transcendental in the unit is the double function rounded to float.

    The breakage this prevents: that evaluation is what the port used until
    2026-10-07, and it misses WRF on every libm a wrf.exe links (3,660 of
    425,468 chain values against glibc 2.43).  The loader must prepend the
    float32 libm headers, and every tr_* word must call them.
    """
    import re

    from woof.core.kernels import EXTRA_HEADERS, _KDIR
    assert EXTRA_HEADERS["topo_radiation"] == ("glibc_flt32.cuh",
                                               "glibc_trig_flt32.cuh",
                                               "atan_scalar_flt32.cuh")
    src = (_KDIR / "topo_radiation.cu").read_text(encoding="utf-8")
    assert not re.search(
        r"\(float\)\s*(sin|cos|tan|asin|acos|atan|exp|pow)\s*\(", src)
    words = dict(re.findall(
        r"__forceinline__ float (tr_\w+)\(float[^)]*\)\s*\{\s*return "
        r"(\w+)\(", src))
    assert words == {"tr_sin": "glibc_sinf", "tr_cos": "glibc_cosf",
                     "tr_tan": "glibc_tanf", "tr_asin": "glibc_asinf",
                     "tr_acos": "glibc_acosf", "tr_atan": "glibc_atanf",
                     "tr_exp": "gfk_exp", "tr_pow": "gfk_pow"}


def test_every_field_the_port_publishes_has_a_health_policy():
    """The first real topo_shading forecast refused at initialization.

    TopoShortwave publishes its arrays in the driver's surface fields, which
    the health gate walks; the int32 shadow mask had no integer policy, so
    StateHealthValidator raised before step one.  WRF writes it 0 or 1 only
    (module_radiation_driver.F:4505, :4661, :4694, :4712, :4730, :4748).
    """
    from types import SimpleNamespace

    from woof.core.health import (FieldRule, collect_state_fields,
                                   gpu_integer_policy, validate_fields_cpu)

    shape = (3, 4)
    fields = {name: np.zeros(shape, np.float32)
              for name in ("slope", "slp_azi", "diffuse_frac",
                           "topo_coszen", "hrang", "swnorm", "ht_shad")}
    fields["topo_declin"] = np.zeros((1,), np.float32)
    fields["shadowmask"] = np.zeros(shape, np.int32)
    state = SimpleNamespace(physics=SimpleNamespace(fields=fields))
    collected = {f.name: f for f in collect_state_fields(state)}
    for name in fields:
        field = collected[f"surface.{name}"]
        assert gpu_integer_policy(field.name, field.values.dtype) is not None
    mask = collected["surface.shadowmask"]
    assert mask.rule == FieldRule("surface", 0.0, 1.0)
    fields["shadowmask"][1, 2] = 1
    assert validate_fields_cpu([mask]).ok
    fields["shadowmask"][1, 2] = -1
    assert not validate_fields_cpu([mask]).ok


def test_the_carrier_is_rebuilt_and_its_arrays_ride_the_fields():
    """The first real slope_rad forecast ended at its final state digest:
    PhysicsDriver.topo_shortwave was in no restart class.  It is rebuilt by
    initialize_physics; what it evolves lives in the serialized fields."""
    import inspect

    from woof.core.topo_radiation import TopoShortwave
    from woof.io import restart

    assert "topo_shortwave" in restart.DRIVER_REBUILT_ATTRS
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(TopoShortwave)))
    rebound = []
    for method in tree.body[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name == "__init__":
            continue
        for node in ast.walk(method):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(
                           node, (ast.AugAssign, ast.AnnAssign)) else [])
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    rebound.append(f"{method.name}: self.{target.attr}")
    # After construction the carrier writes only into its fields mapping.
    assert rebound == [], rebound


def test_the_checkpoint_echo_carries_the_slope_keys_only_when_on():
    """Off, a checkpoint is byte-identical to one written before the keys
    existed; on, all three bind the resume."""
    from woof.io.restart import _drop_inert_topo_radiation

    off = {"slope_rad": 0, "topo_shading": 0, "shadlen": 25000.0, "dt": 9}
    _drop_inert_topo_radiation(off)
    assert off == {"dt": 9}
    flat = {"slope_rad": 1, "topo_shading": 0, "shadlen": 25000.0}
    _drop_inert_topo_radiation(flat)
    assert flat == {"slope_rad": 1}
    shaded = {"slope_rad": 1, "topo_shading": 1, "shadlen": 30000.0}
    _drop_inert_topo_radiation(shaded)
    assert shaded == {"slope_rad": 1, "topo_shading": 1, "shadlen": 30000.0}

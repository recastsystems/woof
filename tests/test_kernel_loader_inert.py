"""Prove the mp=28 kernel-loader hook is inert for every other module.

``woof/core/kernels/__init__.py`` grew one additive ``_EXTRA_HEADERS``
allow-list so the six aerosol-aware Thompson translation units can share
``thompson_aerosol_common.cuh``; there is no ``#include`` path under
``cupy.RawModule``.

That hook touches the ONE file ``mp_physics=8`` also uses, so its inertness is
what keeps WP-00's freeze gate meaningful.  The guarantee claimed in the port
spec is *by construction*, not by measurement: a module absent from the
allow-list contributes the empty string and therefore assembles a
byte-identical source string, hence identical PTX, hence identical FP32
results.  This module enumerates every ``.cu`` in the tree and asserts exactly
that -- no spot checks.

These tests are CPU-only on purpose.  They read and compare source strings and
never compile anything, so they run in environments with no CUDA device.
"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from woof.core import kernels as kernel_loader


_KDIR = Path(kernel_loader.__file__).parent

#: The only modules that may appear in the allow-list.  Kept literal so a
#: future package cannot quietly add itself to the dict.
_ALLOWED_AEROSOL_MODULES = frozenset({
    "thompson_aerosol_probe",
    "thompson_aerosol_state",
    "thompson_aerosol_sat",
    "thompson_aerosol_cold",
    "thompson_aerosol_warm",
    "thompson_aerosol_sed",
})

#: The whole allow-list, module -> its exact header tuple.  Closed and
#: literal, which is the property these tests hold: NOTHING receives a
#: device header without being named here, in a second place.
#: rrtmgp_rte joined when the LW solver started deriving its own Planck
#: sources; rrtmgp_gas joined (with rrtmgp_libm.cuh on both) when the two
#: units were made word for word the upstream single-precision Fortran and
#: the Planck words moved into the shared header.
#: gf joined when the glibc 2.39 float32 transcendentals were lifted out of
#: gf.cu into glibc_flt32.cuh for New Tiedtke to share.  It is the one entry
#: whose module KNOWINGLY gives up the byte-identical-source guarantee the
#: unlisted modules keep -- gf's correctness is held by its 396 parity tests
#: at max_ulp 0, which is a stronger gate than source identity and the only
#: reason the trade was takeable.
_EXPECTED_HEADERS = {
    "gsd_initial_reflectivity": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                                 "thompson_fork_constants.cuh", "thompson_fork.cu"),
    # A new full-column generation unit; existing classic and aerosol units
    # keep their assembled source bytes and their published oracle gates.
    "thompson_column": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                        "thompson_aerosol_libm.cuh", "thompson_column_support.cuh",
                        "thompson_column_radar.cuh", "thompson_column_generated.cuh"),
    "wsm7": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # Selected diffusion and surface sources have independently pinned headers.
    "acoustic": ("glibc_trig_flt32.cuh",),
    "myjsfc": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    "mynn_surface": ("mynn_libm.cuh", "surface_subnormal.cuh"),
    "sfclay": ("glibc_flt32.cuh", "sfclay_classic.cuh"),
    "smag2d": ("glibc_flt32.cuh",),
    "lin": ("glibc_flt32.cuh", "lin_support.cuh", "lin_wrf.cuh"),
    "lin_refl": ("refl.cu",),
    # Goddard 2017 radiation (ra_lw/ra_sw_physics = 5): a new unit.
    "goddard": ("glibc_flt32.cuh", "portable_libm64.cuh", "goddard_bands.cuh", "goddard_sounding.cuh"),
    # WRF lightning diagnostics (glibc float32 powers): a new unit.
    "lightning": ("glibc_flt32.cuh",),
    # aer_opt = 1 six-type climatology optics: named, no extra header.
    "rrtmg_aer1": (),
    # aer_opt = 2 column aerosol (WRF float32 math): a new unit.
    "aer_opt2": ("glibc_flt32.cuh",),
    "upper_wind_limiter": ("glibc_flt32.cuh",),
    # New order-five scalar eta entry, qualified against the native fork.
    # No existing translation unit gains a header through this grant.
    "pd_vertical_sl": ("pd_advection.cu",),
    # The SFIRE port (lane/ec-sfire): native ideal initialization and the
    # geographic coupling take glibc words, and the ideal fire state takes
    # the CORE-MATH tanhf of sfire_libm.cuh; new units, no existing grant.
    "sfire_coupling": ("glibc_trig_flt32.cuh",),
    "sfire_ideal": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "sfire_libm.cuh"),
    "sfire_ideal_atmos": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    # noah.cu carries WRF's seaice_noah (lane/290-A5-fixes), which calls
    # glibc's float32 expf/powf; the ordinary land column calls neither.
    "noah_init": ("glibc_flt32.cuh", "noah.cu"),
    "noah": ("glibc_flt32.cuh",),
    "horizontal": ("portable_libm64.cuh",),
    # Test-only bit grading, with the same shared header future callers use.
    "portable_libm64_grade": ("portable_libm64.cuh",),
    "vert_interp": ("glibc_flt32.cuh",),
    "thompson_cold_start": ("glibc_flt32.cuh", "portable_libm64.cuh"),
    "chem_prep": ("glibc_flt32.cuh",),
    # The GOCART ports (lane/aq-gocart) take WRF-Chem's float32 exp/log/pow
    # from the same header; each is held by its own oracle parity tests at
    # 0 ULP (tests/test_chem_*_wrf471_parity.py), not by source identity.
    **{name: ("glibc_flt32.cuh",) for name in (
        "chem_dust", "chem_seasalt", "chem_rrtmgp_aerosol",
        "chem_ageing", "chem_settling", "chem_drydep_gocart", "chem_optics")},
    # The mp=28 units take WOOF's own float32 and binary64 libm words
    # (thompson_aerosol_libm.cuh on glibc_flt32/flt64.cuh) ahead of their
    # shared header; the 0 ULP column oracle grades them, not source identity.
    **{name: ("glibc_flt32.cuh", "glibc_flt64.cuh",
              "thompson_aerosol_libm.cuh", "thompson_aerosol_common.cuh")
       for name in _ALLOWED_AEROSOL_MODULES},
    # The exact chain's strict headers (a default process composes
    # rrtmgp_rte from rrtmgp_planck_common_default.cuh and gf from
    # glibc_flt32.cuh alone; kernels/__init__.py _extra_header_text).
    "rrtmgp_rte": ("rrtmgp_libm.cuh", "rrtmgp_planck_common.cuh"),
    "gf": ("glibc_flt32.cuh", "gf_wrf_normalization.cuh"),
    "g3": ("glibc_flt32.cuh",),
    "ntiedtke": ("glibc_flt32.cuh",),
    # The original cu6 kernel: a new module whose GPU column oracle
    # (tools/tiedtke_wrf461_oracle/check.py --gpu) grades it bitwise.
    "tiedtke": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                "tiedtke_support.cuh", "tiedtke_routines.cuh"),
    # A new module, so no pre-hook source to stay identical to; its WRF
    # v4.7.1 parity suites grade it bitwise.
    "urban_ucm": ("glibc_flt32.cuh",),
    "urban_ucm480": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                     "urban_ucm.cu", "urban_ucm480_support.cuh", "urban_ucm480.cuh"),
    "urban_bep": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "urban_bep_couple": ("glibc_flt32.cuh",),
    "myjurb": ("glibc_flt32.cuh",),
    # The MYJ PBL, graded bitwise against the WRF v4.6.1 MYJPBL column
    # oracle (tests/test_myjpbl_wrf461_parity.py): gfk_pow and the
    # contracted gfk_exp_fma.
    "myjpbl": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    # YSU's topo_wind arm (ysu_column_topo) takes glibc's powf; the default
    # entry points compile to byte-identical PTX with the header present
    # (tests/test_mp8_frozen.py, the ysu re-pin).
    "ysu": ("glibc_flt32.cuh", "ysu_topo_default.cuh"),
    # WRF 4.8 YSU generation (ysu_version = "wrf_480"), its own unit.
    "ysu_480": ("glibc_flt32.cuh", "ysu_480_topo.cuh"),
    # Shin-Hong takes glibc's powf/expf since lane/parity-pbl-libm; its WRF
    # v4.6.1 oracle grades the unit bitwise (tests/test_shinhong_wrf461_parity.py).
    "shinhong": ("glibc_flt32.cuh",),
    # Lane 286-aer-swint: WRF swint_opt = 1 and aer_opt = 3, graded
    # bitwise against the operational HRRR fork's gfortran/glibc Fortran.
    "swint": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "rrtmg_legacy_solar": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "rrtmg_aer3": ("glibc_flt32.cuh",),
    # Kessler (mp_physics=1), graded bitwise against the byte-unmodified
    # WRF v4.6.1 column oracle by tests/test_kessler_wrf461_parity.py.
    "kessler": ("glibc_flt32.cuh",),
    # Dudhia shortwave (ra_sw_physics=1): a new module, graded bitwise
    # against the WRF v4.6.1 column oracle (tests/test_dudhia_sw_wrf461_parity.py).
    "dudhia_sw": ("glibc_flt32.cuh",),
    # The UW moist-turbulence PBL: a new module, so no existing unit moves.
    "uwpbl": ("glibc_flt64.cuh", "uwpbl_common.cuh", "uwpbl_wvsat.cuh",
              "uwpbl_vdiff.cuh", "uwpbl_zisocl.cuh", "uwpbl_caleddy.cuh",
              "uwpbl_eddy.cuh", "uwpbl_driver.cuh"),
    "real_init_math": ("real_init_common.cuh", "portable_libm64.cuh"),
    # gp-device: only the new REAL unit receives its NaN/rounding vocabulary.
    "real_init": ("real_init_common.cuh",),
    # A new module; the WRF v4.7.1 Noah mosaic column oracle grades it
    # bitwise.
    "noah_mosaic": ("glibc_flt32.cuh",),
    # RUC LOG/EXP mixture words and the new lake driver's REAL32 power
    # use the shared WRF-oracle float32 functions.  RUC's soil-resistance
    # COS takes glibc_cosf (lane/verify-ruc-lsm).
    "ruc": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "lake": ("glibc_flt32.cuh", "f64cr.cuh", "lake_support.cuh", "lake_wrf.cuh"),
    # Kain-Fritsch: WOOF's own float32 expf/logf/powf for its WRF 4.6.1
    # column oracle (2026-10-07); the unit compiles on woof/core/kf.py's
    # own route from this same assembled source.
    "kf": ("glibc_flt32.cuh",),
    "wsm3": ("glibc_flt32.cuh",),
    "chem_fire": ("glibc_flt32.cuh",),
    "chem_plumerise": ("glibc_flt32.cuh",),
    "wsm5": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    "chem_drydep_wesely": ("glibc_flt32.cuh",),
    # WSM6 graded word for word against WRF v4.6.1
    # (tests/test_wsm6_wrf461_column_exact.py): WOOF's own float32 words
    # for the scheme, its own binary64 words for refl10cm_wsm6.
    "wsm6": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # Milbrandt-Yau (mp_physics = 9) and its radar operator: WOOF's own
    # float32 exp/log/pow and log10, graded bitwise against the WRF v4.6.1
    # column oracle (tests/test_milbrandt2_wrf461_exact_gpu.py).
    "milbrandt2": ("glibc_flt32.cuh", "log10_flt32.cuh"),
    "milbrandt2_zet": ("glibc_flt32.cuh", "log10_flt32.cuh"),
    # WDM6 and its reflectivity, graded word for word against WRF v4.6.1
    # by tools/wdm6_wrf461_oracle: WOOF's own float32 exp/log/pow, and for
    # the reflectivity also WOOF's own binary64 exp/log/pow.
    "wdm6": ("glibc_flt32.cuh",),
    "wdm6_refl": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    'nsas': ('glibc_flt32.cuh', 'nsas_constants.cuh'),
    "wsm6r": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # The exact chain's units (intake 2.9.1); the nssl2 and rrtmgp_gas
    # headers reach only a strict process.
    "chem_sulfur": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "physics_exner": ("glibc_flt32.cuh",),
    "topo_radiation": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                       "atan_scalar_flt32.cuh"),
    "thompson_classic": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh",
                         "glibc_flt64.cuh", "thompson_classic_constants.cuh"),
    "morrison": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_flt64.cuh",
                 "glibc_trig_flt32.cuh", "morrison_constants.cuh"),
    "thompson_fork": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                      "thompson_fork_constants.cuh"),
    **{name: ("glibc_flt32.cuh", "nssl2_wrf_math.cuh") for name in (
        "nssl2_nucond", "nssl2", "nssl2_fused_gs", "nssl2_driver_support",
        "nssl2_diagnostics")},
    "rrtmgp_gas": ("rrtmgp_libm.cuh", "rrtmgp_planck_common.cuh"),
}

# Retained overrides from the earlier merge pass.


def _module_names() -> list[str]:
    names = sorted({path.stem for path in _KDIR.glob("*.cu")}
                   | {"ysu", "mynn_dmp_sibling"})
    assert names, "no CUDA translation units found"
    assert "thompson" in names
    return names


@pytest.mark.parametrize("strict", [False, True])
def test_lake_compile_options_do_not_change_existing_modules(monkeypatch, strict):
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", strict)
    # These are closed literal grants for the selected source: the six
    # no-FMA units, four surface units and five diffusion units. In
    # particular MYNN PBL carries both the no-FMA and no-FTZ requirements.
    # A global option change must still fail for every other actual module.
    no_fmad = {"lake", "g3", "ysu", "shinhong", "mynn_pbl", "mynn_dmp_sibling",
               "wsm6", "kessler", "myjpbl", "milbrandt2", "milbrandt2_zet",
               "wdm6", "wdm6_refl", "wsm3", "wsm6r", "noah", "climate_diagnostics"}
    if strict:
        no_fmad |= {"morrison", "rrtmgp_rte", "noah_init", "wdm6",
            "milbrandt2_zet", "wdm6_refl", "rrtmgp_gas", "milbrandt2",
            "rrtmgp_cloud", "noah", "myjsfc", "kessler", "thompson_classic",
            "myjpbl", "wsm6",
            # labs/windfarm (2.9.11): the Fitch unit joins the strict no-FMA set.
            "wind_farm"}
    no_ftz = {"g3", "sfclay", "myjsfc", "mynn_surface", "mynn_pbl", "thompson_column",
              "ysu_480", "wsm6r", "aer_opt2"}
    diffusion = {"smag2d", "diffusion", "diff_opt1", "diff6", "diff6_seam"}
    expected = {name: ("-std=c++17", "--fmad=false") for name in no_fmad}
    expected.update({name: ("-std=c++17", "--fmad=false", "--ftz=false")
                     for name in no_ftz})
    expected.update({name: ("-std=c++17", "--fmad=false", "--ftz=false",
                           "--prec-div=true", "--prec-sqrt=true",
                           "-DGPUWM_WRF_EXACT_C_DIFFUSION=1")
                     for name in diffusion})
    # The two solar geometry units keep subnormals only (WRF 4.8 solar
    # generation, graded against both releases' radconst/calc_coszen).
    expected.update({name: ("-std=c++17", "--ftz=false")
                     for name in ("rrtmg_legacy_solar", "swint")})
    # The urban units keep the urban lanes' marker route and options.
    expected.update({name: ("-std=c++17", "--ftz=false")
                     for name in ("urban_ucm", "urban_bep", "urban_bep_couple")})
    expected["urban_ucm480"] = ("-std=c++17", "--ftz=false", "--fmad=false")
    expected['nsas'] = ('-std=c++17', '--ftz=false', '--fmad=false')
    # The lightning unit's own direct NVRTC route (lane order of flags).
    expected["lightning"] = ("-std=c++17", "--ftz=false", "--fmad=false")
    # WRF's ocean mixed layer (2.9.1) compiles as the column-exact ports do.
    expected["oml"] = ("-std=c++17", "--fmad=false", "--ftz=false")
    # WRF's HAILCAST diagnostics (lane/cap-hail 97e9a3ca061) compile without
    # FMA, like the other column-exact WRF ports.
    expected["hail"] = ("-std=c++17", "--fmad=false")
    names = _module_names()
    assert set(expected) <= set(names), sorted(set(expected) - set(names))
    for name in names:
        assert kernel_loader.module_options(name) == expected.get(
            name, ("-std=c++17",)), name


def _strict_header_text(name: str) -> str:
    from woof.wrf_exact import ENABLED, DIAGNOSTICS_ENABLED
    headers = ()
    if ENABLED and name == "acoustic":
        headers = ("glibc_trig_flt32.cuh",)
    if DIAGNOSTICS_ENABLED and name == "diagnostics":
        headers = ("glibc_flt32.cuh",)
    return "".join((_KDIR / header).read_text(encoding="utf-8")
                   for header in headers)


def _pre_hook_source(name: str) -> str:
    """The exact string the loader assembled BEFORE _EXTRA_HEADERS existed.

    Written out literally rather than derived from the loader, so a change to
    the loader cannot make this test agree with itself.

    The encoding is named for the same reason the loader names it: this test
    asserts two independent reads of one file are byte-identical, so if the
    two sides can disagree about how to decode it the assertion is about the
    locale rather than about the hook.  On Windows both sides used to decode
    the em dashes in acoustic.cu, advection.cu and coriolis_map.cu as cp1252
    and agree with each other while both being wrong.
    """
    return (kernel_loader._preamble() + _strict_header_text(name)
            + _unit_file_text(name))


def _unit_file_text(name: str) -> str:
    """The unit's own text, written out literally like the rest of this test.

    Slimming wave 3 (lane 13) folded mynn_dmp_sibling.cu into mynn_pbl.cu's
    MYNN_EXPORT_PLUME_EDGE build, so that logical module has no .cu of its
    own: its text is the one #define line followed by mynn_pbl.cu, byte for
    byte.
    """
    if name == "mynn_dmp_sibling":
        return ("#define MYNN_EXPORT_PLUME_EDGE 1\n"
                + (_KDIR / "mynn_pbl.cu").read_text(encoding="utf-8"))
    return (_KDIR / f"{name}.cu").read_text(encoding="utf-8")


def _pre_hook_source_int_defines(name: str, prefix: str) -> str:
    return (kernel_loader._preamble() + _strict_header_text(name) + prefix + "\n"
            + _unit_file_text(name))


@pytest.mark.parametrize("name", _module_names())
def test_non_aerosol_module_source_is_byte_identical(name):
    """Every module outside the allow-list assembles the pre-hook string."""
    if name in kernel_loader.EXTRA_HEADERS:
        pytest.skip(f"{name} is an allow-listed aerosol module")
    assembled = kernel_loader.module_source(name)
    expected = _pre_hook_source(name)
    assert assembled == expected, (
        f"{name}.cu no longer assembles its pre-hook source string; the "
        "mp=8 numerics guarantee rests on this being byte-identical")


@pytest.mark.parametrize("name", _module_names())
def test_non_aerosol_int_define_source_is_byte_identical(name):
    """The specialized-integer loader is inert for the same modules."""
    if name in kernel_loader.EXTRA_HEADERS:
        pytest.skip(f"{name} is an allow-listed aerosol module")
    defines = (("KMAX", 64), ("NLEVELS", 3))
    prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
    assembled = kernel_loader.module_source_int_defines(name, defines)
    assert assembled == _pre_hook_source_int_defines(name, prefix)


def test_thompson_cu_source_is_byte_identical():
    """The single most essential case, asserted without parametrization."""
    assert "thompson" not in kernel_loader.EXTRA_HEADERS
    assert (kernel_loader.module_source("thompson")
            == _pre_hook_source("thompson"))


def _declared_header_extensions():
    path = Path(__file__).resolve().parents[1] / "tests/data/kernel_source_pins.json"
    return json.loads(path.read_text(encoding="utf-8")).get("loader_headers", {})


def test_allow_list_is_a_closed_literal_mapping():
    extra = kernel_loader.EXTRA_HEADERS
    expected = dict(_EXPECTED_HEADERS)
    for name, headers in _declared_header_extensions().items():
        assert isinstance(headers, list) and headers, name
        assert name not in expected or expected[name] == tuple(headers), (
            f"extension replaces established loader header identity for {name}")
        expected[name] = tuple(headers)
    assert dict(extra) == expected, (
        "the loader header mapping differs from its declared source-pin "
        "manifest; a new binding must carry source authority there")
    for name, headers in extra.items():
        assert isinstance(headers, tuple)


def test_allow_listed_headers_exist_and_translation_units_are_explicit():
    for module, headers in kernel_loader.EXTRA_HEADERS.items():
        for header in headers:
            path = _KDIR / header
            assert path.is_file(), f"missing device header {header}"
            assert path.suffix == ".cuh" or (module, header) in (
                ("noah_init", "noah.cu"), ("pd_vertical_sl", "pd_advection.cu"),
                ("urban_ucm480", "urban_ucm.cu"),
                ("lin_refl", "refl.cu"),
                ("gsd_helicity", "uh_diag.cu"),
                ("gsd_initial_reflectivity", "thompson_fork.cu")), (
                "only the explicitly reused initialization and scalar units may supply a .cu header")


def test_allow_list_has_no_implicit_filesystem_behaviour():
    """A module not named in the dict gets nothing, even if a .cuh exists."""
    assert kernel_loader._extra_header_text("thompson") == ""
    assert kernel_loader._extra_header_text("no_such_module_at_all") == ""
    # A same-named .cuh sitting next to a .cu must NOT be picked up.
    assert (_KDIR / "thompson_aerosol_common.cuh").is_file()
    assert kernel_loader._extra_header_text("thompson_aerosol_common") == ""


def test_allow_listed_module_actually_receives_the_header():
    """The hook must not be inert for the modules it is FOR."""
    name = "thompson_aerosol_probe"
    assert name in kernel_loader.EXTRA_HEADERS
    header = "".join((_KDIR / h).read_text(encoding="utf-8") for h in (
        "glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh",
        "thompson_aerosol_common.cuh"))
    assembled = kernel_loader.module_source(name)
    assert assembled != _pre_hook_source(name)
    assert assembled == (kernel_loader._preamble() + header
                         + (_KDIR / f"{name}.cu").read_text(encoding="utf-8"))
    # The header must precede the module body, or the helpers are undeclared.
    assert assembled.index("thompson_aa_droplet_bin") < assembled.index(
        "thompson_aa_probe_droplet_bin")


def test_launch_module_names_agree_with_the_allow_list():
    from woof.core import thompson_aerosol_launch as launch

    assert set(launch.AEROSOL_KERNEL_MODULES) == (
        set(kernel_loader.EXTRA_HEADERS) & _ALLOWED_AEROSOL_MODULES)
    assert launch.CLASSIC_MODULE not in kernel_loader.EXTRA_HEADERS
    assert launch.AEROSOL_COMMON_HEADER == "thompson_aerosol_common.cuh"


def test_preamble_is_unchanged_by_the_hook():
    """_preamble() still ends with common.cuh and carries no aerosol text."""
    preamble = kernel_loader._preamble()
    assert preamble.endswith(
        (_KDIR / "common.cuh").read_text(encoding="utf-8") + "\n")
    assert "thompson_aa_" not in preamble


def test_the_noahmp_runtime_route_assembles_the_loader_source_byte_for_byte():
    """``load_module`` sends a standalone Noah-MP unit through
    ``noahmp_kernel_sources.compile_runtime_unit``; the string that route
    compiles must be exactly ``module_source(name)``.

    The two assemblers agree today because no ``noahmp_*`` module is in the
    header allow-list.  The breakage this prevents is the day one is added:
    ``module_source`` would gain a header the runtime unit does not carry,
    the recorded frame row would describe one image and the loader another,
    and nothing would say so.  ``noahmp_vegeflux`` is excluded on purpose --
    its runtime unit is C++14 without the preamble and takes its own
    factory, and ``load_module`` does not route it.
    """
    from woof.core.noahmp_kernel_sources import (
        NOAHMP_TRANSLATION_UNITS, runtime_unit)

    routed = [name for name, parts in NOAHMP_TRANSLATION_UNITS.items()
              if len(parts) == 1 and name != "noahmp_vegeflux"]
    assert routed, "the loader route covers no unit at all"
    for name in routed:
        unit = runtime_unit(name)
        assert unit.source == kernel_loader.module_source(name), name
        # Noah-MP units compile without FTZ (land no-FTZ route carried by
        # the WRF 4.7 night snow-aging port, woof/core/ieee_module.py).
        assert unit.options == ("-std=c++17", "--ftz=false"), name
        assert name not in kernel_loader.EXTRA_HEADERS, (
            f"{name} joined the header allow-list; the Noah-MP runtime unit "
            "must carry the same header or the loader must stop routing it")


def test_mosaic_ucm_composes_one_header_and_unchanged_ucm():
    from woof.core.noah_mosaic import mosaic_ucm_source
    source = mosaic_ucm_source()
    assert source == ("#define NOAH_MOSAIC_UCM 1\n"
                      + kernel_loader.module_source("urban_ucm") + "\n"
                      + (_KDIR / "noah_mosaic.cu").read_text(encoding="utf-8"))
    assert source.count("struct UcmCol {") == 1
    assert "NOAH_MOSAIC_UCM" not in kernel_loader.module_source("urban_ucm")


def test_noah_initialization_reuses_device_physics_without_moving_forecast_source():
    from woof.wrf_exact import ENABLED
    glibc = (_KDIR / "glibc_flt32.cuh").read_text(encoding="utf-8")
    strict_prefix = "#define NOAH_WRF_SCALAR_RCT 1\n" if ENABLED else ""
    assert kernel_loader.EXTRA_HEADERS["noah"] == ("glibc_flt32.cuh",)
    assert kernel_loader.module_source("noah") == (
        kernel_loader._preamble() + glibc + strict_prefix
        + (_KDIR / "noah.cu").read_text(encoding="utf-8"))
    assert kernel_loader.EXTRA_HEADERS["noah_init"] == ("glibc_flt32.cuh", "noah.cu")
    reused = (_KDIR / "noah.cu").read_text(encoding="utf-8")
    body = (_KDIR / "noah_init.cu").read_text(encoding="utf-8")
    assert kernel_loader.module_source("noah_init") == (
        kernel_loader._preamble() + glibc + reused + body)
    assert "__device__ static real noah_frh2o(" not in body

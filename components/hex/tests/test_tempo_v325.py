"""TEMPO v3.2.5 (tempo_v325) is the default TEMPO; v3.0.3 (tempo_v303) stays selectable and unchanged (CPU tier).

THE BREAKAGES THESE PREVENT: a misspelt ``config_tempo_version`` silently running the default; the v3.0.3 pin
picking up a v3.2.5 statement (its strict module is what the RRFS 44c0849 landing gate and every bitwise RRFS
claim measured); the warp and serial copies of a v3.2.5 fall-speed or sedimentation routine drifting apart (both
shapes come from one text and production runs the warp one); the fast layers no longer applying to the new
column (fast is the default numerics). The word-for-word proof on real inputs is the native module replay
recorded in the port's report (strict hex warp kernel vs pristine TEMPO tempo_v3.2.5 + the labelled
final_weight definition).
"""
import json
import re
from importlib import resources

import pytest

from woof.hex.mpasphys.rrfs import tempo_v303 as v303
from woof.hex.mpasphys.rrfs import tempo_v325 as v325


def test_version_option_default_pin_and_refusal():
    assert v303.DEFAULT_TEMPO_VERSION == "v3.2.5" and v303.tempo_version({}) == "v3.2.5"
    assert v303.tempo_version({"config_tempo_version": "v3.0.3 "}) == "v3.0.3"
    with pytest.raises(ValueError, match="config_tempo_version"):
        v303.tempo_version({"config_tempo_version": "v3.2"})


def test_profile_declares_both_options_on_both_rrfs_routes():
    table = json.loads(resources.files("woof.hex").joinpath("data/profiles/mpas-physics-suites.v1.json").read_text())
    for key in ("mpasphys_rrfs", "mpasphys_rrfs_44c0849"):
        section = table[key]
        assert section["registry_defaults"]["config_tempo_version"] == {"type": "character", "value": "v3.2.5"}
        assert section["registry_defaults"]["config_tempo_hailhyperaware"] == {"type": "logical", "value": True}
        record = section["registry_groups"]["physics_mp_tempo"]
        assert "config_tempo_version" in record and "config_tempo_hailhyperaware" in record


def test_v303_strict_text_carries_no_v325_statement():
    for token in ("hail_check_and_update", "compute_hail_fraction", "hailhyperaware_flag", "prh_hcw", "rho_not))"):
        assert token not in v303._COLUMN, token
    assert "first && sub==0 && c==0" in v303._WARP and "first && c==0" in v303._DRIVER


def test_v325_column_carries_the_upstream_changes():
    c = v325._COLUMN
    for token in ("hail_check_and_update(", "compute_hail_fraction(", "hail_fallspeed(", "int hailhyperaware_flag;",
                  "TT(prh_hcw)", "delta_aw = ssatw + 1.0f - a_w_i;", "using TK_max_ni = float;",
                  "const float roverrv = PS(rdry) / PS(rv);", ".014f);", "0.28f * PS(sc3)",
                  "(sed_r[k + 1] - sed_r[k]) * odz", "continue;  // v3.2.5 main:2303",
                  "xni = ns[k] + ni[k]", "PS(c_sqrd) * t1_subl[k]", "0.504843467198652f * PS(rho_not)",
                  "final_weight = 1.0f;"):
        assert token in c, token
    assert "PS(roverrv)" not in c and "Fortran quirk: RETURN" not in c
    # every rain/graupel/ice/hail sedimentation runs inside its substep loop, with no fall-speed recompute there
    main = c[c.index("TEMPO_HDN inline void tempo_main_column("):]
    assert re.search(r"for \(n = 1; n <= substeps_sedi; \+\+n\) \{\n\s+sedimentation\(P, C, nz, W\.ri,", main)
    serial_rain = main[main.index("// sedimentation: rain"):main.index("// graupel (main:685-743)")]
    assert serial_rain.split("} else {")[-1].count("rain_fallspeed") == 0


def test_v325_drivers_take_five_flags_and_first_step_on_every_column():
    assert "flags[4]};" in v325._DRIVER and "tempo_call(nz,dt,s,work[c],blob,flags,first," in v325._DRIVER
    assert "int melt,int hyper,int first,int presence," in v325._WARP
    assert "last,first,0,hyper};" in v325._WARP and "first && sub==0" not in v325._WARP


ROUTINES = ("rain_fallspeed", "graupel_fallspeed", "hail_fallspeed", "snow_fallspeed", "ice_fallspeed",
            "cloud_fallspeed", "sedimentation")
WARP_ONLY = ("for (", "TEMPO_", "tempo_next_active", "tempo_warp_", "__shfl", "rows", "ktop", "substeps", "[j]",
             "const int j", "return", "tempo_active_count", "tempo_active_level",
             "int rank=TEMPO_LANE", "rank+=32")
SERIAL_ONLY = ("ktop_sedi", "substeps_sedi", "vt[k + 1]", "vtn[k + 1]", "int k = nz", "for (", "return")
ASSIGNMENT = re.compile(r"(?<![<>=!])=(?!=)")


def _statements(text, skip):
    text = re.sub(r"//[^\n]*", "", text)
    out = set()
    for piece in re.sub(r"\s+", " ", text).split(";"):
        piece = re.split(r"[{}]", piece)[-1].strip()
        piece = re.sub(r"^(if|else if) \(.*?\) ", "", piece)
        piece = re.sub(r"^(?:--|\+\+)\w+\) ", "", piece)
        piece = re.sub(r"^TEMPO_LEVELS\(\w+, ?[\w +-]+\) ", "", piece)
        if ASSIGNMENT.search(piece) and not any(token in piece for token in skip):
            out.add(piece)
    return out


@pytest.mark.parametrize("name", ROUTINES)
def test_v325_warp_and_serial_copies_compute_the_same_statements(name):
    text = v325._COLUMN
    start = text.index(f"inline void {name}(")
    body = text[start:text.index("\n}\n", start)]
    warp = re.search(r"#if TEMPO_WARP\n(.*?)\n#(?:else|endif)\n", body, re.S)
    assert warp, name
    warp_statements = _statements(warp.group(1), WARP_ONLY)
    serial_statements = _statements(body[warp.end():], SERIAL_ONLY)
    assert warp_statements and warp_statements == serial_statements, name


def test_v325_fast_layers_compose():
    from woof.hex.mpasphys.rrfs.tempo_fast_v325 import fast_column
    fast = fast_column(v325._COLUMN)
    assert "tempo_fsed<1>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls,\n" \
           "                    &W.hail_liquid_equiv_precip);" in fast
    assert not re.search(r"^using TK_\w+ = double;$", fast, re.M) and "__constant__ TempoScalarsF tempo_k;" in fast
    strict = v325.warp_source("// maths\n", max_levels=59, numerics="strict")
    assert v325._COLUMN in strict and "tempo_fsed" not in strict


@pytest.mark.parametrize("table_key", ["mpasphys_rrfs", "mpasphys_rrfs_44c0849"])
def test_exact_mode_runs_the_pinned_tempo_unless_named(table_key):
    """Exact mode reproduces the RRFS pin (TEMPO 9adf9ef = v3.0.3); correct mode (the door default) takes the
    latest upstream TEMPO; a version the namelist names wins in either mode.  The gate's native decks name none."""
    from woof.hex.mpasphys.rrfs import driver_rrfs
    suite = "'hrrrv5'" if table_key == "mpasphys_rrfs_44c0849" else "'rrfs_convection_permitting'"
    spec = driver_rrfs.specification(table_key=table_key)
    if suite.strip("'") not in spec["suite_routes"]:
        suite = repr(next(iter(spec["suite_routes"])))
    base = f"&physics\n config_physics_suite = {suite}\n/\n"
    named = base + "&physics_mp_tempo\n config_tempo_version = 'v3.2.5'\n/\n"

    def version(text, mode):
        config = driver_rrfs.resolve_namelist(text, physics_mode=mode, table_key=table_key)
        return v303.tempo_version(config.options)

    assert version(base, "exact") == "v3.0.3"
    assert version(base, "correct") == "v3.2.5"
    assert version(named, "exact") == "v3.2.5"
    assert version(named.replace("v3.2.5", "v3.0.3"), "correct") == "v3.0.3"


def test_streams_record_a_non_pin_tempo_and_rediagnose_reads_it(tmp_path):
    """THE BREAKAGE: a v3.0.3 state rediagnosed with v3.2.5's operator (or the reverse) because the file did not say
    which TEMPO wrote it. Only non-pin runs carry the attributes, so exact-mode headers stay identical to native."""
    from types import SimpleNamespace
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.hex.mpasphys.rrfs.tempo_refl_v303 import file_tempo

    def cfg(**options):
        return SimpleNamespace(selectors={"microphysics": "mp_tempo"}, options=options)

    assert v303.output_attributes(cfg()) == {"hex_tempo_version": "v3.2.5", "hex_tempo_hailhyperaware": "YES"}
    assert v303.output_attributes(cfg(config_tempo_hailhyperaware=False))["hex_tempo_hailhyperaware"] == "NO"
    assert v303.output_attributes(cfg(config_tempo_version="v3.0.3")) == {}
    assert v303.output_attributes(SimpleNamespace(selectors={"microphysics": "mp_thompson"}, options={})) == {}
    for name, attrs, want in (("new.nc", v303.output_attributes(cfg()), ("v3.2.5", True)),
                              ("pin.nc", {}, ("v3.0.3", False))):
        with netCDF4.Dataset(tmp_path / name, "w") as d:
            for k, v in attrs.items():
                d.setncattr(k, v)
        assert file_tempo(tmp_path / name)[:2] == want

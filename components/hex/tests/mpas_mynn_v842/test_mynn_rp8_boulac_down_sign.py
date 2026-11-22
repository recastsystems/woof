"""RP8 (correct mode): the BouLac downward mixing length takes the upstream sign.

THE BREAKAGE: boulac_length0 and boulac_length (MPAS-A v8.4.2 bl_mynn_subroutines.F90:1327 and 1482) take the
downward search's linear branch (two adjacent levels of exactly equal theta) as
tl = (qtke - zdo_sup)/(beta*(theta(izz) - theta(k))).  The parcel can stop there only when theta(k) > theta(izz),
so tl is negative and the downward length is short by twice the distance the parcel travels into that layer.
NCAR/MYNN-EDMF PR 94 (v6.1.1) and WRF PR 2431 use theta(k) - theta(izz).  BouLac runs only with
config_mynn_mixlength = 1, which the hrrr-like profile sets; MPAS-A's default 2 never reaches it.

Exact mode keeps native MPAS v8.4.2's text byte for byte; correct mode with mixlength 1 applies RP8 on the
CUDA MYNN, and a native-cpu MYNN (pristine v8.4.2) lists it as a named gap, as for RP1 and RP7.
"""
from types import SimpleNamespace

from woof.hex.mpasphys import bl_mynn_v842 as m
from woof.hex.mpasphys.driver_v842 import CORRECTIONS_V842, correct_source, resolve_namelist

ROWS = m.MYNN_TEXT_CORRECTIONS["RP8"]


def deck(mixlength=None, mode="correct"):
    physics = {"config_physics_suite": "convection_permitting"}
    if mixlength is not None:
        physics["config_mynn_mixlength"] = mixlength
    return ("&nhyd_model\nconfig_physics_mode = %r\n/\n" % mode
            + "&physics\n" + "\n".join(f"{k} = {v!r}" for k, v in physics.items()) + "\n/")


def test_rp8_text_is_the_upstream_sign_in_both_boulac_routines():
    assert len(ROWS) == 2
    for old, new, count in ROWS:
        assert count == 1 and m.MYNN_MODULE_CUDA.count(old) == 1
        # only the sign of the theta difference moves
        assert old.replace("mpas_sub(theta(izz), theta(k))", "mpas_sub(theta(k), theta(izz))") \
                  .replace("mpas_sub(theta(izz), theta(iz))", "mpas_sub(theta(iz), theta(izz))") == new
        assert "zdo_sup" in old          # the downward search; the upward one is not touched
    corrected = correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, {"RP1", "RP7", "RP8"}, "test")
    assert all(old not in corrected and corrected.count(new) == 1 for old, new, _ in ROWS)
    rp7_only = correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, {"RP1", "RP7"}, "test")
    assert all(rp7_only.count(old) == 1 for old, _, _ in ROWS)
    assert correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, (), "test") == m.MYNN_MODULE_CUDA


def test_rp8_is_applied_only_where_boulac_runs():
    assert CORRECTIONS_V842["RP8"]["slots"] == {"pbl": ("bl_mynn",)} and CORRECTIONS_V842["RP8"]["breakage"]
    cuda = {"pbl": "hex-cuda"}
    on = resolve_namelist(deck(1), backends=cuda)
    assert "RP8" in on.corrections and not [g for g in on.gaps if g.finding == "RP8"]
    assert "RP8" not in resolve_namelist(deck(2), backends=cuda).corrections
    assert "RP8" not in resolve_namelist(deck(), backends=cuda).corrections       # MPAS-A's default, 2
    assert resolve_namelist(deck(1, mode="exact"), backends=cuda).corrections == frozenset()
    native = resolve_namelist(deck(1))
    assert "RP8" not in native.corrections
    assert ("RP8", "pbl") in {(g.finding, g.slot) for g in native.gaps}


def test_the_hrrr_like_profile_applies_rp8():
    from woof.hex.mpasphys.forecast_v842 import native_route
    config, _ = native_route(SimpleNamespace(physics_profile="hrrr-like"), default_mode="correct")
    assert config.options["config_mynn_mixlength"] == 1 and "RP8" in config.corrections
    plain, _ = native_route(SimpleNamespace(physics_profile="hrrr-like-aerosol"), default_mode="correct")
    assert "RP8" not in plain.corrections


def test_the_kernel_pack_recipe_declares_every_mynn_text_set():
    # tests/test_hex_aot.py counts the declared bl_mynn images: one per subset of the text corrections.
    import inspect
    from woof.hex.cuda_backend import aot_build
    assert "combinations(_ids, n)" in inspect.getsource(aot_build)
    assert set(m.MYNN_TEXT_CORRECTIONS) == {"RP7", "RP8"}

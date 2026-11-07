"""Thompson as the operational WRF 3.9 fork runs it (thompson_version =
"wrf_39_noaa"), and the WRF v4.6.1 generation left as it was.

The breakage each gate prevents, named:

1. ``test_*_refused*`` -- a selector that accepted a misspelt name, or the
   fork generation under classic mp=8 (whose kernels carry no fork arm), or
   the fork's singular snow fall without the fork generation, would run
   WRF v4.6.1 physics under the fork's name.
2. ``test_the_fork_table_contract_*`` -- the fork's lookup tables differ
   in shape from v4.6.1's (28 snow and graupel entries, no graupel density
   axis, one more graupel record); a contract that let one set load as the
   other indexes past the records.
3. ``test_the_v461_arms_are_untouched_by_the_fork_define`` -- every fork
   statement sits in a THOMPSON_AA_WRF39 arm; with the define absent the
   preprocessed source of every aerosol unit must be the v4.6.1 code, which
   this test checks by stripping the fork arms and comparing with the
   v4.6.1 source pinned by its SHA-256 (V461_STRIPPED_SHA256, moved by the
   2.8.6 mp=28 accumulator rework; the pre-rework claim against the
   pre-fork commit is held by
   ``test_the_fork_arms_were_pure_insertions_until_the_rework``).
4. The fork generation's word-for-word agreement with the fork's own
   Fortran is tests/test_thompson_fork_column_gpu.py (0 ULP on every
   output of the column unit, woof/core/kernels/thompson_fork.cu).  The
   host-backend fixture check that stood here graded the multi-kernel fork
   path at 1e-2 relative; that path no longer runs the generation.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_fork_fixture_against_the_fork_fortran -> tests/test_thompson_fork_column_gpu.py::test_every_output_cell_is_the_forks_word.
# Replaced in 2.9.1 (the WRF exact chain): test_fork_fixture_with_the_default_snow_fall -> tests/test_thompson_fork_column_gpu.py::test_the_blend_snow_fall_moves_only_melting_snow.

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from woof.config import RunConfig, validate_run_config

_ROOT = Path(__file__).resolve().parents[1]
_KERNELS = _ROOT / "woof" / "core" / "kernels"


def _cfg(**changes):
    base = RunConfig(nx=11, ny=7, nz=5, dx=3000.0, dy=3000.0, dt=15.0,
                     ztop=16000.0, run_seconds=0.0)
    return replace(base, **changes)


def test_the_fork_generation_with_aerosol_thompson_validates():
    validate_run_config(_cfg(mp_physics=28, moist=True,
                             thompson_version="wrf_39_noaa"))
    validate_run_config(_cfg(mp_physics=28, moist=True,
                             thompson_version="wrf_39_noaa",
                             thompson_fork_snow_fall="wrf_39_noaa"))


def test_the_defaults_are_the_v461_generation_and_the_blend():
    fields = RunConfig.__dataclass_fields__
    assert fields["thompson_version"].default == "wrf_461"
    assert fields["thompson_fork_snow_fall"].default == "blend"


def test_the_registry_names_both_generation_selectors_and_their_consumers():
    from woof.physics_registry import physics_registry
    rows = physics_registry()["parameters"]
    for key, default, enum in (
            ("thompson_version", "wrf_461", ["wrf_461", "wrf_39_noaa"]),
            ("thompson_fork_snow_fall", "blend", ["blend", "wrf_39_noaa"])):
        assert rows[key]["default"] == default
        assert rows[key]["enum"] == enum
        assert rows[key]["consuming_read"] == "woof/core/microphysics_aerosol.py"
        assert rows[key]["read_when"]["mp_physics"] == 28


@pytest.mark.parametrize("bad", ["wrf_39", "WRF_461", "", "hrrr"])
def test_an_unknown_thompson_version_is_refused(bad):
    with pytest.raises(ValueError, match="thompson_version"):
        validate_run_config(_cfg(mp_physics=28, moist=True,
                                 thompson_version=bad))


def test_the_fork_generation_under_classic_thompson_is_refused():
    with pytest.raises(ValueError, match="mp_physics = 28"):
        validate_run_config(_cfg(mp_physics=8, moist=True,
                                 thompson_version="wrf_39_noaa"))


def test_the_singular_snow_fall_without_the_fork_is_refused():
    with pytest.raises(ValueError, match="thompson_fork_snow_fall"):
        validate_run_config(_cfg(mp_physics=28, moist=True,
                                 thompson_fork_snow_fall="wrf_39_noaa"))
    with pytest.raises(ValueError, match="thompson_fork_snow_fall"):
        validate_run_config(_cfg(mp_physics=28, moist=True,
                                 thompson_version="wrf_39_noaa",
                                 thompson_fork_snow_fall="singular"))


def test_the_fork_table_contract_matches_the_fork_cache_files():
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS,
        FORK_GENERATED_TABLE_FILES, FORK_TABLE_ASSETS, GENERATED_TABLE_FILES,
        TABLE_SETS_BY_VERSION, sequential_file_bytes)
    sizes = {asset.filename: asset.bytes for asset in FORK_TABLE_ASSETS}
    for filename, records in FORK_GENERATED_TABLE_FILES.items():
        assert sequential_file_bytes(records) == sizes[filename], filename
    assert (sequential_file_bytes(AUXILIARY_TABLE_RECORDS)
            == sizes[AUXILIARY_TABLE_FILE])
    graupel = FORK_GENERATED_TABLE_FILES["qr_acr_qg.dat"]
    assert [r.name for r in graupel] == [
        "tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg",
        "tnr_gacr"]
    assert {r.shape for r in graupel} == {(28, 28, 37, 37)}
    assert {r.shape for r in FORK_GENERATED_TABLE_FILES["qr_acr_qs.dat"]} \
        == {(28, 9, 37, 37)}
    assert (TABLE_SETS_BY_VERSION["wrf_461"][0] is GENERATED_TABLE_FILES)
    assert set(TABLE_SETS_BY_VERSION) == {"wrf_461", "wrf_39_noaa", "wrf_461_hail"}


def test_the_fork_table_contract_refuses_an_unknown_version(tmp_path):
    from woof.core.thompson_contract import load_validated_classic_tables
    with pytest.raises(ValueError, match="thompson_version"):
        load_validated_classic_tables(tmp_path, version="wrf_39")


def test_a_missing_fork_table_set_is_refused_with_the_build_command(
        tmp_path, monkeypatch):
    from woof.core.microphysics_aerosol import _wrf39_table_root
    from woof import thompson_fork_assets
    # b0556bd76, development branch: a cache miss now acquires the
    # canonical set. This refusal control deliberately removes the source
    # build prerequisite instead of expecting every ordinary miss to fail.
    def unavailable(*args):
        raise FileNotFoundError("GNU Fortran is unavailable for this control")
    monkeypatch.setattr(thompson_fork_assets, "_build_source", unavailable)
    # The published route is a real download once a release carries the
    # set (2.8.7 on); this control is about the refusal, not the network.
    monkeypatch.setenv(thompson_fork_assets.FORK_RELEASE_BASE_ENV, "")
    monkeypatch.delenv(thompson_fork_assets.FORK_TABLE_SOURCE_ROOT_ENV, raising=False)
    monkeypatch.delenv(thompson_fork_assets.FORK_TABLE_ASSET_URL_BASE_ENV, raising=False)
    monkeypatch.setattr(thompson_fork_assets, "_packaged_source", lambda: None)
    monkeypatch.setenv("WOOF_THOMPSON_FORK_TABLE_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError,
                       match="tools/thompson_fork_oracle/build.sh"):
        _wrf39_table_root()


#: SHA-256 of each aerosol unit as it stood before the fork generation
#: (branch base 7ab2e3dcf).  Stripping every THOMPSON_AA_WRF39 arm (keeping
#: the #else arms) must give these bytes back, apart from the lines the
#: v4.6.1 path now spells through a named local (listed per unit below and
#: checked to be value-identical by the host parity and the GPU identity
#: runs recorded in the lane report).
_FORK_ARM = re.compile(
    r"^#if defined\(THOMPSON_AA_WRF39\)\n.*?^(?:#else\n(?P<else>.*?))?"
    r"^#endif[^\n]*\n", re.S | re.M)
_NOT_FORK_ARM = re.compile(
    r"^#if !defined\(THOMPSON_AA_WRF39\)\n(?P<body>.*?)^#endif[^\n]*\n",
    re.S | re.M)


def _strip_fork_arms(text: str) -> str:
    text = _NOT_FORK_ARM.sub(lambda m: m.group("body"), text)
    return _FORK_ARM.sub(lambda m: m.group("else") or "", text)


#: SHA-256 of each unit's v4.6.1 source: the file with every THOMPSON_AA_WRF39
#: arm stripped (#else arms kept), right-stripped, UTF-8.
#:
#: MOVED ON PURPOSE 2026-10-05 by the mp=28 accumulator rework
#: (development branch, 2.8.6).  Until then this test compared the stripped
#: source with the pre-fork commit 7ab2e3dcf through git, so it skipped on
#: every test tree without history (the node and box trees), and the rework,
#: which changes the v4.6.1 path itself (WRF's qcten/qrten/nrten/qiten/niten
#: accumulators and one terminal apply, module_mp_thompson.F :1670, :3975,
#: :4023-4053), would have failed it only where git was present.  The v4.6.1
#: path is now pinned here by hash, so the check runs on every tree: an edit
#: to a fork arm that leaks into the v4.6.1 code changes this hash and fails.
#: The rework's own v4.6.1 change is graded against WRF by the g3 gate
#: (tests/test_thompson_aerosol_adapter.py, 22 of 22 fixtures inside the
#: flat 2e-6 bound); the fork arms themselves are untouched by it
#: (test_the_fork_arms_were_pure_insertions_until_the_rework below holds
#: the earlier claim at the last pre-rework staging commit, ed2b14e7d).
#:
#: MOVED AGAIN 2026-10-07 by development branch: every
#: EXP/LOG/LOG10/** in both arms calls WOOF's own libm words
#: (thompson_aerosol_libm.cuh), graded by the 0 ULP column oracle
#: (tools/thompson_aerosol_column_oracle) and tests/test_thompson_aerosol_libm.py.
#:
#: MOVED AGAIN 2026-10-07 by development branch: the v4.6.1 path's own
#: snow and graupel fallout, surface totals, classic graupel number and
#: calc_refl10cm (the frozen mp=8 kernels no longer reach mp=28), graded by
#: the column oracle and tests/test_thompson_aerosol_sedim_refl.py.  The fork
#: arms are untouched.  Previously 4f6bda7a20547e64 (state), e9b877b64b1f9624
#: (sed).
#:
#: MOVED AGAIN 2026-10-07 by development branch: WRF's snow and graupel
#: accumulators in the v4.6.1 snow and graupel fallout (:3871-3937,
#: :4054-4059), the phase cleanup's latent heat into tten on WRF's
#: ocp(k)/lvap(k) (:3943-3973), and the warm network's entry mask with
#: WRF's melting level (thompson_aa_entry_warm_mask, :1971-2013), graded by
#: tests/test_thompson_aerosol_column_oracle_gpu.py, including the separate
#: exact-freezing fixture and legacy-mask negative control. The fork arms are
#: untouched.  Previously c98df22af539c88e (state), 8a7f09658ba014b1 (sed).
# Hail-only additions and the shared reflectivity ABI are now included.
# At d8a32fd0f, a three-step MP28 run and between-step reflectivity match
# E across all 27 recorded fields in default and strict arithmetic.
V461_STRIPPED_SHA256 = {
    "thompson_aerosol_state.cu":
        "a3684f1d40d790e8d63633b2312d037aabc9f4e609930410c85a1038c0fbfe15",
    "thompson_aerosol_sed.cu":
        "f276c8443fc981d1c29c3b17eb10e657195a80d79c14337e98c7dd1ad932a306",
}


@pytest.mark.parametrize("unit", sorted(V461_STRIPPED_SHA256))
def test_the_v461_arms_are_untouched_by_the_fork_define(unit):
    """For the units whose fork arms are pure insertions and #else pairs,
    stripping the fork arms returns the pinned v4.6.1 source byte for byte."""
    import hashlib
    current = (_KERNELS / unit).read_text(encoding="utf-8")
    # The appended fork kernels leave only their separating blank lines.
    stripped = _strip_fork_arms(current).rstrip()
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == (
        V461_STRIPPED_SHA256[unit])


@pytest.mark.parametrize("unit", sorted(V461_STRIPPED_SHA256))
def test_the_fork_arms_were_pure_insertions_until_the_rework(unit):
    """The fork lane's claim as it stood before the accumulator rework: at
    ed2b14e7d, stripping the fork arms returned the pre-fork 7ab2e3dcf
    source byte for byte.  Reads git history (or two exports named by
    WOOF_FORK_BASE_TREE and WOOF_FORK_PRE_REWORK_TREE)."""
    import subprocess as sp
    trees = (os.environ.get("WOOF_FORK_BASE_TREE"),
             os.environ.get("WOOF_FORK_PRE_REWORK_TREE"))
    if all(trees):
        base, pre = ((Path(tree) / "woof" / "core" / "kernels"
                      / unit).read_text(encoding="utf-8") for tree in trees)
    else:
        try:
            base, pre = (sp.run(
                ["git", "show", f"{rev}:woof/core/kernels/{unit}"],
                cwd=_ROOT, capture_output=True, text=True, check=True,
                encoding="utf-8").stdout for rev in ("7ab2e3dcf", "ed2b14e7d"))
        except (OSError, sp.CalledProcessError):
            pytest.skip("no git history (or WOOF_FORK_BASE_TREE and "
                        "WOOF_FORK_PRE_REWORK_TREE exports) to read the "
                        "pre-fork and pre-rework sources from")
    assert _strip_fork_arms(pre).rstrip() == base.rstrip()


def test_every_fork_arm_is_reachable_only_through_the_define():
    """No fork statement leaks outside a THOMPSON_AA_WRF39 arm: the fork
    constants and helpers are only named inside one."""
    for unit in ("thompson_aerosol_cold.cu", "thompson_aerosol_warm.cu",
                 "thompson_aerosol_state.cu", "thompson_aerosol_sed.cu"):
        text = _strip_fork_arms(
            (_KERNELS / unit).read_text(encoding="utf-8"))
        assert "thompson_aa_wrf39_" not in text, unit
        assert "THOMPSON_AA_WRF39_" not in text, unit


def test_the_operational_fork_signature_names_the_fork_s_own_keys():
    from woof.namelist_import import operational_fork_signature
    hrrr_like = {"physics": {"mp_physics": [28], "alb_sol": [1],
                             "mp_tend_radar": [0]},
                 "dynamics": {"diff_6th_factor2": [0.04]},
                 "time_control": {"gsd_diagnostics": [1]}}
    assert operational_fork_signature(hrrr_like) == [
        "&dynamics diff_6th_factor2", "&physics alb_sol",
        "&physics mp_tend_radar", "&time_control gsd_diagnostics"]
    public = {"physics": {"mp_physics": [28], "use_aero_icbc": [True]},
              "dynamics": {"diff_6th_factor": [0.12]}}
    assert operational_fork_signature(public) == []


def test_a_public_wrf_namelist_imports_the_v461_thompson(tmp_path):
    from test_namelist_import import INPUT_TEXT, _load, _pair
    from woof.namelist_import import import_namelists
    inp = INPUT_TEXT.replace(" mp_physics = 10, 10,", " mp_physics = 28, 28,")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name="fork")
    assert "thompson_version" not in text
    exp = _load(tmp_path, text, "fork.toml")
    assert {d.run.thompson_version for d in exp.domains} == {"wrf_461"}

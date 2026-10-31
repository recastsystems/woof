"""The licence layer, pinned the way every other claim in this repository is.

THE BREAKAGE THIS PREVENTS
--------------------------
The 2.7.0 licence lane found two present breaches of PERMISSIVE licences --
NumPy's BSD-3-Clause and MPAS's -- in a tree that already carried a careful
NOTICE, a licenses/ directory and a 670 KB binary-form notice.  Neither was
found by the licence census, because the census sorted third-party code into
"vendored crate" and "pip dependency" and both of these are neither: they are
FIRST-PARTY CRATES THAT PORT THIRD-PARTY CODE INTO THEMSELVES.
``static-fields`` ports four NumPy SIMD kernels; ``rw-mpas`` is 44,768 lines
of MPAS-Atmosphere v8.4.1.  Both ship compiled, in the release bundles and in
platform wheels, and until 2.7.0 neither appeared in any notice.

Nothing checked any of it.  Before this file, ``grep -rIln 'licenses/' tests/``
returned one unrelated hit: no test asserted that ``licenses/*`` exists, that
it reaches a wheel, that the nineteen per-file notice headers survive an edit,
that the kernel notice's file lists still match the tree, or that the
binary-form notice's two copies still agree.  In a project whose discipline is
"pin it or it drifts", the licence layer was the one thing unpinned.

WHAT THIS FILE ASSERTS
----------------------
1.  Every text in ``licenses/`` is non-empty and is NAMED in the root NOTICE,
    and every ``licenses/...`` path the NOTICE names exists.  A text nothing
    points at, and a pointer at nothing, are both failures.
2.  Both distributions declare ``licenses/*`` in ``license-files``, and the
    kernel-directory notice is named in package-data.  This is the only route
    by which a ``pip install`` user receives any of it.
3.  The binary-form notice's two copies are byte-identical and the bundle copy
    sits under a directory ``build_bridge_bundle`` actually walks.
4.  The nineteen files that carry a per-file notice still carry it.
5.  The Arm scope list in ``woof/core/kernels/LICENSE-third-party.txt`` is
    DERIVED FROM THE TREE, not typed: the eighteen files are exactly the ones
    that reproduce Arm's coefficient tables.  The FDLIBM list is pinned, and a
    kernel that starts defining an FDLIBM routine without joining it fails.
    Both walks read every subdirectory of the kernel tree (tempo/ included).
6.  Every first-party crate carrying a third-party licence marker, and every
    upstream Fortran project a first-party crate cites, is named in the NOTICE.
    This is the check that would have caught NumPy and MPAS.
7.  The earlier Grell-Freitas gamma replaced at 2.7.0 stays deleted, and so
    do the two glibc-only libm fragments removed at the same release.
8.  No shipped licence text, kernel header or document describes the
    Grell-Freitas gamma in ``glibc_flt32.cuh`` as a transcription, as LGPL,
    as FSF copyright or as glibc's work.  Both gammas, the earlier one and
    the current one, are this project's own work (ruling of 2026-09-12); an
    earlier development record said otherwise and was wrong, and the claim
    came back into the tree three times from that record before this pin.
9.  The two generated notices survive their own generators.  The terminal
    section's generator, which every version bump runs, replaces that
    section and nothing after it; and the Zarr reader's notice reproduces
    the binary notice as it is now, resolved from the lock as it is now.

It reads text.  No CUDA, no compiler, no network, no device.
"""

from __future__ import annotations

import pathlib
import hashlib
import importlib.util
import json
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTICE = ROOT / "NOTICE"
LICENSES = ROOT / "licenses"
KERNELS = ROOT / "woof" / "core" / "kernels"
KERNEL_NOTICE = KERNELS / "LICENSE-third-party.txt"
COMPANION = ROOT / "recast-woof-data"


def _requires_source_tree() -> None:
    """Skip only when this is genuinely not a source checkout.

    The earlier form skipped whenever ``NOTICE`` or ``licenses/`` was absent,
    which made the whole gate fail open: deleting the entire licence layer
    turned 31 assertions into 31 skips and left the suite green.  That is the
    defect class this branch exists to remove -- an absent measurement scored
    as agreement -- so the two cases are separated here.  A tree carrying
    ``pyproject.toml`` IS a source checkout, and in one the licence layer
    missing is a failure, not an absence of evidence.
    """
    if not (ROOT / "pyproject.toml").is_file():
        pytest.skip("not a source checkout; the licence layer is not present")
    missing = [name for name, ok in (("NOTICE", NOTICE.is_file()),
                                     ("licenses/", LICENSES.is_dir())) if not ok]
    if missing:
        raise AssertionError(
            "source checkout is missing the licence layer: "
            + ", ".join(missing)
            + " -- these ship with the tree and their absence is a failure, "
              "not a reason to skip")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _kernel_sources() -> list[pathlib.Path]:
    """Every CUDA source the wheel ships from the kernel tree, subdirectories
    included.  The breakage the recursion prevents: tempo/tempo_common.cuh
    carried FDLIBM's binary64 log10 in 2.8.9 outside both FDLIBM lists, and
    the scope gates could not see it because they read the top level only."""
    return sorted(list(KERNELS.rglob("*.cu")) + list(KERNELS.rglob("*.cuh")))


def _kernel_name(path: pathlib.Path) -> str:
    """The name the notices use: the file name at the top level, the path
    under woof/core/kernels/ below it (tempo/tempo_common.cuh)."""
    return path.relative_to(KERNELS).as_posix()


def _decommented(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


# ---------------------------------------------------------------------------
# 1.  the texts exist, are non-empty, and the NOTICE points at each of them
# ---------------------------------------------------------------------------
def test_every_licence_text_is_named_by_the_notice_and_vice_versa() -> None:
    _requires_source_tree()
    notice = _read(NOTICE)

    # Files, not entries: licenses/soundings/ is a directory (the public sounding
    # renderer's dependency notices).  Reading entries, 2.9.3's public CI run
    # 39368051825 reported "licence texts with no text: ['soundings']" on
    # windows-2025, where a directory's st_size is 0 (4096 on ext4, so the Linux
    # job passed and the desktop's filesystem hid it too).
    texts = [p.relative_to(ROOT).as_posix() for p in sorted(LICENSES.rglob("*")) if p.is_file()]
    empty = [name for name in texts if (ROOT / name).stat().st_size == 0]
    assert empty == [], f"licence texts with no text: {empty}"

    unnamed = [name for name in texts if name not in notice]
    assert unnamed == [], (
        "the NOTICE names no path for these licence texts, so a reader has no "
        f"way to find them: {unnamed}")

    # A path in this repository, not the tail of a URL: the SoilGrids credit
    # cites https://creativecommons.org/licenses/by/4.0/, which is no file.
    cited = {m.rstrip(".,;") for m in
             re.findall(r"(?<![/\w.])licenses/[A-Za-z0-9._/-]+", notice)}
    cited.discard("licenses/licenses")       # the PEP 639 wheel path, not a file
    cited = {name.rstrip("/") for name in cited}   # "licenses/soundings/: ..." names the directory
    dangling = sorted(name for name in cited if not (ROOT / name).exists())
    assert dangling == [], f"the NOTICE points at licence texts that are not here: {dangling}"


def test_the_notice_names_every_grant_that_conditions_reproduction() -> None:
    """A named-grant checklist, so a section cannot quietly disappear.

    Each entry is a grant whose text this distribution is required to
    reproduce, and the token is the shortest string that identifies its
    section.  MPAS and NumPy are on the list because 2.7.0 is when they were
    found missing, and a checklist that only holds what was already right
    would have held in 2.6.5 too.
    """
    _requires_source_tree()
    notice = _read(NOTICE)
    required = (
        "Arm Limited",                    # MIT
        "CORE-MATH",                      # MIT
        "Sun Microsystems",               # FDLIBM
        "Atmospheric and Environmental Research",   # AER RRTMG, RTE+RRTMGP
        "UChicago Argonne",               # Py-ART
        "Los Alamos National Security",   # MPAS
        "NumPy Developers",               # NumPy
        "Numerical Recipes",
        "SIL Open Font License",
        "public domain",                  # WRF / UCAR
        "WRF Preprocessing System",       # WPS
        "BowEcho",                       # vendored simulation, MIT OR Apache-2.0
        "recast-radar-tools",             # compiled writers, MIT OR Apache-2.0
        "recast-radar-bzip2",             # retained libbzip2 randomisation table
    )
    missing = [token for token in required if token not in notice]
    assert missing == [], f"the NOTICE no longer names: {missing}"


# ---------------------------------------------------------------------------
# 2.  the texts reach a built distribution
# ---------------------------------------------------------------------------
def test_both_distributions_ship_the_licence_directory() -> None:
    _requires_source_tree()
    for pyproject, licences in ((ROOT / "pyproject.toml", LICENSES),
                                (COMPANION / "pyproject.toml",
                                 COMPANION / "licenses")):
        if not pyproject.is_file():
            pytest.skip(f"{pyproject} is not in this checkout")
        with pyproject.open("rb") as stream:
            config = tomllib.load(stream)
        declared = config["project"].get("license-files", [])
        # `licenses/*` does not recurse: the 2.9.3 candidate wheel shipped none of
        # licenses/soundings/'s three texts while NOTICE named them.
        subdirs = sorted(p.relative_to(pyproject.parent).as_posix() for p in licences.rglob("*") if p.is_dir())
        unshipped = [f"{d}/*" for d in subdirs if f"{d}/*" not in declared]
        assert unshipped == [], (
            f"{pyproject} does not ship these licence subdirectories; add each to license-files: {unshipped}")
        assert "licenses/*" in declared, (
            f"{pyproject} does not ship licenses/, so a pip install of it "
            f"performs no reproduction condition at all: {declared}")
        assert licences.is_dir() and any(licences.iterdir()), (
            f"{pyproject} globs {licences}, which is missing or empty")


def test_the_kernel_directory_notice_is_declared_as_package_data() -> None:
    """The .cu files are digest-pinned, so their notice sits beside them.

    That only works if it ships.  ``package-data`` is the one thing that
    carries a non-Python file inside a package directory into the wheel.
    """
    _requires_source_tree()
    assert KERNEL_NOTICE.is_file(), f"{KERNEL_NOTICE} is missing"
    with (ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    entries = config["tool"]["setuptools"]["package-data"]["woof.core.kernels"]
    assert "LICENSE-third-party.txt" in entries, entries


def test_the_binary_form_notice_has_two_byte_identical_copies() -> None:
    """One copy for the wheel, one for the bundle; a drifted pair is a bundle
    shipping a different notice from the wheel built at the same commit."""
    _requires_source_tree()
    wheel_copy = LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt"
    bundle_copy = (ROOT / "tools" / "rustwx" / "assets" / "basemap"
                   / "THIRD-PARTY-LICENSES.txt")
    assert wheel_copy.is_file(), f"{wheel_copy} is missing"
    assert bundle_copy.is_file(), f"{bundle_copy} is missing"
    assert wheel_copy.read_bytes() == bundle_copy.read_bytes(), (
        "the two copies of the binary-form notice have drifted; regenerate "
        "both together")

    # This distribution carries no terminal program, so the notice has no
    # terminal section.
    assert "tools/arwen-tui/" not in _read(wheel_copy)

    from woof import bridge_assets
    assert bundle_copy.parent.name in bridge_assets.REQUIRED_ASSET_SUBDIRS, (
        "the bundle copy no longer sits under a directory "
        "build_bridge_bundle.collect_assets() walks, so it stops travelling "
        f"with the binaries: {bridge_assets.REQUIRED_ASSET_SUBDIRS}")


def test_the_notices_count_the_artifacts_the_bundle_carries() -> None:
    """The binary-form notice and NOTICE say how many compiled artifacts they
    cover. Both said 27 while the bundle carried 31 (rw_simradar was the
    31st), telling a recipient the notice covered fewer binaries than they
    were handed. The count is woof.bridge_assets.BUNDLED_ARTIFACTS'."""
    _requires_source_tree()
    from woof import bridge_assets

    count = len(bridge_assets.BUNDLED_ARTIFACTS)
    for copy in (LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt",
                 ROOT / "tools" / "rustwx" / "assets" / "basemap" / "THIRD-PARTY-LICENSES.txt"):
        header = " ".join(_read(copy).split("\n")[:8])
        assert f"wheels: {count} Rust executables and libraries" in header, (
            f"{copy} does not count the {count} bundled artifacts")
    notice = " ".join(_read(ROOT / "NOTICE").split())
    assert (f"{count} Rust executables and libraries "
            "(woof.bridge_assets .BUNDLED_ARTIFACTS)") in notice, (
        f"NOTICE does not count the {count} bundled artifacts; run "
        "python -m tools.update_bridge_notice_inventory then "
        "python tools/update_zarr_license_notice.py")


def test_inventory_refresh_preserves_notices_when_the_bundle_grows() -> None:
    tool = _notice_tool("update_bridge_notice_inventory.py")
    for relative in tool.COPIES:
        old = _read(ROOT / relative)
        root_notice = relative == "NOTICE"
        updated = tool.render(old, 61, root_notice=root_notice)
        assert tool.render(updated, 60, root_notice=root_notice) == tool.render(
            old, 60, root_notice=root_notice)
        assert tool.render(updated, 61, root_notice=root_notice) == updated
    with pytest.raises(ValueError, match="exactly one"):
        tool.render("no inventory header", 61, root_notice=True)


def test_reservoir_attribution_reaches_binary_consumers() -> None:
    _requires_source_tree()
    binary = _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    for name in ("LICENSE-t-route.txt", "NOTICE-t-route-reservoirs.txt",
                 "NOTICE-t-route-NOAA-terms.md"):
        assert _read(LICENSES / name) in binary, (
            f"binary consumers do not receive the retained t-route text {name}")


def test_the_binary_form_notice_covers_the_first_party_ports() -> None:
    _requires_source_tree()
    text = _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    for token in ("UChicago Argonne", "Los Alamos National Security",
                  "NumPy Developers", "SIL OPEN FONT LICENSE",
                  "rw-libm", "Arm Limited", "Alexei Sibidanov",
                  "tools/grib1_bridge/src/glibc239_math.rs"):
        assert token in text, (
            f"the binary-form notice does not reproduce {token!r}; a compiled "
            "consumer never sees the source tree, so this is the only copy "
            "that reaches them")


# ---------------------------------------------------------------------------
# 3.  the per-file notices survive an edit
# ---------------------------------------------------------------------------
#: Files that open with a THIRD-PARTY NOTICE block because they carry
#: transcribed work.  Measured on the 2.7.0 tree; a file that carries
#: transcription joins the list, and a file that loses its header fails.
NOTICE_CARRIERS: tuple[str, ...] = (
    "woof/core/spp_kernel_sources.py",
    "woof/ensemble/stochastic.py",
    "woof/core/kernels/glibc_flt32.cuh",
    "woof/core/kernels/glibc_trig_flt32.cuh",
    "woof/core/kernels/thompson_aerosol_common.cuh",
    "woof/core/milbrandt2_constants.py",
    "woof/core/mynn_pbl.py",
    "woof/core/noahmp_libm.py",
    "woof/core/rrtm.py",
    "woof/core/rrtm_lw.py",
    "woof/core/rrtm_tables.py",
    "woof/core/rrtm_taumol.py",
    "woof/core/rrtmg_legacy.py",
    "woof/core/rrtmg_legacy_prep.py",
    "woof/core/rrtmg_lw.py",
    "woof/core/rrtmg_mcica.py",
    "woof/core/rrtmg_sw.py",
    # The UW PBL's binary64 libm: Arm exp/log/pow and CORE-MATH cos/acos,
    # both MIT, notice inline (root NOTICE, "FP64 libm transcriptions").
    "woof/core/kernels/glibc_flt64.cuh",
    "woof/core/rrtmgp.py",
    "woof/core/ruc.py",
    "woof/core/thompson_aerosol_contract.py",
    "woof/obs/dealias_region.py",
    # The single-layer UCM's CPU reference embeds WRF v4.7.1's
    # module_sf_urban.F statement listing and carries WRF's notice.
    "woof/verify/urban_ucm_ref.py",
)


@pytest.mark.parametrize("relative", NOTICE_CARRIERS)
def test_every_notice_carrier_still_opens_with_its_notice(relative) -> None:
    _requires_source_tree()
    path = ROOT / relative
    assert path.is_file(), f"{relative} is gone; the NOTICE still cites it"
    head = "".join(_read(path).splitlines(True)[:12])
    assert "THIRD-PARTY NOTICE" in head, (
        f"{relative} no longer opens with its third-party notice")


def test_no_file_carries_a_notice_this_list_does_not_know_about() -> None:
    """The other direction: a new carrier must join the list above."""
    _requires_source_tree()
    found = []
    for path in sorted(list((ROOT / "woof").rglob("*.py"))
                       + list(KERNELS.glob("*.cuh"))):
        head = "".join(_read(path).splitlines(True)[:12])
        if "THIRD-PARTY NOTICE" in head:
            found.append(path.relative_to(ROOT).as_posix())
    assert sorted(found) == sorted(NOTICE_CARRIERS), (
        "NOTICE_CARRIERS is out of step with the tree:\n"
        f"  only in tree: {sorted(set(found) - set(NOTICE_CARRIERS))}\n"
        f"  only in list: {sorted(set(NOTICE_CARRIERS) - set(found))}")


# ---------------------------------------------------------------------------
# 4.  the kernel notice's scope lists match the tree
# ---------------------------------------------------------------------------
#: Any spelling of Arm's logf / exp2f / powf or sinf / cosf tables.  A file that
#: reproduces one of these reproduces Arm's work and is inside the MIT grant.
_ARM_TABLE = re.compile(
    r"exp2f_tab|logf_tab|logf_invc|powf_log2_tab|powf_invc|gt_sincos_table", re.I)


def _arm_files() -> list[str]:
    return sorted(_kernel_name(p) for p in _kernel_sources()
                  if _ARM_TABLE.search(_read(p)))


def test_the_arm_scope_list_is_derived_from_the_tree() -> None:
    """The NOTICE says this list is machine-derived.  This is the machine.

    The breakage it prevents: 7f00a76db (2026-10-08) deleted this test and
    the FDLIBM one below, and the same day's mp28-exact and MYNN surface
    intakes shipped mynn_libm.cuh and thompson_aerosol_libm.cuh, which
    reproduce Arm's tables, outside the Arm scope list of both notices in
    2.8.8 and 2.8.9.  Restored for 2.9.0 with both added (eighteen files,
    glibc_trig_flt32.cuh's sinf/cosf tables among them).
    """
    _requires_source_tree()
    derived = _arm_files()
    assert derived, "no Arm table carriers found"
    from tools.update_kernel_license_scopes import generated_notices
    for path, expected in generated_notices(ROOT).items():
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path} is stale; run python -m tools.update_kernel_license_scopes")
    assert "glibc_trig_flt32.cuh" in derived
    listed = _read(KERNEL_NOTICE)
    missing = [name for name in derived if name not in listed]
    assert missing == [], (
        "woof/core/kernels/LICENSE-third-party.txt does not name every file "
        f"that reproduces Arm's tables: {missing}")
    notice = _read(NOTICE)
    missing = [name for name in derived if name not in notice]
    assert missing == [], f"the root NOTICE's Arm list is short of: {missing}"


#: The kernel translation units carrying an FDLIBM-descended reduction
#: (expm1f, tanhf, atanf, log10f, log1pf, and binary64 log10, log1p, atan,
#: atan2), by the name the notices use (_kernel_name).
#: Pinned rather than derived because each file names the routine
#: differently -- r_atan, ng_atanf, glibc_atanf, mynn_tanhf, nmpe_expm1f,
#: ruc_tanhf_glibc -- and the derivation below is what keeps the pin accurate.
from tools.update_kernel_license_scopes import FDLIBM_FILES
#: Device routines whose NAME looks like an FDLIBM one but which are not
#: FDLIBM transcriptions: these are libm calls rounded once or CORE-MATH,
#: as identified beside each entry.  Listed with the reason so that adding a
#: real transcription here is a deliberate act.
_NOT_FDLIBM = {
    # Terrain drag (lane/282-terrain-drag): CUDA's double atan2 rounded once
    # to float, standing for glibc 2.43's correctly rounded atan2f
    # (CORE-MATH); no FDLIBM reduction is transcribed.
    "terrain_drag.cu": ("td_atan2f",),
    # The unused optical stub calls the CUDA builtin without transcribing libm.
    "chem_optics.cu": ("chem_log10f",),
    "glibc_trig_flt32.cuh": ("glibc_atanf",),    # CORE-MATH s_atanf.c, a8066a5, MIT
    "sfire_libm.cuh": ("sfire_tanhf",),          # CORE-MATH s_tanhf.c, b1ecd83, MIT (NOTICE)
    "p3.cu": ("p3_log10",),                      # (float)log10((double)x)
    "thompson_aerosol_warm.cu": ("thompson_aa_log10f_cr",),
    "ruc.cu": ("ruc_log10f_rn", "ruc_expm1f_glibc"),
    # BEP+BEM: ubm_atanf calls glibc_atanf (CORE-MATH); ubm_log10f and its
    # special-case arm transcribe CORE-MATH log10f.c bc385c2, MIT (NOTICE).
    "urban_bem.cuh": ("ubm_atanf", "ubm_log10f", "ubm_log10f_special"),
    "urban_ucm.cu": ("u_atan", "u_log10"),       # (float)atan/log10((double)x)
    # Dudhia shortwave: WOOF's own float32 log10 on its own gfk_log (an
    # exponent split and a two-part log10(2)); no third-party source is
    # copied.  Word-equal to the C library's log10f on every float32 in
    # [1, +inf] (tools/dudhia_sw_wrf461_oracle/sweep_libm.py).
    "dudhia_sw.cu": ("dsw_log10f",),
    # Noah mosaic: noah_log10 and its special-case arm transcribe CORE-MATH
    # log10f.c bc385c2, MIT (NOTICE), not FDLIBM.
    "noah_mosaic.cu": ("noah_log10", "noah_log10_special"),
    # Noah (lane/verify-noah-lsm): noah_log10f is WOOF's own log10, composed
    # on WOOF's own float32 logf (gfk_log) as k*log10(2) + log10(e)*ln(m);
    # graded exhaustively against the oracle host's log10f.
    "noah.cu": ("noah_log10f",),
    # slope_rad / topo_shading: (float)atan((double)x), (float)atan2((double)y, (double)x)
    "topo_radiation.cu": ("tr_atan", "tr_atan2",
                          # the exact chain's slope atan2: the correctly
                          # rounded double-double side test, and the
                          # scalar quadrant arm on sfc_classic_atanf
                          # (atan_scalar_flt32.cuh, on FDLIBM_FILES)
                          "tr_atan2_resolve", "tr_slope_atan2_scalar"),
    # mp28 column support: tc_atan calls CUDA's atan/atanf; tc_log10
    # delegates to thompson_aa_log10f / thompson_aa_log10, whose
    # transcription lives in thompson_aerosol_libm.cuh (on FDLIBM_FILES).
    "thompson_column_support.cuh": ("tc_atan", "tc_log10"),
    # Goddard bands: g_log10 is gfk_log(x) * log10(e), WOOF's own
    # composition; no FDLIBM reduction is transcribed.
    "goddard_bands.cuh": ("g_log10",),
    # WOOF's own float32 log10 (gfk_log10) and Morrison's binary64 log10
    # (mo_log10d, WRF's DOUBLE ALOG10 in refl10cm_hm): WOOF's own libm
    # routines, graded word for word against the oracle host's C library
    # (tools/glibc_flt32_extra_proof, tools/morrison_wrf461_oracle).
    "glibc_flt32_extra.cuh": ("gfk_log10",),
    "morrison.cu": ("mo_log10d",),
    # The fork Thompson's DOUBLE LOG10 is CUDA's binary64 log10.
    "thompson_fork.cu": ("tf_log10",),
    # HAILCAST support (cap-hail, 2.9.4): lake_log10 is the HAILCAST
    # transcription's LOG10 kind shim, CUDA's log10f/log10 called directly.
    "hailcast_support.cuh": ("lake_log10",),
}

_FDLIBM_DEF = re.compile(
    r"__device__[^;{]{0,160}?\b(\w*(?:atan|tanh|log10|expm1|log1p)\w*)\s*\(")


def test_the_fdlibm_scope_list_matches_the_tree() -> None:
    """The breakage it prevents: 7f00a76db (2026-10-08) deleted this test,
    and thompson_aerosol_libm.cuh, mynn_libm.cuh, sfclay.cu,
    sfclay_classic.cuh and ysu_480_topo.cuh shipped FDLIBM reductions in
    2.8.8/2.8.9 (ysu_480_topo.cuh new on the 2.9.0 line) outside the FDLIBM
    scope list of both notices, which cover what they name and nothing
    else.  The root NOTICE is read as well as the kernel notice, because
    both lists were short.  The walk is recursive and knows log1p, because
    tempo/tempo_common.cuh (binary64 log10, below the top level) and
    portable_libm64.cuh (log1p) were outside the kernel notice too, and
    dudhia_sw.cu's FDLIBM log10f had been excused as first-party."""
    _requires_source_tree()
    assert all((KERNELS / name).is_file() for name in FDLIBM_FILES)
    listed = _read(KERNEL_NOTICE)
    missing = [name for name in FDLIBM_FILES if name not in listed]
    assert missing == [], (
        f"the kernel notice's FDLIBM list is short of: {missing}")
    notice = _read(NOTICE)
    missing = [name for name in FDLIBM_FILES if name not in notice]
    assert missing == [], (
        f"the root NOTICE's FDLIBM list is short of: {missing}")

    unnotified = {}
    for path in _kernel_sources():
        name = _kernel_name(path)
        names = sorted({m.group(1) for m
                        in _FDLIBM_DEF.finditer(_decommented(_read(path)))})
        real = [n for n in names if n not in _NOT_FDLIBM.get(name, ())]
        if real and name not in FDLIBM_FILES:
            unnotified[name] = real
    assert unnotified == {}, (
        "these kernels define an FDLIBM-shaped routine and are not on the "
        "FDLIBM notice list; either they transcribe FDLIBM and the notice "
        "must say so, or they are not FDLIBM and belong in _NOT_FDLIBM with the "
        f"reason beside them: {unnotified}")


# ---------------------------------------------------------------------------
# 5.  first-party crates that carry third-party expression
# ---------------------------------------------------------------------------
#: Source roots ArWen WROTE.  Not vendor trees and not pip dependencies --
#: the third category, which is where both 2.7.0 breaches were.
_FIRST_PARTY_ROOTS = (
    "tools/rustwx/crates", "tools/rw_wps/crates", "tools/grib1_bridge/src",
    "tools/region_global_dealias/src",
)

_LICENCE_MARKER = re.compile(
    r"Copyright\s*\((?:c|C)\)|SPDX-License-Identifier|BSD-3-Clause"
    r"|BSD 3-Clause|MIT License")

#: ``file that carries a marker -> a token the root NOTICE must contain``.
#: A first-party file announcing someone else's copyright and no NOTICE entry
#: is the exact shape of the NumPy breach.
MARKER_FILES: dict[str, str] = {
    # Its pinned FP32 power is a native copy of the existing Arm MIT code.
    # A source header alone would leave the compiled bridge unnotified.
    "tools/grib1_bridge/src/glibc239_math.rs":
        "tools/grib1_bridge/src/glibc239_math.rs",
    "tools/region_global_dealias/src/solver.rs": "UChicago Argonne",
    "tools/rustwx/crates/static-fields/src/projection/npmath.rs":
        "NumPy Developers",
    # Arm optimized-routines and CORE-MATH, the WPS orographic projection's
    # libm (public CI run 37036982597 failed it on the platform libm).
    "tools/rustwx/crates/rw-libm/src/lib.rs":
        "Portable libm in Rust -- Arm optimized-routines and CORE-MATH",
}


def _first_party_rust() -> list[pathlib.Path]:
    out = []
    for relative in _FIRST_PARTY_ROOTS:
        root = ROOT / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.rs")):
            text = path.as_posix()
            if "/vendor/" in text or "/target/" in text:
                continue
            out.append(path)
    return out


def test_first_party_code_announcing_a_foreign_copyright_is_notified() -> None:
    _requires_source_tree()
    sources = _first_party_rust()
    if not sources:
        pytest.skip("the Rust trees are not in this checkout")
    notice = _read(NOTICE)
    found = {}
    for path in sources:
        if _LICENCE_MARKER.search(_read(path)):
            found[path.relative_to(ROOT).as_posix()] = True
    unknown = sorted(set(found) - set(MARKER_FILES))
    assert unknown == [], (
        "these first-party files announce a third-party copyright or licence "
        "and no NOTICE entry is mapped to them.  A first-party crate that "
        "ports third-party code is in neither the vendor census nor the pip "
        f"census, which is how NumPy went unnotified until 2.7.0: {unknown}")
    for relative, token in MARKER_FILES.items():
        assert (ROOT / relative).is_file(), f"{relative} is gone"
        assert token in notice, (
            f"{relative} carries third-party expression and the NOTICE no "
            f"longer names {token!r}")


#: Upstream Fortran projects a first-party crate transcribes, and the token
#: the NOTICE must carry for each.  The classifier below is what says which
#: project a cited file belongs to.
UPSTREAM_TOKENS = {
    "MPAS": "Los Alamos National Security",
    "WPS": "WRF Preprocessing System",
    "WRF": "public domain",
    "UPP": "NOAA EMC Unified Post Processor (UPP)",
    "TEMPO": "NCAR/TEMPO",
}

#: TEMPO's sources as rw-tempo cites them: the src/module_mp_tempo_*.F90
#: modules, by full name or by their short names (params.F90, utils.F90,
#: driver.F90), all in NCAR/TEMPO, not in WRF.
_TEMPO_SHORT = frozenset({"params.F90", "utils.F90", "driver.F90"})

#: WOOF's own Fortran: the TEMPO reference driver in tools/tempo_ref, which
#: links unmodified TEMPO and dumps its init state.  First-party, no notice.
_OWN_FORTRAN = frozenset({"tempo_ref_driver.F90"})

#: Cited Fortran sources with NO counterpart in the WRF distribution, so they
#: are WPS's own.  Measured against a WRF v4.6.1 checkout.
_WPS_ONLY = frozenset({
    "rrpr.F", "gribcode.F", "output.F", "rd_grib1.F", "rd_grib2.F",
    "new_storage.F", "process_tile_module.F", "interp_module.F",
    "read_met_module.F90",
    # geogrid's terrain smoother (WPS geogrid/src/smooth_module.F), which
    # the static-fields crate transcribes for GEOGRID.TBL smooth_option.
    "smooth_module.F",
})

#: MPAS init_atmosphere sources cited by their short names in rw-mpas's RRFS
#: init port (rrfs_init/, rrfs_static_gwd.rs, tests/gsl_oro.rs): MPAS v8.4.2
#: and the RRFS fork RRFSx/MPAS-Model (MPAS's BSD licence) mpas_init_atm_cases.F,
#: mpas_init_atm_gwd.F, mpas_init_atm_read_met.F, the fork's initialize_real.F and
#: mpas_gsl_oro_data_sm_scale.F.  MPAS's notice covers them.
_MPAS_SHORT = frozenset({"cases.F", "gwd.F", "initialize_real.F", "read_met.F", "sm_scale.F"})

#: WRF sources outside the module_/mp_ naming, cited by a first-party crate,
#: each NAMED in the NOTICE as well as covered by WRF's declaration.
#: share/output_wrf.F: the renderer's local import reads SIMULATION_START_DATE
#: as WRF's history writer defines it (A137, rw-wrfbatch local_import.rs).
_WRF_OTHER = frozenset({"output_wrf.F"})

#: NCAR/TEMPO's own source files, cited bare (src/params.F90 and so on) by
#: the rw-tempo crate that ports its table builders.
_TEMPO = frozenset({"driver.F90", "params.F90", "utils.F90"})

#: This repository's own Fortran: the TEMPO column reference driver under
#: tools/tempo_ref, which the crate reads the records of.  No upstream.
_FIRST_PARTY_FORTRAN = frozenset({"tempo_ref_driver.F90"})

_FORTRAN_CITATION = re.compile(r"\b([A-Za-z0-9_]+\.(?:F90|F))\b")


def test_every_upstream_a_first_party_crate_transcribes_is_notified() -> None:
    """MPAS was found this way: 44,768 lines of port, no notice, five shipped
    binaries.  Nothing in the tree announced it except the Fortran file names
    in its own doc comments, which is what this reads."""
    _requires_source_tree()
    sources = _first_party_rust()
    if not sources:
        pytest.skip("the Rust trees are not in this checkout")
    cited = set()
    for path in sources:
        cited.update(m.group(1) for m in _FORTRAN_CITATION.finditer(_read(path)))

    projects = set()
    unclassified = []
    for name in sorted(cited):
        if name in _OWN_FORTRAN:
            continue
        if name.startswith("mpas_") or name in _MPAS_SHORT:
            projects.add("MPAS")
        elif name.startswith("module_mp_tempo_") or name in _TEMPO_SHORT:
            projects.add("TEMPO")
        elif name in {"INITPOST.F", "MDL2P.F", "params.F"}:
            projects.add("UPP")
        elif name in _WPS_ONLY:
            projects.add("WPS")
        elif name in _TEMPO or name.startswith("module_mp_tempo"):
            projects.add("TEMPO")
        elif name in _FIRST_PARTY_FORTRAN:
            continue
        elif (name.startswith("module_") or name.startswith("mp_")
              or name in _WRF_OTHER):
            projects.add("WRF")
        else:
            unclassified.append(name)
    assert unclassified == [], (
        "a first-party crate cites Fortran sources this classifier does not "
        "recognise.  Decide which upstream they belong to and give that "
        f"upstream a NOTICE entry before adding them here: {unclassified}")

    notice = _read(NOTICE)
    unnamed = sorted(name for name in cited & _WRF_OTHER
                     if name not in notice)
    assert unnamed == [], (
        "a first-party crate cites these WRF sources outside the module_ "
        f"naming and the NOTICE does not name them: {unnamed}")
    missing = sorted(project for project in projects
                     if UPSTREAM_TOKENS[project] not in notice)
    assert missing == [], (
        f"first-party crates transcribe {missing} and the NOTICE does not "
        "name them")


# ---------------------------------------------------------------------------
# 6.  what 2.7.0 removed stays removed
# ---------------------------------------------------------------------------
_EARLIER_GAMMA_DEFINITION = re.compile(
    r"^\s*(?:__device__|static|float|double)\s+.*"
    r"\b(gfk_gammaf_positive|gfk_gamma_product|gfk_lgamma_pos)\s*\(", re.M)


def test_the_earlier_gamma_stays_deleted() -> None:
    """The earlier gamma returned glibc's tgammaf words, rounding errors
    included, and the root NOTICE records its removal and that of its lgammaf
    helper.  Its return would bring the errors back and make that record
    wrong."""
    _requires_source_tree()
    back = [p.name for p in _kernel_sources()
            if _EARLIER_GAMMA_DEFINITION.search(_read(p))]
    assert back == [], f"the earlier gamma is back in: {back}"

    fingerprints = ("exp2_adj", "x_adj_mant", "gamma_coeff", "0xBB360B61",
                    "0x3A500D01")
    for path in _kernel_sources():
        if path.name == "gf_wrf_normalization.cuh":
            # Accepted strict-only GF normalization has an independent source pin.
            import hashlib
            assert hashlib.sha256(path.read_bytes()).hexdigest() == "2f258d559bda0b6c97a7220f2eb4c27473817ad25daf69769e858d63a1731942"
            continue
        text = _decommented(_read(path))
        hit = [token for token in fingerprints if token in text]
        assert hit == [], f"{path.name} carries the earlier gamma's fingerprints: {hit}"


def test_the_two_glibc_only_libm_fragments_stay_removed() -> None:
    """Both were glibc's own edits to FDLIBM code, both did no work, and both
    were compared against their originals on all 4,294,967,296 float32 bit
    patterns with zero differing outputs before removal."""
    _requires_source_tree()
    tanh_guard = ("woof/core/kernels/noahmp_energy.cu", "if (ix == 0) return x;")
    log10_div = ("woof/core/kernels/noahmp_leaves.cu",
                 "DV(-two25, fabsf(x))")
    py_log10 = ("woof/core/noahmp_libm.py", "-_TWO25 / abs(value)")
    for relative, fragment in (tanh_guard, log10_div, py_log10):
        # Comments are stripped first: the sites that lost these fragments
        # each carry a comment QUOTING what was removed and why, which is the
        # point of the edit, and a naive substring search would read the
        # explanation as the offence.
        raw = _read(ROOT / relative)
        text = (re.sub(r"#[^\n]*", "", raw) if relative.endswith(".py")
                else _decommented(raw))
        assert fragment not in text, (
            f"{relative} has taken glibc's {fragment!r} back; it is "
            "expression this Apache-2.0 distribution removed at 2.7.0")

    python_tanh = re.sub(r"#[^\n]*", "",
                         _read(ROOT / "woof/core/noahmp_libm.py"))
    body = python_tanh.split("def tanhf(", 1)[1].split("\ndef ", 1)[0]
    assert "if ix == 0:" not in body, (
        "woof/core/noahmp_libm.py's tanhf has taken glibc's redundant zero "
        "guard back")


# ---------------------------------------------------------------------------
# 8. The Grell-Freitas gamma is this project's own work
# ---------------------------------------------------------------------------
#
# The 2.7.0 development record described the earlier gamma in
# glibc_flt32.cuh as a line-for-line transcription of glibc's tgammaf under
# the LGPL.  That finding was wrong: both the earlier gamma and the current
# one were written here, under this project's licence.  The wrong record was
# copied into NOTICE, the kernel headers, the gamma note and the tests, and
# every later reader of the tree re-derived it from those copies; the claim
# was removed and returned three times before 2.7.6.  This pin makes the
# fourth return a red test instead of a rediscovery.  Earlier releases'
# changelog entries are history and are not read here.

_GAMMA_SURFACES = (
    "NOTICE", "README.md", "PROVENANCE.md", "docs/gf_gamma_known_delta.md",
    "woof/core/kernels/LICENSE-third-party.txt",
    "woof/core/kernels/glibc_flt32.cuh", "woof/core/kernels/gf.cu",
)
_GAMMA_SURFACE_GLOBS = ("licenses/*", "recast-woof-data/licenses/*",
                        "docs/public/**/*.md", "docs/manual/*.md")

# Phrases that only ever appeared in the withdrawn description.
_WITHDRAWN_PHRASES = (
    "transcription of glibc's own", "transcription of glibc's tgammaf",
    "glibc's own tgammaf", "lgpl gamma", "gamma transcription",
    "e_gammaf_r.c", "gamma_productf.c", "glibc-authored", "glibc-derived gamma",
)

# A sentence that names the Grell-Freitas gamma and, in the same breath,
# calls it transcribed, LGPL, FSF copyright or glibc's work.
_GAMMA_NAMES = re.compile(
    r"(gfk_tgamma|gfk_gammaf_positive|gfk_gamma_product|gammaf_positive|"
    r"gamma_product|grell-freitas gamma|gf gamma|own gamma|earlier gamma|"
    r"current gamma|the gamma in glibc_flt32|tgamma wrapper)")
_PROVENANCE_CLAIMS = re.compile(
    r"(transcri\w*|\blgpl\b|\bfsf\b|glibc's own work|glibc-authored|"
    r"derived from glibc|glibc-derived)")

# A sentence that names the gamma next to one of those words in order to deny
# the claim, or to describe the helper routines it once called, is not a hit.
_DENIALS = ("not a transcription", "own work", "own gamma", "arwen's earlier gamma",
            "arwen's current gamma", "whose only caller", "withdrawn",
            "that description was wrong")


def _gamma_surface_files() -> list[pathlib.Path]:
    files = [ROOT / rel for rel in _GAMMA_SURFACES]
    for pattern in _GAMMA_SURFACE_GLOBS:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(set(files))


def _prose_sentences(text: str):
    """Sentences of a text or a comment block, with comment markers dropped."""
    lines = [re.sub(r"^\s*(//|#|\*)\s?", "", line) for line in text.splitlines()]
    flat = re.sub(r"\s+", " ", " ".join(lines))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def test_the_grell_freitas_gamma_is_never_described_as_glibc_work() -> None:
    """Assertion 8: the withdrawn description does not come back anywhere
    a user or a licence reader can see it."""
    _requires_source_tree()
    hits = []
    for path in _gamma_surface_files():
        text = _read(path)
        lowered = text.lower()
        for phrase in _WITHDRAWN_PHRASES:
            if phrase in lowered:
                hits.append(f"{path.relative_to(ROOT).as_posix()}: phrase {phrase!r}")
        for sentence in _prose_sentences(text):
            low = sentence.lower()
            if _GAMMA_NAMES.search(low) and _PROVENANCE_CLAIMS.search(low):
                if any(marker in low for marker in _DENIALS):
                    continue  # a sentence saying the opposite, or the retraction
                hits.append(f"{path.relative_to(ROOT).as_posix()}: {sentence[:160]!r}")
    assert hits == [], (
        "the Grell-Freitas gamma is this project's own work (ruling of "
        "2026-09-12); shipped text describes it otherwise:\n" + "\n".join(hits))


def test_core_math_trig_grant_is_reproduced_and_scoped() -> None:
    """Pin the new grant's exact text, copyright lines and routine scope."""
    _requires_source_tree()
    text = _read(LICENSES / "LICENSE-CORE-MATH-MIT.txt")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "9700243118309ce16f64fab4680507bf713caed4f77e8aa6bc08a324b4f663c1"
    header = _read(KERNELS / "glibc_trig_flt32.cuh")
    notice = _read(NOTICE)
    adjacent = _read(KERNEL_NOTICE)
    for line in text.splitlines():
        if line:
            assert line in notice
            assert line in adjacent
    for token in ("Copyright (c) 2022-2024 Alexei Sibidanov.",
                  "Copyright (c) 2023-2024 Alexei Sibidanov.",
                  "LICENSE-CORE-MATH-MIT.txt", "bc385c2", "56dd347",
                  "a8066a5", "59d21d7"):
        assert token in header
        assert token in notice
        assert token in adjacent
    for function in ("glibc_tanf", "glibc_asinf", "glibc_acosf", "glibc_atanf"):
        assert function in header
        assert function in notice
        assert function in adjacent


def test_core_math_log10f_grant_is_reproduced_and_scoped() -> None:
    """Pin the Noah mosaic column's CORE-MATH log10f grant and scope."""
    _requires_source_tree()
    kernel = _read(KERNELS / "noah_mosaic.cu")
    notice = _read(NOTICE)
    adjacent = _read(KERNEL_NOTICE)
    for token in ("Copyright (c) 2022-2023 Alexei Sibidanov.", "bc385c2",
                  "Permission is hereby granted, free of charge"):
        assert token in kernel
    for token in ("2022-2023 Alexei Sibidanov", "bc385c2",
                  "LICENSE-CORE-MATH-MIT.txt"):
        assert token in notice
        assert token in adjacent
    for function in ("noah_log10", "noah_log10_special"):
        assert function in kernel
        assert function in notice
        assert function in adjacent


@pytest.mark.parametrize("vendor,label", (
    ("bowecho", "BowEcho"),
    ("recast-radar-tools", "recast-radar-tools"),
))
def test_simulated_radar_grants_reach_source_and_binary_consumers(vendor, label) -> None:
    """A compiled radar bridge must retain the grants from both source trees."""
    _requires_source_tree()
    source = ROOT / "tools" / "rustwx" / "vendor" / vendor
    notice = _read(NOTICE)
    binary = _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    assert (source / "SOURCE.json").is_file(), "radar source revision record is missing"
    assert vendor in notice and label in binary
    for upstream, suffix in (("LICENSE-MIT", "MIT"), ("LICENSE-APACHE", "Apache-2.0")):
        shipped = LICENSES / f"LICENSE-{label}-{suffix}.txt"
        assert shipped.read_bytes() == (source / upstream).read_bytes(), (
            f"{shipped.name} differs from the vendored grant")
        assert f"licenses/{shipped.name}" in notice
        for line in _read(shipped).splitlines():
            if line:
                assert line in binary, f"binary notice omits a line from {shipped.name}"
    if vendor == "bowecho":
        assert (source / "PYART-LICENSE.txt").read_bytes() == (
            LICENSES / "LICENSE-PyART-Argonne-BSD-3-Clause.txt").read_bytes()

    scoped = LICENSES / f"NOTICE-{label}.txt"
    assert scoped.read_bytes() == (source / "NOTICE").read_bytes()
    assert f"licenses/{scoped.name}" in notice
    assert _read(scoped) in binary, "compiled radar loses the scoped vendor notice"


def test_simulated_radar_dependency_grants_include_native_library_notices() -> None:
    """New radar crates and their native libraries must reach binary consumers."""
    _requires_source_tree()
    path = ROOT / "tools/rustwx/crates/rw-simradar/data/dependency-licenses.json"
    inventory = json.loads(_read(path))
    notice = _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    assert inventory["schema"] == "woof.simulated-radar-dependency-grants/v1"
    assert inventory["packages"], "radar dependency notice inventory is empty"
    nested_native_grants = []
    for package in inventory["packages"]:
        assert f"{package['name']} {package['version']} | {package['license']}" in notice
        if package["name"] == "wrf-core":
            assert package["declaration_source"] in notice
            assert package["license"] == "MIT"
            continue
        assert package["grants"], f"no grant retained for {package['name']}"
        for grant in package["grants"]:
            source = ROOT / grant["path"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == grant["sha256"]
            assert grant["path"] in notice
            assert f"[{grant['label']}] sha256 {grant['sha256']}" in notice
            for line in _read(source).splitlines():
                if line:
                    assert line.rstrip() in notice, f"binary notice omits grant text from {grant['path']}"
            if "/libdeflate/" in grant["path"]:
                nested_native_grants.append(grant["path"])
    assert nested_native_grants, "native libdeflate grant is missing from radar notices"


def test_simulated_radar_bzip2_table_grant_reaches_binary_consumers() -> None:
    """The decoder's copied randomisation table needs its original bzip2 grant."""
    _requires_source_tree()
    vendor = ROOT / "tools/rustwx/vendor/recast-radar-tools"
    upstream = _read(vendor / "THIRD_PARTY_NOTICES.md")
    section = upstream.split("## bzip2 / libbzip2\n", 1)[1].split("\n## ", 1)[0]
    grant = section.split("```\n", 1)[1].split("```", 1)[0]
    assert _read(LICENSES / "LICENSE-recast-radar-tools-bzip2.txt") == grant
    assert grant in _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    assert "randtable.c" in _read(vendor / "crates/recast-radar-bzip2/src/rand.rs")


@pytest.mark.parametrize("vendor", ("bowecho", "recast-radar-tools"))
def test_simulated_radar_vendor_inventory_is_complete_and_exportable(vendor) -> None:
    """Missing source, drifted grants or pruned vendors break public offline builds."""
    _requires_source_tree()
    from tools.release_exclusions import matches, read_exclusions

    source = ROOT / "tools/rustwx/vendor" / vendor
    record = json.loads(_read(source / "SOURCE.json"))
    assert record["schema_version"] == 2
    assert record.get("licence", record.get("license")) == "MIT OR Apache-2.0"
    revision = record.get("extraction_commit", record.get("revision"))
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    scope = _read(source / "NOTICE")
    assert revision in scope
    if vendor == "bowecho":
        assert re.fullmatch(r"[0-9a-f]{40}", record["source_commit"])
        assert record["source_commit"] in scope
        assert record["local_commits"][-1] == revision
    # The inventory is the vendored SOURCE.  ci.yml's syntax step,
    # `python -m compileall -q -j 0 woof tilestream tools tests
    # conftest.py`, writes __pycache__ beside the Python scripts bowecho
    # vendors (nexrad_io's fixture generators, the pytmatrix LUT tools);
    # .gitignore keeps that bytecode out of every commit and so out of
    # every export, and SOURCE.json rightly does not list it.  Counting it
    # turned this gate red in any checkout that had run the syntax step,
    # the 2.8.9 E3 a1388eae7 public CI reproduction among them.
    expected = {p.relative_to(source).as_posix() for p in source.rglob("*")
                if p.is_file() and p.name != "SOURCE.json"
                and "__pycache__" not in p.relative_to(source).parts}
    entries = record["files"]
    assert entries and len(entries) == len({entry["path"] for entry in entries})
    assert {entry["path"] for entry in entries} == expected
    rules = read_exclusions(ROOT)
    for entry in entries:
        relative = pathlib.PurePosixPath(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = source / relative
        raw = path.read_bytes()
        assert len(raw) == entry["bytes"], entry["path"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["path"]
        if entry.get("origin") == "vendoring":
            assert entry["change"]
        else:
            assert entry["origin_revision"] == revision
            assert entry["origin_path"] == entry["path"]
        shipped = path.relative_to(ROOT).as_posix()
        assert matches(shipped, rules) is None, f"public snapshot drops {shipped}"
    assert set(record["notices"]) <= expected
    assert {"NOTICE", "LICENSE-MIT", "LICENSE-APACHE"} <= set(record["notices"])
    for relative in (source / "SOURCE.json", ROOT / "tools/rustwx/Cargo.lock",
                     ROOT / "tools/rustwx/.cargo/config.toml",
                     ROOT / "tools/rustwx/crates/rw-simradar/Cargo.toml"):
        assert matches(relative.relative_to(ROOT).as_posix(), rules) is None


def test_simulated_radar_writer_crates_resolve_from_the_public_source_tree() -> None:
    """Unpublished writer crates must build from paths without Git or registry fetches."""
    _requires_source_tree()
    dependencies = tomllib.loads(_read(ROOT / "pyproject.toml"))["project"]["dependencies"]
    # The writers are the vendored Rust crates; the PyPI distribution had no
    # consumer in gpuwm/, tools/ or tests/, so declaring it only made every
    # install download recast_radar, xarray and pandas for nothing.
    assert not any(item.replace(" ", "").lower().startswith("recast-radar")
                   for item in dependencies), "no recast-radar Python requirement is declared"
    assert not any("recast-radar" in item and ("git+" in item or " @ " in item)
                   for item in dependencies), "public installs require the declared PyPI version"
    rust = ROOT / "tools/rustwx"
    source = rust / "vendor/recast-radar-tools"
    record = json.loads(_read(source / "SOURCE.json"))
    crates = set(record["crates"])
    assert len(crates) == 6
    locked = tomllib.loads(_read(rust / "Cargo.lock"))["package"]
    writer_locks = [package for package in locked if package["name"] in crates]
    assert {package["name"] for package in writer_locks} == crates
    for package in writer_locks:
        assert package["version"] == record["version"]
        assert "source" not in package, f"writer crate needs a fetch: {package['name']}"
    manifests = [rust / "crates/rw-simradar/Cargo.toml", source / "Cargo.toml"]
    manifests.extend(source / "crates" / crate / "Cargo.toml" for crate in sorted(crates))

    def dependencies(table):
        for key, value in table.items():
            if key in ("dependencies", "dev-dependencies", "build-dependencies"):
                yield from value.items()
            elif isinstance(value, dict):
                yield from dependencies(value)

    seen = set()
    for manifest in manifests:
        for name, spec in dependencies(tomllib.loads(_read(manifest))):
            if name not in crates:
                continue
            seen.add(name)
            assert isinstance(spec, dict) and "path" in spec, name
            assert "git" not in spec, f"writer crate uses a Git URL: {name}"
            destination = (manifest.parent / spec["path"]).resolve()
            assert destination == (source / "crates" / name).resolve()
            assert (destination / "Cargo.toml").is_file()
    assert seen == crates
    config = tomllib.loads(_read(rust / ".cargo/config.toml"))["source"]
    for registry in {"crates-io"} | {package["source"].split("#", 1)[0]
                                    for package in locked
                                    if package.get("source", "").startswith("git+")}:
        replacement = config[registry]["replace-with"]
        directory = rust / config[replacement]["directory"]
        assert directory.is_dir(), f"offline source replacement is absent: {directory}"


# ---------------------------------------------------------------------------
# 9.  the generated notices survive their own generators
# ---------------------------------------------------------------------------
def _notice_tool(name: str):
    """A generator under tools/, loaded by path: it is a script, not a module."""
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location("licence_notice_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_terminal_notice_generator_replaces_only_its_own_section() -> None:
    """Every version bump runs tools/update_tui_license_notice.py.  It rewrote
    the binary notice from section 5's heading to the end of the file, and the
    simulated-radar vendor notices follow that section: opening 2.8.5 would
    have deleted the BowEcho and recast-radar-tools notices and the bzip2 table
    grant from the notice a compiled consumer receives."""
    _requires_source_tree()
    if not (ROOT / "tools" / "update_tui_license_notice.py").is_file():
        # Recast WOOF builds no terminal workspace: its generator stays with the regional engine's own
        # distribution, and the binary notice WOOF ships has no section 5 for it to rewrite.
        notice = (LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt").read_bytes().decode("utf-8")
        assert "LOCKED DEPENDENCIES AND LICENCE TEXTS" not in notice
        assert "update_tui_license_notice" not in notice
        return
    tool = _notice_tool("update_tui_license_notice.py")
    path = LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt"
    notice = path.read_bytes().decode("utf-8")
    assert notice.count(tool.MARKER) == 1 and notice.count(tool.END) == 1, (
        "section 5 of the binary notice needs one heading and one END line; "
        "without the END line its generator cannot tell where to stop")
    head, _, rest = notice.partition(tool.MARKER)
    section, _, after = rest.partition(tool.END)
    # The section the file carries, written back, is the file.
    assert tool.compose(notice, tool.MARKER + section.rstrip() + "\n") == notice
    # A new section leaves both sides of it byte for byte.
    replaced = tool.compose(notice, tool.MARKER + "a new lock\n")
    assert replaced == head + tool.MARKER + "a new lock\n\n" + tool.END + after
    for name in ("NOTICE-BowEcho.txt", "NOTICE-recast-radar-tools.txt",
                 "LICENSE-recast-radar-tools-bzip2.txt", "LICENSE-t-route.txt",
                 "NOTICE-t-route-reservoirs.txt", "NOTICE-t-route-NOAA-terms.md"):
        assert _read(LICENSES / name) in replaced, (
            f"regenerating section 5 drops {name} from the binary notice")
    # With no END line it refuses; it does not truncate.
    with pytest.raises(SystemExit, match="END line"):
        tool.compose(head + tool.MARKER + section + after, tool.MARKER + "a new lock\n")
    # A notice that has no section 5 yet gains one and loses nothing.
    assert tool.compose(head, tool.MARKER + "a new lock\n") == (
        head + tool.MARKER + "a new lock\n\n" + tool.END)


def test_the_zarr_notice_reproduces_the_binary_notice_as_it_is_now() -> None:
    """The Zarr reader's notice reproduces the binary notice whole, as the
    retained record of the mapped engine it links, and names the lock it was
    resolved from.  Nothing regenerated it when either moved: through 2.8.4
    it shipped the binary notice of 2026-09-08, which by then counted 27
    binaries in a bundle of 31 and lacked every notice added since, beside a
    lock digest two lock edits old.  tools/release/bump_version.py now runs
    its generator after the binary notice's; an edit to either file outside a
    bump has to run it too."""
    _requires_source_tree()
    wheel_copy = LICENSES / "THIRD-PARTY-LICENSES-zarr-binary.txt"
    bundle_copy = (ROOT / "tools" / "rustwx" / "assets" / "basemap"
                   / "THIRD-PARTY-LICENSES-zarr-binary.txt")
    assert wheel_copy.read_bytes() == bundle_copy.read_bytes(), (
        "the two copies of the Zarr notice have drifted; regenerate both "
        "with python tools/update_zarr_license_notice.py")
    notice = wheel_copy.read_bytes().decode("utf-8")
    remedy = "run python tools/update_zarr_license_notice.py"

    lock = hashlib.sha256((ROOT / "tools/zarr_bridge/Cargo.lock").read_bytes()).hexdigest()
    assert f"\nCargo.lock sha256 {lock}\n" in notice, (
        "the Zarr notice was resolved from a different tools/zarr_bridge/Cargo.lock; "
        + remedy)

    binary = (LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt").read_bytes()
    carried = re.findall(
        r"^\[ZARR-\d+\] sha256 ([0-9a-f]{64})\nAs carried by:\n"
        r"  licenses/THIRD-PARTY-LICENSES-bridge-binaries\.txt\n", notice, re.M)
    assert len(carried) == 1, "the Zarr notice no longer reproduces the binary notice"
    assert carried[0] == hashlib.sha256(binary).hexdigest(), (
        "the Zarr notice reproduces an earlier binary notice; " + remedy)
    assert binary.decode("utf-8").rstrip() in notice, (
        "the Zarr notice names the binary notice's digest without its text; " + remedy)


def test_the_sounding_notice_generator_is_idempotent_on_the_tree() -> None:
    """tools/update_soundings_license_notice.py run on an untouched tree must
    change nothing.  It rewrote ~3,466 lines of both binary notices while this
    file stayed green: the committed section lacked cudarc (rustwx-render gained
    it in 529053024f without a refresh), a merge had folded the head of the
    t-route block into the sounding section, and the tool's own output varied
    with Python's per-process hash seed (a path crate and its git twin share a
    name and version) and, on Windows, with case-insensitive path order.  Two
    runs under different hash seeds, each in --check mode, must both find the
    committed copies current."""
    _requires_source_tree()
    import os
    import shutil
    import subprocess
    import sys
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not on PATH; the generator resolves the lock with cargo metadata")
    tool = _notice_tool("update_soundings_license_notice.py")
    notice = (LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt").read_bytes().decode("utf-8")
    assert notice.count(tool.MARKER) == 1 and notice.count(tool.END) == 1
    _, _, rest = notice.partition(tool.MARKER)
    section, _, after = rest.partition(tool.END)
    assert "NEXTGEN RESERVOIR ADAPTATION" not in section, (
        "the t-route block sits inside the generated sounding section; the "
        "generator would delete it")
    assert "NEXTGEN RESERVOIR ADAPTATION" in after
    for seed in ("0", "1"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "update_soundings_license_notice.py"), "--check"],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONHASHSEED": seed})
        assert result.returncode == 0, (
            f"PYTHONHASHSEED={seed}: the sounding notices are not what the generator writes; "
            f"run python tools/update_soundings_license_notice.py\n{result.stderr[-2000:]}")


def test_the_hydro_output_notice_generator_is_current_on_the_tree() -> None:
    """librw_nextgen_output joined the bridge roster at 2.9.4 (on demand), so its
    locked closure (parquet, thrift, snap, crc32fast, ...) reaches users in binary
    form. tools/update_nextgen_output_license_notice.py writes that closure's
    inventory and texts into both binary notices; run in --check mode it must find
    both committed copies current, or the bundle ships the library without the
    licence texts its dependencies require."""
    _requires_source_tree()
    import shutil
    import subprocess
    import sys
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not on PATH; the generator resolves the lock with cargo metadata")
    tool = _notice_tool("update_nextgen_output_license_notice.py")
    notice = (LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt").read_bytes().decode("utf-8")
    assert notice.count(tool.MARKER) == 1 and notice.count(tool.END) == 1
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "update_nextgen_output_license_notice.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        "the hydro output notices are not what the generator writes; run python "
        f"tools/update_nextgen_output_license_notice.py\n{result.stderr[-2000:]}")

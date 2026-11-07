"""The per-thread local-frame table is a RECORDING, and it says whose.

``KERNEL_MAX_LOCAL_SIZE_BYTES`` prices the launch-time local-memory
backing store, which is the largest single term in the non-pool device
budget (5.7 GiB for one kf launch on a 170-SM card).  Every row of it is
a number NVRTC produced for ONE target architecture with ONE compiler
build, and until 2026-08-20 the module carried those numbers as if they
were a property of the source.

They are not.  Measured the same NVRTC-plus-driver way on three compile
platforms:

  ======================  =========  =========  =========
  module                  sm_120     sm_120     sm_86
                          13.0.48    13.3.33    13.0.48
  ======================  =========  =========  =========
  gf                       22,416     22,416     23,984
  noah                        176        176        224
  thompson_aerosol_warm         0          0        112
  ysu                       9,232      9,232      7,184
  nssl2_fused_gs              112        216        112
  rrtmgp_cloud                  0         40          0
  shinhong (to 2026-09-30) 14,040     17,160     14,040
  noahmp_leaves               272        208        208
  ======================  =========  =========  =========

Four rows move with the ARCHITECTURE at a fixed compiler, four move with
the COMPILER BUILD at a fixed architecture, and one moves with both.  So
the table is a joint property of (target architecture, NVRTC build), and
the only accurate form for it is a set of named recordings plus a ceiling
over them.

These tests need no device: they check the shape of the recording and the
arithmetic of the ceiling.  ``tests/test_preflight.py::
test_the_recorded_local_frames_match_the_driver`` is the leg that puts a
real compiler behind them.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from woof.core import preflight as pf

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "woof" / "core" / "kernels"


@pytest.mark.parametrize("relative_path,names", [
    ("woof/core/preflight.py", {"KERNEL_MAX_LOCAL_SIZE_BYTES"}),
    ("woof/core/kernel_frame_recordings.py", None),
    ("tests/test_kernel_source_freeze_per_module.py",
     {"BASELINE_PINNED", "PINNED_HEADERS"}),
])
def test_kernel_bookkeeping_has_no_shadowed_literal_keys(relative_path, names):
    """A duplicate source key silently discards a pin or frame measurement."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))
    if names is None:
        roots = [tree]
    else:
        roots = []
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            matched = {target.id for target in targets
                       if isinstance(target, ast.Name)} & names
            if matched:
                found.update(matched)
                roots.append(node.value)
        assert found == names, (relative_path, names - found)
    for root in roots:
        for node in ast.walk(root):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for key in node.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                assert key.value not in seen, (
                    f"{relative_path}:{key.lineno}: duplicate {key.value!r}; "
                    f"first recorded at line {seen.get(key.value)}")
                seen[key.value] = key.lineno


def test_every_recording_names_the_box_and_the_compiler_that_made_it():
    """A frame table with no provenance cannot be told from an assumption.

    The breakage this prevents is the one that opened the item: rows
    measured on a card that has since left the machine were carried in
    generic code with the box named only in prose, so a reader on any
    other machine had no way to see that the numbers were somebody
    else's.
    """
    recordings = pf.KERNEL_LOCAL_FRAME_RECORDINGS
    assert len(recordings) >= 2, "one recording cannot show that rows move"
    for row in recordings:
        assert row.box, "a recording must name the box it was read on"
        assert row.device
        assert re.fullmatch(r"\d+", row.compute_capability), row.compute_capability
        assert re.fullmatch(r"\d+\.\d+\.\d+", row.nvrtc_build), row.nvrtc_build
        assert row.platform_family in ("windows", "linux")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.measured), row.measured
        assert row.frames, "a recording with no frames is not a recording"


def test_the_shipped_table_is_the_ceiling_over_every_recording():
    """Never below a measurement: under-pricing is what breaches a rail.

    The reservation is ``(frame - stack) x SMs x threads/SM``, so a row
    one byte under what this box's compiler emits under-charges by one
    byte times the whole resident-thread capacity.  Over-pricing costs
    headroom and is the direction the module has always taken; the
    ceiling makes that promise arithmetic instead of prose.
    """
    ceiling = pf.KERNEL_MAX_LOCAL_SIZE_BYTES
    rows = [measured for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS
            for measured in (row, row.strict_recording) if measured is not None]
    for row in rows:
        for module, frame in row.frames.items():
            assert module in ceiling, (
                f"{module}: measured on {row.box} and absent from the "
                "shipped table")
            assert ceiling[module] >= frame, (
                f"{module}: shipped {ceiling[module]} B is BELOW the "
                f"{frame} B {row.box} measured")
    derived = {}
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        for measured in (row, row.strict_recording):
            if measured is not None:
                for module, frame in measured.frames.items():
                    derived[module] = max(derived.get(module, 0), frame)
    assert dict(ceiling) == derived, (
        "the shipped table must be exactly the element-wise maximum over "
        "the recordings, so no row can drift away from every measurement")


def test_every_kernel_source_has_a_row_or_is_declared_unmeasurable():
    """A ``.cu`` with no row is a module the pricing cannot see.

    Found by running the regeneration gate on a second box:
    ``health_tile.cu`` (the tile-streamed health reduction,
    woof/core/streaming.py:1728) had shipped since the out-of-core merge
    with no row in either table, so the gate that is supposed to notice a
    frame moving could not even enumerate it.
    """
    from woof.core.kernels import _DEFAULT_SOURCE_MODULES
    # <module>_default.cu is the default process's text of <module>,
    # compiled and priced under <module>'s own name.
    sources = {path.stem for path in KERNEL_DIR.glob("*.cu")}
    sources -= {f"{module}_default" for module in _DEFAULT_SOURCE_MODULES}
    # These remain priced logical modules. The first two resolve retained
    # files in kernels._unit_text; rrtmg_lw._lw_coalesced_source generates
    # the third inside the same measured composed longwave translation unit.
    # Observation FDDA composes woof/fddaobs/{kernels,forecast}.cu through
    # woof.fddaobs.forecast.frame_module, outside KERNEL_DIR.
    sources |= {"ysu", "mynn_dmp_sibling", "rrtmg_lw_chain_coalesced", "fddaobs"}
    priced = (set(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
              | set(pf.UNMEASURED_KERNEL_MODULES))
    assert sources - priced == set(), sorted(sources - priced)
    assert priced - sources == set(), sorted(priced - sources)


def test_a_recorded_platform_is_recognised_from_its_own_fingerprint(monkeypatch):
    """The gate has to know whether THIS box is one of the recorded ones."""
    from woof import wrf_exact
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    row = pf.KERNEL_LOCAL_FRAME_RECORDINGS[0]
    fingerprint = {
        "device_compute_capability": row.compute_capability,
        "nvrtc_build": row.nvrtc_build,
    }
    assert pf.kernel_frame_recording_for(fingerprint) is row
    assert pf.kernel_frame_recording_for(
        {"device_compute_capability": "1", "nvrtc_build": "0.0.0"}) is None
    # An unresolved fingerprint is not a match on anything: "unavailable"
    # must never be read as "the reference box".
    assert pf.kernel_frame_recording_for(
        {"device_compute_capability": "unavailable",
         "nvrtc_build": "unavailable"}) is None
    assert pf.kernel_frame_recording_for({}) is None


def test_an_over_wide_frame_is_reported_with_the_bytes_it_under_charges():
    """The refusal names the breakage in the unit that breaks: bytes.

    A module compiling wider than its row does not fail visibly -- the
    run is admitted by a fit gate that under-counted the driver's backing
    store and OOMs later -- so the report has to convert the frame delta
    into the device bytes nobody charged for.
    """
    profile = pf.DeviceLocalMemoryProfile(
        name="probe", multiprocessor_count=68,
        max_threads_per_multiprocessor=1536)
    observed = dict(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
    observed["thompson"] = observed["thompson"] + 8
    over = pf.under_priced_kernel_frames(observed, profile=profile)
    assert set(over) == {"thompson"}
    assert over["thompson"].shipped_bytes == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[
        "thompson"]
    assert over["thompson"].observed_bytes == observed["thompson"]
    assert over["thompson"].unpriced_device_bytes == 8 * 68 * 1536
    assert pf.under_priced_kernel_frames(
        dict(pf.KERNEL_MAX_LOCAL_SIZE_BYTES), profile=profile) == {}


def test_every_chained_translation_unit_says_why_it_has_no_recording():
    """A launched translation unit outside these tables must be a DECISION.

    The recordings key on ``.cu`` files that compile ALONE, because that is
    what both readers enumerate -- ``tools/vram_reserve_probe.py``
    (``mode_frames``) and ``tests/test_preflight.py::
    test_the_recorded_local_frames_match_the_driver`` glob
    ``woof/core/kernels/*.cu`` and drop whatever NVRTC refuses standalone.
    Three units the model really launches are composed rather than
    standalone -- the two legacy-RRTMG chains and ``p3_composed``, which is
    what an ``mp_physics = 50`` domain loads -- so each is priced on EVERY
    platform from ONE reading, and ``under_priced_kernel_frames`` cannot
    report a drifting one, because its ``observed`` argument comes from
    that same glob.

    The breakage this prevents is a fourth composed unit arriving with no
    sentence anywhere: an unrecorded single-platform price reads exactly
    like a measured cross-platform one, and the difference is a
    reservation the fit gate never charged.  P3 is the case that opened
    it -- ``p3_composed`` shipped priced at 0 B off one sm_120 reading with
    nothing in this module mentioning that it existed.
    """
    from woof.core import kernel_frame_recordings as kfr

    chained = set(pf.CHAINED_TRANSLATION_UNIT_FRAMES)
    assert chained, "no composed units at all means this gate measures air"

    # A composed unit may not live in a recording, and that is enforced
    # upstream rather than here: a chained-unit key in a frames mapping
    # makes frame_ceiling() disagree with KERNEL_MAX_LOCAL_SIZE_BYTES and
    # woof.core.preflight raises at import (preflight.py:1663).
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert chained.isdisjoint(row.frames), (
            f"{row.box}: {sorted(chained & set(row.frames))} is a composed "
            "translation unit and cannot carry a standalone frame row")

    recorded = kfr.CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW
    assert set(recorded) == chained, (
        "every composed translation unit must say why it has no recording, "
        "and nothing else may claim to be one; unexplained "
        f"{sorted(chained - set(recorded))}, stale "
        f"{sorted(set(recorded) - chained)}.  Add the reason to "
        "CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW in "
        "woof/core/kernel_frame_recordings.py -- do not delete this "
        "assertion")
    for unit, reason in recorded.items():
        assert len(reason.split()) >= 15, f"{unit}: {reason!r} is a label"
        assert ".cu" in reason or ".py" in reason, (
            f"{unit}: the reason must point at the source that composes the "
            f"unit or at the gate that re-audits its frame, got {reason!r}")

    # P3's own fragment, spelled out: it is UNMEASURABLE rather than
    # unmeasured -- p3.cu borrows the tree's one glibc r_pow/r_exp/r_log
    # from noahmp_leaves.cu and fails NVRTC alone -- so no recording may
    # ever grow a row for it, and a row would be a value nothing measured.
    assert "p3" in pf.UNMEASURED_KERNEL_MODULES
    assert (pf.CHAINED_TRANSLATION_UNIT_FRAMES["p3_composed"].covers
            == frozenset({"p3"}))
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert "p3" not in row.frames, row.box


def test_complete_recordings_cover_the_current_standalone_source_set():
    """A newly shipped module cannot silently invalidate a complete claim."""
    sources = {path.stem for path in KERNEL_DIR.glob("*.cu")}
    standalone = sources - set(pf.UNMEASURED_KERNEL_MODULES)
    for recording in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        if recording.complete:
            assert set(recording.frames) == standalone, recording.box


def test_parameter_scaler_has_a_measured_production_source_and_frame():
    """A measurable initialization shader must retain its physical reading."""
    from woof.core import kernel_frame_recordings as kfr, kernels

    row = kfr.SM120_NVRTC_13_4_92
    assert row.platform_key == ("120", "13.4.92")
    assert row.frames["physics_params"] == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["physics_params"] == 0
    assert "physics_params" not in pf.UNMEASURED_KERNEL_MODULES
    assert kernels.module_options("physics_params") == ("-std=c++17",)
    assert hashlib.sha256(kernels.module_source("physics_params").encode()).hexdigest() == (
        kfr.PHYSICS_PARAMS_MEASURED_SOURCE_SHA256), (
            "parameter shader source changed; re-read driver attributes before replacing its profile")


@pytest.mark.parametrize("module", ["pd_vertical_sl", "upper_wind_limiter"])
def test_production_dycore_frames_keep_their_measured_composed_source(module):
    from woof.core import kernel_frame_recordings as kfr, kernels

    recording = kfr.SM120_NVRTC_13_4_92
    assert recording.platform_key == ("120", "13.4.92")
    assert recording.frames[module] == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[module] == 0
    assert module not in pf.UNMEASURED_KERNEL_MODULES
    assert kernels.module_options(module) == ("-std=c++17",)
    assert hashlib.sha256(kernels.module_source(module).encode()).hexdigest() \
        == kfr.DYCORE_FRAME_MEASURED_SOURCE_SHA256[module], (
            f"{module} composed source changed; read actual driver attributes "
            "during a forecast before replacing its frame recording")


def test_every_noahmp_composed_recording_is_its_own_platform_and_stays_out_of_the_standalone_tables():
    """The Noah-MP composed units are priced from their own table.

    One row per compile platform, each recognised from its own
    fingerprint, and none of the composed keys in any standalone table:
    the standalone census is read by a different instrument (the ``*.cu``
    glob) that cannot compile a composition, so a composed key there
    would be a number nobody measured.  A card on an unrecorded platform
    is priced from the ceiling over THESE rows, with the basis stated
    (tests/test_noahmp_frame_provenance.py holds that, and the shape and
    identity of each row); this is the structural half.
    """
    from woof.core import kernel_frame_recordings as kfr
    from woof.core.noahmp_kernel_sources import NOAHMP_PRICING_MODULES

    rows = kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS
    assert rows, "with no row there is no ceiling, and scheme 4 is refused on every card"
    keys = [row.platform_key for row in rows]
    assert len(keys) == len(set(keys)), "one row per compile platform"
    for row in rows:
        assert isinstance(row, kfr.ComposedUnitFrameRecording)
        assert set(row.frames) == set(NOAHMP_PRICING_MODULES), row.box
        assert row.platform_key == (row.compute_capability, row.nvrtc_build)
        assert kfr.noahmp_composed_recording_for(
            {"device_compute_capability": row.compute_capability,
             "nvrtc_build": row.nvrtc_build}) is row
    composed = {key for key in NOAHMP_PRICING_MODULES
                if key.endswith(("_composed", "_runtime"))}
    assert composed.isdisjoint(kfr.frame_ceiling())
    assert composed.isdisjoint(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
    assert composed.isdisjoint(pf.CHAINED_TRANSLATION_UNIT_FRAMES)
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert composed.isdisjoint(row.frames), row.box
    assert kfr.noahmp_composed_recording_for(
        {"device_compute_capability": "unavailable",
         "nvrtc_build": "unavailable"}) is None


# ---------------------------------------------------------------------------
# The compile platform a fresh install gets is set by the package's own
# dependency spec, and the Noah-MP rows must include it.
# ---------------------------------------------------------------------------

def _gpu_extras() -> dict[str, list[str]]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]
    return {name: list(reqs) for name, reqs in extras.items()
            if name.startswith("gpu-cu")}


def test_every_gpu_extra_has_one_current_resolved_toolchain_pin():
    """The declaration in the tree IS the extra, byte for byte.

    The breakage this prevents: someone widens or re-pins
    ``cupy-cuda13x[ctk]`` in pyproject and the declared resolution -- the
    NVRTC build every Noah-MP admission claim rests on -- silently
    describes a requirement the package no longer ships.  One current pin
    per GPU extra, its requirement equal to the extra's single entry, and
    every superseded pin still naming a requirement that extra carries.
    """
    from woof.core import kernel_frame_recordings as kfr

    extras = _gpu_extras()
    assert extras, "pyproject declares no gpu-cu* extra"
    current = [pin for pin in kfr.RESOLVED_TOOLCHAIN_PINS if pin.current]
    assert sorted(pin.extra for pin in current) == sorted(extras), (
        "exactly one current pin per gpu-cu* extra")
    for pin in current:
        # The CuPy requirement is the extra's first entry; what follows may only narrow the NVRTC it resolves
        # (gpu-cu13 keeps NVRTC below 13.5, 2.9.13), and the narrowed range must still admit the pinned build.
        assert extras[pin.extra][:1] == [pin.requirement], (
            f"pyproject's {pin.extra} is {extras[pin.extra]}; the declared "
            f"resolution is of {pin.requirement!r}.  Re-resolve with "
            "`python tools/measure_noahmp_frames.py resolve` and re-declare")
        from packaging.requirements import Requirement
        for extra_line in extras[pin.extra][1:]:
            constraint = Requirement(extra_line)
            assert constraint.name in ("nvidia-cuda-nvrtc", "nvidia-nvjitlink"), (pin.extra, extra_line)
            assert constraint.specifier.contains(pin.nvrtc_build), (
                f"{pin.extra}'s {extra_line} excludes the pinned NVRTC {pin.nvrtc_build}; re-resolve and re-declare")
    for pin in kfr.RESOLVED_TOOLCHAIN_PINS:
        assert isinstance(pin, kfr.ResolvedToolchainPin)
        assert pin.requirement in extras.get(pin.extra, ()), pin
        assert re.fullmatch(r"\d+\.\d+\.\d+", pin.nvrtc_build), pin.nvrtc_build
        assert re.fullmatch(r"\d+(\.\d+)+", pin.cuda_toolkit), pin.cuda_toolkit
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pin.resolved), pin.resolved
        major = pin.nvrtc_build.split(".")[0]
        assert pin.cuda_toolkit.split(".")[0] == major, (
            "an NVRTC build and its cuda-toolkit share a CUDA major")
        assert pin.extra == f"gpu-cu{major}", (
            f"{pin.extra} cannot resolve to a CUDA {major} compiler")
        assert pin.nvrtc_distribution.startswith("nvidia-cuda-nvrtc")
        assert all(re.fullmatch(r"\d+", arch) for arch in pin.noahmp_architectures)


def test_the_recorded_noahmp_platforms_include_what_the_spec_resolves_to():
    """A fresh install must land on a recorded platform, or the docs lie.

    The breakage this prevents is the one that shipped for a day: every
    Noah-MP row was read on NVRTC 13.3.33 from a venv resolved in the
    cuda-toolkit 13.3.x window, cuda-toolkit 13.4.1 (2026-09-09) moved the
    ``[ctk]`` resolution to nvidia-cuda-nvrtc 13.4.59, and from then on
    `pip install recast-woof[gpu-cu13]` on the very card class the release named
    refused `woof check` with "no Noah-MP local-frame recording exists for
    this card class -- sm_120 at NVRTC 13.4.59".  An unrecorded platform is
    priced from the ceiling now rather than refused, but the docs name the
    platforms that are MEASURED, and a pin that lists an architecture
    without a row makes that claim false.  So: every pin that lists an
    architecture has a composed row at exactly (architecture, its NVRTC
    build); every row's build is one some pin declares, so a reading
    taken from a borrowed library is written down as the resolution window
    it belongs to; every row's architecture is listed by its build's pin,
    so a row cannot arrive without the pin admitting it; and at least one
    current install is measured somewhere, or the release ships every
    Noah-MP price from a ceiling nobody can compare with a reading.
    """
    from woof.core import kernel_frame_recordings as kfr

    pins = kfr.RESOLVED_TOOLCHAIN_PINS
    for pin in pins:
        for arch in pin.noahmp_architectures:
            row = kfr.noahmp_composed_recording_for(
                {"device_compute_capability": arch, "nvrtc_build": pin.nvrtc_build})
            assert row is not None, (
                f"a fresh `pip install recast-woof[{pin.extra}]` compiles with NVRTC "
                f"{pin.nvrtc_build} (cuda-toolkit {pin.cuda_toolkit}, resolved "
                f"{pin.resolved}) and the tree has no Noah-MP row for sm_{arch} "
                "on it: the docs name that platform as measured while every "
                "sf_surface_physics = 4 run on such an install is priced from "
                "the ceiling.  Take the row with `python "
                "tools/measure_noahmp_frames.py measure` inside that environment "
                f"on an sm_{arch} card, or drop the architecture from the pin")
            assert row.compute_capability == arch
            assert row.nvrtc_build == pin.nvrtc_build
    declared_builds = {pin.nvrtc_build for pin in pins}
    for row in kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS:
        assert row.nvrtc_build in declared_builds, (
            f"the {row.box} row was read on NVRTC {row.nvrtc_build}, which no "
            "declared resolution of any GPU extra installs; say which "
            "resolution window it belongs to in RESOLVED_TOOLCHAIN_PINS")
        assert any(row.compute_capability in pin.noahmp_architectures
                   for pin in pins if pin.nvrtc_build == row.nvrtc_build), (
            f"the pin for NVRTC {row.nvrtc_build} does not admit Noah-MP on "
            f"sm_{row.compute_capability}, yet a row exists for it")
    assert any(pin.current and pin.noahmp_architectures for pin in pins), (
        "no current install of any GPU extra lands on a measured Noah-MP "
        "platform: every sf_surface_physics = 4 price this release gives "
        "would come from a ceiling no fresh install can check against a "
        "reading of its own")


@pytest.mark.network
def test_the_declared_resolution_is_what_the_index_resolves_today():
    """The pin declaration is re-checked against the live index.

    Gated twice -- the ``network`` marker and WOOF_NETWORK_TESTS=1 --
    because it talks to PyPI.  Exit 1 here means a new cuda-toolkit release
    moved the compiler every fresh install gets, and the Noah-MP table
    needs a row on that build before the next cut.
    """
    if os.environ.get("WOOF_NETWORK_TESTS") != "1":
        pytest.skip("live index resolution needs WOOF_NETWORK_TESTS=1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/measure_noahmp_frames.py"), "resolve"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_assumed_bound_is_never_below_a_reading():
    """What a module with no reading anywhere is charged.

    It must be a BOUND: not below any frame this tree has recorded, for
    a standalone kernel or for a Noah-MP composed unit.  Pricing from it
    is what a missing reading costs; refusing the run was the defect it
    replaced.
    """
    from woof.core import kernel_frame_recordings as kfr

    bound = kfr.assumed_frame_bound()
    assert bound > 0
    for module, frame in kfr.frame_ceiling().items():
        assert bound >= frame, f"{module} is recorded above the assumed bound"
    for row in kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS:
        for key, frame in row.frames.items():
            assert bound >= frame, f"{key} is recorded above the assumed bound"
    assert kfr.ASSUMED_BOUND_PHRASE == "assumed bound, not measured"


def test_shinhong_workspace_reading_is_on_every_platform():
    """The workspace source was re-read on every recorded platform.

    Before 2026-09-30 the column arrays sat in a 13,000-17,160 B local
    frame.  A recording still carrying one of those would price a backing
    store the kernel no longer reserves, and a recording that dropped the
    row would lose its completeness claim; both are what this catches.
    """
    rows = [row for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS
            if "shinhong" in row.frames]
    assert len(rows) >= 4
    for row in rows:
        assert row.frames["shinhong"] == 0, (row.box, row.nvrtc_build)
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["shinhong"] == 0


def test_new_partial_recordings_use_public_host_labels():
    from woof.core.kernel_frame_recordings import KERNEL_LOCAL_FRAME_RECORDINGS
    rows = [row for row in KERNEL_LOCAL_FRAME_RECORDINGS if row.box == 'measured-host']
    assert any(dict(row.frames) == {'hail': 7392} for row in rows)
    assert any(dict(row.frames) == {'chem_radiation': 0} for row in rows)
    assert all(not row.complete for row in rows)


def test_frame_census_includes_launch_bounded_production_entries():
    """CAM's 32-thread driver must be counted with its scalar probes."""
    from tools.vram_reserve_probe import _SYMBOL
    names = set(_SYMBOL.findall((KERNEL_DIR / "cam3_emissivity.cu").read_text()))
    assert {"cam_driver", "cam_log10_probe", "cam_log10_variants"} <= names
    assert _SYMBOL.findall('extern "C" __global__ __launch_bounds__(128, 2) void tile() {}') == ["tile"]

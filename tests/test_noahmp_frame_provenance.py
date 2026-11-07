"""Noah-MP composed units: one compile site, per-platform frame rows, and an
estimator that prices the card it was asked about -- from that card's own
row when its compile platform has one, from the ceiling over the recorded
platforms when it does not, saying which beside the number -- and refuses
only when the tree holds no usable reading at all.

Everything here is CPU-only and every frame in the synthetic rows is made
up.  The one test that reads a real card is the opt-in
``test_this_platforms_row_is_what_the_card_compiles_to``; it is the driver
gate for the composed units, the way
``tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver``
is for the standalone ones.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.core import kernel_frame_recordings as kfr
from woof.core import noahmp_frame_provenance as prov
from woof.core import noahmp_kernel_sources as ks
from woof.core import preflight as pf
from woof.core.preflight import GIB
manifest = importlib.import_module("woof.certify.kernel_manifest")
from tools import measure_noahmp_frames as tool

ROOT = Path(__file__).resolve().parents[1]

#: Independent, reviewed runtime inventory.  Not derived from the production
#: table: deleting glacier or the slab there must not delete it here too.
EXPECTED_PARTS = {
    "noahmp_bareflux": ("noahmp_bareflux",),
    "noahmp_driver": ("noahmp_leaves", "noahmp_driver"),
    "noahmp_energy": ("noahmp_leaves", "noahmp_energy"),
    "noahmp_fluxprep": ("noahmp_fluxprep",),
    "noahmp_glacier": ("noahmp_leaves", "noahmp_glacier"),
    "noahmp_leaves": ("noahmp_leaves",),
    "noahmp_libm_slab": ("noahmp_leaves", "noahmp_energy", "noahmp_libm_slab"),
    "noahmp_radiation": ("noahmp_radiation",),
    "noahmp_sflx": ("noahmp_sflx",),
    "noahmp_snow": ("noahmp_snow",),
    "noahmp_soilwater": ("noahmp_soilwater",),
    "noahmp_thermal": ("noahmp_leaves", "noahmp_thermal"),
    "noahmp_vegeflux": ("noahmp_vegeflux",),
    "noahmp_vegprecip": ("noahmp_vegprecip",),
    "noahmp_water": ("noahmp_water",),
    "noahmp_output": ("noahmp_output",),
}

SYNTHETIC_PLATFORM = ("99", "0.0.1")


def experiment(lsm=4, **overrides):
    settings = dict(nx=40, ny=32, nz=40, dx=3000., dy=3000., ztop=16000.,
                    dt=10., run_seconds=600., moist=True, mp_physics=0,
                    bl_pbl_physics=1, sf_sfclay_physics=1,
                    sf_surface_physics=lsm, ra_physics=90, cu_physics=0,
                    num_soil_layers=9 if lsm == 3 else 4)
    settings.update(overrides)
    return experiment_from_run_config(RunConfig(**settings), datetime(2026, 1, 1))


def synthetic_attrs(unit, symbol):
    """An INHERITED export is made the widest on purpose, so a reading that
    only looked at a fragment's own kernels would come out short."""
    frame = 0
    if unit == "noahmp_thermal" and symbol == "noahmp_leaf_thermoprop":
        frame = 4096
    if unit == "noahmp_glacier" and symbol == "noahmp_glacier_column":
        frame = 6144
    if unit == "noahmp_libm_slab" and symbol == "noahmp_energy_assembly":
        frame = 8192
    return dict(local_size_bytes=frame, num_regs=32, shared_size_bytes=0,
                const_size_bytes=0, max_threads_per_block=128,
                ptx_version=99, binary_version=99)


def synthetic_frames() -> dict[str, int]:
    frames = {}
    for name in EXPECTED_PARTS:
        unit = ks.runtime_unit(name)
        frames[unit.key] = max(synthetic_attrs(name, symbol)["local_size_bytes"]
                               for symbol in unit.exports)
    return frames


@pytest.fixture
def row():
    return kfr.ComposedUnitFrameRecording(
        box="synthetic test box", device="SYNTHETIC TEST GPU",
        compute_capability=SYNTHETIC_PLATFORM[0],
        nvrtc_build=SYNTHETIC_PLATFORM[1], platform_family="linux",
        measured="2026-01-01", frames=synthetic_frames(),
        unit_identity=prov.unit_identities())


@pytest.fixture
def recorded(monkeypatch, row):
    """The tree's Noah-MP recordings replaced by the one synthetic row."""
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (row,))
    return row


@pytest.fixture
def present():
    """A card in the machine, bare context measured, platform read."""
    return pf.DeviceLocalMemoryProfile(
        name="SYNTHETIC TEST GPU", multiprocessor_count=4,
        max_threads_per_multiprocessor=32, default_stack_limit_bytes=1024,
        bare_context_bytes=123_456_789, compile_platform=SYNTHETIC_PLATFORM)


@pytest.fixture
def fake_cupy(monkeypatch):
    """Compile-and-attributes only: calling a kernel ALWAYS raises."""
    cp = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    events = []
    sources = {ks.runtime_unit(name).source: name for name in EXPECTED_PARTS}

    class Function:
        def __init__(self, attrs):
            self.attributes = attrs

        def __call__(self, *args, **kwargs):
            raise AssertionError("a compile-only measurement launched a kernel")

    class Module:
        def __init__(self, *, code, options, **kwargs):
            self.code, self.options = code, tuple(options)
            events.append(("RawModule", code, self.options))

        def compile(self, log_stream=None):
            events.append(("nvrtc", self.code, self.options))

        def get_function(self, name):
            assert name in ks.exported_kernels(self.code)
            events.append(("attributes", sources.get(self.code), name))
            return Function(synthetic_attrs(sources.get(self.code), name))

    class Device:
        id = 0
        compute_capability = SYNTHETIC_PLATFORM[0]

        def __init__(self, *args):
            pass

    props = dict(name=b"SYNTHETIC TEST GPU", multiProcessorCount=4,
                 maxThreadsPerMultiProcessor=32)
    cuda.Device = Device
    cuda.runtime = types.SimpleNamespace(
        getDevice=lambda: 0, getDeviceCount=lambda: 1,
        getDeviceProperties=lambda _: props,
        deviceGetLimit=lambda _: 1024, is_hip=False)
    cp.RawModule, cp.cuda = Module, cuda
    for name, module in (("cupy", cp), ("cupy.cuda", cuda)):
        monkeypatch.setitem(sys.modules, name, module)
    from woof.certify import compile_platform
    fingerprint = {key: "SYNTHETIC" for key in compile_platform.FINGERPRINT_KEYS}
    fingerprint.update(device_compute_capability=SYNTHETIC_PLATFORM[0],
                       nvrtc_build=SYNTHETIC_PLATFORM[1])
    monkeypatch.setattr(compile_platform, "compile_platform_fingerprint",
                        lambda: dict(fingerprint))
    # Noah-MP compiles at the direct-NVRTC boundary (woof.core.ieee_module,
    # --ftz=false kept past CuPy's trailing override), not through
    # cp.RawModule.  The stand-in keeps that boundary's contract -- refuse
    # options without --ftz=false, take the strict options under
    # GPUWM_WRF_EXACT, record the manifest row -- and hands the same Module
    # double to the factories, so every count below still sees one compile.
    from woof.core import ieee_module

    def compile_module(source, options, module_key, log_stream=None):
        from woof import wrf_exact
        options = tuple(options)
        if "--ftz=false" not in options:
            raise ValueError("land kernels require --ftz=false")
        if wrf_exact.ENABLED:
            options = wrf_exact.effective_options(options)
        module = Module(code=source, options=options)
        module.compile(log_stream=log_stream)
        manifest.record_module(module_key, source=source, options=options,
                               module=module)
        return module

    monkeypatch.setattr(ieee_module, "compile_module", compile_module)
    manifest.reset_kernel_manifest()
    yield cp, events
    manifest.reset_kernel_manifest()
    from woof.core import kernels
    kernels.load_module.cache_clear()
    kernels.load_module_int_defines.cache_clear()
    kernels.get_kernel.cache_clear()
    for name in ("driver", "energy", "thermal", "glacier", "vegeflux"):
        module = sys.modules.get(f"woof.core.noahmp_{name}_gpu")
        if module is not None:
            for attr in ("_module", "_module_on"):
                fn = getattr(module, attr, None)
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()
    module = sys.modules.get("woof.core.noahmp_slab_libm")
    if module is not None:
        module._MODULE_CACHE.clear()
        module._KERNEL_CACHE.clear()


# ---------------------------------------------------------------------------
# The composition: one authority, one compile site.
# ---------------------------------------------------------------------------

def test_inventory_covers_disk_and_independently_reviewed_compositions():
    assert ks.NOAHMP_TRANSLATION_UNITS == EXPECTED_PARTS
    assert set(ks.kernel_files()) == set(EXPECTED_PARTS)
    expected_keys = {name + "_composed" if len(parts) > 1 else
                     (name + "_runtime" if name == "noahmp_vegeflux" else name)
                     for name, parts in EXPECTED_PARTS.items()}
    assert set(pf._LAND_SURFACE_KERNEL_MODULES[4]) == expected_keys
    assert not expected_keys & pf.UNMEASURED_KERNEL_MODULES
    assert {"noahmp_glacier_composed", "noahmp_libm_slab_composed"} <= expected_keys


def test_reachable_runtime_factories_and_symbols_are_covered():
    """Walk every Noah-MP call site independently, module-name constants
    included.  ``noahmp_kernel_sources`` is the ONE file allowed to
    construct a RawModule; every other ``noahmp*.py`` -- the frame
    provenance module included -- must reach the compiler through it."""
    reached = set()
    for path in sorted((ROOT / "woof/core").glob("noahmp*.py")):
        if path.stem == "noahmp_kernel_sources":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {n.targets[0].id: n.value.value for n in tree.body
                     if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name)
                     and isinstance(n.value, ast.Constant)
                     and isinstance(n.value.value, str)}
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            callee = call.func.id if isinstance(call.func, ast.Name) else (
                call.func.attr if isinstance(call.func, ast.Attribute) else "")
            assert callee not in ("RawModule", "RawKernel"), path.name
            if callee not in ("get_kernel", "compile_runtime_unit") or not call.args:
                continue
            arg = call.args[0]
            name = arg.value if isinstance(arg, ast.Constant) else constants.get(
                arg.id if isinstance(arg, ast.Name) else "")
            if name is None and callee == "compile_runtime_unit":
                # measure_live loops over the inventory itself.
                assert path.stem == "noahmp_frame_provenance", path.name
                continue
            assert name in EXPECTED_PARTS, (path.name, name)
            reached.add(name)
            if callee == "get_kernel" and len(call.args) > 1:
                symbol = call.args[1]
                if isinstance(symbol, ast.Constant):
                    assert symbol.value in ks.runtime_unit(name).exports
    assert {"noahmp_glacier", "noahmp_libm_slab", "noahmp_thermal"} <= reached


def test_export_inventory_includes_inherited_globals():
    units = [ks.runtime_unit(name) for name in EXPECTED_PARTS]
    # 109 since the WRF 4.7 night snow-aging entry
    # (noahmp_radiation_batch_night_age) joined noahmp_radiation's exports;
    # 111 since 2.9.3: noahmp_vegeflux gained k_vegeflux_stc3 (300876fefd5,
    # the WRF-lineage STC=3 leg) and k_vegeflux_masked (b3dd58fc9f3, the
    # fixed-width masked dispatch), both kept by the speed2 merge de768fc2770.
    # Native output accumulation adds its own single exported grid kernel.
    assert sum(len(unit.exports) for unit in units) == 112
    leaves = set(ks.runtime_unit("noahmp_leaves").exports)
    for name in ks.COMPOSED_UNITS:
        assert leaves <= set(ks.runtime_unit(name).exports)
    assert set(ks.runtime_unit("noahmp_energy").exports) <= set(
        ks.runtime_unit("noahmp_libm_slab").exports)
    assert "noahmp_leaf_thermoprop" in ks.runtime_unit("noahmp_thermal").exports


@pytest.mark.parametrize("source", ["", "__global__ void unsupported() {}",
    'extern "C" __global__ void k() {}\nextern "C" __global__ void k() {}'])
def test_unknown_or_duplicate_exports_refuse(source):
    with pytest.raises(ValueError, match="exports"):
        ks.exported_kernels(source)


@pytest.mark.parametrize("name", sorted(EXPECTED_PARTS))
def test_actual_runtime_factory_compiles_authoritative_source(name, fake_cupy, monkeypatch):
    """Each production factory hands NVRTC exactly runtime_unit(name)."""
    cp, events = fake_cupy
    unit = ks.runtime_unit(name)
    if name in {"noahmp_driver", "noahmp_energy", "noahmp_thermal", "noahmp_glacier"}:
        short = name.removeprefix("noahmp_")
        module = importlib.import_module(f"woof.core.{name}_gpu")
        module._module.cache_clear()
        compiled = getattr(module, f"{short}_module")()
    elif name == "noahmp_libm_slab":
        from woof.core import noahmp_slab_libm as module
        module._MODULE_CACHE.clear()
        compiled = module._module()
    elif name == "noahmp_vegeflux":
        from woof.core import noahmp_vegeflux_gpu as module
        module._module_on.cache_clear()
        copied = []
        monkeypatch.setattr(module, "_copy_constant", lambda m, n, v: copied.append(n))
        compiled = module._module()
        assert set(copied) == set(module._constant_tables())
    else:
        from woof.core.kernels import load_module
        load_module.cache_clear()
        compiled = load_module(name)
    assert compiled.code == unit.source
    assert compiled.options == unit.options
    assert [e[0] for e in events].count("RawModule") == 1
    record = next(iter(manifest.kernel_manifest().values()))
    assert record["source_sha256"] == unit.identity()["source_sha256"]
    assert record["options"] == list(unit.options)
    if name == "noahmp_vegeflux":
        assert unit.options == ("-std=c++14", "--ftz=false")
        assert unit.source == (ks.KERNEL_DIR / (name + ".cu")).read_text(encoding="ascii")
    if name == "noahmp_fluxprep":
        assert tuple(p for p, _ in unit.parts) == (name,)


def test_negative_control_sources_still_compile_as_themselves(fake_cupy):
    """The parity suites' perturbed copies are recorded as what they are."""
    from woof.core.noahmp_driver_gpu import driver_module, _module
    _module.cache_clear()
    changed = ks.runtime_unit("noahmp_driver").source + "\n// negative control"
    assert driver_module(source=changed).code == changed
    keys = list(manifest.kernel_manifest())
    assert keys == ["woof.core.noahmp_driver_gpu:driver(substituted-source)"]


# ---------------------------------------------------------------------------
# The rows: shape, and that they describe THIS tree.
# ---------------------------------------------------------------------------

def test_shipped_rows_describe_the_units_in_this_tree():
    """A row read from a different source is not a measurement of this one.

    The breakage this prevents is a stale price: an edit to any Noah-MP
    ``.cu``, to the preamble or to an option tuple changes the frame NVRTC
    emits, and a row that still carried the old number would price the
    reservation short on exactly the platform that trusted it.  Red here
    means "re-read the platform", and the estimator refuses meanwhile.
    """
    rows = kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS
    assert rows, ("no Noah-MP composed-unit recording at all: scheme 4 "
                  "would be refused on every card")
    current = prov.unit_identities()
    assert set(current) == set(ks.NOAHMP_PRICING_MODULES)
    for row in rows:
        assert row.box and row.device
        assert re.fullmatch(r"\d+", row.compute_capability), row.compute_capability
        assert re.fullmatch(r"\d+\.\d+\.\d+", row.nvrtc_build), row.nvrtc_build
        assert row.platform_family in ("windows", "linux")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.measured), row.measured
        assert set(row.frames) == set(ks.NOAHMP_PRICING_MODULES), row.box
        assert all(isinstance(v, int) and v >= 0 for v in row.frames.values())
        assert dict(row.unit_identity) == current, (
            f"{row.box} (sm_{row.compute_capability}, NVRTC {row.nvrtc_build}) "
            "was read from a different Noah-MP source, option tuple or export "
            "set than this tree holds; re-read it with "
            "`python tools/measure_noahmp_frames.py measure` and move the row")
    keys = [row.platform_key for row in rows]
    assert len(keys) == len(set(keys)), "one row per compile platform"


def test_composed_units_never_join_the_standalone_tables():
    """No row of theirs in the standalone ceiling.

    The composed units carry their own table and their own ceiling over
    it; a composed key inside a standalone recording would price it twice
    from two tables, and the standalone one is read by a different
    instrument (the ``*.cu`` census) that cannot compile a composition."""
    composed = {key for key in ks.NOAHMP_PRICING_MODULES
                if key.endswith(("_composed", "_runtime"))}
    assert len(composed) == 6
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert composed.isdisjoint(row.frames), row.box
    assert composed.isdisjoint(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
    assert composed.isdisjoint(pf.CHAINED_TRANSLATION_UNIT_FRAMES)
    # The standalone Noah-MP stems keep their rows in the standalone census
    # (they compile alone), and scheme 4 prices them from the platform row
    # too: the same compiler read both, and the row is this platform's.
    standalone = set(ks.NOAHMP_PRICING_MODULES) - composed
    assert standalone <= set(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)


def test_a_recorded_platform_is_recognised_from_its_own_fingerprint(recorded):
    fingerprint = {"device_compute_capability": SYNTHETIC_PLATFORM[0],
                   "nvrtc_build": SYNTHETIC_PLATFORM[1]}
    assert kfr.noahmp_composed_recording_for(fingerprint) is recorded
    assert kfr.noahmp_composed_recording_for(
        {"device_compute_capability": "99", "nvrtc_build": "0.0.2"}) is None
    assert kfr.noahmp_composed_recording_for(
        {"device_compute_capability": "unavailable",
         "nvrtc_build": "unavailable"}) is None
    assert kfr.noahmp_composed_recording_for({}) is None
    assert kfr.noahmp_composed_recording_for(None) is None


# ---------------------------------------------------------------------------
# The estimator: the card it was asked about, or a refusal by name.
# ---------------------------------------------------------------------------

def test_a_present_card_on_a_recorded_platform_is_priced_from_its_row(recorded, present):
    exp = experiment()
    frames = pf.kernel_local_frame_bytes(exp, profile=present)
    for key, frame in recorded.frames.items():
        assert frames[key] == frame
    widest = max(frames.values())
    assert widest == 8192, "the synthetic slab frame must be the widest"
    assert pf.kernel_local_memory_bytes(exp, profile=present) == (
        present.reservation_bytes(widest)) == (8192 - 1024) * 4 * 32
    estimate = pf.estimate_experiment(exp, profile=present)
    # The card the caller asked about, untouched: its name, its geometry,
    # its MEASURED bare context, its platform.
    assert estimate.local_memory_profile is present
    assert estimate.non_pool_device_bytes == (
        present.cuda_context_bytes
        + present.reservation_bytes(widest)
        + pf.column_workspace_bytes(exp, profile=present))
    basis = pf.non_pool_basis(estimate.local_memory_profile)
    assert "measured on this card" in basis
    assert present.name in basis
    assert "sm_99 / NVRTC 0.0.1" in basis
    assert "absent" not in basis


def test_the_noahmp_path_mutates_no_other_term(recorded, present):
    """Context and column-workspace terms are the same bytes for scheme 4
    as for scheme 2 on the same profile; only the frames half moves."""
    noahmp, noah = experiment(), experiment(lsm=2)
    est4 = pf.estimate_experiment(noahmp, profile=present)
    est2 = pf.estimate_experiment(noah, profile=present)
    other4 = est4.non_pool_device_bytes - pf.kernel_local_memory_bytes(noahmp, profile=present)
    other2 = est2.non_pool_device_bytes - pf.kernel_local_memory_bytes(noah, profile=present)
    assert other4 == other2 == (present.cuda_context_bytes
                                + pf.column_workspace_bytes(noah, profile=present))
    assert est4.local_memory_profile is est2.local_memory_profile is present
    # And scheme 2 on the same card never consults the Noah-MP rows at all.
    assert pf.estimate_experiment(noah, profile=replace(present, compile_platform=None)
                                  ).non_pool_device_bytes == est2.non_pool_device_bytes


def test_another_platform_is_priced_from_the_ceiling_and_says_so(recorded, present, monkeypatch):
    """The frame is a reading of (architecture, NVRTC build); a card on any
    other pair is priced the way every standalone kernel is priced on an
    unrecorded platform -- from the element-wise ceiling over the recorded
    rows -- and the basis says so, naming the recorded platforms, this
    card's own pair and the command that makes the price exact.  Never a
    refusal: the strict rule this replaces refused the shipped desktop
    runtime's card and compiler by name."""
    exp = experiment()
    other = replace(present, name="OTHER GPU", compile_platform=("99", "0.0.2"))
    frames = pf.kernel_local_frame_bytes(exp, profile=other)
    for key, frame in recorded.frames.items():
        assert frames[key] == frame, "one row: the ceiling IS the row"
    estimate = pf.estimate_experiment(exp, profile=other)
    assert estimate.local_memory_profile is other, "the card asked about, untouched"
    assert pf.kernel_local_memory_bytes(exp, profile=other) == other.reservation_bytes(8192)
    basis = pf.non_pool_basis(other, exp)
    assert ("Noah-MP local frames priced from the ceiling over the recorded "
            "platforms sm_99/0.0.1" in basis)
    assert "not measured on this card" in basis
    assert "no reading exists for its compile platform sm_99 / NVRTC 0.0.2" in basis
    assert "measure_noahmp_frames.py measure" in basis
    assert "measured on this card's compile platform" not in basis
    # Two rows: the ceiling is element-wise -- never one row's column and
    # never an average -- and it lists every platform it was taken over.
    wider = replace(recorded, compute_capability="98", box="second synthetic box",
                    frames={**recorded.frames, "noahmp_water": 9000,
                            "noahmp_libm_slab_composed": 16})
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (recorded, wider))
    frames = pf.kernel_local_frame_bytes(exp, profile=other)
    assert frames["noahmp_water"] == 9000
    assert frames["noahmp_libm_slab_composed"] == 8192
    assert pf.kernel_local_memory_bytes(exp, profile=other) == other.reservation_bytes(9000)
    assert "recorded platforms sm_99/0.0.1, sm_98/0.0.1" in pf.non_pool_basis(other, exp)
    # The card whose platform IS recorded still gets its own row, not the
    # ceiling: the exact reading is preferred whenever there is one.
    assert pf.kernel_local_frame_bytes(exp, profile=present)["noahmp_water"] == (
        recorded.frames["noahmp_water"])
    assert ("measured on this card's compile platform sm_99/0.0.1"
            in pf.non_pool_basis(present, exp))


def test_an_absent_card_is_priced_on_its_own_geometry_from_the_ceiling(recorded, present):
    """The CARD_CLASS_MULTIPROCESSORS defect this guards against: a declared
    card silently priced on the recorded card's SMs.  An absent card keeps
    the reference geometry -- never the recorded card's -- takes its
    Noah-MP frames from the ceiling, and its basis says the platform was
    not read.  A present card whose platform was not read is an absent
    card for the frames only, and keeps its own measured geometry."""
    exp = experiment()
    reference = pf.MEASURED_LOCAL_MEMORY_PROFILE
    for gib in (12.0, 32.0):
        estimate = pf.estimate_experiment(exp, vram_gib=gib)
        assert estimate.local_memory_profile is reference
        assert estimate.non_pool_device_bytes == pf.non_pool_device_bytes(exp, profile=reference)
        absent = pf.card_local_memory_profile(gib)
        assert pf.estimate_experiment(exp, profile=absent).local_memory_profile is absent
        basis = pf.non_pool_basis(absent, exp)
        assert basis.startswith("modelled for an absent card")
        assert "priced from the ceiling over the recorded platforms sm_99/0.0.1" in basis
        assert "was not read" in basis and "not in this machine" in basis
        assert present.name not in basis
    # Never more optimistic than the reference geometry: the synthetic
    # 4 x 32 card's own reservation is not what the absent card is charged.
    assert pf.kernel_local_memory_bytes(exp, profile=reference) == reference.reservation_bytes(8192)
    assert reference.reservation_bytes(8192) > present.reservation_bytes(8192)
    unread = replace(present, compile_platform=None)
    assert pf.kernel_local_memory_bytes(exp, profile=unread) == present.reservation_bytes(8192)
    basis = pf.non_pool_basis(unread, exp)
    assert basis.startswith("measured on this card (")
    assert "priced from the ceiling" in basis and "was not read" in basis


def test_this_machine_unread_is_priced_on_the_reference_profile_and_says_so(recorded, present, monkeypatch):
    """No card read here -- the switch, or nothing answered -- prices
    scheme 4 the way it prices scheme 2: on the reference profile, the
    Noah-MP frames from the ceiling, the basis saying the card was not
    read.  The strict rule refused this route by name."""
    exp = experiment()
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    frames = pf.kernel_local_frame_bytes(exp)
    for key, frame in recorded.frames.items():
        assert frames[key] == frame
    reference = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert pf.kernel_local_memory_bytes(exp) == reference.reservation_bytes(8192)
    assert pf.non_pool_device_bytes(exp) == pf.non_pool_device_bytes(exp, profile=reference)
    estimate = pf.estimate_experiment(exp)
    assert estimate.local_memory_profile is reference
    basis = pf.non_pool_basis(estimate.local_memory_profile, exp)
    assert "priced from the ceiling over the recorded platforms sm_99/0.0.1" in basis
    assert "not measured on this card" in basis
    # No switch, no card: the same answer.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU")
    monkeypatch.setattr(pf, "live_device_local_memory_profile", lambda: None)
    assert pf.estimate_experiment(exp).local_memory_profile is reference


def test_with_no_profile_the_question_is_about_this_machine(recorded, present, monkeypatch):
    """``estimate_experiment(exp)`` with no card declared reads this
    machine's card -- the run host's own estimate -- and prices on it,
    from its own row."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(pf, "live_device_local_memory_profile", lambda: present)
    exp = experiment()
    estimate = pf.estimate_experiment(exp)
    assert estimate.local_memory_profile is present
    assert estimate.non_pool_device_bytes == pf.non_pool_device_bytes(exp, profile=present)
    assert pf.kernel_local_memory_bytes(exp) == present.reservation_bytes(8192)
    assert "measured on this card's compile platform sm_99/0.0.1" in pf.non_pool_basis(present, exp)
    # ...but a declared card is a different machine: this one's card is not
    # substituted for it, and the declared one is priced on the reference
    # geometry from the ceiling, saying so.
    declared = pf.estimate_experiment(exp, vram_gib=32.0)
    assert declared.local_memory_profile is pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert "priced from the ceiling" in pf.non_pool_basis(declared.local_memory_profile, exp)


def test_a_declared_card_with_no_reading_anywhere_still_prices(
        recorded, monkeypatch):
    """``--vram-gib`` on a tree that holds no Noah-MP row at all.

    A declared card is hardware somewhere else, so it is priced on the
    reference geometry; with no reading anywhere the Noah-MP units take
    the assumed bound.  Neither is a refusal, and both are said out loud.
    """
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", ())
    exp = experiment()
    declared = pf.estimate_experiment(exp, vram_gib=32.0)
    assert declared.local_memory_profile is pf.MEASURED_LOCAL_MEMORY_PROFILE
    basis = pf.non_pool_basis(declared.local_memory_profile, exp)
    assert "priced at the assumed bound" in basis
    assert kfr.ASSUMED_BOUND_PHRASE in basis


def test_a_stale_row_is_withdrawn_and_no_usable_row_prices_the_bound(
        monkeypatch, row, present):
    """A row read from a different source is not a measurement of this
    one.  Alone it leaves nothing to price FROM, so the units are priced
    at the tree's assumed bound and the basis says so; beside a usable
    row it is withdrawn from the exact match and the ceiling both, and
    the basis says which row went stale."""
    bound = kfr.assumed_frame_bound()
    identity = dict(row.unit_identity)
    identity["noahmp_thermal_composed"] = "0" * 64
    stale = replace(row, unit_identity=identity)
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (stale,))
    # It RUNS: the pricing returns a number instead of refusing.
    assert pf.kernel_local_memory_bytes(experiment(), profile=present) > 0
    frames = pf.kernel_local_frame_bytes(experiment(), profile=present)
    assert frames["noahmp_water"] == bound
    text = pf.non_pool_basis(present, experiment())
    assert f"priced at the assumed bound {bound} B per thread" in text
    assert kfr.ASSUMED_BOUND_PHRASE in text
    assert "noahmp_thermal_composed" in text
    assert "measure_noahmp_frames.py measure" in text
    missing = replace(row, frames={k: v for k, v in row.frames.items()
                                   if k != "noahmp_water"})
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (missing,))
    assert pf.kernel_local_frame_bytes(
        experiment(), profile=present)["noahmp_water"] == bound
    # No row at all is the other shape of the same missing reading.
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", ())
    assert pf.kernel_local_frame_bytes(
        experiment(), profile=present)["noahmp_water"] == bound
    assert "no Noah-MP local-frame recording" in pf.non_pool_basis(
        present, experiment())
    # A stale row for THIS platform beside a usable row for another: this
    # card is priced from the ceiling over the usable row alone.
    other = replace(row, compute_capability="98", box="other synthetic box",
                    frames={**row.frames, "noahmp_water": 5000})
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (stale, other))
    assert prov.usable_recordings() == (other,)
    frames = pf.kernel_local_frame_bytes(experiment(), profile=present)
    assert frames["noahmp_water"] == 5000
    basis = pf.non_pool_basis(present, experiment())
    assert "priced from the ceiling over the recorded platforms sm_98/0.0.1" in basis
    assert "sm_99/0.0.1" not in basis, "the stale row is not a recorded platform"
    assert ("no longer describes the Noah-MP units in this tree: "
            "noahmp_thermal_composed") in basis
    assert "withdrawn" in basis and "measure_noahmp_frames.py measure" in basis


def test_a_module_with_no_reading_is_priced_at_the_bound(
        recorded, present, monkeypatch):
    """Neither shape of missing reading refuses: a fragment that never
    compiles alone, and a module with no row in any table, are both
    priced at the assumed bound and both named in the basis."""
    bound = kfr.assumed_frame_bound()
    exp = experiment()
    selected = pf.physics_kernel_modules(exp)
    monkeypatch.setattr(pf, "physics_kernel_modules", lambda _: selected | {"noahmp_driver"})
    frames = pf.kernel_local_frame_bytes(exp, profile=present)
    assert frames["noahmp_driver"] == bound
    text = pf.non_pool_basis(present, exp)
    assert f"noahmp_driver priced at the assumed bound {bound} B per thread" in text
    assert kfr.ASSUMED_BOUND_PHRASE in text
    monkeypatch.setattr(pf, "physics_kernel_modules", lambda _: selected | {"unknown_future_unit"})
    frames = pf.kernel_local_frame_bytes(exp, profile=present)
    assert frames["unknown_future_unit"] == bound
    assert "unknown_future_unit priced at the assumed bound" in pf.non_pool_basis(
        present, exp)


@pytest.mark.parametrize("lsm", [0, 2, 3])
def test_non_noahmp_pricing_is_untouched(lsm, monkeypatch):
    """Every other scheme prices exactly as before: no device read, the
    reference profile for absent and undeclared cards, no platform needed."""
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    exp = experiment(lsm=lsm)
    assert pf._noahmp_pricing_profile(exp, None) is None
    frames = pf.kernel_local_frame_bytes(exp)
    assert frames and not set(frames) & set(ks.NOAHMP_PRICING_MODULES)
    assert pf.estimate_experiment(exp, vram_gib=32.0).local_memory_profile is (
        pf.MEASURED_LOCAL_MEMORY_PROFILE)
    assert pf.estimate_experiment(exp).local_memory_profile is (
        pf.MEASURED_LOCAL_MEMORY_PROFILE)
    absent = pf.card_local_memory_profile(12.0)
    assert pf.estimate_experiment(exp, profile=absent).local_memory_profile is absent


# ---------------------------------------------------------------------------
# The profile carries where it was read.
# ---------------------------------------------------------------------------

def test_the_probe_payload_carries_the_compile_platform():
    payload = {"free_bytes": 1, "profile": {
        "name": "NVIDIA GeForce RTX 5070 Ti", "multiprocessor_count": 70,
        "max_threads_per_multiprocessor": 1536, "default_stack_limit_bytes": 1024,
        "bare_context_bytes": 480_000_000, "compile_platform": ["120", "13.3.33"]}}
    profile = pf.profile_from_device_probe(payload)
    assert profile.compile_platform == ("120", "13.3.33")
    assert profile.platform_is_read
    for broken in (None, ["120"], ["unavailable", "13.3.33"], ["", "13.3.33"], "120/13.3.33"):
        payload["profile"]["compile_platform"] = broken
        assert pf.profile_from_device_probe(payload).compile_platform is None
    del payload["profile"]["compile_platform"]
    assert pf.profile_from_device_probe(payload).compile_platform is None
    # The probe source reads the platform itself and still imports no woof.
    source = pf._DEVICE_MEMORY_PROBE_SOURCE
    assert '"compile_platform": _platform' in source
    statements = [line for line in source.splitlines()
                  if not line.lstrip().startswith("#")]
    assert not [line for line in statements
                if line.lstrip().startswith(("import woof", "from woof"))]


def test_the_basis_sentence_names_the_platform_the_frames_were_read_at(present):
    measured = pf.non_pool_basis(present)
    assert "measured on this card" in measured
    assert "frames read at its compile platform sm_99 / NVRTC 0.0.1" in measured
    unmeasured_context = replace(present, bare_context_bytes=None)
    text = pf.non_pool_basis(unmeasured_context)
    assert text.startswith("read on this card")
    assert "modelled" in text and "absent" not in text
    assert "sm_99 / NVRTC 0.0.1" in text
    absent = pf.card_local_memory_profile(32.0)
    assert pf.non_pool_basis(absent).startswith("modelled for an absent card")
    assert "compile platform" not in pf.non_pool_basis(absent)


def test_the_basis_sentence_says_whether_noahmp_frames_were_measured_on_this_card(recorded, present):
    """Plan review states the basis in the memory verdict: the same
    sentence names the card the grid-independent terms were priced on and,
    for scheme 4, whether the Noah-MP frames were read on this card's
    compile platform or taken from the ceiling over the recorded ones."""
    exp = experiment()
    measured = pf.non_pool_basis(present, exp)
    assert "measured on this card (" in measured
    # The device and the reading date are the basis; the host the reading
    # was taken on is recorded on the row but is not user-facing text.
    assert measured.endswith(
        "Noah-MP local frames measured on this card's compile platform "
        "sm_99/0.0.1 (SYNTHETIC TEST GPU, read 2026-01-01)")
    # Scheme 2 on the same card: no Noah-MP clause at all, and no
    # experiment means the profile half alone, exactly as before.
    assert "Noah-MP" not in pf.non_pool_basis(present, experiment(lsm=2))
    assert pf.non_pool_basis(present) == pf.non_pool_basis(present, experiment(lsm=2))
    basis = prov.noahmp_frame_basis(pf.physics_kernel_modules(exp), present)
    assert basis.measured and basis.recording is recorded
    assert basis.platform == SYNTHETIC_PLATFORM
    assert prov.noahmp_frame_basis(pf.physics_kernel_modules(experiment(lsm=2)), present) is None
    ceiling = prov.frame_basis_for_profile(replace(present, compile_platform=("99", "0.0.2")))
    assert not ceiling.measured and ceiling.recording is None
    assert ceiling.recorded_platforms == ("sm_99/0.0.1",)
    assert dict(ceiling.frames) == dict(recorded.frames)
    assert ceiling.platform == ("99", "0.0.2")


def test_read_compile_platform_refuses_an_unresolved_half(monkeypatch):
    from woof.certify import compile_platform
    good = {key: "x" for key in compile_platform.FINGERPRINT_KEYS}
    good.update(device_compute_capability="120", nvrtc_build="13.3.33")
    monkeypatch.setattr(compile_platform, "compile_platform_fingerprint", lambda: dict(good))
    assert pf.read_compile_platform() == ("120", "13.3.33")
    for key in ("device_compute_capability", "nvrtc_build"):
        broken = dict(good, **{key: compile_platform.UNRESOLVED})
        monkeypatch.setattr(compile_platform, "compile_platform_fingerprint", lambda b=broken: dict(b))
        assert pf.read_compile_platform() is None

    def boom():
        raise RuntimeError("no nvrtc here")
    monkeypatch.setattr(compile_platform, "compile_platform_fingerprint", boom)
    assert pf.read_compile_platform() is None


# ---------------------------------------------------------------------------
# Taking a row: the measurement, with a compile-only fake and for real.
# ---------------------------------------------------------------------------

def test_measure_live_reads_every_export_and_never_launches(fake_cupy):
    cp, events = fake_cupy
    reading = prov.measure_live()
    assert reading["platform"]["device_compute_capability"] == SYNTHETIC_PLATFORM[0]
    assert reading["platform"]["nvrtc_build"] == SYNTHETIC_PLATFORM[1]
    assert reading["device"] == dict(name="SYNTHETIC TEST GPU", multiprocessor_count=4,
                                     max_threads_per_multiprocessor=32,
                                     default_stack_limit_bytes=1024)
    assert reading["frames"] == synthetic_frames()
    assert reading["unit_identity"] == prov.unit_identities()
    assert len(reading["functions"]) == 16
    # 111 exports since 2.9.3 (k_vegeflux_stc3 and k_vegeflux_masked; see
    # test_export_inventory_includes_inherited_globals).
    assert sum(len(rows) for rows in reading["functions"].values()) == 112
    assert len([e for e in events if e[0] == "attributes"]) == 112
    assert [e[0] for e in events].count("RawModule") == 16
    for name in EXPECTED_PARTS:
        unit = ks.runtime_unit(name)
        assert ("RawModule", unit.source, unit.options) in events
    # "no launches" is structural: the fake's kernels raise when called.


def test_measure_live_requires_a_fresh_process_and_one_device(fake_cupy, monkeypatch):
    cp, events = fake_cupy
    manifest.record_module("woof.core.kernels:acoustic", source="x", options=("-std=c++17",))
    with pytest.raises(RuntimeError, match="fresh process"):
        prov.measure_live()
    manifest.reset_kernel_manifest()
    for count in (0, 2):
        monkeypatch.setattr(cp.cuda.runtime, "getDeviceCount", lambda c=count: c)
        with pytest.raises(RuntimeError, match="exactly one visible CUDA device"):
            prov.measure_live()
    assert not [e for e in events if e[0] == "RawModule"]


def test_measure_live_refuses_an_unresolved_platform(fake_cupy, monkeypatch):
    from woof.certify import compile_platform
    fingerprint = compile_platform.compile_platform_fingerprint()
    fingerprint["nvrtc_build"] = compile_platform.UNRESOLVED
    monkeypatch.setattr(compile_platform, "compile_platform_fingerprint", lambda: fingerprint)
    with pytest.raises(RuntimeError, match="nvrtc_build is unavailable"):
        prov.measure_live()


def test_rendered_row_is_the_recording_it_describes(fake_cupy):
    reading = prov.measure_live()
    text = prov.render_row(reading, box="synthetic test box",
                           platform_family="linux", measured="2026-01-01")
    namespace = {"ComposedUnitFrameRecording": kfr.ComposedUnitFrameRecording,
                 "MappingProxyType": types.MappingProxyType}
    rendered = eval(text.strip().rstrip(","), namespace)  # noqa: S307 -- our own text
    assert dict(rendered.frames) == reading["frames"]
    assert dict(rendered.unit_identity) == reading["unit_identity"]
    assert rendered.platform_key == SYNTHETIC_PLATFORM
    assert rendered.device == "SYNTHETIC TEST GPU"


def test_compare_with_tree_names_every_difference(fake_cupy, recorded, monkeypatch):
    reading = prov.measure_live()
    assert prov.compare_with_tree(reading) == []
    wider = replace(recorded, frames={**recorded.frames, "noahmp_water": 4})
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (wider,))
    problems = prov.compare_with_tree(reading)
    assert problems == ["noahmp_water: row 4 B, this card compiles 0 B"]
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", ())
    assert prov.compare_with_tree(reading) == [
        "no row for sm_99 / NVRTC 0.0.1 in woof/core/kernel_frame_recordings.py"]


def test_the_tool_writes_the_reading_and_prints_its_row(fake_cupy, recorded, tmp_path, capsys):
    out = tmp_path / "reading.json"
    assert tool.main(["measure", "--worker", "--output", str(out),
                      "--box", "synthetic test box", "--platform-family", "linux"]) == 0
    captured = capsys.readouterr()
    reading = json.loads(out.read_text(encoding="utf-8"))
    assert reading["frames"] == synthetic_frames()
    assert reading["row"] == captured.out.rstrip("\n")
    assert "already holds exactly this reading" in captured.err
    assert "ComposedUnitFrameRecording(" in captured.out
    # A reading is never overwritten.
    assert tool.main(["measure", "--output", str(out)]) == 3
    assert "never overwritten" in capsys.readouterr().err


@pytest.mark.parametrize("fault", ["compile", "function", "attribute"])
def test_a_faulted_reading_writes_nothing(fake_cupy, tmp_path, fault, monkeypatch, capsys):
    cp, events = fake_cupy
    if fault == "compile":
        def broken(*args, **kwargs):
            raise RuntimeError("synthetic compile failure")
        monkeypatch.setattr(cp.RawModule, "compile", broken)
    elif fault == "function":
        def broken(*args, **kwargs):
            raise KeyError("synthetic missing exported function")
        monkeypatch.setattr(cp.RawModule, "get_function", broken)
    else:
        monkeypatch.setattr(cp.RawModule, "get_function",
                            lambda *a: types.SimpleNamespace(attributes={}))
    out = tmp_path / "faulted.json"
    assert tool.main(["measure", "--worker", "--output", str(out)]) == 3
    assert not out.exists()
    assert "refused" in capsys.readouterr().err


def test_verify_reports_the_three_outcomes(fake_cupy, recorded, monkeypatch, capsys):
    # Each call is a fresh process in the real tool; here the manifest the
    # fake compiles fill is reset between them to stand in for that.
    assert tool.main(["verify", "--worker"]) == 0
    assert "exactly what this card compiles to" in capsys.readouterr().out
    wider = replace(recorded, frames={**recorded.frames, "noahmp_water": 4})
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", (wider,))
    manifest.reset_kernel_manifest()
    assert tool.main(["verify", "--worker"]) == 1
    assert "gone stale" in capsys.readouterr().err
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", ())
    manifest.reset_kernel_manifest()
    assert tool.main(["verify", "--worker"]) == 2
    assert "no row for sm_99" in capsys.readouterr().err
    # And a second reading in the SAME process is what the tool refuses.
    assert tool.main(["verify", "--worker"]) == 3
    assert "fresh process" in capsys.readouterr().err


def test_resolve_compares_the_declared_pins_with_a_live_resolution(monkeypatch, capsys):
    """``resolve`` says whether the index still hands out the declared compiler.

    Without the network: the resolution is a table handed in.  Exit 0 when
    every current pin's cuda-toolkit and NVRTC versions are what pip
    resolves, 1 when either moved -- naming the move, the architectures
    that then have no row, and the row-taking command -- and 3 when an
    extra nobody declared is asked for.  A superseded pin (current=False)
    is never resolved: its window is over.
    """
    pins = (
        kfr.ResolvedToolchainPin(
            extra="gpu-cu13", requirement="cupy-cuda13x[ctk]>=14.0",
            cuda_toolkit="13.4.1.0", nvrtc_distribution="nvidia-cuda-nvrtc",
            nvrtc_build="13.4.59", resolved="2026-09-10", current=True,
            noahmp_architectures=("120",)),
        kfr.ResolvedToolchainPin(
            extra="gpu-cu13", requirement="cupy-cuda13x[ctk]>=14.0",
            cuda_toolkit="13.3.1", nvrtc_distribution="nvidia-cuda-nvrtc",
            nvrtc_build="13.3.33", resolved="2026-09-08", current=False,
            noahmp_architectures=("120",)),
        kfr.ResolvedToolchainPin(
            extra="gpu-cu12", requirement="cupy-cuda12x[ctk]>=14.0",
            cuda_toolkit="12.9.2.0", nvrtc_distribution="nvidia-cuda-nvrtc-cu12",
            nvrtc_build="12.9.86", resolved="2026-09-10", current=True,
            noahmp_architectures=()),
    )
    monkeypatch.setattr(tool, "RESOLVED_TOOLCHAIN_PINS", pins)
    index = {
        "cupy-cuda13x[ctk]>=14.0": {"cupy-cuda13x": "14.2.0",
                                    "cuda-toolkit": "13.4.1.0",
                                    "nvidia-cuda-nvrtc": "13.4.59"},
        "cupy-cuda12x[ctk]>=14.0": {"cupy-cuda12x": "14.2.0",
                                    "cuda-toolkit": "12.9.2.0",
                                    "nvidia-cuda-nvrtc-cu12": "12.9.86"},
    }
    asked: list[str] = []

    def resolve(requirement, *, python=None):
        asked.append(requirement)
        return dict(index[requirement])

    monkeypatch.setattr(tool, "resolve_requirement", resolve)
    assert tool.main(["resolve"]) == 0
    out = capsys.readouterr()
    assert asked == ["cupy-cuda13x[ctk]>=14.0", "cupy-cuda12x[ctk]>=14.0"]
    assert "gpu-cu13: cupy-cuda13x[ctk]>=14.0 -> cuda-toolkit 13.4.1.0, nvidia-cuda-nvrtc 13.4.59" in out.out
    assert "13.3.33" not in out.out, "a superseded pin is not resolved"
    assert out.err == ""
    # The index moves the compiler: the row for sm_120 is owed.
    index["cupy-cuda13x[ctk]>=14.0"] = {"cupy-cuda13x": "14.3.0",
                                        "cuda-toolkit": "13.5.0.0",
                                        "nvidia-cuda-nvrtc": "13.5.12"}
    assert tool.main(["resolve", "--extra", "gpu-cu13"]) == 1
    out = capsys.readouterr()
    assert "declared 13.4.59, the index resolves 13.5.12" in out.err
    assert "declared 13.4.1.0, the index resolves 13.5.0.0" in out.err
    assert "no Noah-MP row for sm_120" in out.err
    assert "measure_noahmp_frames.py measure" in out.err
    assert "RESOLVED_TOOLCHAIN_PINS" in out.err
    # An extra that admits Noah-MP nowhere owes no row when it moves.
    index["cupy-cuda12x[ctk]>=14.0"]["nvidia-cuda-nvrtc-cu12"] = "12.9.90"
    assert tool.main(["resolve", "--extra", "gpu-cu12"]) == 1
    err = capsys.readouterr().err
    assert "12.9.90" in err and "no row is owed" in err
    # The NVRTC wheel vanishing from the resolution is named too.
    del index["cupy-cuda12x[ctk]>=14.0"]["nvidia-cuda-nvrtc-cu12"]
    assert tool.main(["resolve", "--extra", "gpu-cu12"]) == 1
    assert "no longer installs nvidia-cuda-nvrtc-cu12" in capsys.readouterr().err
    # Nobody declared this extra.
    assert tool.main(["resolve", "--extra", "gpu-cu11"]) == 3
    assert "no current pin declared" in capsys.readouterr().err


def test_compare_resolution_reads_the_pin_it_is_given():
    pin = kfr.RESOLVED_TOOLCHAIN_PINS[0]
    exact = {"cuda-toolkit": pin.cuda_toolkit,
             pin.nvrtc_distribution: pin.nvrtc_build}
    assert tool.compare_resolution(pin, exact) == []
    # A fourth version component is a different build, not the same one.
    assert tool.compare_resolution(
        pin, {**exact, pin.nvrtc_distribution: pin.nvrtc_build + ".1"})


@pytest.mark.skipif(os.environ.get("ARWEN_TEST_NOAHMP_GPU") != "1",
                    reason="opt-in: reads the real card in a fresh process")
def test_this_platforms_row_is_what_the_card_compiles_to(tmp_path):
    """The driver gate for the composed units.

    Exact equality against the tree's row for THIS compile platform, the
    same strength ``test_the_recorded_local_frames_match_the_driver``
    holds the standalone rows to.  A platform with no row fails here with
    the row printed, which is the way to add it.
    """
    pytest.importorskip("cupy")
    out = tmp_path / "reading.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/measure_noahmp_frames.py"), "measure",
         "--output", str(out)], capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    reading = json.loads(out.read_text(encoding="utf-8"))
    assert prov.compare_with_tree(reading) == [], (
        "this platform's Noah-MP row is missing or stale; the reading's row:\n"
        + reading["row"])


# ---------------------------------------------------------------------------
# The CPU-only check route and the once-per-process live read.
# ---------------------------------------------------------------------------

def test_the_cpu_only_check_route_prices_noahmp_from_the_ceiling_and_names_the_unread_card(
        recorded, present, monkeypatch):
    """``woof check`` with GPU readiness unjudged and no card answering
    its probe prices from metadata on the reference profile.  For scheme
    4 that used to be a refusal naming the reference card; now it is the
    estimate every other scheme gets on this route, with a basis that
    says the Noah-MP frames came from the ceiling, what kept this
    machine's card unread, and how to make the price exact.  A declared
    card on this route is priced the same way."""
    import argparse

    exp = experiment()
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(pf, "config_forcing_schedule", lambda *a, **k: (3600.0, 2))
    monkeypatch.setattr(pf, "config_forcing_source", lambda *a, **k: None)
    args = argparse.Namespace(config="unused.toml", column_chunk=None,
                              forcing_interval_s=None, vram_gib=None)
    required = pf._required_memory_without_kernels(
        exp, args, readiness="this machine's GPU readiness is info "
                             "(device not touched)")
    assert required["status"] == "estimated"
    assert required["alloc_estimate_bytes"] > 0
    basis = required["basis"]
    assert basis.startswith("CPU-only metadata; resident alternative with "
                            "conservative reference GPU overhead")
    assert "GPU readiness is info" in basis
    assert "was not read on this route" in basis
    assert ("Noah-MP local frames priced from the ceiling over the recorded "
            "platforms sm_99/0.0.1" in basis)
    assert "not measured on this card" in basis
    assert "measure_noahmp_frames.py measure" in basis
    assert pf.MEASURED_LOCAL_MEMORY_PROFILE.name not in basis
    assert "GPUWM_NO_LOCAL_GPU is set" in basis
    assert required["device_read"] is False
    assert required["local_memory_profile"] == pf.MEASURED_LOCAL_MEMORY_PROFILE.name
    # No verdict handed in: the route still says the card was not read.
    assert "was not read on this route" in pf._required_memory_without_kernels(exp, args)["basis"]
    # A declared card on this route is a machine that is elsewhere: priced
    # from the ceiling like any other unread card.
    args.vram_gib = 24.0
    declared = pf._required_memory_without_kernels(exp, args)
    assert declared["status"] == "estimated"
    assert "priced from the ceiling" in declared["basis"]
    # Scheme 2 on the same route is untouched: the plain basis, no
    # Noah-MP clause.
    required = pf._required_memory_without_kernels(experiment(lsm=2), args)
    assert required["status"] == "estimated"
    assert required["basis"] == ("CPU-only metadata; resident alternative with "
                                 "conservative reference GPU overhead")


def _cpu_only_check_parser(monkeypatch, exp, tmp_path):
    """The parser and the configuration path ``check_main`` is handed.

    The experiment is pinned, but the door reads the file's own text for
    its ``[fetch]`` hints before it loads anything, so the path has to
    be a readable TOML file and not a name."""
    import argparse

    from woof import doctor

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(pf, "_warn_unstaged_physics_tables", lambda *_: None)
    monkeypatch.setattr(pf, "config_forcing_schedule", lambda *a, **k: (3600.0, 2))
    monkeypatch.setattr(pf, "config_forcing_source", lambda *a, **k: None)
    monkeypatch.setattr(pf, "_load_experiment_any", lambda *a, **k: exp)
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda **_kw: doctor.Check(
        "CUDA kernel headers", "info", "not judged -- device not touched",
        brief="device not touched", blocking=False))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname = \"pinned\"\n", encoding="utf-8")
    return parser, str(config)


def test_check_main_on_the_cpu_only_route_prices_noahmp_and_prints_the_basis(
        recorded, present, monkeypatch, capsys, tmp_path):
    """The door a user without a judged card runs: the estimate is made,
    the basis is printed where the number is, and the portable planning
    command is offered exactly as it is for every other scheme."""
    parser, config = _cpu_only_check_parser(monkeypatch, experiment(), tmp_path)
    for flags in ([], ["--json"]):
        args = parser.parse_args(["check", config, *flags])
        code = pf.check_main(args)
        out = capsys.readouterr().out
        assert code == 2
        if args.json:
            report = json.loads(out)
            assert report["required_memory"]["status"] == "estimated"
            basis = report["required_memory"]["basis"]
            assert "priced from the ceiling over the recorded platforms sm_99/0.0.1" in basis
            assert "was not read on this route" in basis
            assert "--free-gib" in report["cpu_planning"]["command"]
        else:
            assert "CPU required-memory estimate (resident alternative)" in out
            assert "priced from the ceiling over the recorded platforms sm_99/0.0.1" in out
            assert "--free-gib FREE_GIB" in out


def test_check_main_estimates_and_states_the_bound_when_no_reading_exists(
        monkeypatch, capsys, tmp_path):
    """A tree with no usable Noah-MP row still produces an estimate and
    still offers the portable planning command.  The units are priced at
    the assumed bound and the basis says so where the number is."""
    monkeypatch.setattr(kfr, "NOAHMP_COMPOSED_FRAME_RECORDINGS", ())
    parser, config = _cpu_only_check_parser(monkeypatch, experiment(), tmp_path)
    for flags in ([], ["--json"]):
        args = parser.parse_args(["check", config, *flags])
        code = pf.check_main(args)
        out = capsys.readouterr().out
        assert code == 2
        if args.json:
            report = json.loads(out)
            assert report["required_memory"]["status"] == "estimated"
            basis = report["required_memory"]["basis"]
            assert "priced at the assumed bound" in basis
            assert kfr.ASSUMED_BOUND_PHRASE in basis
            assert "no Noah-MP local-frame recording" in basis
            assert "--free-gib" in report["cpu_planning"]["command"]
        else:
            assert "CPU required-memory estimate (resident alternative)" in out
            assert "priced at the assumed bound" in out
            assert "--free-gib FREE_GIB" in out


def test_the_live_card_is_read_once_per_process(monkeypatch):
    """Every no-profile Noah-MP estimate used to spawn nvidia-smi twice and
    an NVRTC probe; the first read is also the only one that can measure
    the bare context, so it is the one kept."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    pf.forget_live_device_local_memory_profile()
    reads = []
    read = pf.DeviceLocalMemoryProfile(
        name="READ ONCE", multiprocessor_count=4, max_threads_per_multiprocessor=32,
        bare_context_bytes=1 << 20, compile_platform=SYNTHETIC_PLATFORM)

    def from_device(cp):
        reads.append(cp)
        return read
    monkeypatch.setattr(pf, "local_memory_profile_from_device", from_device)
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    try:
        assert pf.live_device_local_memory_profile() is read
        assert pf.live_device_local_memory_profile() is read
        assert pf.live_device_local_memory_profile() is read
        assert len(reads) == 1
        # The switch keeps its meaning after a read.
        monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
        assert pf.live_device_local_memory_profile() is None
        monkeypatch.delenv("GPUWM_NO_LOCAL_GPU")
        assert pf.live_device_local_memory_profile() is read
        assert len(reads) == 1
        # A failed read caches nothing, so the next call tries again.
        pf.forget_live_device_local_memory_profile()

        def broken(cp):
            reads.append(cp)
            raise RuntimeError("no device")
        monkeypatch.setattr(pf, "local_memory_profile_from_device", broken)
        assert pf.live_device_local_memory_profile() is None
        assert pf.live_device_local_memory_profile() is None
        assert len(reads) == 3
        assert pf._LIVE_DEVICE_PROFILE == []
    finally:
        pf.forget_live_device_local_memory_profile()


# ---------------------------------------------------------------------------
# Every estimate that is not Noah-MP is byte-for-byte the baseline's.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsm", [0, 2, 3, 4])
@pytest.mark.parametrize("mp", [0, 1, 6, 8, 9, 10, 16, 18, 28, 50])
def test_existing_estimator_inventories_match_the_baseline(lsm, mp, monkeypatch):
    """40 domain inventories and 30 non-Noah-MP full estimates, hashed
    against independently reviewed snapshots (fixture ``baseline_commit``).
    The batch008 refresh receipt explains each allocation and frame change.
    New metadata absent from the recorded estimate is asserted empty
    before projection: ``compile_platform`` and the uncoupled ``hydro_memory``
    roster. All allocation numbers and frame inventories remain compared.
    Coupled allocations are checked in test_hydro_memory_contract. Re-record each
    scheme with ``python proof/node-reds-276/regen_estimator_baseline.py
    . <mp> --write`` after explaining the allocation change."""
    monkeypatch.setattr(pf.sys, "platform", "linux")
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    fixture = json.loads((ROOT / "tests/fixtures/noahmp_estimator_baseline_snapshots.json")
                         .read_text(encoding="utf-8"))
    exp = experiment(lsm=lsm, mp_physics=mp)
    row = {"domain": asdict(pf.estimate_domain(exp.root))}

    def snapshot(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    key = f"lsm{lsm}-mp{mp}"
    assert snapshot(row["domain"]) == fixture["domain_sha256"][key], row
    if lsm != 4:
        estimate = pf.estimate_experiment(exp)
        as_dict = asdict(estimate)
        assert as_dict["local_memory_profile"].pop("compile_platform") is None
        assert as_dict.pop("hydro_memory") is None
        row.update(frames=pf.kernel_local_frame_bytes(exp), estimate=as_dict,
                   alloc=estimate.alloc_estimate_bytes, peak=estimate.peak_envelope_bytes)
        assert snapshot(row) == fixture["full_estimate_sha256"][key], row

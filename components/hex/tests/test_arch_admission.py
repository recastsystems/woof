"""Architecture admission: open by default, with the anchors kept as record.

Every architecture NVRTC can compile the kernels for runs, with no flag, the
way any other model runs on the card it is given.  The per-architecture
anchors (the numerical contract measured again on real hardware of that
architecture) stay as the evidence record: a receipt says whether its card's
architecture holds one and names the pin, and never claims one it does not
hold.  The architecture refusals left each name a concrete breakage: a card
NVRTC cannot compile for, on which no kernel could be built, and, measured
by the forecast door on an unanchored card, ordinary arithmetic that is not
IEEE, on which every number the run produced would be wrong.  The sm_120
pin a few FTZ instruments still pass is held below with the recorded byte
pins that keep it there.  The sm_120 path is unchanged.

The evidence tests further down hold every registered anchor to the record
it rests on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import struct
import sys
import types
from fractions import Fraction
from pathlib import Path, PurePosixPath

import pytest

from _layout import PACKAGE_DIR
from woof.hex.cuda_backend import arch_admission
from woof.hex.cuda_backend.arch_admission import (
    ADMITTED_BELOW_FLOOR,
    ArchAnchor,
    PROVEN_COMPUTE,
    below_floor_refusal,
)

TREE_ROOT = Path(__file__).resolve().parent.parent

#: The targets CUDA 13's NVRTC lists, as the fake stack reports them.
FAKE_NVRTC_TARGETS = (75, 80, 86, 89, 90, 100, 120, 121)


def _fake_cupy(monkeypatch, *, major, minor, name="Fake GPU", targets=FAKE_NVRTC_TARGETS):
    """Install a minimal fake cupy stack so require_cuda runs CPU-only."""

    # (Recast WOOF) a fake CuPy opens no device: the engine backend's local-GPU ban (GPUWM_NO_LOCAL_GPU,
    # which the CPU suites set) guards real device contact, not this stub
    monkeypatch.setattr("woof.local_gpu.no_local_gpu", lambda: False)

    runtime = types.SimpleNamespace(
        is_hip=False,  # a real CuPy runtime binding always answers this
        getDeviceCount=lambda: 1,
        getDevice=lambda: 0,
        getDeviceProperties=lambda device_id: {
            "major": major,
            "minor": minor,
            "name": name,
            "totalGlobalMem": 10_240 * 1024 * 1024,
            "multiProcessorCount": 68,
        },
        runtimeGetVersion=lambda: 13_000,
        driverGetVersion=lambda: 13_030,
    )

    class _Device:
        def __init__(self, device_id):
            self.device_id = device_id

        def use(self):
            return None

    nvrtc = types.ModuleType("cupy.cuda.nvrtc")
    nvrtc.getSupportedArchs = lambda: tuple(targets)
    nvrtc.getVersion = lambda: (13, 0)

    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    cuda.Device = _Device
    cuda.nvrtc = nvrtc

    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    cupy.__version__ = "fake-for-admission-tests"

    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvrtc", nvrtc)
    import woof.backend as engine_backend
    monkeypatch.setattr(engine_backend, "_selected", None)  # the selector memoizes the first cupy it served


def _anchor_sm86(tmp_path: Path) -> ArchAnchor:
    return ArchAnchor(
        compute=(8, 6),
        card="RTX 3080 (test double)",
        admitted_on="2026-08-25",
        contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
        authority_anchor="evidence/sm86-tier-20260825/authority",
        basis="test-registered anchor",
    )


# --- the one architecture refusal left: a card NVRTC cannot compile for ---


def test_nvrtc_refusal_names_the_card_the_targets_and_the_breakage():
    text = arch_admission.nvrtc_refusal((6, 1), (75, 80, 86))
    assert "cuda.compute_capability=6.1" in text
    assert "sm_61" in text
    assert "compute_75, compute_80, compute_86" in text
    assert "not for compute_61" in text
    # The gate names the concrete breakage it prevents.
    assert "none of them could be built" in text


def test_no_refusal_cites_a_missing_anchor(monkeypatch):
    _fake_cupy(monkeypatch, major=6, minor=1)
    for text in (
        arch_admission.nvrtc_refusal((6, 1), FAKE_NVRTC_TARGETS),
        below_floor_refusal((6, 1), (12, 0)),
    ):
        assert "anchor" not in text


def test_a_card_nvrtc_cannot_compile_for_is_refused_naming_nvrtc(
    monkeypatch, tmp_path
):
    _fake_cupy(monkeypatch, major=6, minor=1)
    from woof.hex.cuda_backend.runtime import CudaRefusal, require_cuda

    with pytest.raises(CudaRefusal) as caught:
        require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    message = str(caught.value)
    assert "sm_61" in message
    assert "compute_75" in message
    assert "none of them could be built" in message


# --- require_cuda on each class of architecture ----------------------------


def test_unanchored_architecture_is_admitted_by_default(monkeypatch, tmp_path):
    """Fixed means default: no anchor, no flag, and it runs, even at a call
    site that still passes ``required_compute``."""

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    _fake_cupy(monkeypatch, major=8, minor=6, name="NVIDIA GeForce RTX 3080")
    from woof.hex.cuda_backend.runtime import require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert capability.compute == (8, 6)
    status = arch_admission.architecture_status(capability.compute)
    assert status.status == "unanchored"
    assert status.anchor is None


@pytest.mark.parametrize("compute", [(7, 5), (8, 0), (9, 0), (10, 0)])
def test_every_architecture_nvrtc_compiles_for_is_admitted(
    monkeypatch, tmp_path, compute
):
    _fake_cupy(monkeypatch, major=compute[0], minor=compute[1])
    from woof.hex.cuda_backend.runtime import require_cuda

    capability = require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    assert capability.compute == compute


def test_anchored_architecture_is_admitted_at_min_and_required_sites(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): _anchor_sm86(tmp_path)}
    )
    _fake_cupy(monkeypatch, major=8, minor=6, name="NVIDIA GeForce RTX 3080")
    from woof.hex.cuda_backend.runtime import require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert capability.compute == (8, 6)
    assert capability.sm == "sm_86"
    assert capability.name == "NVIDIA GeForce RTX 3080"
    assert arch_admission.architecture_status((8, 6)).status == "anchored"


def test_sm120_path_is_unchanged(monkeypatch, tmp_path):
    consulted = []
    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})

    def _spy(compute):
        consulted.append(compute)
        return None

    monkeypatch.setattr(arch_admission, "admitted_architecture", _spy)
    _fake_cupy(monkeypatch, major=12, minor=0)
    from woof.hex.cuda_backend import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "admitted_architecture", _spy)
    capability = runtime_module.require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert capability.compute == (12, 0)
    # At or above the floor the registry is never consulted.
    assert consulted == []


#: The call sites that still pass ``required_compute=(12, 0)``, each with
#: the recorded byte pin its bytes are held to, which is why the argument
#: waits there (a named deferral) instead of leaving with the rest.  The
#: package's FTZ instruments are keyed by their path in the package, the
#: tools by ``tools/<name>``.
PINNED_TO_SM120 = {
    "cuda_ftz.py": (
        "the v8.2.3 FTZ contract instrument: every dual-run capsule records "
        "its SHA-256 among its implementation sources, and "
        "cuda_dualrun refuses a capsule whose sources differ from the live "
        "files"
    ),
    "cuda_ftz_v841.py": (
        "the v8.4.1 FTZ audit runner: every FTZ transcript records its "
        "SHA-256 as runner_source_sha256, and cuda_ftz refuses a transcript "
        "not bound to the live runner"
    ),
    "tools/run_cuda_ftz_contract.py": (
        "the frozen FTZ contract tool: its SHA-256 is FROZEN_TOOL_SHA256 in "
        "run_cuda_ftz_v841_trust_measurement.py and frozen_tool in "
        "cuda_ftz_v841_authority_pins.json"
    ),
    "tools/run_real_gfs_cuda_x1_163842.py": (
        "the x1.163842 runner: its SHA-256 is EXPECTED_RUNNER_SHA256 in "
        "run_cuda_x1_163842_stabilized_products.py, which the certified "
        "x1.163842 evidence is checked against"
    ),
}


def test_the_sm120_pin_refuses_only_a_part_newer_than_sm120(monkeypatch, tmp_path):
    """What ``required_compute=(12, 0)`` does where it is still passed.

    ``require_cuda`` reads the pin only for a card at or above
    ``min_compute`` (12.0): a card below it is admitted first when NVRTC
    compiles for it, and then the pin is skipped.  So the pin passes sm_120
    and refuses only a newer part, sm_121 and anything after it, by a
    message that names no breakage.  It stays only on the call sites in
    PINNED_TO_SM120, whose bytes recorded pins hold, and this test holds
    what it does there until those pins move."""

    from woof.hex.cuda_backend.runtime import CudaRefusal, require_cuda

    targets = FAKE_NVRTC_TARGETS + (130,)
    for compute in ((12, 1), (13, 0)):
        _fake_cupy(monkeypatch, major=compute[0], minor=compute[1], targets=targets)
        with pytest.raises(CudaRefusal) as caught:
            require_cuda(
                min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
            )
        assert (
            f"cuda.compute_capability={compute[0]}.{compute[1]} != required 12.0"
            in str(caught.value)
        )
    for compute in ((7, 5), (8, 6), (8, 9), (10, 0), (12, 0)):
        _fake_cupy(monkeypatch, major=compute[0], minor=compute[1], targets=targets)
        capability = require_cuda(
            min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
        )
        assert capability.compute == compute
    # Without the pin the newer part runs, as it does at every other site.
    _fake_cupy(monkeypatch, major=12, minor=1, targets=targets)
    assert require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path)).compute == (12, 1)


def test_no_call_site_pins_one_architecture_except_the_named_ones():
    """``required_compute=(12, 0)`` refused every card that was not sm_120
    and named no breakage; with the open admission it refuses only parts
    newer than sm_120.  The forecast door, the driver, the proof harness and
    every tool whose bytes nothing pins no longer pass it.  It is left only
    where PINNED_TO_SM120 names the pin that holds the call's bytes; a call
    site anywhere else in the package or tools/ is the retired refusal
    coming back."""

    tools = TREE_ROOT / "tools"
    pinned = sorted(
        [
            path.relative_to(PACKAGE_DIR).as_posix()
            for path in PACKAGE_DIR.rglob("*.py")
            if "required_compute=(12, 0)" in path.read_text(encoding="utf-8")
        ]
        + [
            f"tools/{path.relative_to(tools).as_posix()}"
            for path in tools.rglob("*.py")
            if "required_compute=(12, 0)" in path.read_text(encoding="utf-8")
        ]
    )
    assert pinned == sorted(PINNED_TO_SM120), pinned
    # The scan reached the tools: without them it would pass on src alone.
    assert any(name.startswith("tools/") for name in pinned)


# --- what a receipt records ------------------------------------------------


def test_a_receipt_row_claims_only_the_anchor_its_architecture_holds():
    sm89 = arch_admission.architecture_status((8, 9)).as_dict()
    assert sm89["compute_capability"] == "8.9"
    assert sm89["anchor_status"] == "anchored"
    assert sm89["evidence_sha256"] == ADMITTED_BELOW_FLOOR[(8, 9)].evidence_sha256
    assert sm89["contract_receipt"] == "evidence/sm89-tier-20260924/RECEIPT.md"

    floor = arch_admission.architecture_status((12, 0)).as_dict()
    assert floor["anchor_status"] == "anchored"
    assert floor["anchor"] == "the proven contract floor"
    assert floor["evidence_sha256"] is None

    for compute in ((7, 5), (8, 0), (8, 7), (9, 0), (10, 0), (12, 1)):
        row = arch_admission.architecture_status(compute).as_dict()
        assert row["anchor_status"] == "unanchored", compute
        assert row["anchor"] is None
        assert row["evidence_sha256"] is None
        assert row["contract_receipt"] is None
        json.dumps(row)


def test_the_door_sentence_names_the_pin_or_says_unanchored():
    sm89 = arch_admission.architecture_status((8, 9)).describe()
    assert sm89.startswith("anchored: the 2026-09-24 anchor")
    assert ADMITTED_BELOW_FLOOR[(8, 9)].evidence_sha256[:16] in sm89
    sm75 = arch_admission.architecture_status((7, 5)).describe()
    assert sm75.startswith("unanchored:")
    assert "record it as unanchored" in sm75


# --- doctor reads the card ---------------------------------------------------


def _doctor_on(monkeypatch, probe):
    """``doctor``'s architecture finding for a card probe that answered
    ``probe`` (``None``: the probe could not run)."""

    from woof.hex import doctor

    monkeypatch.setattr(doctor, "_no_local_gpu", lambda: False)
    monkeypatch.setattr(doctor, "_card_probe", lambda: probe)
    findings = doctor.check_gpu_architecture()
    assert len(findings) == 1
    return doctor, findings[0]


def test_doctor_names_an_anchored_card_and_its_pin(monkeypatch):
    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "NVIDIA GeForce RTX 4090", "compute": [8, 9],
        "nvrtc_targets": list(FAKE_NVRTC_TARGETS),
    })
    assert finding.status == doctor.VERIFIED
    assert "NVIDIA GeForce RTX 4090, compute capability 8.9 (sm_89)" in finding.detail
    assert "admitted, anchored: the 2026-09-24 anchor" in finding.detail
    assert finding.evidence["anchor_status"] == "anchored"
    assert finding.evidence["evidence_sha256"] == ADMITTED_BELOW_FLOOR[(8, 9)].evidence_sha256
    assert "compute_89" in finding.evidence["nvrtc_targets"]
    assert not finding.remedy


@pytest.mark.parametrize("compute", [(7, 5), (8, 0), (10, 0), (12, 1)])
def test_doctor_reports_an_unanchored_card_and_never_as_a_gap(monkeypatch, compute):
    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "Some GPU", "compute": list(compute),
        "nvrtc_targets": list(FAKE_NVRTC_TARGETS),
    })
    assert finding.status == doctor.VERIFIED, finding
    assert f"compute capability {compute[0]}.{compute[1]}" in finding.detail
    assert "admitted, unanchored" in finding.detail
    assert "measures the numeric route" in finding.detail
    assert finding.evidence["anchor_status"] == "unanchored"
    assert finding.evidence["evidence_sha256"] is None
    assert not doctor.blocking_gaps([finding])
    compact = doctor.render([finding], explain=False)
    assert compact.startswith("OK") and "unanchored" in compact, compact


def test_doctor_names_the_proven_floor_without_a_pin(monkeypatch):
    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "NVIDIA GeForce RTX 5090", "compute": [12, 0],
        "nvrtc_targets": list(FAKE_NVRTC_TARGETS),
    })
    assert finding.status == doctor.VERIFIED
    assert "admitted, anchored: the proven contract floor" in finding.detail
    assert finding.evidence["evidence_sha256"] is None


def test_doctor_gap_is_only_a_card_nvrtc_cannot_compile_for(monkeypatch):
    """REFUSES.  The forecast door's own NVRTC refusal, reported at install
    time with the door's words; no pip command closes an older card."""

    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "Old GPU", "compute": [6, 1],
        "nvrtc_targets": list(FAKE_NVRTC_TARGETS),
    })
    assert finding.status == doctor.MISSING
    assert arch_admission.nvrtc_refusal((6, 1), FAKE_NVRTC_TARGETS) in finding.detail
    assert "none of them could be built" in finding.detail
    assert "NOT a pip problem" in finding.remedy
    assert "pip install" not in doctor.render([finding], explain=False)
    # Reported, not fatal: the render door needs no card.
    assert not finding.required and not doctor.blocking_gaps([finding])


def test_doctor_names_the_nvrtc_upgrade_for_a_card_newer_than_its_nvrtc(monkeypatch):
    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "Next GPU", "compute": [13, 0],
        "nvrtc_targets": list(FAKE_NVRTC_TARGETS),
    })
    assert finding.status == doctor.MISSING
    assert "not for compute_130" in finding.detail
    assert "pip install --upgrade nvidia-cuda-nvrtc" in doctor.render([finding], explain=False)


def test_doctor_reports_an_nvrtc_that_cannot_be_asked(monkeypatch):
    doctor, finding = _doctor_on(monkeypatch, {
        "device_count": 1, "name": "NVIDIA GeForce RTX 4090", "compute": [8, 9],
        "nvrtc_error": "OSError: libnvrtc.so.13: cannot open shared object file",
    })
    assert finding.status == doctor.MISSING
    assert "libnvrtc.so.13" in finding.detail and "no kernel could be built" in finding.detail
    floor = doctor.cuda_runtime_floor()
    extra = doctor._GPU_EXTRA_BY_MAJOR[floor // 1000]
    assert f"[{extra}]" in doctor.render([finding], explain=False)


@pytest.mark.parametrize(
    "probe",
    [None, {"device_count": 0}, {"error": "CUDARuntimeError: cudaErrorNoDevice"}],
)
def test_doctor_without_a_readable_card_is_context_not_a_gap(monkeypatch, probe):
    doctor, finding = _doctor_on(monkeypatch, probe)
    assert finding.status == doctor.INFO
    assert finding.detail.startswith("no card read")


def test_doctor_does_not_touch_a_card_on_a_no_gpu_box(monkeypatch):
    from woof.hex import doctor

    def _refuse():
        raise AssertionError("the card probe ran on a no-local-GPU box")

    monkeypatch.setattr(doctor, "_no_local_gpu", lambda: True)
    monkeypatch.setattr(doctor, "_card_probe", _refuse)
    (finding,) = doctor.check_gpu_architecture()
    assert finding.status == doctor.INFO and finding.detail.startswith("not read")


def test_doctor_reports_the_architecture_after_the_cuda_lane(monkeypatch):
    from woof.hex import doctor

    marker = doctor.Finding(subject="cupy (the CUDA lane)", status=doctor.INFO, detail="")
    for name in dir(doctor):
        if name.startswith("check_") and name != "check_gpu_architecture":
            monkeypatch.setattr(
                doctor, name, (lambda: [marker]) if name == "check_gpu_runtime" else (lambda: [])
            )
    monkeypatch.setattr(doctor, "_no_local_gpu", lambda: True)
    subjects = [finding.subject for finding in doctor.collect()]
    assert subjects == ["cupy (the CUDA lane)", "GPU architecture (the card and NVRTC)"]


def test_doctor_card_probe_reads_the_card_and_nvrtc_in_its_own_process(
    monkeypatch, tmp_path
):
    """The real subprocess, on a stand-in CuPy placed first on its path: it
    reads device 0 and asks NVRTC through arch_admission, and nothing else."""

    import importlib

    from woof.hex import doctor

    fake = tmp_path / "cupy"
    (fake / "cuda").mkdir(parents=True)
    (fake / "__init__.py").write_text("from . import cuda\n", encoding="utf-8")
    (fake / "cuda" / "__init__.py").write_text("from . import runtime, nvrtc\n", encoding="utf-8")
    (fake / "cuda" / "runtime.py").write_text(
        "is_hip = False\n"
        "def getDeviceCount():\n    return 1\n"
        "def getDeviceProperties(device):\n"
        "    return {'name': b'Stand-in GPU', 'major': 8, 'minor': 9}\n",
        encoding="utf-8",
    )
    (fake / "cuda" / "nvrtc.py").write_text(
        "def getSupportedArchs():\n    return (120, 75, 89)\n", encoding="utf-8"
    )
    top = importlib.import_module(doctor.__package__.split(".")[0])
    package_root = Path(top.__file__).resolve().parent.parent
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(package_root)]))
    probe = doctor._card_probe()
    assert probe == {
        "device_count": 1, "name": "Stand-in GPU", "compute": [8, 9],
        "nvrtc_targets": [75, 89, 120],
    }, probe


# --- the numeric route -----------------------------------------------------


def _anchored_route_bits():
    """The raw output an anchored card gives: the route flushes, the
    guarded fallback restores the subnormal, a*b+c separately rounded."""

    expected = arch_admission._expected_bits()
    sub = expected["subnormal"]
    f32 = (0, 0, sub, sub, expected["separately_rounded"],
           expected["x_times_y"], expected["x_plus_y"])
    f64 = (expected["dx_times_dy"], expected["dx_plus_dy"], expected["subnormal_in_fp64"])
    return list(f32), list(f64)


def test_the_anchored_route_classifies_as_the_anchored_route():
    f32, f64 = _anchored_route_bits()
    record = arch_admission.classify_numeric_route(tuple(f32), tuple(f64))
    assert record["ftz_route"] == "flush-to-zero"
    assert record["guarded_fallback"] == "ieee"
    assert record["contraction"] == "separately-rounded"
    assert record["normal_range"] == "ieee"
    assert record["matches_anchored_route"] is True
    assert arch_admission.numeric_route_refusal(record) is None


def test_the_contraction_witness_separates_fused_from_rounded():
    """The witness is exact: (1+2^-23)^2 - (1+2^-22) is 0 rounded twice and
    2^-46 fused."""

    expected = arch_admission._expected_bits()
    assert expected["separately_rounded"] == 0
    f32, f64 = _anchored_route_bits()
    f32[4] = struct.unpack("<I", struct.pack("<f", 2.0**-46))[0]
    record = arch_admission.classify_numeric_route(tuple(f32), tuple(f64))
    assert record["contraction"] == "not-separately-rounded"
    assert record["matches_anchored_route"] is False
    # A contracted a*b+c moves a result by one rounding: recorded, not refused.
    assert arch_admission.numeric_route_refusal(record) is None


def test_an_ieee_route_is_recorded_not_refused():
    f32, f64 = _anchored_route_bits()
    sub = arch_admission._expected_bits()["subnormal"]
    f32[0] = f32[1] = sub
    record = arch_admission.classify_numeric_route(tuple(f32), tuple(f64))
    assert record["ftz_route"] == "ieee"
    assert arch_admission.numeric_route_refusal(record) is None


def test_ordinary_arithmetic_that_is_not_ieee_refuses_naming_it():
    f32, f64 = _anchored_route_bits()
    f32[5] ^= 1
    record = arch_admission.classify_numeric_route(tuple(f32), tuple(f64))
    assert record["normal_range"] == "differs"
    refusal = arch_admission.numeric_route_refusal(record)
    assert refusal is not None
    assert "IEEE 754" in refusal and "every number the run produced" in refusal


def test_numeric_route_on_the_real_card():
    """Run the probe on the card in hand: one compile, two launches."""

    cupy = pytest.importorskip("cupy")
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("no CUDA device")
    if count < 1:
        pytest.skip("no CUDA device")
    record = arch_admission.measure_numeric_route()
    assert record["normal_range"] == "ieee"
    assert record["dual_run_identical"] is True
    assert record["schema"] == arch_admission.NUMERIC_ROUTE_SCHEMA
    compute = tuple(
        int(v) for v in (
            cupy.cuda.runtime.getDeviceProperties(0)["major"],
            cupy.cuda.runtime.getDeviceProperties(0)["minor"],
        )
    )
    if arch_admission.architecture_status(compute).anchored:
        # An anchored architecture measured this route when it was anchored.
        assert record["matches_anchored_route"] is True, record


# --- the contraction pin travels with every architecture -----------------


def test_kernel_cache_contraction_pin_is_architecture_independent():
    from woof.hex import numerics
    from woof.hex.cuda_backend.runtime import KernelCache

    # The NVRTC contraction pin is part of the strict numerical contract on
    # every architecture, not an sm_120 accident.  The cache's default base is
    # the process's numerics mode (fast by default, strict the opt-in: owner
    # ruling 2026-10-16), and the probe compiles with the same mode.
    assert KernelCache.__init__.__kwdefaults__["base_options"] is None
    assert numerics.kernel_base("strict") == ("--std=c++17", "--fmad=false")
    assert arch_admission.NUMERIC_ROUTE_OPTIONS == numerics.kernel_base("strict")


# --- registry entries are records, not switches --------------------------


def test_every_registered_anchor_carries_its_evidence_in_tree(receipts):
    for compute, anchor in ADMITTED_BELOW_FLOOR.items():
        assert tuple(compute) == tuple(anchor.compute)
        assert tuple(anchor.compute) < PROVEN_COMPUTE, (
            "anchors exist only below the proven floor"
        )
        receipt = TREE_ROOT / anchor.contract_receipt
        authority = TREE_ROOT / anchor.authority_anchor
        assert receipt.exists(), (
            f"{anchor.sm} names a contract receipt that is not in the tree: "
            f"{anchor.contract_receipt}"
        )
        assert authority.exists(), (
            f"{anchor.sm} names an authority anchor that is not in the tree: "
            f"{anchor.authority_anchor}"
        )
        assert anchor.card and anchor.admitted_on and anchor.basis


def test_the_sm89_anchor_names_the_campaign_and_the_card_that_earned_it():
    """sm_89 was anchored 2026-09-24 on the RTX 4090 by the anchor campaign,
    whose runner the pinned evidence carries as
    evidence/sm89-tier-20260924/instruments/arch_anchor_campaign.py.  The
    row is a record: it names the card and driver the campaign read from
    nvidia-smi and the evidence directory the campaign wrote, and it anchors
    nothing else."""

    anchor = arch_admission.architecture_anchor((8, 9))
    assert anchor is not None and anchor.sm == "sm_89"
    assert anchor.admitted_on == "2026-09-24"
    assert anchor.contract_receipt == "evidence/sm89-tier-20260924/RECEIPT.md"
    assert anchor.authority_anchor == "evidence/sm89-tier-20260924/authority"
    assert "RTX 4090" in anchor.card and "128 SM" in anchor.card
    assert "driver 610.57.04" in anchor.card
    assert "arch_anchor_campaign" in anchor.basis
    # The runner is cited where the evidence the row pins carries it; the
    # tools/ path it ran from is not in this tree.
    assert f"({anchor.evidence_directory}/instruments/arch_anchor_campaign.py)" in anchor.basis
    assert "tools/arch_anchor_campaign.py" not in anchor.basis
    assert "sm_89" in arch_admission.admitted_summary()
    # The sm_89 row anchors nothing else.
    assert arch_admission.architecture_anchor((8, 0)) is None
    assert arch_admission.architecture_anchor((8, 7)) is None
    assert arch_admission.architecture_anchor((9, 0)) is None


# --- the real card, when one is present ----------------------------------


def test_real_device_admission_state_is_truthful():
    cupy = pytest.importorskip("cupy")
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("no CUDA device")
    if count < 1:
        pytest.skip("no CUDA device")
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    compute = (int(properties["major"]), int(properties["minor"]))

    from woof.hex.cuda_backend.runtime import require_cuda

    # Any card this NVRTC compiles for is admitted; its status is a record.
    capability = require_cuda(min_compute=(12, 0))
    assert capability.compute == compute
    status = arch_admission.architecture_status(compute)
    assert status.anchored == (
        compute == PROVEN_COMPUTE
        or arch_admission.architecture_anchor(compute) is not None
    )


def test_json_receipt_paths_are_relative(tmp_path):
    anchor = _anchor_sm86(tmp_path)
    record = anchor.as_dict()
    assert record["sm"] == "sm_86"
    assert not Path(record["contract_receipt"]).is_absolute()
    json.dumps(record)  # a receipt row must serialize as-is


# ---------------------------------------------------------------------------
# the FTZ guard-cost timing ceiling is per-architecture (stale-guard audit, finding 8)
# ---------------------------------------------------------------------------
def test_performance_ceiling_registry_carries_both_admitted_architectures():
    """The single global 1.25x ceiling was calibrated when sm_120 was the
    only architecture; on the admitted sm_86 tier the guard cost MEASURES
    1.47-1.57x at transport_edge_values while bitwise identity holds, so a
    global 1.25 hard-refuses a tier the port admits.  Each architecture
    carries its own measured/recorded row."""

    registry = arch_admission.PERFORMANCE_RATIO_CEILINGS
    assert arch_admission.performance_ratio_ceiling("sm_120") == 1.25
    assert arch_admission.performance_ratio_ceiling("sm_86") == 1.75
    sm86 = registry["sm_86"]
    # The row cites the recorded deviation it is set from, not taste.
    assert "transport_edge_values" in sm86["basis"]
    assert "1.471975" in sm86["basis"] and "1.565028" in sm86["basis"]
    assert "perf-control-stability-sm86" in sm86["basis"]
    assert "follow-up" in sm86["basis"], (
        "the sm_86 ceiling is set from the recorded band, not a fresh "
        "calibration; the calibration run must stay named"
    )
    for row in registry.values():
        assert row["basis"], "a ceiling with no basis is an asserted constant"


def test_the_sm89_ceiling_is_this_cards_own_reading_not_a_borrowed_row():
    """sm_89's guard-cost ceiling comes from readings taken on the RTX 4090,
    and says which; it is not the sm_120 row and not sm_86's.  Since the
    2026-09-24 evidence review it rests on a cross-process calibration (10
    processes, 200 readings); the first 24 readings from one warm process
    set 1.35, which 5 of the 200 exceed."""

    ceiling = arch_admission.performance_ratio_ceiling("sm_89")
    basis = arch_admission.PERFORMANCE_RATIO_CEILINGS["sm_89"]["basis"]
    assert 1.25 <= ceiling < arch_admission.performance_ratio_ceiling("sm_86")
    assert "RTX 4090" in basis and "transport.transport_edge_values" in basis
    assert "calibration receipt perf-calibration-sm89.json" in basis
    assert "separate processes" in basis and "1.392857" in basis
    assert "sm89-tier-20260924/contract/perf-calibration-sm89.json" in basis and "1.285714" in basis
    assert "sm86-tier" not in basis and "3080" not in basis
    # 52 of 200 readings on this card breached the proven-floor 1.25x, so
    # the borrowed sm_120 row would refuse this card on timing alone.
    assert ceiling > arch_admission.performance_ratio_ceiling("sm_120")


def test_unregistered_architecture_ceiling_refuses_by_name():
    with pytest.raises(LookupError) as caught:
        arch_admission.performance_ratio_ceiling("sm_99")
    message = str(caught.value)
    assert "sm_99" in message
    assert "sm_120" in message and "sm_86" in message, (
        "the refusal must name the registered roster"
    )


def test_cuda_ftz_constant_is_the_sm120_registry_row():
    from woof.hex import cuda_ftz

    assert cuda_ftz.PERFORMANCE_RATIO_CEILING == (
        arch_admission.performance_ratio_ceiling("sm_120")
    )


def test_binding_claim_bytes_are_stable_for_sm120():
    """Saved sm_120 bindings are validated by canonical re-hash: the claim
    string rebuilt for capability 120 must reproduce the pre-change bytes
    exactly, or every archived receipt goes red on a text edit."""

    from woof.hex import cuda_ftz

    claim = cuda_ftz._mpas_ftz_claim("120", 1.25)
    assert "declared 1.25x median ceiling" in claim
    assert claim == (
        "The five MPAS RawModule translation units execute under the same "
        "measured terminal -ftz=true route for which woof's "
        "sm_120 probe "
        "observes FP32 DAZ/FTZ. The production transport deck verifies "
        "the guarded subnormal-only FP64 fallback at all 12 transport "
        "kernels and 44 answer-changing non-transport arithmetic classes. "
        "Eight copy/invariant/native-FP64 classes stay green, all 44 "
        "disabled-fallback controls go red. Five named representative "
        "normalized-kernel microbenchmarks remain bitwise identical and "
        "each stays below the declared 1.25x median ceiling; that timing "
        "ceiling is not a whole-step or all-guarded-kernel claim."
    )


# ---------------------------------------------------------------------------
# every registered anchor is pinned to its evidence, and the evidence says
# what the row says
# ---------------------------------------------------------------------------
# THE BREAKAGE THIS PREVENTS.  The only evidence test used to check that the
# receipt file and the authority directory EXIST.  A mutation run against the
# sm_89 receipts (2026-09-24 review) found 9 of 11 edits passing it: the
# authority digest file deleted or one of its digests changed, VERDICT.json
# flipped to refuse, the calibration file or the whole contract/ directory
# deleted, the calibration maximum raised above the ceiling, RECEIPT.md
# emptied, the card changed to another GPU, the determinism record set to
# not identical.  An entry resting on edited evidence would still admit its
# architecture.  These tests open the evidence: each registered row is pinned
# to one exact record (``evidence_sha256``), and in a tree that carries the
# receipts the record has to say what the row says: the card and compute
# capability, the FTZ route grid and the contraction pin, the contract decks,
# the authority digests and their twice-reproduced status, and the readings
# the guard-cost ceiling is set from.
#
# A second mutation run (2026-09-24, content tests only, the pin re-set to
# each mutated directory) found 16 of 18 further edits passing: a frame digest
# changed consistently in the three summary files while both runs still record
# the real hash; one run's own snapshot hash changed; one run's receipt
# replaced by a copy of the other's; one run recording 180 of 360 steps or
# another host; the P3 companion pair not identical; and, in the review-fix
# addendum, a regional contract, a transport deck, an engine FTZ receipt or a
# door preflight saying it failed, the addendum's 45-entrypoint audit losing
# the Smagorinsky record it exists to supply, a leg's rc disagreeing with the
# addendum's table, a raw calibration process file raised or deleted.  So the
# twice-reproduced claim is now read from each door run's own records (its
# driver receipt, its DOOR line, the chain's leg window), and every receipt
# outside attempts/ is held to its own verdict.
#
# A third mutation run (2026-09-24, same method) found 15 of 22 further edits
# passing, because a receipt's summary flags were read and the records under
# them were not.  A regional contract kept "8/8 decks bitwise" with one
# payload's device hash no longer its host hash, a pass-2 payload not
# bitwise, or a declared kernel dropped from every deck.  A guard-cost
# reading could record an enabled/disabled identity break.  The engine FTZ
# receipt's route grid and its filed artifacts were never read.  The record
# session's chain.log rcs, VERDICT.json's states and its record count were
# never reconciled.  An instrument could differ from its SHA256SUMS and from
# the runner hash its identity records, a forecast receipt's driver seal was
# never recomputed, and the nvidia-smi census was never parsed.  So every
# payload a regional contract carries is held to its hash, every identity
# claim in a timing receipt is held true, the engine receipt is held to its
# route grid and its artifacts, the verdict is reconciled with the chain and
# its own receipt, and the instruments, the driver seals and the census are
# checked against what recorded them.
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_TIMING_BREACH = re.compile(r"regressed at \S+: ([0-9.]+)x >")
_NAMED_JSON = re.compile(r"evidence/[A-Za-z0-9_./-]+?\.json")
#: A calibration a ceiling's basis cites by its file name inside a
#: calibration/ folder of the evidence its row pins, so the row text names no
#: evidence folder that is not public.
_NAMED_CALIBRATION = re.compile(
    r"calibration receipt (perf-calibration-[A-Za-z0-9_.-]+?\.json)"
)
_QUOTED_READING = re.compile(r"\b(\d\.\d{6})x")
_CAPABILITY_KEYS = (
    "compute_capability", "compute_cap", "compute", "sm", "device_compute_capability",
)
_DOOR_LINE = re.compile(
    r"^DOOR mesh=(?P<mesh>\S+) steps=(?P<steps>\d+) frames=(?P<frames>\d+) "
    r"status=(?P<status>\S+) out=(?P<out>\S+)[ \t]*$",
    re.MULTILINE,
)
_CHAIN_LEG = re.compile(
    r"^\[(?P<utc>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] (?P<leg>[\w.-]+): "
    r"(?:(?P<start>start)\b|rc=(?P<rc>-?\d+)\b)",
    re.MULTILINE,
)
_SESSION_ROW = re.compile(r"^\| (?P<leg>[a-z0-9-]+) \| (?P<rc>-?\d+) \| (?P<result>.*) \|[ \t]*$",
                          re.MULTILINE)
_AUDIT_COUNTS = re.compile(
    r"(?P<kernels>\d+) (?:release-specific )?entrypoints "
    r"\((?P<guarded>\d+) guarded, (?P<invariant>\d+) invariant\)"
)
_NOW_RECORDED = re.compile(r"((?:`\w+`(?:, | and ))*`\w+`) now ha(?:s|ve) an on-card record")
_CHAIN_SKIPPED = re.compile(r"^\[[^\]]+\] (?P<leg>[\w.-]+): SKIPPED\b", re.MULTILINE)
_SUMS_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<name>\S.*)")
_SMI_DRIVER = re.compile(r"NVIDIA-SMI (?P<driver>\S+)")
_SMI_DEVICE = re.compile(
    r"^\|\s+(?P<index>\d+)\s+(?P<name>\S.*?)\s+(?:On|Off)\s+\|\s+"
    r"(?P<bus>[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])\s",
    re.MULTILINE,
)
_FAMILY_ROW = re.compile(
    r"^\| (?P<family>[a-z0-9-]+) \| (?P<records>\d+)(?: device witnesses)? \| "
    r"(?P<verdict>[a-z-]+) \| (?P<why>.*) \|[ \t]*$",
    re.MULTILINE,
)
#: Every key under which a guard-cost receipt states that the FP64 fallback,
#: enabled against disabled, left the answer bitwise unchanged.
_IDENTITY_KEYS = frozenset((
    "identical",
    "all_identical",
    "identity_held_every_reading",
    "all_normalized_outputs_bitwise_identical",
    "normalized_output_bitwise_identical",
))
#: The engine FTZ probe's route grid, as every campaign receipt states it:
#: the loader RawModule, its explicit --ftz=true control and the CuPy
#: reduction flush; direct NVRTC under --ftz=false keeps IEEE results; inline
#: PTX without .ftz keeps them where it can express the mechanism at all.
_ENGINE_ROUTE_GRID = {
    "R1": ("flush-to-zero",),
    "R1-ftztrue": ("flush-to-zero",),
    "R2": ("ieee-agreement",),
    "R3": ("ieee-agreement",),
    "R4": ("flush-to-zero",),
    "R5": ("ieee-agreement", "not-applicable"),
}
#: The decks tool's decks, each with the verdict family (or the supplementary
#: entry) that has to account for it when it does not measure.  The tool
#: exits 0 only when every deck measured.
_DECK_FAMILY = {
    "transport_deck": "transport-deck",
    "guarded_kernel_audit": "guarded-audit",
    "normalized_performance_control": "normalized-control",
    "v841_kernel_audit": "v841-four-pass",
    "compile_manifest_v823": "compile-manifest-v823",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _driver_seal(document) -> str:
    """The forecast driver's seal over its own receipt: the SHA-256 of the
    canonical JSON (sorted keys, no whitespace, UTF-8, no NaN) of everything
    but the seal.  Restated here, as the ceiling rule is, so a tree that
    carries the receipts recomputes it without the driver."""

    body = {key: value for key, value in document.items() if key != "payload_sha256"}
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identity_claims(node, where: str = ""):
    """(where, value) for every boolean an identity key holds in a receipt."""

    if isinstance(node, dict):
        for key, value in node.items():
            if key in _IDENTITY_KEYS and isinstance(value, bool):
                yield f"{where}/{key}", value
            else:
                yield from _identity_claims(value, f"{where}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _identity_claims(value, f"{where}[{index}]")


def _rowed_readings(node, where: str = ""):
    """(where, reading) for every guard-cost reading that states a maximum
    over per-benchmark rows."""

    if isinstance(node, dict):
        rows = node.get("rows")
        if isinstance(rows, dict) and rows and "maximum" in node and all(
            isinstance(row, dict) and "ratio" in row for row in rows.values()
        ):
            yield where, node
        for key, value in node.items():
            yield from _rowed_readings(value, f"{where}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _rowed_readings(value, f"{where}[{index}]")


def _filed(directory: Path, pattern: str):
    """(relative path, document) for every JSON receipt matching ``pattern``
    that is not a kept failed attempt (attempts/ is history, not a claim)."""

    for path in sorted(directory.rglob(pattern)):
        relative = path.relative_to(directory)
        if "attempts" not in relative.parts:
            yield relative.as_posix(), _json(path)


def _numbers(node):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield float(node)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _numbers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numbers(value)


def _strings_under(node, key: str):
    """Every string value stored under ``key`` anywhere in a receipt."""

    if isinstance(node, dict):
        for name, value in node.items():
            if name == key and isinstance(value, str):
                yield value
            else:
                yield from _strings_under(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _strings_under(value, key)


def _leg_windows(chain_log: str) -> dict[str, tuple[str, str, int]]:
    """leg -> (start UTC, end UTC, rc), as the campaign chain logged it."""

    starts: dict[str, str] = {}
    windows: dict[str, tuple[str, str, int]] = {}
    for match in _CHAIN_LEG.finditer(chain_log):
        if match["start"]:
            starts[match["leg"]] = match["utc"]
        elif match["leg"] in starts:
            windows[match["leg"]] = (starts[match["leg"]], match["utc"], int(match["rc"]))
    return windows


def _card_name(anchor: ArchAnchor) -> str:
    return anchor.card.split(" (", 1)[0]


def _card_driver(anchor: ArchAnchor) -> str:
    return anchor.card.rsplit("driver ", 1)[-1]


def _timing_values(node) -> list[float]:
    """Every guard-cost ratio a timing receipt records, breaches included."""

    found: list[float] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("ratio", "enabled_over_disabled", "maximum") \
                    and isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append(float(value))
            elif isinstance(value, str):
                found += [float(m) for m in _TIMING_BREACH.findall(value)]
            else:
                found += _timing_values(value)
    elif isinstance(node, list):
        for value in node:
            found += _timing_values(value)
    return found


def _timing_receipts(directory: Path) -> list[Path]:
    """The files in an evidence directory that carry guard-cost readings."""

    return sorted(
        path for path in directory.rglob("*.json")
        if path.name.startswith(("perf-", "arch-ftz-decks-", "process-"))
        or path.name == "VERDICT.json"
    )


def _cited_receipts(anchor: ArchAnchor, basis: str) -> list[str]:
    """The receipts a ceiling's basis cites, as paths relative to the tree.

    A basis cites a receipt by its path under evidence/, or a calibration by
    its file name inside a calibration/ folder of the evidence its row pins.
    Every test that reads a basis resolves both the same way, and a
    calibration named by file name has to be exactly one file of that
    evidence.
    """

    cited = set(_NAMED_JSON.findall(basis))
    directory = TREE_ROOT / anchor.evidence_directory
    for name in _NAMED_CALIBRATION.findall(basis):
        found = sorted(
            path.relative_to(TREE_ROOT).as_posix()
            for path in directory.glob(f"**/calibration/{name}")
        )
        assert len(found) == 1, (
            f"{anchor.sm}'s ceiling cites the calibration receipt {name}; the "
            f"evidence under {anchor.evidence_directory} holds {len(found)}: {found}"
        )
        cited.update(found)
    return sorted(cited)


def _capability_of(key: str, value) -> tuple[int, int] | None:
    """The compute capability one device field states, or None."""

    if key == "sm":
        if isinstance(value, str) and re.fullmatch(r"sm_\d{2,3}", value):
            return int(value[3:-1]), int(value[-1])
        return None
    if key == "device_compute_capability":
        text = str(value)
        return (int(text[:-1]), int(text[-1])) if re.fullmatch(r"\d{2,3}", text) else None
    if isinstance(value, str) and re.fullmatch(r"\d{1,2}\.\d", value):
        major, minor = value.split(".")
        return int(major), int(minor)
    if isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value):
        return value[0], value[1]
    return None


def _device_records(node, where: str = ""):
    """(where, card name or None, capability, nvidia-smi driver or None) for
    every mapping in a receipt that states a compute capability."""

    if isinstance(node, dict):
        name = node.get("name") if isinstance(node.get("name"), str) else None
        driver = node.get("driver_version")
        driver = driver if isinstance(driver, str) and "." in driver else None
        for key in _CAPABILITY_KEYS:
            if key in node:
                capability = _capability_of(key, node[key])
                if capability is not None:
                    yield f"{where}/{key}", name, capability, driver
        for key, value in node.items():
            yield from _device_records(value, f"{where}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _device_records(value, f"{where}[{index}]")


def _ceiling_rule(maxima: list[float], floor: float = 1.25):
    """The registry's ceiling rule: the top reading plus the spread of the
    readings above the proven-floor ceiling, rounded up to 0.05 (the floor
    itself when none breaches it).  Restated here so a tree that carries the
    receipts recomputes a calibrated ceiling without the campaign runner."""

    breaches = [value for value in maxima if value > floor]
    if not breaches:
        return floor, 0, max(maxima), 0.0
    top = max(breaches)
    spread = top - min(breaches)
    ceiling = round(math.ceil(round((top + spread) / 0.05, 9)) * 0.05, 2)
    return ceiling, len(breaches), top, spread


def _campaign_anchors(receipts_root: Path):
    for anchor in ADMITTED_BELOW_FLOOR.values():
        directory = TREE_ROOT / anchor.evidence_directory
        if (directory / "VERDICT.json").is_file():
            yield anchor, directory


def test_every_registered_anchor_pins_the_evidence_it_rests_on():
    for anchor in ADMITTED_BELOW_FLOOR.values():
        assert _SHA256_HEX.fullmatch(anchor.evidence_sha256), (
            f"{anchor.sm} names no evidence digest: a tree without receipts "
            f"could not say which record admitted it"
        )
        assert anchor.evidence_directory == Path(anchor.authority_anchor).parent.as_posix()
        assert anchor.evidence_directory == f"evidence/{anchor.sm.replace('_', '')}-tier-" \
            f"{anchor.admitted_on.replace('-', '')}"


def test_the_evidence_digest_moves_with_any_file(tmp_path):
    (tmp_path / "authority").mkdir()
    (tmp_path / "RECEIPT.md").write_text("receipt\n", encoding="utf-8")
    (tmp_path / "authority" / "digests.json").write_text("{}\n", encoding="utf-8")
    first = arch_admission.evidence_digest(tmp_path)
    assert _SHA256_HEX.fullmatch(first)
    (tmp_path / "authority" / "digests.json").write_text("{ }\n", encoding="utf-8")
    assert arch_admission.evidence_digest(tmp_path) != first
    (tmp_path / "authority" / "digests.json").write_text("{}\n", encoding="utf-8")
    assert arch_admission.evidence_digest(tmp_path) == first
    (tmp_path / "authority" / "README.md").write_text("added\n", encoding="utf-8")
    assert arch_admission.evidence_digest(tmp_path) != first
    (tmp_path / "authority" / "README.md").unlink()
    (tmp_path / "authority" / "digests.json").rename(tmp_path / "digests.json")
    assert arch_admission.evidence_digest(tmp_path) != first, "a moved file moves it"
    (tmp_path / "digests.json").rename(tmp_path / "authority" / "digests.json")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert arch_admission.evidence_digest(tmp_path) == first
    with pytest.raises(FileNotFoundError):
        arch_admission.evidence_digest(tmp_path / "absent")


def test_every_registered_anchor_matches_its_evidence_byte_for_byte(receipts):
    for anchor in ADMITTED_BELOW_FLOOR.values():
        directory = TREE_ROOT / anchor.evidence_directory
        assert arch_admission.evidence_digest(directory) == anchor.evidence_sha256, (
            f"{anchor.sm}: the evidence under {anchor.evidence_directory} is not "
            f"the evidence the registry row was admitted on"
        )


def test_every_registered_ceiling_covers_every_reading_its_card_recorded(receipts):
    from woof.hex.cuda_backend.arch_admission import PERFORMANCE_RATIO_CEILINGS

    for anchor in ADMITTED_BELOW_FLOOR.values():
        ceiling = arch_admission.performance_ratio_ceiling(anchor.sm)
        basis = PERFORMANCE_RATIO_CEILINGS[anchor.sm]["basis"]
        named = _cited_receipts(anchor, basis)
        assert named, f"{anchor.sm}'s ceiling names no timing receipt"
        cited: list[float] = []
        for relative in named:
            assert (TREE_ROOT / relative).is_file(), (
                f"{anchor.sm}'s ceiling cites {relative}, which is not in the tree"
            )
            cited += _timing_values(_json(TREE_ROOT / relative))
        # Every reading the basis quotes is one its cited receipts record.
        recorded = {f"{value:.6f}" for value in cited}
        for quoted in _QUOTED_READING.findall(basis):
            assert quoted in recorded, (
                f"{anchor.sm}'s ceiling quotes {quoted}x, which no receipt it cites records"
            )
        directory = TREE_ROOT / anchor.evidence_directory
        readings = [
            (value, path.relative_to(directory).as_posix())
            for path in _timing_receipts(directory)
            for value in _timing_values(_json(path))
        ]
        assert readings, f"{anchor.sm}: no timing reading found in its evidence"
        over = [(value, where) for value, where in readings if value > ceiling]
        assert not over, (
            f"{anchor.sm}'s registered ceiling {ceiling} is below readings its own "
            f"card recorded: {over}"
        )


def test_a_calibrated_ceiling_is_its_rule_applied_to_its_calibration(receipts):
    """A ceiling set from a cross-process calibration recomputes from it:
    the processes are separate (their own pid and kernel cache), each names
    the anchor's card, identity held in every reading, and the figures the
    basis states are the calibration's."""

    from woof.hex.cuda_backend.arch_admission import PERFORMANCE_RATIO_CEILINGS

    calibrated = set()
    for anchor in ADMITTED_BELOW_FLOOR.values():
        basis = PERFORMANCE_RATIO_CEILINGS[anchor.sm]["basis"]
        for relative in _cited_receipts(anchor, basis):
            document = _json(TREE_ROOT / relative)
            if not isinstance(document.get("processes"), list):
                continue
            calibrated.add(anchor.sm)
            processes = document["processes"]
            assert len(processes) >= 2, f"{relative}: one process is not a calibration"
            maxima: list[float] = []
            for process in processes:
                device = process["device"]
                assert device["name"] == _card_name(anchor), (relative, device["name"])
                assert device["sm"] == anchor.sm and device["multiprocessors"] > 0
                assert device["nvidia_smi"]["driver_version"] == _card_driver(anchor)
                for reading in process["readings"]:
                    rows = reading["rows"].values()
                    assert reading["all_identical"] is True
                    assert all(row["identical"] is True for row in rows)
                    assert reading["maximum"] == max(row["ratio"] for row in rows)
                    maxima.append(float(reading["maximum"]))
            assert len({p["device"]["pid"] for p in processes}) == len(processes)
            assert len({p["device"]["kernel_cache"] for p in processes}) == len(processes)
            # The pooled record is the processes' own files, every one of
            # them, and every process exited cleanly.
            assert document.get("process_rcs") == [0] * len(processes), (
                f"{relative}: process rcs {document.get('process_rcs')}"
            )
            raw = (TREE_ROOT / relative).parent / "processes"
            names = sorted(path.name for path in raw.glob("process-*.json"))
            assert names == [f"process-{n:02d}.json" for n in range(1, len(processes) + 1)], (
                f"{relative}: pools {len(processes)} processes, {raw} holds {names}"
            )
            for number, process in enumerate(processes, start=1):
                assert _json(raw / f"process-{number:02d}.json") == process, (
                    f"{relative}: processes[{number - 1}] is not process-{number:02d}.json "
                    f"as that process wrote it"
                )
            ceiling, breaches, top, spread = _ceiling_rule(maxima)
            assert arch_admission.performance_ratio_ceiling(anchor.sm) == ceiling, (
                f"{anchor.sm}: {relative} gives {ceiling} by the rule"
            )
            for figure in (
                f"{len(maxima)} readings",
                f"{len(processes)} separate processes",
                f"median {statistics.median(maxima):.3f}x",
                f"top {top:.6f}x",
                f"{breaches} readings breached 1.25x",
                f"(~{spread:.3f})",
            ):
                assert figure in basis, f"{anchor.sm}'s basis does not state {figure!r}"
    assert "sm_89" in calibrated, "sm_89's ceiling must rest on its cross-process calibration"


def test_the_sm86_evidence_is_the_legacy_layout_and_says_sm86(receipts):
    anchor = ADMITTED_BELOW_FLOOR[(8, 6)]
    directory = TREE_ROOT / anchor.evidence_directory
    # sm_86 predates the campaign runner: no VERDICT.json, a chain script's
    # receipt.  Its content is held by the digest; its face is checked here.
    assert not (directory / "VERDICT.json").exists()
    receipt = (directory / anchor.contract_receipt.split("/")[-1]).read_text(encoding="utf-8")
    assert receipt.startswith("# The sm_86 tier") and "RTX 3080" in receipt
    assert "driver 610.74" in receipt and "driver 610.74" in anchor.card
    digests = _json(directory / "authority" / "x1-forecast-sm86-masked-digests.json")
    assert digests and all(frame.get("variables") for frame in digests.values())


def test_a_campaign_anchor_verdict_names_its_architecture_and_card(receipts):
    anchors = list(_campaign_anchors(receipts))
    assert {anchor.sm for anchor, _ in anchors} >= {"sm_89"}
    for anchor, directory in anchors:
        verdict = _json(directory / "VERDICT.json")
        assert verdict["verdict"] == "admit", f"{anchor.sm}: VERDICT.json does not admit"
        assert verdict["sm"] == anchor.sm
        assert verdict["compute"] == f"{anchor.compute[0]}.{anchor.compute[1]}"
        assert verdict["tier"] == directory.name and not verdict["failures"]
        card = verdict["identity"]["card"]
        assert card["available"] is True
        assert card["compute_cap"] == verdict["compute"]
        assert card["name"] == _card_name(anchor)
        assert card["driver_version"] == _card_driver(anchor)
        total = int(str(card["memory_total"]).split()[0])
        assert f"{total:,} MiB" in anchor.card
        for family, record in verdict["families"].items():
            assert record["verdict"] in ("identical", "deviation"), (anchor.sm, family)
            if record["verdict"] == "deviation":
                assert any(item.startswith(family + ":") for item in verdict["deviations"])
        receipt = (directory / "RECEIPT.md").read_text(encoding="utf-8")
        assert receipt.startswith(f"# The {anchor.sm} tier")
        assert f"Card: {card['name']}, compute {verdict['compute']}" in receipt
        assert f"driver {card['driver_version']}" in receipt
        assert f"**{anchor.sm} earned its anchor.**" in receipt


def test_every_device_record_in_a_campaign_anchor_evidence_is_its_card(receipts):
    """Every mapping in the evidence that states a compute capability states
    the anchor's; every one that also names a card names the anchor's card
    (and its nvidia-smi driver, where it records one); and every host a
    receipt records is the host the campaign's identity read."""

    for anchor, directory in _campaign_anchors(receipts):
        host = _json(directory / "VERDICT.json")["identity"]["host"]
        assert host, f"{anchor.sm}: the campaign identity names no host"
        named = 0
        for path in sorted(directory.rglob("*.json")):
            document = _json(path)
            for recorded in _strings_under(document, "host"):
                assert recorded == host, (
                    f"{anchor.sm}: {path.relative_to(directory).as_posix()} records host "
                    f"{recorded!r}, the campaign ran on {host!r}"
                )
            for where, name, capability, driver in _device_records(document):
                label = f"{path.relative_to(directory).as_posix()}{where}"
                assert capability == anchor.compute, f"{anchor.sm}: {label} states {capability}"
                if name is not None:
                    named += 1
                    assert name == _card_name(anchor), f"{anchor.sm}: {label} names {name!r}"
                if driver is not None:
                    assert driver == _card_driver(anchor), f"{anchor.sm}: {label} driver {driver}"
        assert named >= 9, f"{anchor.sm}: only {named} device records name a card"


def test_a_campaign_anchor_measured_its_ftz_route_grid_and_contraction_pin(receipts):
    from woof.hex.cuda_backend.runtime import KernelCache

    port_options = list(KernelCache.__init__.__kwdefaults__["base_options"])
    preserved = {"nvrtc_ftz_false_REPO_ROUTE", "nvrtc_default_no_flag"}
    flushed = {"nvrtc_ftz_true_NEGATIVE_CTL", "cupy_elementwise_arraymul", "port_kernelcache_route"}
    forms = {
        "nvrtc_fmad_false": "unfused",
        "port_kernelcache_route": "unfused",
        "nvrtc_fmad_true_CONTROL": "fused",
    }
    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        probes = dict(_filed(directory, f"route-probe-{tag}.json"))
        # The session of record's probe, and any later session's on the same
        # card, each hold the whole grid.
        assert f"contract/route-probe-{tag}.json" in probes, (anchor.sm, sorted(probes))
        for relative, probe in probes.items():
            where = f"{anchor.sm} {relative}"
            assert probe["device"]["name"] == _card_name(anchor), where
            assert _capability_of("compute_capability", probe["device"]["compute_capability"]) \
                == anchor.compute, where
            ftz = probe["ftz"]
            reference = ftz["host_numpy_reference"]
            inputs = ftz["inputs_float32"]
            assert inputs[0] < 1.17549435e-38 and inputs[1] < 1.17549435e-38, "two subnormal inputs"
            arms = {name: arm for name, arm in ftz.items() if isinstance(arm, dict)}
            assert set(arms) == preserved | flushed, (where, sorted(arms))
            for name, arm in arms.items():
                assert arm["status"] == "measured" and arm["dual_run_byte_identical"] is True, \
                    (where, name)
                assert arm["subnormals_preserved"] is (name in preserved), (where, name)
                outputs = [arm["mul_by_one"]] + ([arm["add_zero"]] if "add_zero" in arm else [])
                for output in outputs:
                    if name in preserved:
                        assert output == reference, (where, name)
                    else:
                        assert output[:2] == [0.0, 0.0] and output[2:] == reference[2:], \
                            (where, name)
            assert arms["port_kernelcache_route"]["base_options"] == port_options, where
            contraction = probe["contraction"]
            lanes = contraction["lanes"]
            assert lanes["expected_fused_bits"] != lanes["expected_unfused_bits"], \
                f"{where}: the probe can fail"
            for name, form in forms.items():
                arm = contraction[name]
                assert arm["status"] == "measured" and arm["dual_run_byte_identical"] is True, \
                    (where, name)
                assert arm["anchors_match"] is True, (where, name)
                assert arm["plain_is"] == form, (where, name)
                assert arm["plain_bits"] == lanes[f"expected_{form}_bits"], (where, name)
            assert contraction["port_kernelcache_route"]["base_options"] == port_options, where
        arms = {
            name for name, arm in probes[f"contract/route-probe-{tag}.json"]["ftz"].items()
            if isinstance(arm, dict)
        }
        families = _json(directory / "VERDICT.json")["families"]
        assert families["route-ftz"]["verdict"] == "identical"
        assert families["route-ftz"]["records"] == len(arms)
        assert families["route-contraction"]["verdict"] == "identical"
        assert families["route-contraction"]["records"] == len(forms)


def test_a_campaign_anchor_contract_decks_are_the_records_its_verdict_counts(receipts):
    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        families = _json(directory / "VERDICT.json")["families"]
        decks = _json(directory / "contract" / f"arch-ftz-decks-{tag}.json")
        assert decks["capability"]["name"] == _card_name(anchor)
        transport = decks["decks"]["transport_deck"]
        assert transport["status"] == "measured"
        result = transport["result"]
        assert result["dual_run_byte_identical"] is True and result["fallback_verified"] is True
        assert families["transport-deck"]["verdict"] == "identical"
        assert families["transport-deck"]["records"] == len(result["kernel_localization"])
        guarded = decks["decks"]["guarded_kernel_audit"]
        assert guarded["status"] == "measured"
        result = guarded["result"]
        assert result["dual_run_byte_identical"] is True and result["fallback_verified"] is True
        assert families["guarded-audit"]["verdict"] == "identical"
        assert families["guarded-audit"]["records"] == result["kernel_count"] == len(result["kernels"])
        audit = _json(directory / "contract" / f"v841-specific-kernels-{tag}.json")
        assert audit["status"] == "measured" and audit["violations"] == []
        assert audit["enabled_dual_run_identical"] is True
        assert families["v841-release-specific"]["verdict"] == "identical"
        assert families["v841-release-specific"]["records"] == audit["kernel_count"]
        regional = sorted((directory / "contract" / "regional-contract").glob("*.json"))
        assert len(regional) == 1, f"{anchor.sm}: {len(regional)} regional contract receipts"
        record = _json(regional[0])
        for key in ("all_decks_bitwise", "all_kernels_covered", "all_controls_have_teeth",
                    "dual_run_identical"):
            assert record[key] is True, (anchor.sm, key)
        summary = record["summary"]
        assert summary["decks_passed"] == summary["decks"] == families["regional-contract"]["records"]
        assert summary["kernels_covered"] == summary["kernels_declared"]
        assert families["regional-contract"]["verdict"] == "identical"


def test_every_contract_receipt_in_a_campaign_anchor_holds_its_own_verdict(receipts):
    """The session of record and every later session on the same card (the
    addenda, the committed-rows verify run) file receipts that each state a
    verdict of their own.  Each has to say it passed: an addendum is what the
    ceiling and the review fixes now rest on, so a failed receipt there is as
    disqualifying as one in contract/.  A failed attempt before the session of
    record is kept under attempts/ as history and is not a claim."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        card = _card_name(anchor)
        seen = {"regional": 0, "decks": 0, "audit": 0, "engine": 0, "preflight": 0}

        for relative, record in _filed(directory, "*.json"):
            if PurePosixPath(relative).parent.name != "regional-contract":
                continue
            seen["regional"] += 1
            where = f"{anchor.sm} {relative}"
            assert record["card"] == card and record["device"]["name"] == card, where
            assert record["device"]["sm"] == anchor.sm, where
            for key in ("all_decks_bitwise", "all_kernels_covered", "all_controls_have_teeth",
                        "dual_run_identical"):
                assert record[key] is True, (where, key)
            summary = record["summary"]
            assert summary["decks_passed"] == summary["decks"] == len(record["decks"]) > 0, where
            declared = record["translation_unit"]["declared_kernels"]
            assert summary["kernels_covered"] == summary["kernels_declared"] == len(declared), where
            assert record["translation_unit"]["kernels_without_a_deck"] == [], where
            assert summary["all_dual_run_identical"] is True, where
            assert summary["all_controls_have_teeth"] is True, where
            covered: set[str] = set()
            for deck in record["decks"]:
                label = (where, deck["deck"])
                assert deck["verdict"] is True and deck["dual_run_identical"] is True, label
                assert deck["pass_1"]["bitwise_equal"] is True, label
                assert deck["pass_2"]["bitwise_equal"] is True, label
                assert deck["mutation"]["control_has_teeth"] is True, label
                assert deck["mutation"]["deck_still_matches_host"] is False, label
                assert deck["kernels"], label
                covered |= set(deck["kernels"])
                # The flags above are the deck's summary of these payloads:
                # every payload of both passes is its host oracle's bytes,
                # the two passes hash the same device bytes, and the
                # mutation turns at least one payload red.
                first = deck["pass_1"]["payloads"]
                second = deck["pass_2"]["payloads"]
                names = [payload["payload"] for payload in first]
                assert names and names == [payload["payload"] for payload in second], label
                for payload in first + second:
                    item = (label, payload["payload"])
                    assert payload["bitwise_equal"] is True, item
                    assert _SHA256_HEX.fullmatch(payload["device_sha256"]), item
                    assert payload["device_sha256"] == payload["host_sha256"], item
                for one, two in zip(first, second):
                    assert one["device_sha256"] == two["device_sha256"], (label, one["payload"])
                mutated = deck["mutation"]["payloads"]
                assert [payload["payload"] for payload in mutated] == names, label
                for payload in mutated:
                    assert (payload["bitwise_equal"] is False) == (payload["mismatch_count"] > 0), \
                        (label, payload["payload"])
                assert any(payload["bitwise_equal"] is False for payload in mutated), (
                    f"{label}: the mutation control turned no payload red"
                )
            assert covered == set(declared), (
                f"{where}: the decks cover {len(covered)} of the {len(declared)} declared kernels "
                f"({sorted(set(declared) - covered)} missing)"
            )

        for relative, record in _filed(directory, f"arch-ftz-decks-{tag}.json"):
            seen["decks"] += 1
            where = f"{anchor.sm} {relative}"
            assert record["capability"]["name"] == card, where
            assert record["statuses"] == {
                name: deck["status"] for name, deck in record["decks"].items()
            }, where
            for name in ("transport_deck", "guarded_kernel_audit"):
                deck = record["decks"][name]
                assert deck["status"] == "measured", (where, name)
                result = deck["result"]
                assert result["dual_run_byte_identical"] is True, (where, name)
                assert result["fallback_verified"] is True, (where, name)
            guarded = record["decks"]["guarded_kernel_audit"]["result"]
            assert guarded["kernel_count"] == len(guarded["kernels"]) > 0, where
            control = record["decks"].get("normalized_performance_control")
            if control is not None and control["status"] == "measured":
                assert control["result"]["all_normalized_outputs_bitwise_identical"] is True, where

        for relative, audit in _filed(directory, f"v841-specific-kernels-{tag}.json"):
            seen["audit"] += 1
            where = f"{anchor.sm} {relative}"
            assert audit["status"] == "measured" and audit["violations"] == [], where
            assert audit["enabled_dual_run_identical"] is True, where
            enabled, disabled = audit["records_enabled"], audit["records_disabled"]
            assert len(enabled) == len(disabled) == audit["kernel_count"] > 0, where
            assert set(enabled) == set(disabled), where
            classes = [record["classification"] for record in enabled.values()]
            assert classes.count("guarded_fallback_required") \
                == audit["guarded_fallback_required"], where
            assert classes.count("fallback_invariant") == audit["fallback_invariant"], where
            assert audit["guarded_fallback_required"] + audit["fallback_invariant"] \
                == audit["kernel_count"], where
            for key, record in enabled.items():
                assert record["matches_expected"] is True, (where, key)
                assert record["observed_bits"] == record["expected_bits"], (where, key)
                off = disabled[key]
                assert off["classification"] == record["classification"], (where, key)
                # The disabled-fallback arm goes red exactly where the
                # classification says the guard is required.
                assert off["matches_expected"] is (
                    record["classification"] == "fallback_invariant"
                ), (where, key)

        for relative, receipt in _filed(directory, "receipt.json"):
            if not PurePosixPath(relative).parent.name.startswith("engine-ftz-receipt-"):
                continue
            seen["engine"] += 1
            where = f"{anchor.sm} {relative}"
            assert receipt["device"]["name"] == card, where
            dual = receipt["dual_run"]
            assert dual["runs"] == 2 and dual["byte_identical"] is True, where
            assert len(dual["bit_table_sha256"]) == 2, where
            assert len(set(dual["bit_table_sha256"])) == 1, where
            # The route grid the receipt and RECEIPT.md state: every route
            # times every mechanism, once each, on the verdict its route
            # requires.
            assert set(receipt["routes"]) == set(_ENGINE_ROUTE_GRID), where
            cells = {(cell["route"], cell["mechanism"]): cell for cell in receipt["cells"]}
            assert len(cells) == len(receipt["cells"]), f"{where}: a cell is recorded twice"
            assert set(cells) == {
                (route, mechanism)
                for route in _ENGINE_ROUTE_GRID for mechanism in receipt["mechanisms"]
            }, where
            inputs = len(receipt["inputs"])
            for (route, mechanism), cell in cells.items():
                assert cell["verdict"] in _ENGINE_ROUTE_GRID[route], (
                    f"{where}: {route} {mechanism!r} measured {cell['verdict']!r}, the route "
                    f"requires {_ENGINE_ROUTE_GRID[route]}"
                )
                assert cell["input_count"] == (0 if cell["verdict"] == "not-applicable" else inputs), \
                    (where, route, mechanism)
            # The options each route's own NVRTC calls received say the same:
            # a final ftz=true flushes, ftz=false or none keeps IEEE.  R5 is
            # inline PTX in R1's module, so R1's options do not decide it.
            for route, calls in receipt["effective_option_capture"]["per_route"].items():
                if route == "R5":
                    continue
                for call in calls:
                    assert call["fired"] is True, (where, route)
                    ftz = [flag.lstrip("-")[len("ftz="):] for flag in call["effective_flags"]
                           if flag.lstrip("-").startswith("ftz=")]
                    requires = "flush-to-zero" if ftz and ftz[-1] == "true" else "ieee-agreement"
                    assert _ENGINE_ROUTE_GRID[route] == (requires,), (where, route, call["effective_flags"])
            # The flag arms are distinguishable: no flush arm shares its bit
            # table with an IEEE arm.
            flags = receipt["flag_sensitivity"]
            tables = flags["arm_bit_table_sha256"]
            assert flags["arms_differ"] is True, where
            assert set(tables) == set(_ENGINE_ROUTE_GRID), where
            assert flags["distinct_arm_bit_tables"] == len(set(tables.values())) > 1, where
            flushed = {tables[route] for route, grid in _ENGINE_ROUTE_GRID.items()
                       if grid == ("flush-to-zero",)}
            kept = {tables[route] for route, grid in _ENGINE_ROUTE_GRID.items()
                    if grid == ("ieee-agreement",)}
            assert not flushed & kept, f"{where}: a flush arm and an IEEE arm wrote one bit table"
            # The bit table the dual run hashed is the one filed, and every
            # file filed beside the receipt is an artifact it records, on
            # the hash it records.
            artifacts = receipt["artifacts"]
            assert dual["bit_table_sha256"][0] == artifacts["bitpatterns.csv"]["sha256"], where
            folder = (directory / relative).parent
            recorded = {name: item["sha256"] for name, item in artifacts.items() if "sha256" in item}
            for path in sorted(folder.rglob("*")):
                name = path.relative_to(folder).as_posix()
                if not path.is_file() or name == "receipt.json":
                    continue
                assert name in recorded, f"{where}: {name} is filed and the receipt records no hash for it"
                assert _file_sha256(path) == recorded[name], (
                    f"{where}: {name} is not the artifact the receipt recorded"
                )

        for relative, receipt in _filed(directory, "pf-*.json"):
            seen["preflight"] += 1
            where = f"{anchor.sm} {relative}"
            assert receipt["status"] == "preflight_passed", where
            assert receipt["preflight_problems"] == [], where
            assert receipt["admission"]["admitted"] is True, where
            assert receipt["admission"]["card"]["name"] == card, where

        assert all(seen.values()), f"{anchor.sm}: no receipt of some contract kind: {seen}"


def test_a_session_table_is_what_its_legs_wrote(receipts):
    """A session README's leg table states each leg's rc and what it measured.
    The rc is the one the leg wrote, and a v8.4.1 release-specific audit leg's
    receipt counts the entrypoints the table states and records every kernel
    the table says it now has an on-card record for."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        tables = 0
        for readme in sorted(directory.rglob("README.md")):
            if "attempts" in readme.relative_to(directory).parts:
                continue
            rows = {match["leg"]: match for match in _SESSION_ROW.finditer(
                readme.read_text(encoding="utf-8"))}
            if not rows:
                continue
            tables += 1
            session = readme.parent
            logs = session / "chain" if (session / "chain").is_dir() else session
            label = readme.relative_to(directory).as_posix()
            written = {path.stem: path for path in logs.glob("*.rc")}
            for leg, row in rows.items():
                assert leg in written, f"{anchor.sm} {label}: leg {leg} wrote no rc"
                rc = int(written[leg].read_text(encoding="utf-8").strip())
                assert rc == int(row["rc"]), (
                    f"{anchor.sm} {label}: the table says {leg} rc {row['rc']}, "
                    f"the leg wrote {rc}"
                )
                if not leg.endswith("-v841-audit"):
                    continue
                audit = _json(session / leg[: -len("-v841-audit")] / "contract"
                              / f"v841-specific-kernels-{tag}.json")
                counts = _AUDIT_COUNTS.search(row["result"])
                assert counts, f"{anchor.sm} {label}: {leg} states no entrypoint count"
                assert audit["kernel_count"] == int(counts["kernels"]), (label, leg)
                assert len(audit["records_enabled"]) == int(counts["kernels"]), (label, leg)
                assert audit["guarded_fallback_required"] == int(counts["guarded"]), (label, leg)
                assert audit["fallback_invariant"] == int(counts["invariant"]), (label, leg)
                kernels = {key.rsplit("::", 1)[-1] for key in audit["records_enabled"]}
                for claim in _NOW_RECORDED.findall(row["result"]):
                    for kernel in re.findall(r"`(\w+)`", claim):
                        assert kernel in kernels, (
                            f"{anchor.sm} {label}: {leg} claims an on-card record for "
                            f"{kernel}, which its receipt does not hold"
                        )
            # A leg the table leaves out (a census, say) cannot have failed.
            for leg, path in written.items():
                if leg not in rows:
                    assert path.read_text(encoding="utf-8").strip() == "0", (
                        f"{anchor.sm} {label}: {leg} failed and the table omits it"
                    )
        if (directory / "addendum").is_dir():
            assert tables >= 1, f"{anchor.sm}: an addendum with no leg table"


def _held_to_its_own_runs(anchor, directory, host, label, record, digests):
    """Assert a reproduced pair is two door runs, read from what each run
    wrote itself: its forecast receipt (the snapshot hashes its driver took
    of the frames it wrote, its step count and health, its card and host),
    the DOOR line its process printed, and the chain's window for its leg."""

    arms = record["rcs"]
    assert len(arms) == 2 and set(arms.values()) == {0}, (anchor.sm, label, arms)
    windows = _leg_windows((directory / "chain" / "chain.log").read_text(encoding="utf-8"))
    extra = list(record.get("extra_args") or ())
    backend = extra[extra.index("--physics-backend") + 1] if "--physics-backend" in extra else None
    created, scratch, outputs = set(), set(), set()
    for arm, rc in arms.items():
        where = f"{anchor.sm} {label} {arm}"
        receipt = _json(directory / "payoff" / f"{arm}-forecast-receipt.json")
        assert receipt["status"] == "passed", where
        assert receipt["host"] == host, where
        assert receipt["mesh"]["name"] == record["mesh"], where
        if backend is not None:
            assert receipt["physics"]["backend"] == backend, where
        driver = receipt["driver_receipt"]
        assert driver["status"] == "passed", where
        # The driver sealed what it wrote: a frame hash, a step count or a
        # health entry edited after the run no longer recomputes the seal.
        assert _SHA256_HEX.fullmatch(driver["payload_sha256"]), where
        assert _driver_seal(driver) == driver["payload_sha256"], (
            f"{where}: its driver receipt's seal does not recompute, so the receipt is "
            f"not what the driver wrote"
        )
        forecast = driver["forecast"]
        assert forecast["refusal"] is None, where
        assert forecast["full_physics_cuda_executed"] is True, where
        capability = forecast["capability"]
        assert capability["name"] == _card_name(anchor) and capability["sm"] == anchor.sm, where
        steps = forecast["steps_executed"]
        health = forecast["step_health"]
        assert steps == forecast["steps_requested"] == forecast["step_receipt_count"] \
            == len(health) == forecast["schedule"]["steps"] > 0, where
        assert [entry["step"] for entry in health] == list(range(1, steps + 1)), where
        for entry in health:
            assert entry["finite"] is True, (where, entry["step"])
            assert all(math.isfinite(value) for value in _numbers(entry)), (where, entry["step"])
        captures = forecast["schedule"]["capture_steps"]
        assert captures[-1] == steps, where
        assert sorted(int(step) for step in forecast["physical_gates"]) == captures, where
        assert all(gate["finite_all_fields"] is True
                   for gate in forecast["physical_gates"].values()), where
        # The frames this run wrote, hashed by this run's own driver.
        snapshots = forecast["snapshot_files"]
        assert sorted(int(step) for step in snapshots) == captures, where
        written = {PurePosixPath(item["path"]).name: item["sha256"] for item in snapshots.values()}
        assert written == digests, (
            f"{where}: the frames its own driver hashed are not the pair's digests"
        )
        assert {PurePosixPath(item["path"]).parent.as_posix() for item in snapshots.values()} \
            == {receipt["out"]}, where
        assert sorted(PurePosixPath(item).name for item in receipt["history"]) == sorted(digests)
        # The line the door process printed, and the chain's record of its leg.
        log = (directory / "chain" / f"forecast-{arm}.log").read_text(encoding="utf-8")
        doors = [match.groupdict() for match in _DOOR_LINE.finditer(log)]
        assert doors == [{
            "mesh": record["mesh"], "steps": str(steps), "frames": str(len(digests)),
            "status": "passed", "out": receipt["out"],
        }], (where, doors)
        # The driver process printed a summary as it wrote its receipt: the
        # seal, and the hash of the receipt file as written (sorted keys,
        # indent 2, one trailing newline).  A receipt resealed after the run
        # no longer matches what the process printed.
        summaries = [json.loads(line) for line in log.splitlines()
                     if line.startswith("{") and '"payload_sha256"' in line]
        assert len(summaries) == 1, (where, len(summaries))
        printed = summaries[0]
        as_written = json.dumps(driver, indent=2, sort_keys=True, allow_nan=False) + "\n"
        assert printed["payload_sha256"] == driver["payload_sha256"], where
        assert printed["receipt_sha256"] == hashlib.sha256(as_written.encode("utf-8")).hexdigest(), (
            f"{where}: the driver receipt is not the file its process wrote"
        )
        assert printed["status"] == "passed" and printed["steps"] == steps, where
        assert printed["history_frames"] == len(digests), where
        assert PurePosixPath(printed["receipt"]).parent.as_posix() == receipt["out"], where
        start, end, chain_rc = windows[f"forecast-{arm}"]
        assert chain_rc == rc, where
        assert start <= receipt["created_utc"] <= end, (where, start, receipt["created_utc"], end)
        created.add(receipt["created_utc"])
        scratch.add(receipt["scratch"])
        outputs.add(receipt["out"])
    assert len(created) == len(scratch) == len(outputs) == 2, (
        f"{anchor.sm} {label}: the pair is one run recorded twice "
        f"(created {sorted(created)}, scratch {sorted(scratch)}, out {sorted(outputs)})"
    )


def _determinism_digests(anchor, label, determinism) -> dict[str, str]:
    assert determinism["all_identical"] is True, (anchor.sm, label)
    rows = {row["file"]: row for row in determinism["rows"]}
    assert determinism["frames"] == determinism["frames_identical"] == len(rows) >= 2, \
        (anchor.sm, label)
    for frame, row in rows.items():
        assert row["identical"] is True and row["in_arm_a"] is True and row["in_arm_b"] is True, \
            (anchor.sm, label, frame)
        assert _SHA256_HEX.fullmatch(row["digest_a"]), (anchor.sm, label, frame)
        assert row["digest_a"] == row["digest_b"], (anchor.sm, label, frame)
    return {frame: row["digest_a"] for frame, row in rows.items()}


def test_a_campaign_anchor_authority_digests_are_the_pairs_own(receipts):
    for anchor, directory in _campaign_anchors(receipts):
        verdict = _json(directory / "VERDICT.json")
        host = verdict["identity"]["host"]
        authority = verdict["authority"]
        assert authority["verdict"] in ("re-anchored", "identical")
        files = sorted((directory / "authority").glob("*-masked-digests.json"))
        assert len(files) == 1, f"{anchor.sm}: {len(files)} authority digest files"
        assert files[0].name.startswith(authority["mesh"])
        digests = _json(files[0])
        determinism = _json(directory / "payoff" / "determinism.json")
        assert determinism == authority["determinism"]
        pair = _determinism_digests(anchor, "authority", determinism)
        assert len(digests) == len(pair), f"{anchor.sm}: an authority set of {len(digests)} frame"
        for frame, entry in digests.items():
            assert entry["identical_across_pair"] is True, (anchor.sm, frame)
            assert entry["digest"] == pair[frame], (anchor.sm, frame)
        # Twice reproduced: two separate door runs, each passed, each hashing
        # every frame of the set it wrote into its own output directory.
        _held_to_its_own_runs(
            anchor, directory, host, "authority", authority,
            {frame: entry["digest"] for frame, entry in digests.items()},
        )
        # A companion pair (another physics backend on the same mesh) is held
        # to the same standard, and is its own measurement, not the
        # authority's frames again.
        for name, companion in (verdict.get("companions") or {}).items():
            label = f"companion {name}"
            assert companion["verdict"] in ("re-anchored", "identical"), (anchor.sm, label)
            assert companion["preflight_rc"] == 0, (anchor.sm, label)
            frames = _determinism_digests(anchor, label, companion["determinism"])
            assert not set(frames.values()) & set(pair.values()), (anchor.sm, label)
            _held_to_its_own_runs(anchor, directory, host, label, companion, frames)


def _proposed_basis(anchor: ArchAnchor, directory: Path) -> str:
    """The registered basis as its campaign proposed it.

    The one difference a registered row may carry: the campaign named its
    runner by the path it ran from, tools/arch_anchor_campaign.py, which
    this tree does not carry, and the row cites the copy the campaign filed
    in its own evidence instead.  That copy has to be there and be the file
    the evidence's SHA256SUMS names.
    """

    filed = f"{anchor.evidence_directory}/instruments/arch_anchor_campaign.py"
    if f"({filed})" not in anchor.basis:
        return anchor.basis
    runner = directory / "instruments" / "arch_anchor_campaign.py"
    assert runner.is_file(), f"{anchor.sm} cites {filed}, which its evidence does not carry"
    sums = {
        line["name"]: line["digest"]
        for line in _SUMS_LINE.finditer(
            (directory / "instruments" / "SHA256SUMS").read_text(encoding="utf-8")
        )
    }
    assert sums.get("arch_anchor_campaign.py") == _file_sha256(runner), (
        f"{anchor.sm} cites {filed}, which is not the runner its SHA256SUMS names"
    )
    return anchor.basis.replace(f"({filed})", "(tools/arch_anchor_campaign.py)")


def test_a_campaign_anchor_row_is_the_row_its_campaign_proposed(receipts):
    for anchor, directory in _campaign_anchors(receipts):
        proposed = _json(directory / "rows.json")["arch_admission"]
        fields = proposed["fields"]
        assert tuple(fields["compute"]) == anchor.compute
        for key in ("card", "admitted_on", "contract_receipt", "authority_anchor"):
            assert fields[key] == getattr(anchor, key), (anchor.sm, key)
        basis = _proposed_basis(anchor, directory)
        assert fields["basis"] == basis, (anchor.sm, "basis")
        rows_md = (directory / "ROWS.md").read_text(encoding="utf-8")
        for key in ("card", "admitted_on", "contract_receipt", "authority_anchor"):
            assert repr(getattr(anchor, key)) in rows_md, (anchor.sm, key)
        assert repr(basis) in rows_md, (anchor.sm, "basis")
        # A registered ceiling that differs from the campaign's proposal rests
        # on a calibration that replaced it, and says so.
        from woof.hex.cuda_backend.arch_admission import PERFORMANCE_RATIO_CEILINGS

        registered = PERFORMANCE_RATIO_CEILINGS[anchor.sm]
        if registered["ceiling"] != proposed["ceiling"]["ceiling"]:
            cited = [
                relative for relative in _cited_receipts(anchor, registered["basis"])
                if "perf-calibration" in relative
            ]
            assert cited, f"{anchor.sm}'s ceiling departs from the proposal on no calibration"


def test_every_guard_cost_reading_held_its_bitwise_identity(receipts):
    """A guard-cost reading times the FP64 fallback enabled against disabled
    and states that the answer stayed bitwise identical.  An identity break
    is a correctness failure, not a timing one: the fallback changed an
    answer on this card.  So every identity claim in every timing receipt,
    attempts/ included, is held true (not only in the cross-process
    calibration the ceiling test opens), and a reading's maximum is the
    largest ratio of its own rows."""

    for anchor in ADMITTED_BELOW_FLOOR.values():
        directory = TREE_ROOT / anchor.evidence_directory
        claims = 0
        for path in _timing_receipts(directory):
            document = _json(path)
            label = path.relative_to(directory).as_posix()
            for where, value in _identity_claims(document):
                claims += 1
                assert value is True, (
                    f"{anchor.sm}: {label}{where} is {value}: the fallback changed an answer "
                    f"on this card"
                )
            for where, reading in _rowed_readings(document):
                assert reading["maximum"] == max(row["ratio"] for row in reading["rows"].values()), \
                    (anchor.sm, label, where)
        if (directory / "VERDICT.json").is_file():
            assert claims > 0, f"{anchor.sm}: no timing receipt states its identity"


def test_a_campaign_anchor_verdict_is_what_its_chain_and_decks_wrote(receipts):
    """The record session's VERDICT.json and RECEIPT.md are reconciled with
    the rc each leg wrote to chain/chain.log and with the decks receipt: a
    leg whose receipt these tests hold as passed exited 0, every state the
    verdict reports is its leg's rc, every leg that exited non-zero is named
    as not runnable or accounted for by a declared deviation, a reading that
    breached its ceiling sits under a deviation, and the record count is the
    identical families' own."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        verdict = _json(directory / "VERDICT.json")
        chain = (directory / "chain" / "chain.log").read_text(encoding="utf-8")
        rcs = {leg: rc for leg, (_, _, rc) in _leg_windows(chain).items()}
        skipped = {match["leg"] for match in _CHAIN_SKIPPED.finditer(chain)}
        families = verdict["families"]
        supplementary = verdict["supplementary"]
        decks = _json(directory / "contract" / f"arch-ftz-decks-{tag}.json")

        # Every leg whose receipt the tests above hold as passed exited 0.
        held = ["route-probe", "engine-probe", "v841-audit", "preflight"]
        if "regional-contract" in families:
            held.append("regional-contract")
        held += [f"forecast-{arm}" for arm in verdict["authority"]["rcs"]]
        for name, companion in (verdict.get("companions") or {}).items():
            held.append(f"preflight-{name}")
            assert companion["preflight_rc"] == rcs.get(f"preflight-{name}"), (anchor.sm, name)
            held += [f"forecast-{arm}" for arm in companion["rcs"]]
        for leg in held:
            assert rcs.get(leg) == 0, (
                f"{anchor.sm}: chain.log records leg {leg} rc {rcs.get(leg)}, and its receipt "
                f"is held as passed"
            )

        # The states the verdict reports are the rcs the chain logged.
        for row in verdict["sm86_sequence"]:
            for leg, state in row["states"].items():
                if leg in skipped:
                    assert state.startswith("skipped"), (anchor.sm, leg, state)
                elif leg not in rcs:
                    assert state.startswith("not in this campaign"), (anchor.sm, leg, state)
                else:
                    assert state == ("passed" if rcs[leg] == 0 else f"rc {rcs[leg]}"), (
                        f"{anchor.sm}: the verdict says {leg} {state!r}, chain.log rc {rcs[leg]}"
                    )
        manifest = decks["decks"].get("compile_manifest_v823") or {}
        relation = manifest.get("relation") or {}
        bound = (relation.get("result") or {}).get("device_compute_capability")
        from_decks = {
            "compile-manifest-v823": manifest.get("status", "not-run"),
            "compile-manifest-relation": (
                "not-runnable" if manifest.get("status") != "measured"
                else "passed" if relation.get("status") == "measured" and bound not in (None, "")
                else "refused"
            ),
        }
        for name, row in supplementary.items():
            if name in from_decks:
                assert "rc" not in row and row["state"] == from_decks[name], (anchor.sm, name, row)
                continue
            assert name in rcs or name in skipped, f"{anchor.sm}: supplementary {name} is no chain leg"
            rc = "skipped" if name in skipped else rcs[name]
            assert row["rc"] == rc, f"{anchor.sm}: the verdict says {name} rc {row['rc']}, chain.log {rc}"
            assert row["state"] == (
                "passed" if rc == 0 else "not-runnable" if rc == "skipped" else "refused"
            ), (anchor.sm, name, row["state"], rc)

        # What the verdict lists as not runnable, and as deviations, is what
        # its supplementary states and its families say.
        assert sorted(item.split(":", 1)[0] for item in verdict["not_runnable"]) == sorted(
            name for name, row in supplementary.items() if row["state"] in ("not-runnable", "refused")
        ), anchor.sm
        assert sorted(verdict["deviations"]) == sorted(
            f"{name}: {family['why']}" for name, family in families.items()
            if family["verdict"] == "deviation"
        ), anchor.sm

        # Every leg that exited non-zero is accounted for.  The decks tool
        # exits 0 only when every deck measured; a deck that did not is a
        # declared deviation or a refused supplementary entry.
        named = {item.split(":", 1)[0] for item in verdict["not_runnable"]}
        declared = {item.split(":", 1)[0] for item in verdict["deviations"]}
        unmeasured = sorted(name for name, status in decks["statuses"].items() if status != "measured")
        if "decks" in rcs:
            assert (rcs["decks"] == 0) == (not unmeasured), (anchor.sm, rcs["decks"], unmeasured)
        for deck in unmeasured:
            family = _DECK_FAMILY[deck]
            assert family in declared or family in named, (
                f"{anchor.sm}: deck {deck} did not measure and the verdict declares no {family}"
            )
        for leg, rc in rcs.items():
            if rc != 0 and leg != "decks":
                assert leg in named, (
                    f"{anchor.sm}: leg {leg} exited {rc} and the verdict names it nowhere"
                )

        # A reading that breached its ceiling is under a deviation, and a
        # reading's status is what its ratio and ceiling say.
        for name, family in families.items():
            for reading in family.get("readings") or ():
                assert reading["status"] in ("within", "breach"), (anchor.sm, name, reading)
                if "ceiling" in reading:
                    assert (reading["status"] == "breach") == (reading["ratio"] > reading["ceiling"]), \
                        (anchor.sm, name, reading)
                if reading["status"] == "breach":
                    assert family["verdict"] == "deviation", (
                        f"{anchor.sm}: {name} records a breach at {reading['ratio']} and says "
                        f"{family['verdict']}"
                    )

        # The record count is the identical families' own.
        assert verdict["records_identical"] == sum(
            family["records"] for family in families.values()
            if family["verdict"] == "identical"
            and family.get("unit", "contract records") == "contract records"
        ), anchor.sm
        assert verdict["device_witnesses_identical"] == sum(
            family["records"] for family in families.values()
            if family["verdict"] == "identical" and family.get("unit") == "device witnesses"
        ), anchor.sm

        # RECEIPT.md's contract table is the verdict's.
        receipt = (directory / "RECEIPT.md").read_text(encoding="utf-8")
        contract = receipt.split("\n## The contract\n", 1)[1].split("\n## ", 1)[0]
        rows = {match["family"]: match for match in _FAMILY_ROW.finditer(contract)}
        assert set(rows) == set(families), (anchor.sm, sorted(rows))
        for name, row in rows.items():
            family = families[name]
            assert (int(row["records"]), row["verdict"], row["why"]) == (
                family["records"], family["verdict"], family["why"]
            ), (anchor.sm, name)
        assert f"\n{verdict['records_identical']} contract records identical" in contract, anchor.sm


def test_a_campaign_anchor_instruments_and_census_are_what_ran(receipts):
    """Every instrument an anchor files is the one its SHA256SUMS hashed,
    the campaign runner is the one whose hash the identity recorded, and
    every nvidia-smi census outside attempts/ names the identity's card at
    the identity's bus, on the anchor's driver."""

    for anchor, directory in _campaign_anchors(receipts):
        identity = _json(directory / "VERDICT.json")["identity"]
        assert _json(directory / "chain" / "identity.json") == identity, anchor.sm
        listed: set[Path] = set()
        sums = sorted(directory.rglob("SHA256SUMS"))
        assert sums, f"{anchor.sm}: no SHA256SUMS"
        for path in sums:
            label = path.relative_to(directory).as_posix()
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                match = _SUMS_LINE.fullmatch(line)
                assert match, (anchor.sm, label, line)
                target = path.parent / match["name"]
                assert target.is_file(), f"{anchor.sm}: {label} lists {match['name']}, not filed"
                assert _file_sha256(target) == match["digest"], (
                    f"{anchor.sm}: {match['name']} is not the file {label} hashed"
                )
                listed.add(target)
        for folder in sorted(directory.rglob("instruments")):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts:
                    assert path in listed, (
                        f"{anchor.sm}: {path.relative_to(directory).as_posix()} is an instrument "
                        f"no SHA256SUMS hashes"
                    )
        runner = directory / "instruments" / "arch_anchor_campaign.py"
        assert _file_sha256(runner) == identity["tree"]["runner_sha256"], (
            f"{anchor.sm}: the filed runner is not the runner the campaign identity hashed"
        )
        card = identity["card"]
        censuses = [
            path for path in sorted(directory.rglob("census*.log"))
            if "attempts" not in path.relative_to(directory).parts
        ]
        assert censuses, f"{anchor.sm}: no nvidia-smi census"
        for path in censuses:
            label = path.relative_to(directory).as_posix()
            text = path.read_text(encoding="utf-8")
            drivers = {match["driver"] for match in _SMI_DRIVER.finditer(text)}
            assert drivers == {_card_driver(anchor)}, (anchor.sm, label, drivers)
            devices = [
                match for match in _SMI_DEVICE.finditer(text)
                if match["bus"].upper() == card["pci_bus_id"].upper()
            ]
            assert len(devices) == 1, f"{anchor.sm}: {label} shows no device at {card['pci_bus_id']}"
            name = devices[0]["name"]
            assert name == _card_name(anchor) or (
                name.endswith("...") and _card_name(anchor).startswith(name[:-3])
            ), f"{anchor.sm}: {label} names {name!r}"


# ---------------------------------------------------------------------------
# what a receipt derives from its own inputs is recomputed from them
# ---------------------------------------------------------------------------
# A fourth mutation run (2026-09-24, same method) found 9 of 10 further edits
# passing, each confined to one file or to one self-consistent set of files.
# A route probe's host reference forged to the flushed values its preserved
# arms were then made to show; the contraction lanes forged so --fmad=false
# read fused; a v8.4.1 audit record's observed and expected bits forged, its
# records seal left as the audit wrote it; the footprint peak raised above the
# derived row's prediction with the row still called conservative; the
# normalized-control breach hidden in the verdict while its perf receipt still
# records it; a regional contract under-declaring its kernel set; the
# authority verdict claiming identity with no reference recorded; the
# committed-rows session deleted, leaving the calibration receipt it wrote with
# no process that printed it; an engine FTZ receipt naming another card.  Each
# of those receipts carries the inputs its figures follow from, or a sibling
# record of the same measurement, so these tests recompute the figures.
#
# What no content test sees is a frame digest forged consistently in every
# file that records it, with both driver seals recomputed and both door logs
# rewritten to print them: the frames are NetCDF files the evidence does not
# carry.  The byte pin holds that record.
_FLT_MIN = 1.1754943508222875e-38  # the smallest normal float32
_GATED_BREACH = re.compile(
    r"regressed at (?P<kernel>\S+): (?P<ratio>[0-9.]+)x > (?P<ceiling>[0-9.]+)x"
)
_REGIONAL_WHY = re.compile(
    r"(?P<passed>\d+) of (?P<decks>\d+) decks bitwise\b.*?"
    r"(?P<covered>\d+) of (?P<declared>\d+) kernels covered"
)
_PRINTED_READING = re.compile(r"\d+\.\d+")
_UUID = re.compile(r"[0-9a-f]{16,32}")


def _f32_bits(value: float) -> int:
    """The IEEE bits of ``value`` rounded to float32 (to nearest, ties even)."""

    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f32_nearest_bits(exact: Fraction) -> int:
    """The float32 nearest an exact value, ties to even: ONE rounding, as a
    fused multiply-add rounds a*b+c.  The float64 guess is within one float32
    step of the answer, so its two neighbours settle it."""

    guess = _f32_bits(float(exact))
    candidates = [
        bits for bits in (guess - 1, guess, guess + 1)
        if 0 <= bits < 1 << 32 and math.isfinite(_f32(bits))
    ]
    return min(candidates, key=lambda bits: (abs(Fraction(_f32(bits)) - exact), bits & 1))


def _canonical_sha256(document) -> str:
    """``woof.hex.cuda_ftz.canonical_sha256`` restated (sorted keys, no
    whitespace, UTF-8), as the driver seal is, so a tree that carries the
    receipts recomputes an audit's record seals without the audit."""

    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()


def _section(markdown: str, heading: str) -> str:
    return markdown.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_a_route_probe_expects_what_its_own_inputs_give(receipts):
    """A route probe judges its FTZ arms against the float32 rounding of the
    inputs it records, and its contraction arms against a*b+c rounded twice
    (unfused) or once (fused) from the a, b and c it records.  Both
    expectations are recomputed here, so a reference forged to what a flushed
    arm printed, or lanes forged so an unfused arm reads fused, are no longer
    the probe's expectation."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        probes = dict(_filed(directory, f"route-probe-{tag}.json"))
        assert probes, f"{anchor.sm}: no route probe"
        for relative, probe in probes.items():
            where = f"{anchor.sm} {relative}"
            ftz = probe["ftz"]
            reference = [_f32(_f32_bits(value)) for value in ftz["inputs_float32"]]
            assert ftz["host_numpy_reference"] == reference, (
                f"{where}: the host reference is not the float32 rounding of the probe's "
                f"own inputs"
            )
            assert 0.0 < reference[0] < _FLT_MIN and 0.0 < reference[1] < _FLT_MIN, (
                f"{where}: the first two inputs are not float32 subnormals, so no arm "
                f"could show a flush"
            )
            lanes = probe["contraction"]["lanes"]
            operands = (lanes["a_bits"], lanes["b_bits"], lanes["c_bits"])
            assert len({len(column) for column in operands}) == 1, where
            unfused, fused = [], []
            for a_bits, b_bits, c_bits in zip(*operands):
                a, b, c = _f32(a_bits), _f32(b_bits), _f32(c_bits)
                # float64 holds a*b exactly and rounds a float32 sum once more
                # innocuously (53 >= 2*24 + 2), so this is float32 arithmetic.
                unfused.append(_f32_bits(_f32(_f32_bits(a * b)) + c))
                fused.append(_f32_nearest_bits(Fraction(a) * Fraction(b) + Fraction(c)))
            assert lanes["expected_unfused_bits"] == unfused, (
                f"{where}: the expected unfused lanes are not a*b+c rounded twice from the "
                f"recorded a, b and c"
            )
            assert lanes["expected_fused_bits"] == fused, (
                f"{where}: the expected fused lanes are not a*b+c rounded once from the "
                f"recorded a, b and c"
            )
            differing = [lane for lane, (one, two) in enumerate(zip(unfused, fused)) if one != two]
            assert differing and lanes["discriminating_lanes"] == differing, where
            for name, arm in probe["contraction"].items():
                if name == "lanes":
                    continue
                # Each arm's own anchors are the same host lanes.
                assert arm["unfused_bits"] == unfused and arm["fused_bits"] == fused, (where, name)


def test_a_release_specific_audit_is_sealed_and_agrees_across_sessions(receipts):
    """A v8.4.1 release-specific audit seals its enabled and disabled records
    (``enabled_records_sha256``, ``disabled_records_sha256``); the seals
    recompute here.  Its expected bits are the host oracle's answer for a
    released v8.4.1 kernel on a named lane, which no session on the card can
    change: the enabled and disabled arms expect the same bits, and a kernel
    lane audited by more than one session expects the same bits in each."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        expected: dict[tuple[str, str], tuple[str, dict]] = {}
        audits = 0
        for relative, audit in _filed(directory, f"v841-specific-kernels-{tag}.json"):
            audits += 1
            where = f"{anchor.sm} {relative}"
            for arm in ("enabled", "disabled"):
                assert _canonical_sha256(audit[f"records_{arm}"]) \
                    == audit[f"{arm}_records_sha256"], (
                    f"{where}: the {arm} records do not hash to the seal the audit wrote"
                )
            for key, record in audit["records_enabled"].items():
                bits = record["expected_bits"]
                disabled = audit["records_disabled"][key]
                assert bits and disabled["expected_bits"] == bits, (where, key)
                lane = (key, record["lane"])
                if lane in expected:
                    first, recorded = expected[lane]
                    assert bits == recorded, (
                        f"{where}: {key} expects other bits than {first} on the same lane: "
                        f"the host derives one answer"
                    )
                else:
                    expected[lane] = (relative, bits)
        assert audits, f"{anchor.sm}: no v8.4.1 release-specific audit"


def test_a_campaign_anchor_footprint_is_its_samplers_and_its_door(receipts):
    """The footprint the verdict and ``probe/row.json`` state is the samplers'
    record (each run's baseline, peak and sample count, the peak the largest
    run's), its "conservative" flag is its prediction against that peak, the
    prediction is the one the door priced the card at in every preflight on
    that mesh and card, and RECEIPT.md states the same figures."""

    for anchor, directory in _campaign_anchors(receipts):
        verdict = _json(directory / "VERDICT.json")
        footprint = verdict["footprint"]
        assert _json(directory / "probe" / "row.json") == footprint, anchor.sm
        samplers = {
            path.name[len("sampler-forecast-"):-len(".json")]: _json(path)
            for path in sorted((directory / "probe").glob("sampler-forecast-*.json"))
        }
        arms = set(verdict["authority"]["rcs"]) | {
            arm for companion in (verdict.get("companions") or {}).values()
            for arm in companion["rcs"]
        }
        assert set(samplers) == arms, (anchor.sm, sorted(samplers), sorted(arms))
        assert footprint["measured"] is True, anchor.sm
        assert footprint["sampled_runs"] == sorted(set(samplers) - set(footprint["excluded_runs"]))
        for run, sampler in samplers.items():
            where = f"{anchor.sm} probe/sampler-forecast-{run}.json"
            baseline = sorted(sampler["baseline_samples_mib"])
            assert sampler["baseline_mib"] == baseline[len(baseline) // 2], where
            series = [value for _, value in sampler["series_mib"]]
            assert series and max(series) <= sampler["peak_mib"], where
            if len(series) == sampler["samples"]:
                assert max(series) == sampler["peak_mib"], (
                    f"{where}: the peak is not the largest value the sampler recorded"
                )
            assert sampler["peak_over_baseline_mib"] \
                == sampler["peak_mib"] - sampler["baseline_mib"], where
            if run in footprint["sampled_runs"]:
                assert footprint["samples"][run] == {
                    key: sampler[key]
                    for key in ("baseline_mib", "peak_mib", "peak_over_baseline_mib", "samples")
                }, where
        peaks = {run: footprint["samples"][run]["peak_over_baseline_mib"]
                 for run in footprint["sampled_runs"]}
        assert footprint["peak_mib"] == max(peaks.values()) == peaks[footprint["peak_run"]]
        conservative = footprint["derived_row_predicted_mib"] >= footprint["peak_mib"]
        assert footprint["derived_row_conservative"] is conservative, (
            f"{anchor.sm}: the derived row predicted {footprint['derived_row_predicted_mib']} MiB "
            f"against a peak of {footprint['peak_mib']} MiB and calls itself "
            f"{'conservative' if footprint['derived_row_conservative'] else 'under'}"
        )
        doors = 0
        for relative, preflight in _filed(directory, "pf-*.json"):
            admission = preflight["admission"]
            shape = (admission["cells"], admission["configuration"],
                     admission["card"]["multiprocessors"])
            if shape != (footprint["cells"], footprint["configuration"],
                         footprint["card"]["multiprocessors"]):
                continue
            doors += 1
            priced = footprint["row_predicted_mib" if admission["row_is_measured"]
                               else "derived_row_predicted_mib"]
            assert abs(admission["predicted_peak_mib"] - priced) < 0.05, (
                f"{anchor.sm} {relative}: the door priced {admission['predicted_peak_mib']} MiB, "
                f"the footprint says {priced} MiB"
            )
        assert doors, f"{anchor.sm}: no preflight priced the footprint's mesh on this card"
        section = _section((directory / "RECEIPT.md").read_text(encoding="utf-8"), "The footprint")
        assert f"peak {footprint['peak_mib']:,.1f} MiB" in section, anchor.sm
        assert (
            f"derived row predicted {footprint['derived_row_predicted_mib']:,.1f} MiB "
            f"({'conservative' if conservative else 'UNDER the measurement'})"
        ) in section, anchor.sm


def test_a_campaign_anchor_timing_readings_are_its_perf_receipts(receipts):
    """The normalized-control readings the verdict judges are the ones its
    receipts recorded, in order: the decks' gated reading, each v8.4.1 audit
    attempt (within at its measured maximum, or a breach at the kernel, ratio
    and ceiling its refusal names), and the identity-only reading (the rows
    it identity-checked, a breach at its worst row when that row is over the
    borrowed ceiling)."""

    for anchor, directory in _campaign_anchors(receipts):
        tag = anchor.sm.replace("_", "")
        family = _json(directory / "VERDICT.json")["families"]["normalized-control"]
        decks = _json(directory / "contract" / f"arch-ftz-decks-{tag}.json")
        stability = _json(directory / "contract" / f"perf-control-stability-{tag}.json")
        recorded = []
        deck = decks["decks"].get("normalized_performance_control")
        if deck:
            recorded.append(("decks", deck))
        attempts = stability.get("attempts") or {}
        recorded += [
            (f"v841-audit {key}", attempts[key]) for key in sorted(attempts) if attempts[key]
        ]
        if stability.get("identity_only"):
            recorded.append(("v841-audit identity_only", stability["identity_only"]))
        readings = family["readings"]
        assert [reading["source"] for reading in readings] == [source for source, _ in recorded], (
            anchor.sm, [reading["source"] for reading in readings]
        )
        for reading, (source, row) in zip(readings, recorded):
            where = f"{anchor.sm} normalized-control {source}"
            if source == "v841-audit identity_only":
                assert reading["mode"] == "identity-only" and row["status"] == "measured", where
                rows = row["rows"]
                assert reading["ceiling"] == row["borrowed_ceiling"], where
                assert set(reading["identity_held"]) == {
                    name for name, measured in rows.items() if measured["identical"] is True
                }, where
                worst = max(rows, key=lambda name: rows[name]["ratio"])
                if rows[worst]["ratio"] > row["borrowed_ceiling"]:
                    assert (reading["status"], reading["kernel"], reading["ratio"]) \
                        == ("breach", worst, rows[worst]["ratio"]), where
                else:
                    assert (reading["status"], reading["ratio"]) \
                        == ("within", row["maximum"]), where
                continue
            assert reading["mode"] == "gated", where
            if row["status"] == "measured":
                result = row.get("result") or row
                maximum = result.get("maximum_enabled_over_disabled", result.get("maximum"))
                assert (reading["status"], reading["ratio"]) == ("within", maximum), (
                    f"{where}: the verdict reads {reading['status']} at {reading['ratio']}, the "
                    f"receipt measured {maximum}"
                )
                continue
            breach = _GATED_BREACH.search(str(row.get("error", "")))
            assert breach, f"{where}: refused for a reason other than timing: {row.get('error')}"
            assert (reading["status"], reading["kernel"], reading["ratio"], reading["ceiling"]) == (
                "breach", breach["kernel"], float(breach["ratio"]), float(breach["ceiling"])
            ), (
                f"{where}: the verdict reads {reading['status']} at {reading['ratio']}, the "
                f"receipt refused at {breach['ratio']}x"
            )


def test_every_regional_contract_on_one_kernel_set_declares_one_kernel_list(receipts):
    """``kernel_set_sha256`` digests the sources the regional step launches
    through, so two regional contracts on one kernel set declare one kernel
    list; and the verdict's regional-contract family states the decks and
    kernel counts, class and mask of the session of record's receipt."""

    for anchor, directory in _campaign_anchors(receipts):
        declared: dict[str, tuple[str, list[str]]] = {}
        for relative, record in _filed(directory, "*.json"):
            if PurePosixPath(relative).parent.name != "regional-contract":
                continue
            kernels = sorted(record["translation_unit"]["declared_kernels"])
            assert len(set(kernels)) == len(kernels), (
                f"{anchor.sm} {relative}: a kernel declared twice"
            )
            key = record["kernel_set_sha256"]
            if key in declared:
                first, listed = declared[key]
                assert kernels == listed, (
                    f"{anchor.sm} {relative} declares {len(kernels)} kernels on kernel set "
                    f"{key[:12]}, {first} declares {len(listed)}: "
                    f"{sorted(set(kernels) ^ set(listed))}"
                )
            else:
                declared[key] = (relative, kernels)
        family = _json(directory / "VERDICT.json")["families"].get("regional-contract")
        if family is None:
            continue
        (path,) = sorted((directory / "contract" / "regional-contract").glob("*.json"))
        record = _json(path)
        summary = record["summary"]
        stated = _REGIONAL_WHY.search(family["why"])
        assert stated, (anchor.sm, family["why"])
        assert tuple(int(stated[key]) for key in ("passed", "decks", "covered", "declared")) == (
            summary["decks_passed"], summary["decks"], summary["kernels_covered"],
            summary["kernels_declared"],
        ), f"{anchor.sm}: the verdict says {stated[0]!r}, the receipt {summary}"
        assert record["decks_selected"] is False, anchor.sm
        assert (family["class_id"], family["bdy_mask_sha256"]) \
            == (record["class_id"], record["bdy_mask_sha256"]), anchor.sm


def test_an_authority_verdict_claims_no_identity_its_reference_does_not_record(receipts):
    """A pair that reproduced itself on this card is re-anchored: a
    per-architecture digest set.  A cross-card identity claim stands only on
    a reference digest set the campaign recorded (and the evidence files),
    every frame of which the pair matched.  RECEIPT.md states the verdict the
    VERDICT.json records, for the authority and every companion."""

    for anchor, directory in _campaign_anchors(receipts):
        verdict = _json(directory / "VERDICT.json")
        receipt = (directory / "RECEIPT.md").read_text(encoding="utf-8")
        pairs = [("authority", verdict["authority"])] + [
            (f"companion {name}", companion)
            for name, companion in (verdict.get("companions") or {}).items()
        ]
        for label, record in pairs:
            where = f"{anchor.sm} {label}"
            reference = record.get("reference")
            if reference is None:
                assert record["verdict"] == "re-anchored", (
                    f"{where}: verdict {record['verdict']!r} with no reference digest set recorded"
                )
                continue
            assert record["verdict"] in ("re-anchored", "identical"), where
            filed = sorted(
                path for path in directory.rglob(PurePosixPath(reference["path"]).name)
                if path.is_file()
            )
            assert filed, f"{where}: the reference digest set {reference['path']} is not filed"
            frames = {
                name: value.get("digest") if isinstance(value, dict) else value
                for name, value in _json(filed[0]).items()
            }
            pair = {row["file"]: row["digest_a"] for row in record["determinism"]["rows"]}
            matched = sum(1 for name, digest in frames.items() if pair.get(name) == digest)
            assert (reference["frames_in_reference"], reference["frames_identical"]) \
                == (len(frames), matched), where
            assert reference["identical"] is (bool(frames) and matched == len(frames)), where
            if record["verdict"] == "identical":
                assert reference["identical"] is True, where
        authority = verdict["authority"]
        stated = next(line for line in _section(receipt, "The authority anchor").splitlines()
                      if line.strip())
        assert stated == f"{authority['verdict']}: {authority['why']}", (
            f"{anchor.sm}: RECEIPT.md says {stated!r}"
        )
        companions = verdict.get("companions") or {}
        if companions:
            listed = _section(receipt, "Companion pairs (supplementary)")
            for name, companion in companions.items():
                assert f"- {name} (`" in listed and \
                    f"): {companion['verdict']}. {companion['why']}" in listed, (anchor.sm, name)


def test_a_calibration_receipt_is_what_its_process_printed(receipts):
    """A calibration process prints each reading's maximum as it takes it (a
    one-process calibration also prints its summary line).  Every calibration
    receipt that holds its own readings is held to the log of the process
    that printed them, in order, and to that leg's rc: a receipt edited after
    its run, or one whose session was removed from the evidence (the
    committed-rows run that wrote ``contract/perf-calibration-sm89.json``),
    has no process that printed it."""

    for anchor, directory in _campaign_anchors(receipts):
        printed: dict[Path, tuple[list[float], list[dict]]] = {}
        for path in sorted(directory.rglob("*.log")):
            if "attempts" in path.relative_to(directory).parts:
                continue
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            values = [float(line) for line in lines if _PRINTED_READING.fullmatch(line)]
            if values:
                printed[path] = (values, [
                    json.loads(line) for line in lines
                    if line.startswith("{") and line.endswith("}")
                    and '"identity_held_every_reading"' in line
                ])
        calibrations = 0
        for relative, document in _filed(directory, "*.json"):
            if not str(document.get("schema", "")).startswith("gpuwm-hex.arch-perf-calibration") \
                    or not isinstance(document.get("readings"), list):
                continue
            calibrations += 1
            where = f"{anchor.sm} {relative}"
            maxima = [float(reading["maximum"]) for reading in document["readings"]]
            logs = [path for path, (values, _) in printed.items() if values == maxima]
            assert logs, (
                f"{where}: no process log in the evidence printed its {len(maxima)} readings"
            )
            for path in logs:
                label = path.relative_to(directory).as_posix()
                rc = path.with_suffix(".rc")
                if rc.is_file():
                    assert rc.read_text(encoding="utf-8").strip() == "0", (where, label)
                for summary in printed[path][1]:
                    assert {key: document.get(key) for key in summary} == summary, (
                        f"{where}: its summary is not the one {label} printed"
                    )
        assert calibrations, f"{anchor.sm}: no calibration receipt holds its own readings"


def test_every_uuid_an_anchor_records_names_one_device(receipts):
    """The engine FTZ receipts record the device UUID, the one card-unique
    witness in the evidence: every UUID recorded anywhere in an anchor's
    evidence outside attempts/ names the same device."""

    for anchor, directory in _campaign_anchors(receipts):
        recorded: dict[str, str] = {}
        for relative, document in _filed(directory, "*.json"):
            for uuid in _strings_under(document, "uuid"):
                assert _UUID.fullmatch(uuid), (anchor.sm, relative, uuid)
                recorded.setdefault(uuid, relative)
        assert recorded, f"{anchor.sm}: no receipt records a device UUID"
        assert len(recorded) == 1, (
            f"{anchor.sm}: the evidence names {len(recorded)} devices: {recorded}"
        )

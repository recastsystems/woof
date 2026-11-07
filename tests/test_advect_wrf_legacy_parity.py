"""Advection at WRF vert_order 5 against the compiled HRRR fork routines.

The fixture (tests/data/wrf_legacy_advect) runs the eight real-state cases
of the WRF 4.7.1 fixture, by the same input archives, at
``v_sca_adv_order = v_mom_adv_order = 5`` (operational HRRR).  The
reference of record is NOAA-EMC/HRRR 40ee6058c's WRFV3.9 module_advect_em
(tools/advect_wrf_legacy_oracle); WRF 4.7.1 at the same orders is the
cross-check (``<case>-wrf471.npz``).  Every defined output word is
measured; the CUDA words are pinned to their receipt for exact equality.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.advect_oracle import (
    ADVECT_ORACLE_DIR, ROUTINES, SUPPORTED_ROUTINES, SUPPORTED_OUTPUTS,
    advection_orders, load_advect_cases, validate_case, write_fortran_input,
    defined_output_mask, measure_words, measure_advect_parity, advect_port_outputs,
    advect_gpu_identity, advect_nvrtc_key, architecture_is_certified, check_receipt_format,
    receipt_staleness, check_uncertified_stale_records, uncertified_stale_report,
    UncertifiedReceiptStale, UNCERTIFIED_STALE_KEY)

LEGACY_DIR = ADVECT_ORACLE_DIR.parent / "wrf_legacy_advect"
COLUMN_DIR = ADVECT_ORACLE_DIR.parent / "wrf_legacy_vertical_columns"
EXPLICIT = ("advect_scalar", "advect_u", "advect_v", "advect_w")
#: The modules every receipt of this fixture pins by assembled source.
RECEIPT_MODULES = ("advection", "pd_advection", "pd_vertical_sl", "openbc")


def test_native_column_inputs_and_references_hold_their_hashes():
    pins = json.loads((COLUMN_DIR / "input-pins.json").read_text())
    cases = load_advect_cases(COLUMN_DIR)
    assert {c.shape[0] for c in cases} == {7, 50}
    assert len(cases) == 4
    for case in cases:
        row = pins[case.name]
        for suffix, key in ((".npz", "input_sha256"), ("-wrf.npz", "reference_sha256")):
            assert hashlib.sha256((COLUMN_DIR / (case.name + suffix)).read_bytes()).hexdigest() == row[key]
# The fixture's own index (tests/test_advect_wrf471_parity.py reads it the
# same way); a first capture on a new card is added there by
# tools/advect_wrf471_oracle/recapture_receipt.py --install.
_INDEX = json.loads((LEGACY_DIR / "gpu-receipts.json").read_text(encoding="ascii"))
assert _INDEX["schema_version"] == 1
GPU_RECEIPTS = dict(_INDEX["cards"])
ARCH_RECEIPTS = {tuple(int(part) for part in key.split(".")): name
                 for key, name in _INDEX["architectures"].items()}
#: (NVRTC "major.minor", architecture) -> receipt.  The production advection
#: compile allows multiply-add contraction, so its words follow the NVRTC
#: build as well as the card: NVRTC 13.2.51 and 13.4.x put 338 of 2,867,200
#: HRRR-fork words in different places on the same sm_120 card.  A compiler
#: with its own capture replays it ahead of the card/architecture receipt.
COMPILER_RECEIPTS = {(nvrtc, tuple(int(part) for part in key.split("."))): name
                     for nvrtc, rows in _INDEX.get("compilers", {}).items()
                     for key, name in rows.items()}


def _nvrtc_key():
    """``"major.minor"`` of the NVRTC this process compiles with, or None."""
    return advect_nvrtc_key()


def _receipt_for_gpu(gpu_name, compute_capability=None, nvrtc=None):
    architecture = tuple(compute_capability) if compute_capability is not None else None
    name = COMPILER_RECEIPTS.get((nvrtc, architecture)) if nvrtc is not None else None
    if name is None:
        name = GPU_RECEIPTS.get(gpu_name)
    if name is None:
        name = ARCH_RECEIPTS.get(architecture)
    assert name is not None, (
        f"The HRRR-fork advection fixture has no recorded measurements for GPU {gpu_name!r} "
        f"or architecture {architecture}; receipt mappings cover {sorted(GPU_RECEIPTS)} "
        f"and {sorted(ARCH_RECEIPTS)}. This is a test coverage gap.")
    path = LEGACY_DIR / name
    assert path.is_file(), f"HRRR-fork advection receipt {name!r} is missing for GPU {gpu_name!r}"
    return json.loads(path.read_text(encoding="ascii"))


def _device_receipt():
    """The receipt this card's runtime words are compared with.

    Blackwell and newer gate exactly as before (no receipt is a coverage
    gap).  An older card is not certified (ruling 2026-10-04): with no
    receipt, or one that trails the tree, the word comparison is skipped
    and says why; a current receipt still compares every word.
    """
    gpu, capability = advect_gpu_identity()
    if not architecture_is_certified(capability):
        name = GPU_RECEIPTS.get(gpu) or ARCH_RECEIPTS.get(tuple(capability))
        if name is None:
            pytest.skip(f"{gpu} (sm_{capability[0]}{capability[1]}) has no HRRR-fork advection "
                        "receipt and is not certified: byte identity is certified on Blackwell "
                        "and newer only (ruling 2026-10-04)")
        receipt = _receipt_for_gpu(gpu, capability, _nvrtc_key())
        moved = receipt_staleness(receipt, LEGACY_DIR, RECEIPT_MODULES)
        if moved:
            pytest.skip(uncertified_stale_report(
                name, receipt, moved, _INDEX.get(UNCERTIFIED_STALE_KEY, {}).get(name)))
        return receipt
    return _receipt_for_gpu(gpu, capability, _nvrtc_key())


def test_device_receipt_gates_blackwell_and_skips_older_cards_only_when_stale(monkeypatch):
    """The device half of the 2026-10-04 ruling for the HRRR-fork fixture.

    The breakage this prevents: an H100 (no receipt here) or a 4090 whose
    receipt trails the tree failing the device word tests, and the converse,
    a Blackwell card losing its coverage-gap failure or its comparison.
    """
    module = sys.modules[__name__]
    # The card receipts, whichever NVRTC this process has (compiler receipts
    # have their own row below).
    monkeypatch.setattr(module, "_nvrtc_key", lambda: None)

    def card(name, capability, moved=None):
        monkeypatch.setattr(module, "advect_gpu_identity", lambda: (name, capability))
        monkeypatch.setattr(module, "receipt_staleness", lambda *args: dict(moved or {}))

    stale = {"pd_vertical_sl": ("e" * 64, "a" * 64)}
    card("NVIDIA GeForce RTX 5090", (12, 0), stale)
    assert _device_receipt()["gpu"] == "NVIDIA GeForce RTX 5090"
    card("unmeasured Blackwell", (10, 0))
    with pytest.raises(AssertionError, match="test coverage gap"):
        _device_receipt()
    card("NVIDIA H100 80GB HBM3", (9, 0))
    with pytest.raises(pytest.skip.Exception, match="not certified"):
        _device_receipt()
    card("NVIDIA GeForce RTX 4090", (8, 9), stale)
    with pytest.raises(pytest.skip.Exception, match="informational, not a gate"):
        _device_receipt()
    card("NVIDIA GeForce RTX 4090", (8, 9))
    assert _device_receipt()["gpu"] == "NVIDIA GeForce RTX 4090"


def _available_receipts():
    for gpu_name, name in GPU_RECEIPTS.items():
        if (LEGACY_DIR / name).is_file():
            yield name, _receipt_for_gpu(gpu_name)
    for (nvrtc, architecture), name in sorted(COMPILER_RECEIPTS.items()):
        yield name, _receipt_for_gpu(None, architecture, nvrtc)


def test_a_compiler_capture_is_selected_ahead_of_the_card_receipt(monkeypatch):
    """STRICT-04: a compiler with its own capture replays it; every other
    compiler keeps the card/architecture receipt."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "COMPILER_RECEIPTS",
                        {("13.2", (12, 0)): "gpu-receipt-sm120.json"})
    monkeypatch.setattr(module, "GPU_RECEIPTS", {"Card A": "gpu-receipt.json"})
    assert _receipt_for_gpu("Card A", (12, 0), "13.2")["gpu"] == "NVIDIA GeForce RTX 5090"
    assert _receipt_for_gpu("Card A", (12, 0), "13.4")["gpu"] == "NVIDIA GeForce RTX 4090"
    assert _receipt_for_gpu("Card A", (12, 0))["gpu"] == "NVIDIA GeForce RTX 4090"


def _cases():
    return load_advect_cases(LEGACY_DIR)


def _cross_reference(case):
    with np.load(LEGACY_DIR / f"{case.name}-wrf471.npz", allow_pickle=False) as data:
        return {key: np.ascontiguousarray(data[key]) for key in data.files}


def test_fixture_reuses_the_pinned_inputs_at_order_five():
    manifest = json.loads((LEGACY_DIR / "cases.json").read_text(encoding="ascii"))
    base = json.loads((ADVECT_ORACLE_DIR / "cases.json").read_text(encoding="ascii"))
    assert manifest["advection_orders"] == {"h_mom_adv_order": 5, "v_mom_adv_order": 5,
                                            "h_sca_adv_order": 5, "v_sca_adv_order": 5}
    assert manifest["source_sha256"] == base["source_sha256"]
    rows = {row["name"]: row for row in manifest["cases"]}
    base_rows = {row["name"]: row for row in base["cases"]}
    assert set(rows) == set(base_rows)
    for case in _cases():
        validate_case(case)
        row = rows[case.name]
        assert advection_orders(case.metadata) == manifest["advection_orders"]
        assert row["input_sha256"] == base_rows[case.name]["input_sha256"]
        assert hashlib.sha256((LEGACY_DIR / row["file"]).read_bytes()).hexdigest() == row["input_sha256"]
        for key, values in case.inputs.items():
            assert hashlib.sha256(values.tobytes()).hexdigest() == row["array_sha256"][key], (case.name, key)
        kept = {k: v for k, v in case.metadata.items() if not k.endswith("_adv_order")}
        assert kept == base_rows[case.name]["metadata"]
        assert set(ROUTINES).issubset(case.reference)


def test_native_header_carries_the_fixture_orders(tmp_path):
    case = _cases()[0]
    write_fortran_input(case, "advect_scalar", tmp_path / "s.in.bin")
    words = np.fromfile(tmp_path / "s.in.bin", dtype="<i4", count=27)
    assert tuple(words[20:25]) == (1, 5, 5, 5, 5)
    base = load_advect_cases()[0]
    write_fortran_input(base, "advect_scalar", tmp_path / "b.in.bin")
    words = np.fromfile(tmp_path / "b.in.bin", dtype="<i4", count=27)
    assert tuple(words[20:25]) == (1, 5, 3, 5, 3)


def test_fork_reference_equals_wrf471_on_the_explicit_routines_and_not_on_pd():
    """The fork's vert_order 5 arms of advect_scalar, advect_u, advect_v and
    advect_w are the same text as WRF 4.7.1's, so the two compiled
    references agree word for word; advect_scalar_pd differs (the fork's
    GSL semi-Lagrangian upwind low-order vertical flux), so the fixture's
    reference is the fork's own and not a copy."""
    for case in _cases():
        cross = _cross_reference(case)
        for routine in EXPLICIT + ("advect_scalar_mono",):
            row = measure_words(case.reference[routine], cross[routine])
            assert row["different_words"] == 0, (case.name, routine, row)
        row = measure_words(case.reference["advect_scalar_pd"], cross["advect_scalar_pd"])
        assert row["different_words"] > 0, case.name


def test_fork_low_order_flux_has_the_recorded_downstream_cell_behavior():
    """The fork's low-order vertical flux in advect_scalar_pd takes the cell
    on the other side of the face from WRF 4.7.1's ``flux_upwind`` (the
    downstream one) when the face Courant number is at most 1.  The native
    probe receipt (tools/advect_wrf_legacy_oracle/pd_low_order_probe.py,
    both compiled routines on the same input words) shows the consequence:
    the fork drains an EMPTY upstream cell, 4.7.1 leaves it at zero; and in
    the fixture's floor-level tracer case the two references differ by a
    tenth of the largest tendency, not by rounding.  Only the strict WRF
    verification build carries this choice; production order 5 keeps the
    upwind flux at Courant <= 1, because the drained cell's negative value
    is clamped to zero after the update and the clamp creates mass (52.5 t
    of smoke in a 12 h 3 km forecast).  The breakage this prevents: the fork
    distance being read as last-bit rounding (the first write-up of this
    fixture said so), or the fork's flux being described as positive
    definite."""
    tool = Path(__file__).resolve().parents[1] / "tools" / "advect_wrf_legacy_oracle"
    receipt = json.loads((tool / "receipts" / "pd-low-order-probe.json").read_text(encoding="ascii"))
    assert len(receipt["rows"]) == 2
    for row in receipt["rows"]:
        assert row["courant"] <= 1.0
        assert row["fork_drains_the_empty_upstream_cell"] is True
        assert row["fork_tendency_in_the_empty_upstream_cell"] < 0.0
        assert row["wrf471_drains_the_empty_upstream_cell"] is False
        assert row["wrf471_tendency_in_the_empty_upstream_cell"] == 0.0
        occupied = row["levels"].index(row["occupied_level"])
        scale = abs(row["wrf471_z_tendency"][occupied])
        # Flux form on both sides: a redistribution, not a leak.
        assert abs(row["fork_mass_weighted_column_sum"]) < 1e-6 * scale
        assert abs(row["wrf471_mass_weighted_column_sum"]) < 1e-6 * scale
    case = next(case for case in _cases() if case.name == "zero_nearzero_tracers")
    fork = case.reference["advect_scalar_pd"].astype(np.float64)
    cross = _cross_reference(case)["advect_scalar_pd"].astype(np.float64)
    defined = (fork != -999999.0) & (cross != -999999.0)
    scale = np.abs(cross[defined]).max()
    assert np.abs(fork - cross)[defined].max() > 0.05 * scale


def test_order_five_reference_is_not_the_order_three_one():
    three = {case.name: case for case in load_advect_cases()}
    for case in _cases():
        for routine in EXPLICIT + ("advect_scalar_pd",):
            row = measure_words(case.reference[routine], three[case.name].reference[routine])
            assert row["different_words"] > 0, (case.name, routine)


def test_cuda_receipts_pin_the_compiled_source_and_replay_their_words():
    """Blackwell receipts gate their pins; older ones are reported when stale
    (byte identity is certified on Blackwell and newer only, ruling
    2026-10-04).  Every receipt's format and committed words are checked."""
    from woof.core.kernels import module_source
    records = check_uncertified_stale_records(_INDEX, LEGACY_DIR, RECEIPT_MODULES)
    cases = _cases()
    found = False
    for name, receipt in _available_receipts():
        found = True
        check_receipt_format(receipt, LEGACY_DIR, RECEIPT_MODULES, cases)
        if not architecture_is_certified(receipt["compute_capability"]):
            moved = receipt_staleness(receipt, LEGACY_DIR, RECEIPT_MODULES)
            if moved:
                warnings.warn(UncertifiedReceiptStale(
                    uncertified_stale_report(name, receipt, moved, records.get(name))))
            if "fixture" in moved:
                continue
        else:
            for module in RECEIPT_MODULES:
                digest = receipt["kernels"][module]
                assert hashlib.sha256(module_source(module).encode("utf-8")).hexdigest() == digest, (name, module)
            assert receipt["fixture_manifest_sha256"] == hashlib.sha256((LEGACY_DIR / "cases.json").read_bytes()).hexdigest(), name
        words_directory = receipt["words_directory"]
        for case in cases:
            row = receipt["cases"][case.name]
            path = LEGACY_DIR / words_directory / row["words_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["words_sha256"], (name, case.name)
            with np.load(path, allow_pickle=False) as data:
                outputs = {key: data[key] for key in data.files}
            got = measure_advect_parity(case, outputs)
            assert got == {key: row["measurements"][key] for key in SUPPORTED_OUTPUTS}, (name, case.name)
        assert receipt["mutation_rejected"] is True
    assert found, "no HRRR-fork advection receipt is recorded"


@pytest.fixture(scope="module")
def device_receipt():
    """This card's receipt, resolved before ``cuda_cases`` launches anything.

    A test that compares receipt words asks for this fixture ahead of
    ``cuda_cases``, so on an uncertified card whose receipt is missing or
    stale it skips before every case's kernels run for nothing.
    """
    return _device_receipt()


@pytest.fixture(scope="module")
def cuda_cases():
    gpu, capability = advect_gpu_identity()
    if architecture_is_certified(capability):
        # A certified card with no receipt is a coverage gap: fail before the launches.
        _receipt_for_gpu(gpu, capability)
    return {case.name: (case, advect_port_outputs(case)) for case in _cases()}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("routine", SUPPORTED_ROUTINES)
def test_cuda_routine_holds_every_measured_fork_word(routine, device_receipt, cuda_cases):
    receipt = device_receipt
    for name, (case, outputs) in cuda_cases.items():
        row = measure_words(outputs[routine], case.reference[routine])
        assert row == receipt["cases"][name]["measurements"][routine], (name, routine, row)


@pytest.mark.gpu
@requires_gpu
def test_cuda_pd_optional_channels_hold_every_defined_word(device_receipt, cuda_cases):
    receipt = device_receipt
    for name, (case, outputs) in cuda_cases.items():
        for key in ("advect_scalar_pd.h_tendency", "advect_scalar_pd.z_tendency"):
            row = measure_words(outputs[key], case.reference[key], defined=defined_output_mask(case, key))
            assert row == receipt["cases"][name]["measurements"][key], (name, key, row)


@pytest.mark.gpu
@requires_gpu
def test_cuda_order_five_words_differ_from_the_order_three_fixture(cuda_cases):
    """The same inputs through the same launchers at vorder 3 are the 4.7.1
    fixture's recorded production words; at vorder 5 every explicit
    routine's words move, so the argument reaches the kernels."""
    three = {case.name: case for case in load_advect_cases()}
    for name, (case, outputs) in cuda_cases.items():
        base = advect_port_outputs(three[name])
        for routine in EXPLICIT + ("advect_scalar_pd",):
            assert measure_words(outputs[routine], base[routine])["different_words"] > 0, (name, routine)


@pytest.mark.gpu
@requires_gpu
def test_exact_build_holds_the_fork_reference_bitwise_on_the_explicit_routines():
    """Under the strict WRF arithmetic process the four explicit routines
    reproduce every word of the fork reference on the specified and
    periodic cases, including the scalar total tendency with the fork's
    semi-Lagrangian low-order vertical flux."""
    from woof.wrf_exact import ADVECTION_ENABLED
    if not ADVECTION_ENABLED:
        pytest.skip("requires an exact advection process")
    from tools.advect_wrf_exact.compare import compare
    outputs = {}
    results = compare(LEGACY_DIR, outputs)
    selected = {case.name: case for case in _cases()
                if case.metadata["specified"] or not (case.metadata["open_x"] or case.metadata["open_y"])}
    assert selected
    for name, case in selected.items():
        for routine in EXPLICIT:
            assert results[name][routine]["different_words"] == 0, (name, routine, results[name][routine])
        row = measure_words(outputs[name]["advect_scalar_pd"], case.reference["advect_scalar_pd"])
        assert row["different_words"] == 0, (name, row)
    columns = compare(COLUMN_DIR)
    for name, rows in columns.items():
        for routine in SUPPORTED_ROUTINES:
            assert rows[routine]["different_words"] == 0, (name, routine, rows[routine])

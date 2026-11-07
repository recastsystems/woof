"""The advection receipt recapture procedure, on synthetic captures.

The breakage this prevents: an advection oracle receipt pins the kernel
source it measured, so a kernel edit leaves the receipts of cards no lane
can reach red (the H100 receipt of the WRF 4.7.1 fixture after the
vertical-order lane, and no H100 receipt at all for the HRRR-fork fixture).
The move happens on a rented box, by whoever has one, and that is where a
receipt gets words that were never measured or a production change on that
architecture gets recorded as the new truth.
``tools/advect_wrf471_oracle/recapture_receipt.py`` is the one procedure:
these rows hold its comparison and install rules (``settle``) without a
GPU, and the last rows hold the committed receipt indexes the device tests
select through.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "advect_wrf471_oracle" / "recapture_receipt.py"
CASES = ("case_a", "case_b")


@pytest.fixture()
def tool(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("recapture_receipt", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO", tmp_path.resolve())
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _words(value):
    return {"advect_scalar": np.full((2, 3), value, dtype=np.float32)}


def _rows(folder, variant, values, *, write=True):
    rows = {}
    for case in CASES:
        name = f"{case}-{variant}.npz"
        words = _words(values[case])
        if write:
            np.savez_compressed(folder / name, **words)
        digest = hashlib.sha256(words["advect_scalar"].tobytes()).hexdigest()
        rows[case] = {"measurements": {"advect_scalar": {"got_sha256": digest, "max_ulp": 0}},
                      "levels": {"advect_scalar": [{"different_words": 0}]},
                      "words_file": name,
                      "words_sha256": _sha(folder / name) if write else "0" * 64}
    return rows


def _receipt(folder, words_directory, gpu, capability, kernels, production, controls,
             *, stored_controls=True, mutation_rejected=True):
    (folder / words_directory).mkdir(parents=True, exist_ok=True)
    words = folder / words_directory
    return {"schema_version": 1, "gpu": gpu, "compute_capability": list(capability),
            "cupy": "14.2.0", "cuda_runtime": 13020, "fixture_manifest_sha256": "f" * 64,
            "kernels": {"advection": kernels, "pd_advection": kernels, "openbc": "o" * 64},
            "words_directory": words_directory,
            "cases": _rows(words, "production", production),
            "controls": {name: _rows(words, name, values, write=stored_controls)
                         for name, values in controls.items()},
            "mutation_rejected": mutation_rejected}


def _write(path, value):
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii"))


BASE = {"case_a": 1.0, "case_b": 2.0}
MUTATED = {"case_a": 1.5, "case_b": 2.5}


def _fixture(tmp_path, name, cards, architectures, receipts):
    tmp_path = tmp_path.resolve()
    fixture = tmp_path / "tests" / "data" / name
    fixture.mkdir(parents=True)
    for file_name, (gpu, capability, folder) in receipts.items():
        receipt = _receipt(fixture, folder, gpu, capability, "old" + "0" * 61, BASE,
                           {"mutation": MUTATED, "wrf_flux": BASE})
        _write(fixture / file_name, receipt)
    _write(fixture / "gpu-receipts.json", {"schema_version": 1, "cards": cards,
                                           "architectures": architectures})
    if name == "wrf471_advect":
        lines = [f"{_sha(p)}  {p.relative_to(tmp_path).as_posix()}"
                 for p in sorted(fixture.rglob("*")) if p.is_file() and p.name != "gpu-receipts.json"]
        (fixture / "oracle-sha256sums.txt").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    return fixture


def _capture(tmp_path, gpu, capability, *, production=BASE, wrf_flux=BASE, mutation_rejected=True):
    staged = tmp_path / "scratch" / "data"
    staged.mkdir(parents=True, exist_ok=True)
    receipt = _receipt(staged, "captured-words", gpu, capability, "new" + "0" * 61, production,
                       {"mutation": MUTATED, "wrf_flux": wrf_flux},
                       mutation_rejected=mutation_rejected)
    _write(staged / "captured-receipt.json", receipt)
    return staged / "captured-receipt.json", staged / "captured-words"


def _snapshot(fixture):
    return {p.relative_to(fixture).as_posix(): _sha(p) for p in sorted(fixture.rglob("*")) if p.is_file()}


def test_a_card_capture_that_reproduces_production_moves_its_pins_and_sums(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    receipt, words = _capture(tmp_path, "Card A", (8, 9), wrf_flux={"case_a": 1.0, "case_b": 2.25})
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert (status, record["mode"], record["production_reproduced"]) == (0, "card", True)
    assert record["every_variant_reproduced"] is False  # the control moved and is re-recorded
    assert record["production_words_compared"] == 12 and record["production_words_different"] == 0
    installed = json.loads((fixture / "gpu-receipt.json").read_text(encoding="ascii"))
    assert installed["kernels"]["advection"].startswith("new")
    assert installed["words_directory"] == "gpu-words"
    captured = json.loads(receipt.read_text(encoding="ascii"))
    assert installed["controls"]["wrf_flux"] == captured["controls"]["wrf_flux"]
    assert installed["cases"] == captured["cases"]
    sums = (fixture / "oracle-sha256sums.txt").read_text(encoding="ascii")
    assert f"{_sha(fixture / 'gpu-receipt.json')}  tests/data/wrf471_advect/gpu-receipt.json" in sums
    assert record["installed"] is True


def test_a_card_capture_whose_production_moved_installs_nothing(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    before = _snapshot(fixture)
    receipt, words = _capture(tmp_path, "Card A", (8, 9), production={"case_a": 1.0, "case_b": 2.0000002})
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert status == 1 and record["installed"] is False
    assert record["production_words_different"] == 6
    assert record["variants"]["production"]["cases"]["case_b"]["differing_measurements"] == ["advect_scalar"]
    assert "finding" in record
    assert _snapshot(fixture) == before


def test_a_verified_native_fix_requalifies_only_its_own_fork_card(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf_legacy_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    receipt, words = _capture(tmp_path, "Card A", (8, 9), production=MUTATED)
    status, record = tool.settle(fixture, receipt, words, install=True, native_fix_verified=True)
    assert status == 0 and record["installed"] is True
    assert record["production_reproduced"] is False
    assert record["native_fix_verified"] is True
    installed = json.loads((fixture / "gpu-receipt.json").read_text())
    assert installed["cases"] == json.loads(receipt.read_text())["cases"]


def test_a_native_fix_cannot_requalify_the_order_three_fixture(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    before = _snapshot(fixture)
    receipt, words = _capture(tmp_path, "Card A", (8, 9), production=MUTATED)
    status, record = tool.settle(fixture, receipt, words, install=True, native_fix_verified=True)
    assert status == 2 and not record["installed"]
    assert _snapshot(fixture) == before


def test_an_architecture_capture_with_a_moved_control_is_refused_then_kept_as_its_own_card(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf_legacy_advect", {"Card A": "gpu-receipt-sm120.json"},
                       {"12.0": "gpu-receipt-sm120.json"},
                       {"gpu-receipt-sm120.json": ("Card A", (12, 0), "gpu-words-sm120")})
    before = _snapshot(fixture)
    receipt, words = _capture(tmp_path, "Card B", (12, 0), wrf_flux={"case_a": 1.25, "case_b": 2.0})
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert (status, record["mode"], record["production_reproduced"]) == (2, "architecture", True)
    assert "Card A" in record["refused"] and _snapshot(fixture) == before
    status, record = tool.settle(fixture, receipt, words, install=True, as_card=True)
    assert (status, record["mode"], record["receipt"]) == (0, "first", "gpu-receipt-card-b.json")
    index = json.loads((fixture / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["cards"]["Card B"] == "gpu-receipt-card-b.json"
    assert index["architectures"]["12.0"] == "gpu-receipt-sm120.json"  # Card A's mapping stays
    own = json.loads((fixture / "gpu-receipt-card-b.json").read_text(encoding="ascii"))
    assert own["gpu"] == "Card B" and own["words_directory"] == "gpu-words-card-b"
    assert all((fixture / "gpu-words-card-b" / f"{case}-{variant}.npz").is_file()
               for case in CASES for variant in ("production", "mutation", "wrf_flux"))


def test_an_architecture_capture_that_reproduces_everything_moves_pins_and_maps_the_card(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card A": "gpu-receipt-h100.json"}, {},
                       {"gpu-receipt-h100.json": ("Card A", (9, 0), "gpu-words-h100")})
    receipt, words = _capture(tmp_path, "Card A2", (9, 0))
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert (status, record["mode"], record["every_variant_reproduced"]) == (0, "architecture", True)
    installed = json.loads((fixture / "gpu-receipt-h100.json").read_text(encoding="ascii"))
    assert installed["gpu"] == "Card A" and installed["kernels"]["advection"].startswith("new")
    index = json.loads((fixture / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["architectures"] == {"9.0": "gpu-receipt-h100.json"}


def test_installing_an_uncertified_receipt_again_drops_its_staleness_record(tool, tmp_path):
    """The parity tests refuse a staleness record whose receipt is current
    again; the one procedure that makes it current must take the record with it."""
    from woof.verify.advect_oracle import UNCERTIFIED_STALE_KEY
    assert tool.STALE_RECORD_KEY == UNCERTIFIED_STALE_KEY
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card H": "gpu-receipt-h100.json"}, {},
                       {"gpu-receipt-h100.json": ("Card H", (9, 0), "gpu-words-h100")})
    index_path = fixture / "gpu-receipts.json"
    index = json.loads(index_path.read_text(encoding="ascii"))
    index[UNCERTIFIED_STALE_KEY] = {"gpu-receipt-h100.json": {
        "staled_by": "b" * 40, "pins": {"advection": "old" + "0" * 61}}}
    tool.write_index(fixture, index)
    receipt, words = _capture(tmp_path, "Card H", (9, 0))
    status, record = tool.settle(fixture, receipt, words, install=False)
    assert status == 0 and UNCERTIFIED_STALE_KEY in json.loads(index_path.read_text(encoding="ascii"))
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert (status, record["mode"], record["stale_record_dropped"]) == (0, "card", True)
    assert UNCERTIFIED_STALE_KEY not in json.loads(index_path.read_text(encoding="ascii"))
    assert "gpu-receipts.json" in record["written"]


def test_a_first_capture_writes_its_receipt_words_and_index(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf_legacy_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    receipt, words = _capture(tmp_path, "Card H", (9, 0))
    status, record = tool.settle(fixture, receipt, words, install=False)
    assert (status, record["mode"], record["installed"]) == (0, "first", False)
    assert not (fixture / "gpu-receipt-sm90.json").exists()
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert (status, record["receipt"]) == (0, "gpu-receipt-sm90.json")
    index = json.loads((fixture / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["cards"]["Card H"] == "gpu-receipt-sm90.json"
    assert index["architectures"]["9.0"] == "gpu-receipt-sm90.json"
    written = json.loads((fixture / "gpu-receipt-sm90.json").read_text(encoding="ascii"))
    for case in CASES:
        row = written["cases"][case]
        assert _sha(fixture / "gpu-words-sm90" / row["words_file"]) == row["words_sha256"]


def test_a_compiler_capture_is_recorded_beside_the_card_receipt(tool, tmp_path):
    """STRICT-04: words that follow the NVRTC build get that build's own
    receipt; the card receipt and its other compilers are untouched."""
    fixture = _fixture(tmp_path, "wrf_legacy_advect", {"Card A": "gpu-receipt.json"},
                       {"12.0": "gpu-receipt.json"},
                       {"gpu-receipt.json": ("Card A", (12, 0), "gpu-words")})
    card = _sha(fixture / "gpu-receipt.json")
    receipt, words = _capture(tmp_path, "Card A", (12, 0), production={"case_a": 1.0, "case_b": 2.5})
    status, record = tool.settle(fixture, receipt, words, install=False, nvrtc="13.2")
    assert (status, record["mode"]) == (1, "card")
    status, record = tool.settle(fixture, receipt, words, install=True, nvrtc="13.2", as_compiler=True)
    assert (status, record["mode"], record["installed"]) == (0, "compiler-first", True)
    assert record["receipt"] == "gpu-receipt-sm120-nvrtc13.2.json"
    assert _sha(fixture / "gpu-receipt.json") == card
    index = json.loads((fixture / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["compilers"] == {"13.2": {"12.0": "gpu-receipt-sm120-nvrtc13.2.json"}}
    assert tool.select_receipt(index, fixture, "Card A", (12, 0), "13.2") == (
        "compiler", "gpu-receipt-sm120-nvrtc13.2.json")
    assert tool.select_receipt(index, fixture, "Card A", (12, 0), "13.4") == ("card", "gpu-receipt.json")
    written = json.loads((fixture / "gpu-receipt-sm120-nvrtc13.2.json").read_text(encoding="ascii"))
    assert written["nvrtc"] == [13, 2]
    for case in CASES:
        row = written["cases"][case]
        assert _sha(fixture / "gpu-words-sm120-nvrtc13.2" / row["words_file"]) == row["words_sha256"]
    # A folder that stores production only gets no control words beside it.
    for control in ("mutation", "wrf_flux"):
        (fixture / "gpu-words" / f"case_a-{control}.npz").unlink()
    index["compilers"] = {}
    (fixture / "gpu-receipt-sm120-nvrtc13.2.json").unlink()
    for path in (fixture / "gpu-words-sm120-nvrtc13.2").iterdir():
        path.unlink()
    (fixture / "gpu-words-sm120-nvrtc13.2").rmdir()
    _write(fixture / "gpu-receipts.json", index)
    status, record = tool.settle(fixture, receipt, words, install=True, nvrtc="13.2", as_compiler=True)
    assert status == 0
    assert sorted(p.name for p in (fixture / "gpu-words-sm120-nvrtc13.2").iterdir()) == [
        "case_a-production.npz", "case_b-mutation.npz", "case_b-production.npz", "case_b-wrf_flux.npz"]
    # The same compiler again must reproduce its own receipt.
    status, record = tool.settle(fixture, receipt, words, install=False, nvrtc="13.2")
    assert (status, record["mode"], record["production_reproduced"]) == (0, "compiler", True)


def test_a_capture_whose_mutation_was_not_rejected_is_refused(tool, tmp_path):
    fixture = _fixture(tmp_path, "wrf471_advect", {"Card A": "gpu-receipt.json"}, {},
                       {"gpu-receipt.json": ("Card A", (8, 9), "gpu-words")})
    before = _snapshot(fixture)
    receipt, words = _capture(tmp_path, "Card A", (8, 9), mutation_rejected=False)
    status, record = tool.settle(fixture, receipt, words, install=True)
    assert status == 2 and "mutation" in record["refused"] and _snapshot(fixture) == before


@pytest.mark.parametrize("name", ("wrf471_advect", "wrf_legacy_advect"))
def test_the_committed_receipt_indexes_name_what_the_receipts_say(name):
    """The device tests select through these files; each mapping must land
    on a receipt measured by that card or of that architecture."""
    fixture = ROOT / "tests" / "data" / name
    index = json.loads((fixture / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["schema_version"] == 1 and "NVIDIA GeForce RTX 4090" in index["cards"]
    for card, receipt_name in index["cards"].items():
        receipt = json.loads((fixture / receipt_name).read_text(encoding="ascii"))
        assert receipt["gpu"] == card, (receipt_name, receipt["gpu"])
        assert (fixture / receipt["words_directory"]).is_dir()
    for key, receipt_name in index["architectures"].items():
        receipt = json.loads((fixture / receipt_name).read_text(encoding="ascii"))
        assert receipt["compute_capability"] == [int(part) for part in key.split(".")]
    for nvrtc, rows in index.get("compilers", {}).items():
        for key, receipt_name in rows.items():
            receipt = json.loads((fixture / receipt_name).read_text(encoding="ascii"))
            assert receipt["compute_capability"] == [int(part) for part in key.split(".")]
            assert receipt["nvrtc"] == [int(part) for part in nvrtc.split(".")]
            assert (fixture / receipt["words_directory"]).is_dir()

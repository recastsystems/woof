"""The registered native-undefined exclusion stays exactly as narrow as it was signed.

Proof-bar ruling 1 admits excluding words where native MPAS-A itself reads
undefined memory only with a full witness and a recorded sign-off for that
case. tools/native_undefined_exclusions.json holds one such case: p21-rad-shared's
exact final ozone-table latitude, ozmixm words 0..707 (the first global cell).
These tests fail if the set widens or moves, if it is asked for on another unit,
case or field, if any word outside it may differ, if the register grows a graded
allowance, or if any module other than the named consumer starts using it.
The witness and all-input receipts are home evidence under the repository-root
receipts/ folder; those checks skip, naming the folder, where it is absent.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]
REPO = TREE.parent
TOOL = TREE / "tools" / "native_undefined_exclusions.py"
EVIDENCE = REPO / "receipts" / "native-undefined" / "p21-rad-shared"
UNIT, CASE, FIELD = "p21-rad-shared", "exact-final-ozone-table-latitude", "ozmixm"
SHAPE = (2562, 59, 12)
LABEL = "DECLARED RESIDUE (native reads undefined memory); never called identical"
needs_evidence = pytest.mark.skipif(
    not EVIDENCE.is_dir(), reason=f"home witness receipts absent: {EVIDENCE.relative_to(REPO)}")


@pytest.fixture(scope="module")
def nue():
    spec = importlib.util.spec_from_file_location("native_undefined_exclusions_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def entry(nue):
    return nue.entry(UNIT, CASE, FIELD)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_register_holds_exactly_the_signed_case(nue):
    entries = nue.load()["entries"]
    assert [e["id"] for e in entries] == ["p21-ozone-exact-final-latitude"]
    e = entries[0]
    assert (e["unit"], e["case"], e["field"], e["cell_index"]) == (UNIT, CASE, FIELD, 0)
    assert e["netcdf_c_words"] == {"first": 0, "last": 707, "count": 708}
    assert e["layout"] == {"netcdf_dims": ["nCells", "nOznLevels", "nMonths"], "order": "C", "shape_tail": [59, 12]}
    assert e["mpas_indices"] == "ozmixm(month=1:12, level=1:59, cell=1)"
    assert e["label"] == nue.LABEL == LABEL
    assert e["native"]["source"].endswith("physics/mpas_atmphys_o3climatology.F")
    assert e["native"]["lines"] == "164-184"
    assert (e["native"]["undefined_read"]["line"], e["native"]["undefined_read"]["index"]) == (165, 65)
    assert e["native"]["commit"] == "f34984b2c89353cf31c235b08d3b4acd96b37bb3"
    assert (e["sign_off"]["date"], e["sign_off"]["verbatim"]) == ("2026-10-14", "the moas thing sure")
    w = e["witness"]
    assert (w["runs"], w["hosts"], w["process_counts"], w["varying_words"]) == (8, 2, [1, 2], 47)
    assert all(i % 12 == 11 for i in w["varying_word_indices"])
    assert (e["process_count"]["native"], e["process_count"]["hex"]) == (1, 1)
    # One consumer, in the p21 replay folder. The register names it; that file stays home, so this
    # shipped test reads the name from the register instead of repeating it.
    assert len(e["consumers"]) == 1 and e["consumers"][0].startswith("tests/radshared_v842/")


def test_mask_is_exactly_the_first_cells_708_words(nue, entry):
    mask = nue.excluded_mask(entry, SHAPE)
    assert mask.shape == (int(np.prod(SHAPE)),)
    assert mask.sum() == 708 and mask[:708].all() and not mask[708:].any()


@pytest.mark.parametrize("unit,case,field", [
    (UNIT, CASE, "o3clim"),
    (UNIT, CASE, "o3vmr"),
    (UNIT, "march-real-lw-pre", FIELD),
    (UNIT, "exact-first-ozone-table-latitude", FIELD),
    ("p22-rrtmg-lw", CASE, FIELD),
])
def test_no_other_unit_case_or_field_gets_the_exclusion(nue, unit, case, field):
    with pytest.raises(nue.ExclusionRefusal):
        nue.entry(unit, case, field)


@pytest.mark.parametrize("shape", [(2562, 60, 12), (2562, 59, 11), (2562, 708), (0, 59, 12), (2562, 12, 59), (1, 2562, 59, 12)])
def test_other_layouts_are_refused(nue, entry, shape):
    with pytest.raises(nue.ExclusionRefusal):
        nue.excluded_mask(entry, shape)


def _pair():
    native = np.random.default_rng(165).standard_normal(SHAPE).astype("<f4")
    return native, native.copy()


def test_every_word_outside_the_set_is_compared(nue, entry):
    native, actual = _pair()
    actual.reshape(-1)[:708] = np.nan
    result = nue.compare(native, actual, entry)
    assert (result["words"], result["excluded_words"], result["compared_words"]) == (1_813_896, 708, 1_813_188)
    assert result["differing_words_outside_set"] == 0
    assert result["differing_words_inside_set_diagnostic"] == 708
    assert result["label"] == LABEL
    nue.require_identical_outside(result)


@pytest.mark.parametrize("word", [708, 709, 719, 2 * 708, 1_813_895])
def test_one_differing_word_outside_the_set_fails(nue, entry, word):
    native, actual = _pair()
    flat = actual.reshape(-1)
    flat[word] = np.nextafter(flat[word], np.float32(np.inf))
    result = nue.compare(native, actual, entry)
    assert result["differing_words_outside_set"] == 1
    assert result["first_differing_outside"] == [word]
    with pytest.raises(nue.ExclusionRefusal):
        nue.require_identical_outside(result)


def test_signed_zero_outside_the_set_is_a_difference(nue, entry):
    native = np.zeros(SHAPE, dtype="<f4")
    actual = native.copy()
    actual.reshape(-1)[708] = np.float32(-0.0)
    assert nue.compare(native, actual, entry)["differing_words_outside_set"] == 1


def test_compare_refuses_mismatched_or_wide_words(nue, entry):
    native, actual = _pair()
    with pytest.raises(nue.ExclusionRefusal):
        nue.compare(native, actual[:-1], entry)
    with pytest.raises(nue.ExclusionRefusal):
        nue.compare(native.astype("<f8"), actual.astype("<f8"), entry)


@pytest.mark.parametrize("fault", [
    "wider", "shifted", "other_cell", "tolerance", "rms", "unsigned", "no_variation",
    "variation_outside", "one_host", "one_process_count", "no_bounds", "label",
    "duplicate", "no_consumer", "no_process_count", "unknown_key", "receipt_digest",
])
def test_register_faults_are_refused(nue, tmp_path, fault):
    register = json.loads(nue.REGISTER.read_text(encoding="utf-8"))
    e = register["entries"][0]
    w = e["witness"]
    if fault == "wider": e["netcdf_c_words"].update(last=708, count=709)
    elif fault == "shifted": e["netcdf_c_words"].update(first=1, last=708)
    elif fault == "other_cell": e["cell_index"] = 1
    elif fault == "tolerance": w["tolerance_ulp"] = 4
    elif fault == "rms": e["process_count"]["rms"] = 1e-7
    elif fault == "unsigned": e["sign_off"]["verbatim"] = ""
    elif fault == "no_variation": w.update(varying_word_indices=[], varying_words=0)
    elif fault == "variation_outside": w.update(varying_word_indices=w["varying_word_indices"] + [719],
                                                varying_words=w["varying_words"] + 1)
    elif fault == "one_host": w["hosts"] = 1
    elif fault == "one_process_count": w["process_counts"] = [1]
    elif fault == "no_bounds": w["bounds_check"] = {}
    elif fault == "label": e["label"] = "BITWISE"
    elif fault == "duplicate": register["entries"].append(dict(e, id="copy"))
    elif fault == "no_consumer": e["consumers"] = []
    elif fault == "no_process_count": e["process_count"] = {}
    elif fault == "unknown_key": e["applies_to_all_cells"] = True
    elif fault == "receipt_digest": w["receipts"][0]["sha256"] = "0"
    path = tmp_path / "register.json"
    path.write_text(json.dumps(register), encoding="utf-8")
    with pytest.raises(nue.ExclusionRefusal):
        nue.load(path)


def test_only_the_named_consumer_uses_the_register(nue):
    e = nue.load()["entries"][0]
    needles = ("native_undefined_exclusions", e["id"], e["case"])
    allowed = {"tools/native_undefined_exclusions.py", "tests/test_native_undefined_exclusions_v842.py", *e["consumers"]}
    users = set()
    for root in ("src", "tools", "tests"):
        for path in (TREE / root).rglob("*"):
            if path.suffix in {".py", ".sh", ".rs"} and path.is_file():
                if any(n in path.read_text(encoding="utf-8", errors="ignore") for n in needles):
                    users.add(path.relative_to(TREE).as_posix())
    assert "tools/native_undefined_exclusions.py" in users
    assert users <= allowed, f"register used outside its named consumer: {sorted(users - allowed)}"
    for consumer in e["consumers"]:
        path = TREE / consumer
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "native_undefined_exclusions" in text and repr(e["case"]) in text and repr(e["field"]) in text


@needs_evidence
def test_witness_receipts_reproduce_the_registered_variation(nue, entry):
    w = entry["witness"]
    for receipt in w["receipts"]:
        assert _sha(REPO / receipt["path"]) == receipt["sha256"], receipt["path"]
    primary = [json.loads((REPO / r["path"]).read_text()) for r in w["receipts"] if r["role"] == "primary"]
    rows = [row for receipt in primary for row in receipt["runs"]]
    assert len(rows) == w["runs"] and len({r["host"] for r in primary}) == w["hosts"]
    assert {r["native_sha256"] for r in primary} == {w["native_binary_sha256"]}
    assert {r["fixture_sha256"] for r in primary} == {w["fixture_sha256"]}
    assert {r["input_sha256"] for r in primary} == {w["input_sha256"]}
    assert sorted({row["ranks"] for row in rows}) == w["process_counts"]
    assert all(row["returncode"] == 0 for row in rows)
    words = np.array([row["fields"]["ozmixm"]["affected_cell_words"] for row in rows], dtype=np.uint64)
    assert words.shape == (8, 708)
    assert np.flatnonzero((words != words[0]).any(axis=0)).tolist() == w["varying_word_indices"]
    bounds = json.loads((REPO / next(r["path"] for r in w["receipts"] if r["role"] == "bounds")).read_text())
    assert any(run.get("kind") == "bounds" and "line 165 of file mpas_atmphys_o3climatology.F" in run.get("stderr", "")
               and "Index '65'" in run["stderr"] for run in bounds["runs"])
    sign = json.loads((REPO / entry["sign_off"]["record"]).read_text())
    assert (sign["entry_id"], sign["date"], sign["verbatim"]) == (entry["id"], entry["sign_off"]["date"],
                                                                 entry["sign_off"]["verbatim"])


@needs_evidence
def test_all_input_check_left_out_only_the_registered_words(nue, entry):
    check = entry["all_input_check"]
    path = REPO / check["receipt"]
    assert _sha(path) == check["sha256"]
    receipt = json.loads(path.read_text())
    assert receipt["register_entry"] == {k: v for k, v in entry.items() if k != "all_input_check"}
    assert receipt["tier"] == LABEL
    c = receipt["all_input_comparison"]
    assert (c["entry_id"], c["field"], c["shape"]) == (entry["id"], FIELD, list(SHAPE))
    assert (c["words"], c["excluded_words"], c["compared_words"]) == (1_813_896, 708, 1_813_188)
    assert c["differing_words_outside_set"] == 0 == check["differing_words_outside_set"]
    assert (check["field_words"], check["excluded_words"], check["compared_words"]) == (1_813_896, 708, 1_813_188)
    assert receipt["excluded_words"]["netcdf_c_word_indices"] == list(range(708))
    assert set(receipt["observed_varying_word_indices"]) <= set(range(708))
    assert (receipt["process_count"]["native"], receipt["process_count"]["hex"]) == (1, 1)
    assert (check["native_process_count"], check["hex_process_count"]) == (1, 1)
    assert receipt["backend"] == check["backend"]
    assert receipt["intended_endpoint_diagnostic"]["differing_words"] == 0
    masked = receipt["full_field_masked_witnesses"]
    assert masked and all(m["native_sha256"] == m["reconstructed_sha256"] for m in masked)

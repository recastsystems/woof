"""The shared edits most easily lost while stacking independent lanes."""
import ast
from pathlib import Path

from woof.core import kernels

ROOT = Path(__file__).resolve().parents[1]


def test_exact_acoustic_restart_retains_allocated_carriers_only():
    import numpy as np
    from woof.io import restart

    class State:
        _scratch = {}

    state = State()
    assert restart._scratch_manifest(state) == {}
    words = np.array([0x3f800001, 0x80000000], np.uint32).view(np.float32)
    state._scratch = {slot: words.copy() for slot in
                      restart.CHECKPOINTED_CARRIED_SCRATCH_SLOTS}
    manifest = restart._scratch_manifest(state)
    assert set(manifest) == {"scratch/wrf_exact_muts", "scratch/wrf_exact_mu_halo"}
    for slot in state._scratch:
        assert restart._restorable_scratch_slot(slot)
        assert manifest["scratch/" + slot].tobytes() == words.tobytes()
    assert not restart._restorable_scratch_slot("rk_ru")


def test_extra_headers_have_one_row_per_module():
    tree = ast.parse((ROOT / "woof/core/kernels/__init__.py").read_text())
    assignment = next(node for node in tree.body
                      if ((isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_EXTRA_HEADERS")
                          or (isinstance(node, ast.Assign) and any(getattr(t,"id",None)=="_EXTRA_HEADERS" for t in node.targets))))
    keys = [ast.literal_eval(key) for key in assignment.value.keys]
    assert len(keys) == len(set(keys)), "a later header row silently shadows a carried recipe"
    assert kernels._EXTRA_HEADERS["sfclay"] == ("glibc_flt32.cuh", "sfclay_classic.cuh")


def test_every_strict_intake_no_contraction_recipe_is_retained(monkeypatch):
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", True)
    modules = ("kessler", "wsm6", "wdm6", "wdm6_refl", "morrison",
               "milbrandt2", "milbrandt2_zet", "myjpbl", "myjsfc",
               "mynn_pbl", "mynn_surface", "sfclay", "noah", "noah_init")
    for module in modules:
        assert "--fmad=false" in kernels.module_options(module), module


def test_mynn_richardson_number_squares_psi_before_multiplying():
    source = (ROOT / "woof/core/kernels/mynn_surface.cu").read_text()
    assert "result = ri * (psix2 * psix2) / psit2;" in source
    assert "result = ri * psix2 * psix2 / psit2;" not in source


def test_shinhong_has_one_tanh_implementation_and_probe():
    source = (ROOT / "woof/core/kernels/shinhong.cu").read_text()
    assert source.count("real sh_tanhf(real x)") == 1
    assert source.count("void shinhong_tanhf_probe(") == 1


def test_boundary_preprocessing_uses_shared_carried_mass_key():
    from woof.state_serialization_contract import WRF_MUTS_SLOT
    assert WRF_MUTS_SLOT == "wrf_exact_muts"
    boundary = (ROOT / "woof/ingest/lateral_bc.py").read_text()
    dycore = (ROOT / "woof/core/dycore.py").read_text()
    shared = "from woof.state_serialization_contract import WRF_MUTS_SLOT"
    assert shared in dycore
    # The boundary reads the carried mass through the contract's own
    # accessor (cold-carrier aware), which keys on WRF_MUTS_SLOT; it spells
    # neither the slot literal nor a dycore-local key.
    assert ("from woof.state_serialization_contract import carried_wrf_muts"
            in boundary)
    import inspect
    from woof import state_serialization_contract as contract
    assert "existing_scratch(WRF_MUTS_SLOT)" in inspect.getsource(
        contract.carried_wrf_muts)
    assert '"wrf_exact_muts"' not in boundary and "'wrf_exact_muts'" not in boundary
    assert "from woof.core.dycore import WRF_MUTS_SLOT" not in boundary


def test_morrison_keeps_chunked_scratch_and_validator_restore():
    from woof.core.morrison_constants import (
        NWORK, WORK_FLOOR_BYTES, work_chunk_columns, work_slot_shape)
    for ncol in (1, 21, 22, 25600):
        assert work_chunk_columns(ncol) == (ncol + NWORK - 1) // NWORK
        assert work_slot_shape(50, ncol)[0] <= max(
            50 * (ncol + NWORK), WORK_FLOOR_BYTES // 4)
    source = (ROOT / "tools/morrison_wrf461_oracle/validate_morrison_oracle.py").read_text()
    assert "finally:" in source
    assert "kernels.load_module = original" in source


def test_registry_check_compares_both_outputs_without_writing(tmp_path):
    from tools.build_registry import CONSUMER_EXPORT_PATH, REGISTRY_PATH, main
    registry = tmp_path / REGISTRY_PATH.name
    export = tmp_path / CONSUMER_EXPORT_PATH.name
    registry.write_bytes(REGISTRY_PATH.read_bytes())
    export.write_bytes(CONSUMER_EXPORT_PATH.read_bytes())
    before = (registry.read_bytes(), export.read_bytes())
    assert main(["--registry", str(registry), "--out", str(registry), "--check"]) == 0
    assert (registry.read_bytes(), export.read_bytes()) == before
    export.write_bytes(b"{}\n")
    assert main(["--registry", str(registry), "--out", str(registry), "--check"]) == 1
    assert registry.read_bytes() == before[0]
    assert export.read_bytes() == b"{}\n"


def test_registry_history_extension_keeps_older_preparation_identities():
    from tools.registry_physics_history import extend
    import pytest
    old = {"schema": "history", "identity_schema": "physics",
           "documents": {"old": {"physics_sha256": "p"}},
           "physics": {"p": {"microphysics": "old"}}}
    new = {"schema": "history", "identity_schema": "physics",
           "documents": {"new": {"physics_sha256": "q"}},
           "physics": {"q": {"microphysics": "new"}}}
    merged = extend(old, new)
    assert set(merged["documents"]) == {"old", "new"}
    assert merged["physics"]["p"] == old["physics"]["p"]
    assert "new" not in old["documents"]
    with pytest.raises(ValueError, match="physics parts differ"):
        extend(old, {**new, "physics": {"p": {"microphysics": "changed"}}})


def test_word_counter_accepts_real_oracle_stat_shapes():
    from dataclasses import dataclass
    import numpy as np
    from tools.verify_lanes_stack_pytest import _counts
    @dataclass
    class FieldStat:
        total: int
        differing: int
        max_ulp: int
    rows = _counts({"fields": {"a": FieldStat(10, 2, 7),
                               "b": {"count": np.int64(8), "mismatch": np.int64(1)}}})
    assert sum(row["words"] for row in rows) == 18
    assert sum(row["differing"] for row in rows) == 3
    assert rows[0]["max_ulp"] == 7


def test_receipt_audit_refuses_missing_default_and_empty_column_comparisons():
    from tools.verify_lanes_stack_audit import lane_errors
    branch = "lane/verify-mynn-sfclay-wrf461"
    row = {"lane": branch, "mode": "strict", "exit_code": 0,
           "compared_words": 10, "differing_words": 0}
    document = {"revision": "r", "source_diff_sha256": "d",
                "passed": True, "results": [row]}
    assert any("default gate absent" in error for error in lane_errors(branch, document, "r", "d"))
    document["results"].append({**row, "mode": "default", "compared_words": 0})
    assert any("no numerical comparison" in error for error in lane_errors(branch, document, "r", "d"))

"""MRF (``bl_pbl_physics = 99``): the fixture, the kernel's pins, the doors.

CPU-only.  The card rows are tests/test_mrf_wrf461_parity.py.

The fixture is WRF v4.6.1's own words for ``module_bl_mrf.F`` (the object
WRF's scalar build compiled, tools/mrf_wrf461_oracle); these gates pin its
bytes, what it covers, and the facts about WRF it records, then pin every
place the scheme is admitted: RunConfig, the WRF v4.6.1 pairing matrix, the
namelist importer, the registry option, the dispatch row, the checkpoint
identity, the vertical contract and the loader's header row.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from woof.verify.mrf_oracle import (MRF_ORACLE_DIR, fixture_sha256,
                                     load_mrf_oracle)

ROOT = Path(__file__).resolve().parents[1]
MRF = 99


def test_mrf_history_diagnostics_round_trip(tmp_path):
    from woof.io.wrf_output_schema import MRF_PBL_OUTPUT_FIELDS
    from test_wrf_output_schema import _schema_frame, _write

    netcdf = pytest.importorskip("netCDF4")
    from woof.io import nc_writer_bridge
    # The default wrfout engine is the Rust NetCDF writer cdylib; public ci's Windows cpu leg builds only the
    # table fetcher, so the writer is absent there (public ci 41990710176, RuntimeError at _write).
    missing = nc_writer_bridge.unavailable_reason()
    if missing:
        pytest.skip(f"the Rust NetCDF writer library is not built: {missing}")
    frame = _schema_frame(5, 4, 6)
    frame["HOL"] = np.linspace(-1., 1., 20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / "wrfout_mrf"
    _write(path, frame, nx=5, ny=4, nz=6)
    with netcdf.Dataset(path) as ds:
        for key, field in MRF_PBL_OUTPUT_FIELDS.items():
            variable = ds.variables[field.netcdf_name]
            assert np.dtype(variable.dtype) == np.dtype(field.dtype)
            assert variable.dimensions == ("Time", "south_north", "west_east")
            np.testing.assert_array_equal(variable[0], frame[field.netcdf_name])

#: sha256 of the packaged fixture (LF), as oracle-sha256sums.txt records it.
FIXTURE_SHA256 = {
    "mrf-levels.csv":
        "bea3e4117aeae2dc2bb0eed308a492166c906d08a4f2acc69f1de966eb669eb6",
    "mrf-surface.csv":
        "3faa0f2e529d9d1f77265e39b415148f0c50af50014ea44b824deed07ab7c40a",
    "mrf-cases.json":
        "fd23baedbfa1bbefef0ac542355b97b316a8cad0df2059e1776b6e9d3d98bd9f",
    "build-receipt.txt":
        "5404366c5398fa961c610117f2aa4a7cff7967992152ce4782509401eb2c7bf8",
}


def test_fixture_is_the_pinned_bytes_and_its_own_receipt():
    assert fixture_sha256() == FIXTURE_SHA256
    recorded = {}
    for line in (MRF_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
            encoding="ascii").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == FIXTURE_SHA256


def test_receipt_shows_wrfs_own_object_and_a_matching_rebuild():
    text = (MRF_ORACLE_DIR / "build-receipt.txt").read_text(encoding="ascii")
    assert ("a1306d0063043d001fa4a53b9219031b931999c7d7daea172fd97518bc9b19eb"
            in text)
    assert "-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize" in text
    assert "referee and rebuild outputs byte-identical" in text
    # WRF as built calls powf four times (THGB, THCON, PHIM, PHIH); the -O0
    # diagnostic arm a fifth time, for the ZFAC**PFAC the port multiplies.
    arms = re.findall(r"# (\w+) libm call sites.*?\n#   powf: (\d+)", text)
    assert dict(arms) == {"referee": "4", "rebuild": "4", "o0": "5"}
    assert "_ZGV" not in text


def test_fixture_covers_the_regimes_the_brief_names():
    o = load_mrf_oracle()
    assert o.ncase >= 64 and o.nstep >= 2 and o.nz == 40
    families = {c["family"] for c in o.cases}
    assert {"convective-land", "stable-night", "neutral-windy",
            "cloud-topped", "very-shallow", "near-model-top", "water",
            "probe", "probe-br"} <= families
    water = np.array([c["xland"] for c in o.cases]) >= 1.5
    assert water.any() and (~water).any()
    assert o.flag_qi.any() and (~o.flag_qi).any()
    dts = {float(x) for x in o.surface["dt"][0]}
    assert {10.0, 30.0, 60.0, 90.0, 120.0, 180.0} <= dts
    # WRF's KPBL word spans the whole column, top-down: KL (stable, or a
    # thermal that filled the column) down to deep convective PBLs.
    assert o.kpbl.max() == o.nz and o.kpbl.min() <= 20


def test_wrf_kpbl_is_top_down_and_pbl_is_the_first_layer_when_pblflg_clears():
    """The two facts about WRF the port's KPBL divergence rests on."""
    o = load_mrf_oracle()
    fam = np.array([c["family"] for c in o.cases])
    zq_kl = o.levels["dz8w"][:, 0, :]          # ZQ(KL) = lowest layer depth
    stable = fam == "stable-night"
    assert np.all(o.kpbl[:, stable] == o.nz)
    assert np.array_equal(o.surface["pbl"][:, stable].view(np.uint32),
                          zq_kl[:, stable].view(np.uint32))
    top = fam == "near-model-top"
    assert np.all(o.kpbl[:, top] == o.nz)
    assert np.array_equal(o.surface["pbl"][:, top].view(np.uint32),
                          zq_kl[:, top].view(np.uint32))


def test_regime_zol_and_wspd_pass_through_untouched():
    o = load_mrf_oracle()
    for name in ("regime", "zol", "wspd"):
        assert np.array_equal(o.surface[name].view(np.uint32),
                              o.surface[f"{name}_in"].view(np.uint32)), name


def test_kernel_constants_are_wrfs_float32_words():
    f = np.float32
    r_d = f(287.0)
    cp = f(7.0) * r_d / f(2.0)
    r_v = f(461.6)
    want = {"MRF_G": f(9.81), "MRF_RD": r_d, "MRF_CP": cp, "MRF_RV": r_v,
            "MRF_ROVCP": r_d / cp, "MRF_EP1": r_v / r_d - f(1.0),
            "MRF_XLV": f(2.5e6), "MRF_KARMAN": f(0.4)}
    source = (ROOT / "woof/data/mrf/constants.cuh").read_text(encoding="utf-8")
    for name, value in want.items():
        m = re.search(rf"#define {name}\s+__int_as_float\((0x[0-9a-f]+)\)",
                      source)
        assert m, name
        word = int(np.array([value], np.float32).view(np.uint32)[0])
        assert int(m.group(1), 16) == word, name


def test_workspace_geometry_is_one_authority():
    from woof.core.physics_inventory import (MRF_BLOCK, MRFWS_SLOTS,
                                              mrf_workspace_floats)
    source = (ROOT / "woof/core/kernels/mrf.cu").read_text(encoding="utf-8")
    assert f"#define MRFWS_SLOTS {MRFWS_SLOTS}" in source
    assert f"#define MRFWS_LANES {MRF_BLOCK}" in source
    assert f"#define MRF_BLOCK {MRF_BLOCK}" in source
    # 1-based levels with ZQ(KLP1): nz + 2 per slot, whole blocks.
    assert mrf_workspace_floats(40, 33) == 2 * MRFWS_SLOTS * 42 * MRF_BLOCK


def test_admission_includes_the_column_and_deferred_validator():
    from woof.core.preflight import _PBL_KERNEL_MODULES
    assert _PBL_KERNEL_MODULES[MRF] == ("mrf", "mrf_validation")


def test_the_kernel_loads_through_the_no_ftz_route():
    """RawModule appends -ftz=true after the caller's options; the exact
    kernels compile through NVRTC directly (terrain_drag, ruc_spp)."""
    from woof.core.kernels import EXTRA_HEADERS
    assert EXTRA_HEADERS["mrf"] == ("glibc_flt32.cuh",)
    source = (ROOT / "woof/core/mrf.py").read_text(encoding="utf-8")
    assert ('MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")'
            in source)
    assert "nvrtc_ptx_cache" in source
    import ast
    tree = ast.parse(source)
    compile_source = ast.get_source_segment(source, next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_mrf_module"))
    assert "RawModule(" not in compile_source and "get_kernel(" not in compile_source


def _cfg(**overrides):
    from woof.config import RunConfig
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=MRF, sf_sfclay_physics=1,
                sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_native_mrf_restart_rebuilds_hol_and_normalizes_kpbl(monkeypatch):
    from types import SimpleNamespace
    from woof.ingest import wrfrst
    from woof.io import restart

    cfg = _cfg(nz=49, diff_opt=2)
    fields = {"fields/hol": np.full((1, 4), np.nan, np.float32),
              "fields/kpbl": np.zeros((1, 4), np.int32)}
    raw = {"KPBL": np.array([[0, 49, 39, 1]], np.int32)}
    # Exercise the actual overlay walk without allocating a GPU atmosphere.
    # Dynamics and held-tendency restoration have their own restart tests.
    monkeypatch.setattr(restart, "state_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_driver_manifest", lambda driver: fields)
    monkeypatch.setattr(wrfrst, "_copy_into", lambda target, value:
                        np.copyto(target, value.astype(target.dtype)))
    monkeypatch.setattr(wrfrst, "_recompose_held_tendencies", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_warm_start_driver_counters", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_record_eos_checks", lambda *args: None)
    report = wrfrst.apply_warm_start_physics(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace(raw=raw), cfg)
    np.testing.assert_array_equal(fields["fields/kpbl"], [[0, 1, 11, 49]])
    np.testing.assert_array_equal(fields["fields/hol"], 0)
    assert "top-down to bottom-up" in report.filled["fields/kpbl"]
    assert "rewritten" in report.rebuilt["fields/hol"]
    # The rule is specific to MRF, and native input words are never mutated.
    assert wrfrst.fields_restore_plan("hol", ()) is None
    np.testing.assert_array_equal(raw["KPBL"], [[0, 49, 39, 1]])


@pytest.mark.parametrize("native_active", [False, True])
def test_native_handoff_adds_hail_outputs_without_erasing_active_stones(
        monkeypatch, native_active):
    from types import SimpleNamespace
    from woof.core.hail import PERSISTENT
    from woof.ingest import wrfrst
    from woof.io import restart

    targets = {f"scratch/{name}": np.full((1, 4), np.nan, np.float32)
               for name in PERSISTENT}
    raw = {"HAILCAST_WUP_MASK": np.zeros((1, 4), np.float32),
           "HAILCAST_WDUR": np.zeros((1, 4), np.float32),
           "HAIL_MAXK1": np.full((1, 4), 0.012, np.float32)}
    if native_active:
        raw["HAILCAST_DHAIL1"] = np.ones((1, 4), np.float32)
    monkeypatch.setattr(restart, "state_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda state: targets)
    monkeypatch.setattr(restart, "_driver_manifest", lambda driver: {})
    monkeypatch.setattr(wrfrst, "_copy_into", lambda target, value:
                        np.copyto(target, value.astype(target.dtype)))
    monkeypatch.setattr(wrfrst, "_recompose_held_tendencies", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_warm_start_driver_counters", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_record_eos_checks", lambda *args: None)
    call = lambda: wrfrst.apply_warm_start_physics(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace(raw=raw),
        _cfg(diff_opt=2))
    if native_active:
        with pytest.raises(ValueError, match="scratch/hailcast_dhail2"):
            call()
        # Restore is in place: valid carry is copied before missing carry refuses.
        np.testing.assert_array_equal(targets["scratch/hailcast_dhail1"],
                                      raw["HAILCAST_DHAIL1"])
        assert np.isnan(targets["scratch/hailcast_dhail2"]).all()
    else:
        report = call()
        np.testing.assert_array_equal(targets["scratch/hail_maxk1"], raw["HAIL_MAXK1"])
        np.testing.assert_array_equal(targets["scratch/hailcast_runmax"], 0)
        assert "restart instant" in report.rebuilt["scratch/hailcast_runmax"]
        assert report.filled["scratch/hailcast_wdur"] == "HAILCAST_WDUR"


def test_mrf_is_admitted_with_both_mm5_surface_layers():
    from woof.config import MRF_PBL_SCHEME, PBL_SCHEMES, validate_run_config
    assert MRF_PBL_SCHEME == MRF and MRF in PBL_SCHEMES
    for sfclay in (1, 91):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


@pytest.mark.parametrize("sfclay", [0, 5])
def test_other_surface_layers_are_wrfs_own_fatal(sfclay):
    from woof.config import validate_run_config
    with pytest.raises(Exception) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=sfclay,
                                 sf_surface_physics=0 if sfclay == 0 else 2))
    assert "bl_pbl_physics" in str(caught.value) or "99" in str(caught.value)


def test_the_wrf461_matrix_carries_wrfs_mrfscheme_case():
    from woof.wrf461_compatibility import (PBL_OPTIONS, WRFVerdict,
                                            pbl_surface_layer_verdict)
    assert MRF in PBL_OPTIONS
    for sfclay, verdict in ((0, WRFVerdict.FATAL), (1, WRFVerdict.LEGAL),
                            (5, WRFVerdict.FATAL), (91, WRFVerdict.LEGAL)):
        got, citation = pbl_surface_layer_verdict(MRF, sfclay)
        assert got is verdict
        assert "3715-3717" in citation.lines


def test_bep_and_bem_urban_are_refused_by_the_existing_urban_law():
    """WRF's MRFSCHEME arm also fatals BEP/BEM (module_physics_init.F:
    3718-3719); woof.config.URBAN_BEP_PBL_SCHEMES already says it for
    every PBL but YSU and MYJ, so MRF needs no second authority."""
    from woof.config import URBAN_BEP_PBL_SCHEMES, validate_run_config
    assert MRF not in URBAN_BEP_PBL_SCHEMES
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(sf_urban_physics=2))
    assert "bl_pbl_physics=99" in str(caught.value)


def test_dispatch_routes_99_to_its_own_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH, PhysicsDriver
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][MRF] == "_run_mrf"
    assert callable(getattr(PhysicsDriver, "_run_mrf"))


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert PBL_ALGORITHM_IDENTITIES[MRF] == "mrf-pbl-wrf-v4.6.1-v1"


def test_the_registry_option_is_the_selector():
    import json
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["mrf"]
    assert option["selectors"] == {"bl_pbl_physics": MRF}
    assert option["implemented"] is True
    assert option["constraints"]["requires_components"]["surface_layer"] \
        == ["revised-mm5", "classic-mm5"]


def test_the_namelist_importer_carries_99_natively():
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[MRF][0] == MRF


def test_the_launcher_bound_is_the_contract_bound():
    from woof.physics_vertical_contract import MRF_VERTICAL_LEVEL_BOUNDS
    assert MRF_VERTICAL_LEVEL_BOUNDS == (2, None)
    source = (ROOT / "woof/core/mrf.py").read_text(encoding="utf-8")
    assert "MRF_VERTICAL_LEVEL_BOUNDS" in source


def test_mrf_constants_use_the_pinned_table_asset(tmp_path):
    import hashlib
    from woof.core.kernels import _table_header_text

    header = _table_header_text("mrf")
    root = ROOT / "woof/data/mrf"
    assert header == (root / "constants.cuh").read_text("utf-8")
    digest, name = (root / "MANIFEST.sha256").read_text("ascii").split()
    assert name == "constants.cuh"
    assert digest == hashlib.sha256(header.encode("utf-8")).hexdigest()
    assert "#define MRF_CFAC    7.8f" in header
    assert "#define MRF_CFAC" not in (ROOT / "woof/core/kernels/mrf.cu").read_text("utf-8")
    (tmp_path / "mrf").mkdir()
    # Bytes, not text: a text-mode write turns LF into CRLF on Windows, so the size check refused the file before
    # the SHA-256 check this test is about (public ci 41990710176, windows-2025).
    (tmp_path / "mrf/constants.cuh").write_bytes(header.replace("7.8f", "7.9f").encode("utf-8"))
    with pytest.raises(ValueError, match="SHA-256"):
        _table_header_text("mrf", data_root=tmp_path)



def test_mrf_registry_carries_the_wrf_driver_contract_and_asset():
    import json
    registry = json.loads((ROOT / "woof/physics_registry_v2.json").read_text("utf-8"))
    row = registry["components"]["pbl"]["options"]["mrf"]
    contract = row["extensions"]["wrf_driver_contract"]
    assert contract["tendencies"] == ["RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN", "RQIBLTEN"]
    assert contract["surface_layer_passthrough"] == ["ZOL", "REGIME"]
    assert contract["cadence_parameter"] == "bldt"
    asset = row["asset_requirements"][0]["assets"][0]
    digest, name = (ROOT / "woof/data/mrf/MANIFEST.sha256").read_text("ascii").split()
    assert asset == {"filename": name,
                     "bytes": (ROOT / "woof/data/mrf" / name).stat().st_size,
                     "sha256": digest}

"""BouLac (bl_pbl_physics=8): its doors, WRF's law for it, and its seams.

No device is opened here.  What is pinned:

* the three doors a user reaches the scheme through: the RunConfig key, the
  WRF namelist importer and the registry option;
* WRF's surface-layer law (isfc 1 or 2, module_physics_init.F:3814-3816)
  and its BEP/BEM admission, each refusal naming what it prevents;
* the dispatch row, the checkpoint identity, the vertical bound, the
  history rows and the preflight pricing, each held equal to the one
  authority it restates;
* the packaged oracle fixtures are the bytes their SHA256SUMS name;
* default-off: a configuration that does not select the scheme allocates
  none of its fields.

The card checks are tests/test_boulac_wrf461_parity.py (every word against
WRF) and tests/test_boulac_driver_seam.py (the driver and the restart).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from woof.config import (BOULAC_PBL_SCHEME, ETA_LAYER_TKE_FIELDS,
                          URBAN_BEP_PBL_SCHEMES, RunConfig,
                          validate_run_config)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "boulac"


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=BOULAC_PBL_SCHEME, sf_sfclay_physics=1,
                sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_boulac_is_admitted_over_every_surface_layer_wrf_allows():
    from woof.config import WRF_UNPORTED_SCHEME_REASONS
    assert BOULAC_PBL_SCHEME not in WRF_UNPORTED_SCHEME_REASONS["bl_pbl_physics"]
    # isfc = 1 (revised and classic MM5) and isfc = 2 (Eta).
    for sfclay in (1, 91, 2):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


@pytest.mark.parametrize("sfclay", [0, 5])
def test_wrf_fatal_surface_layers_are_refused_with_the_citation(sfclay):
    from woof.physics_compat import UnsupportedPhysicsSuiteError
    with pytest.raises((UnsupportedPhysicsSuiteError, ValueError)) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=sfclay,
                                 sf_surface_physics=0))
    if isinstance(caught.value, UnsupportedPhysicsSuiteError):
        assert "module_physics_init.F:" in str(caught.value)
        assert "3814-3816" in str(caught.value)


def test_the_eta_layer_reads_boulacs_own_tke():
    assert ETA_LAYER_TKE_FIELDS[BOULAC_PBL_SCHEME] == "tke_pbl"
    from woof.core.physics_inventory import BOULAC_STATE_3D
    assert ETA_LAYER_TKE_FIELDS[BOULAC_PBL_SCHEME] in BOULAC_STATE_3D
    source = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    assert "ETA_LAYER_TKE_FIELDS[int(cfg.bl_pbl_physics)]" in source


def test_bep_and_bem_admit_boulac():
    assert BOULAC_PBL_SCHEME in URBAN_BEP_PBL_SCHEMES
    from types import SimpleNamespace
    from woof.config import validate_urban_config
    for urban in (2, 3):
        cfg = SimpleNamespace(sf_urban_physics=urban, use_wudapt_lcz=0,
                              num_urban_hi=15, sf_surface_physics=2,
                              bl_pbl_physics=BOULAC_PBL_SCHEME)
        try:
            validate_urban_config(cfg)
        except ValueError as exc:
            # Only the build-presence refusal may remain; never the PBL one.
            assert "bl_pbl_physics" not in str(exc), exc
    refusal = SimpleNamespace(sf_urban_physics=2, use_wudapt_lcz=0,
                              num_urban_hi=15, sf_surface_physics=2,
                              bl_pbl_physics=5)
    with pytest.raises(ValueError) as caught:
        validate_urban_config(refusal)
    assert "8 (BouLac)" in str(caught.value)
    assert "not ported" not in str(caught.value)


def test_dispatch_routes_eight_to_its_own_runner():
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][BOULAC_PBL_SCHEME] == \
        "_run_boulac"
    source = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    assert "def _run_boulac(self" in source


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert PBL_ALGORITHM_IDENTITIES[BOULAC_PBL_SCHEME] == \
        "boulac-pbl-wrf-v4.7.1-v3"


def test_the_registry_option_is_the_selector():
    import json
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["boulac"]
    assert option["selectors"] == {"bl_pbl_physics": BOULAC_PBL_SCHEME}
    assert option["implemented"] is True
    assert sorted(option["constraints"]["requires_components"][
        "surface_layer"]) == ["classic-mm5", "eta-similarity", "revised-mm5"]
    assert option["consumers"]["restart_algorithm_identity"] == \
        "boulac-pbl-wrf-v4.7.1-v3"
    sl = registry["components"]["surface_layer"]["options"]
    assert "boulac" in sl["eta-similarity"]["constraints"][
        "requires_components"]["pbl"]
    urban = registry["components"]["urban"]["options"]
    for option_id in ("bep", "bep-bem"):
        assert "boulac" in urban[option_id]["constraints"][
            "requires_components"]["pbl"]
    gf = registry["components"]["cumulus"]["options"]["grell-freitas"]
    assert "boulac" in gf["constraints"]["requires_components"]["pbl"]


def test_the_namelist_importer_carries_eight_natively():
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[8][0] == BOULAC_PBL_SCHEME


def test_the_launcher_bound_is_the_contract_bound():
    source = (ROOT / "woof/core/boulac.py").read_text(encoding="utf-8")
    assert "VERTICAL_LEVEL_BOUNDS = BOULAC_VERTICAL_LEVEL_BOUNDS" in source
    kernel = (ROOT / "woof/core/kernels/boulac.cu").read_text(
        encoding="utf-8")
    from woof.physics_vertical_contract import BOULAC_VERTICAL_LEVEL_BOUNDS
    assert BOULAC_VERTICAL_LEVEL_BOUNDS == (3, 128)
    assert "#define BOULAC_KMAX 128" in kernel
    assert "nz > BOULAC_KMAX || nz < 3" in kernel


def test_the_kernel_constants_are_wrfs_float32_words():
    """ceps_b = 1/1.4, ck_b = 0.4, temin = 0.0001, the 1/3 exponent."""
    import re

    import numpy as np
    kernel = (ROOT / "woof/core/kernels/boulac.cu").read_text(
        encoding="utf-8")

    def lit(name):
        return np.float32(re.search(rf"#define {name} ([0-9.e+-]+)f",
                                    kernel).group(1))
    f = np.float32
    assert lit("BL_CEPS_B") == f(1) / f(1.4)
    assert lit("BL_CK_B") == f(0.4)
    assert lit("BL_TEMIN") == f(0.0001)
    assert lit("BL_THIRD") == f(1) / f(3)
    assert lit("BL_EP") == f(0.61)


def test_the_history_publishes_wrfs_tke_pbl_row():
    from woof.io.wrf_output_schema import BOULAC_PBL_OUTPUT_FIELDS
    row = BOULAC_PBL_OUTPUT_FIELDS["tke_pbl"]
    assert (row.netcdf_name, row.stagger, row.units) == (
        "TKE_PBL", "Z", "m2 s-2")


def test_the_preflight_prices_the_scheme():
    from woof.core import preflight
    from woof.core.physics_inventory import BOULAC_STATE_3D
    cfg = _cfg()
    shapes = preflight.physics_array_shapes(cfg)
    for name in BOULAC_STATE_3D:
        assert shapes[f"fields/{name}"] == (cfg.nz, cfg.ny, cfg.nx)
    transient = preflight.boulac_output_transient_shapes(cfg)
    assert transient["boulac_output/rthblten"] == (cfg.nz, cfg.ny, cfg.nx)
    for name in ("wu", "wv", "wt", "wq"):
        assert f"boulac_output/{name}" not in transient
    assert preflight.boulac_output_transient_shapes(
        _cfg(bl_pbl_physics=1)) == {}
    assert "boulac" in preflight._PBL_KERNEL_MODULES[BOULAC_PBL_SCHEME]
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["boulac"] > 0


def test_default_off_allocates_nothing():
    """Every BouLac allocation is behind its selector (read off the source)."""
    source = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    block = source.split(
        "if int(cfg.bl_pbl_physics) == BOULAC_PBL_SCHEME:", 1)[1]
    block = block.split("if int(cfg.bl_pbl_physics) ==", 1)[0]
    assert "for name in BOULAC_STATE_3D" in block


def test_work_buffers_rebuild_and_carried_fields_are_checkpointed():
    from woof.io.restart import classify_scratch_slot
    from woof.core.physics_inventory import BOULAC_COLUMN_OUTPUTS
    for name in ("rho", "moist") + BOULAC_COLUMN_OUTPUTS:
        if name not in ("el_pbl", "exch_h", "exch_m"):
            assert classify_scratch_slot("boulac_" + name) == "rebuild"


def test_the_packaged_fixtures_are_the_bytes_their_sums_name():
    sums = (FIXTURES / "SHA256SUMS").read_text(encoding="ascii")
    rows = [line.split() for line in sums.splitlines() if line.strip()]
    assert len(rows) == 11      # five grids x (in, rec) + the index
    for digest, name in rows:
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_the_fixtures_come_from_the_referee_object():
    report = (FIXTURES / "variants-report.txt").read_text(encoding="utf-8")
    assert "fixture of record: wrfobj" in report
    for variant in ("o2", "o0", "snan"):
        assert f"{variant} byte-identical to wrfobj: yes" in report
    from woof.verify.boulac_oracle import grids
    index = grids()
    assert sum(g["ncol"] for g in index) >= 64
    labels = [label for g in index for label in g["labels"]]
    for family in ("convective", "stable-jet", "neutral-windy",
                   "strong-shear", "high-terrain", "bep-frc0.6"):
        assert any(label.startswith(family) for label in labels), family
    assert any(g["flag_bep"] for g in index)
    assert max(g["nsteps"] for g in index) >= 6


@pytest.mark.parametrize("strict", [False, True])
def test_boulac_uses_the_shared_unflushed_compile_route(monkeypatch, strict):
    from woof import wrf_exact
    monkeypatch.setattr(wrf_exact, "ENABLED", strict)
    from woof.core.kernels import module_options
    expected = (("-std=c++17", "--fmad=false", "--ftz=false") if strict
                else ("-std=c++17",))
    assert module_options("boulac") == expected


def test_boulac_constants_are_registered_literal_rows():
    import json
    from woof.physics_params import REGISTRY_PATH, registry_sha256
    table = json.loads(REGISTRY_PATH.read_text())
    rows = [r for r in table["constants"] if r["name"].startswith("boulac.")]
    assert len(rows) == 8
    kernel = (ROOT / "woof/core/kernels/boulac.cu").read_text()
    for row in rows:
        assert row["requires"] == {"bl_pbl_physics": [8]}
        assert row["lower"] < row["default"] < row["upper"]
        import numpy as np
        assert np.float32(row["lower"]) == np.float32(row["upper"])
        site = row["sites"][0]
        assert kernel.count(site["anchor"]) == site["count"] == 1
    assert len(registry_sha256()) == 64

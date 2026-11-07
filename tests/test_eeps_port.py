"""EEPS (bl_pbl_physics=16), the parts that need no device.

The fixtures of record and what they cover, the column's constants against
the oracle's, the column's math vocabulary, the one declared divergence,
and the admission: config, dispatch, registry, namelist import, the WRF
v4.6.1 matrix, the transported-scalar inventory and its restart and nest
classification.  The card grading is tests/test_eeps_wrf461_parity.py and
tests/test_eeps_driver_seam.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from woof.config import (EEPS_PBL_SCHEME, PBL_SCHEMES, RunConfig,
                          validate_run_config)
from woof.verify.eeps_oracle import (PER_REGIME, REGIMES, blend_carried,
                                      read_trace, ulp_distance)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/data/oracles/eeps"
KERNEL = ROOT / "woof/core/kernels/eeps.cu"

#: sha256 of the fixtures of record (tools/eeps_wrf461_oracle/build.sh on
#: the sprint's CPU box: gfortran 13.3.0, glibc 2.39, the scalar WRF
#: build's flags).  A regenerated fixture changes these on purpose.
FIXTURE_SHA256 = {
    "trace-A.bin.gz":
        "ba11016c86ad8325f572fbd89b288d84d2724b36d50521d6c2461ef8b8f40946",
    "trace-B.bin.gz":
        "07f73509f97346093cfc89481c2cd6fe083c876f13f1c664e020b1db84d6a5e9",
    "trace-C.bin.gz":
        "4f88ff9fee65f50c8787dc33e83c8e714d7ec1143ab044cf9ca8c4bbbc90be3a",
    "constants.txt":
        "9335f60232fc4bb6758a33a2d9e7d6132474fd338ee19780ef49be476e42b6ec",
}


def _trace(name):
    return read_trace(ORACLE / f"trace-{name}.bin.gz")


def test_the_fixtures_are_the_recorded_words():
    for name, digest in FIXTURE_SHA256.items():
        data = (ORACLE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_the_fixtures_cover_what_the_brief_asks_for():
    """>= 64 columns, several calls, the first call from EEPSINIT, a
    restart-shaped set, both moisture-argument shapes, and the regimes."""
    a, b, c = _trace("A"), _trace("B"), _trace("C")
    for t in (a, b, c):
        assert t["header"]["ncol"] == len(REGIMES) * PER_REGIME >= 64
        assert t["header"]["nsteps"] >= 4
    assert a["steps"][0]["itimestep"] == 1 and b["steps"][0]["itimestep"] == 1
    assert c["steps"][0]["itimestep"] > 1
    assert a["header"]["has_qs_qg"] and not b["header"]["has_qs_qg"]
    # The cold first call reads EEPSINIT's minpek/minpep and a zero
    # advected pair (WRF's 'i0').
    first = a["steps"][0]["inputs"]
    assert np.all(first["pek"] == np.float32(1.0e-4))
    assert np.all(first["pep"] == np.float32(1.0e-7))
    assert not np.any(first["pek_adv"])
    out = a["steps"][-1]["outputs"]
    kpbl = out["kpbl"].reshape(len(REGIMES), PER_REGIME)
    # Very shallow PBLs at the second level, deep convective ones above
    # 2 km, and the low-topped columns whose PBL reaches the search limit
    # reported at the lowest level (WRF's own fallback).
    assert kpbl[REGIMES.index("very_shallow")].max() <= 2
    assert out["pblh"].max() > 2000.0
    assert 1 in kpbl[REGIMES.index("near_model_top")]
    # Ice below the freezing level on the first call: the melting arm.
    inp = a["steps"][0]["inputs"]
    assert np.any((inp["qi"] > 1e-20) & (inp["t"] >= 273.15))


def test_the_columns_constants_are_the_oracles_words():
    """Every module_model_constants value the column hardcodes is the bit
    pattern gfortran folded (the oracle's constants.txt)."""
    recorded = dict(line.split() for line in
                    (ORACLE / "constants.txt").read_text().splitlines())
    source = KERNEL.read_text(encoding="utf-8")
    pins = dict(re.findall(
        r"#define EEPS_\w+\s+__int_as_float\(0x([0-9A-F]{8})\)\s+//\s+(\w+)",
        source))
    by_name = {v.lower(): k for k, v in pins.items()}
    for name in ("r_d", "prandtl", "ep_1", "ep_2", "g", "karman", "cp",
                 "rcp", "xlv", "xls", "p1000mb"):
        assert by_name[name] == recorded[name], name


def test_the_column_selects_fast_and_strict_math():
    source = KERNEL.read_text(encoding="utf-8")
    strict, fast = source.split("#ifdef GPUWM_WRF_EXACT", 1)[1].split("#else", 1)
    for name in ("pow", "exp", "log"):
        assert f"#define eeps_{name} gfk_{name}" in strict
        assert f"#define eeps_{name} {name}f" in fast
    assert "--fmad=false" not in __import__(
        "woof.core.eeps_pbl", fromlist=["MODULE_OPTIONS"]).MODULE_OPTIONS


def test_the_declared_divergence_is_the_only_source_edit():
    """RADSUM starts at zero in the column; the oracle adds exactly that one
    line to WRF's module, and nothing else."""
    source = KERNEL.read_text(encoding="utf-8")
    assert "float radsum = 0.f;" in source
    build = (ROOT / "tools/eeps_wrf461_oracle/build.sh").read_text()
    assert "radsum(j)  = 0." in build
    assert "did not apply to exactly one line" in build


def test_the_stand_in_handoff_is_the_oracles_arithmetic():
    a = _trace("A")
    for prev, nxt in zip(a["steps"], a["steps"][1:]):
        for name in ("pek_adv", "pep_adv"):
            ours = blend_carried(prev["outputs"][name])
            assert not np.any(ulp_distance(ours, nxt["inputs"][name])), name


# ------------------------------------------------------------- admission

def _cfg(**overrides):
    values = dict(nx=8, ny=6, nz=20, dx=3000.0, dy=3000.0, dt=12.0,
                  ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                  sf_sfclay_physics=1, bl_pbl_physics=EEPS_PBL_SCHEME,
                  sf_surface_physics=2, num_soil_layers=4)
    values.update(overrides)
    return RunConfig(**values)


def test_eeps_is_a_selectable_pbl_value():
    assert EEPS_PBL_SCHEME == 16 and 16 in PBL_SCHEMES


@pytest.mark.parametrize("sfclay", [1, 2, 5, 91])
def test_wrfs_surface_layer_list_is_admitted(sfclay):
    validate_run_config(_cfg(sf_sfclay_physics=sfclay))


@pytest.mark.parametrize("mp", [6, 10, 1, 0])
def test_any_moist_package_is_admitted(mp):
    validate_run_config(_cfg(mp_physics=mp))


def test_no_surface_layer_is_refused_with_wrfs_fatal():
    with pytest.raises(ValueError, match="3871-3873"):
        validate_run_config(_cfg(sf_sfclay_physics=0))


def test_the_eta_layer_owns_its_native_cold_tke_column():
    from woof.core.preflight import physics_array_shapes
    cfg = _cfg(sf_sfclay_physics=2)
    assert physics_array_shapes(cfg)["fields/tke_myj"] == (cfg.nz, cfg.ny, cfg.nx)


def test_a_dry_state_is_refused_naming_the_frozen_pair():
    with pytest.raises(ValueError, match="never move"):
        validate_run_config(_cfg(moist=False, mp_physics=0))


def test_the_dispatch_row_names_the_runner():
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][16] == "_run_eeps"
    from woof.core.physics import PhysicsDriver
    assert callable(getattr(PhysicsDriver, "_run_eeps"))


def test_the_registry_option_is_published():
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["eeps"]
    assert option["selectors"] == {"bl_pbl_physics": 16}
    assert option["implemented"] is True
    assert option["constraints"]["requires_components"]["surface_layer"] \
        == ["revised-mm5", "classic-mm5", "mynn", "eta-similarity"]
    assert option["extensions"]["advected_scalars"]["names"] \
        == ["pek_adv", "pep_adv"]


def test_the_namelist_importer_maps_16_natively():
    from woof import namelist_import
    assert namelist_import._BL_MAP[16][0] == 16


def test_the_wrf461_matrix_carries_eeps_cells():
    from woof.wrf461_compatibility import (PBL_OPTIONS, WRFVerdict,
                                            pbl_surface_layer_verdict)
    assert 16 in PBL_OPTIONS
    assert pbl_surface_layer_verdict(16, 0)[0] is WRFVerdict.FATAL
    for sfclay in (1, 5, 91):
        assert pbl_surface_layer_verdict(16, sfclay)[0] is WRFVerdict.LEGAL


def test_the_transported_pair_is_allocated_classified_and_nested():
    from woof.core.device_inventory import state_array_shapes
    from woof.core.physics_inventory import pbl_transported_scalars
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    from woof.io.restart import STATE_REBUILT_ATTRS
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS

    cfg = _cfg()
    assert pbl_transported_scalars(cfg) == ("pek_adv", "pep_adv")
    shapes = state_array_shapes(cfg)
    for name in ("pek_adv", "pep_adv", "pek_adv0", "pep_adv0"):
        assert shapes[name] == (cfg.nz, cfg.ny, cfg.nx), name
    other = state_array_shapes(_cfg(bl_pbl_physics=1))
    assert "pek_adv" not in other and "pek_adv0" not in other
    assert {"pek_adv", "pep_adv"} <= set(STATE_SERIALIZED_ATTRS)
    assert {"pek_adv0", "pep_adv0"} <= STATE_REBUILT_ATTRS
    assert {"pek_adv", "pep_adv"} <= COUPLED_SCALAR_STATE_FIELDS


def test_the_restart_identity_names_the_divergence():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert "radsum-zero" in PBL_ALGORITHM_IDENTITIES[16]

"""physics.mode at the hex seam (woof.hex.correct_mode).

Exact mode must leave every hex artifact as it was before the key existed:
the driver argument vector, the sealed constructor's values and identity
digest, the receipt.  Correct mode must reach the engine, or, on an engine
that cannot apply a correction, run the matched arithmetic for it and name it
as a loud gap in the receipt (never a refusal, never a silent "applied").
CPU-only; the engine side carries its own tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from woof.hex import correct_mode
from woof.hex.cuda_arwen_physics_v841 import SealedArwenConstructorV841

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ntiedtke_column_row import _mapping  # noqa: E402

def test_validate_refuses_misspellings():
    for bad in ("Correct", "on", "", None):
        with pytest.raises(correct_mode.CorrectModeRefusal, match="physics mode"):
            correct_mode.validate(bad)
    with pytest.raises(correct_mode.CorrectModeRefusal, match="cannot switch"):
        correct_mode.validate("correct", ["WB9"])
    with pytest.raises(correct_mode.CorrectModeRefusal, match="twice"):
        correct_mode.validate("correct", ["WB5", "WB5"])
    with pytest.raises(correct_mode.CorrectModeRefusal, match="exact"):
        correct_mode.validate("exact", ["WB5"])


def test_exact_mode_hands_the_driver_no_flag():
    assert correct_mode.door_argv("exact") == []
    assert correct_mode.door_argv("correct") == ["--physics-mode", "correct"]
    assert correct_mode.door_argv("correct", ["WB5"]) == [
        "--physics-mode", "correct", "--physics-correct-skip", "WB5"]


def test_exact_mode_seals_the_same_constructor_and_identity():
    base = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme=None, cumulus_seconds=None))
    explicit = _mapping(scheme=None, cumulus_seconds=None)
    explicit["physics_mode"] = "exact"
    sealed = SealedArwenConstructorV841.from_mapping(explicit)
    assert sealed.identity_sha256 == base.identity_sha256
    assert set(sealed.arwen_kwargs()) == set(base.arwen_kwargs())
    assert "physics_mode" not in sealed.arwen_kwargs()
    assert sealed.physics_mode == "exact"
    assert "physics_mode" not in sealed.receipt()


def test_correct_mode_reaches_the_engine_keywords_and_the_identity():
    base = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme=None, cumulus_seconds=None))
    values = _mapping(scheme=None, cumulus_seconds=None)
    values["physics_mode"] = "correct"
    values["physics_correct_skip"] = ("WB4",)
    sealed = SealedArwenConstructorV841.from_mapping(values)
    kwargs = sealed.arwen_kwargs()
    assert kwargs["physics_mode"] == "correct"
    assert tuple(kwargs["physics_correct_skip"]) == ("WB4",)
    assert sealed.identity_sha256 != base.identity_sha256
    assert sealed.receipt()["physics_mode"] == "correct"


def test_an_unknown_key_is_still_refused():
    values = _mapping(scheme=None, cumulus_seconds=None)
    values["physics_modes"] = "correct"
    with pytest.raises(ValueError, match="not exhaustive"):
        SealedArwenConstructorV841.from_mapping(values)


def test_engine_accepts_reads_the_seam_signature():
    def old_engine_init(self, *, n_levels, n_columns, wsm6_hail_opt=0):
        pass

    def new_engine_init(self, *, n_levels, physics_mode="exact", physics_correct_skip=()):
        pass

    assert not correct_mode.engine_accepts(old_engine_init)
    assert correct_mode.engine_accepts(new_engine_init)


def _no_engine(monkeypatch):
    # An engine without correct mode: importing woof.correct_mode fails.  The release engine
    # (gpuwm 2.9.0) carries that module, and once anything has imported it the woof package holds it
    # as an attribute, which `from woof import correct_mode` returns without consulting sys.modules;
    # so the binding is hidden too (on 2.9.0 these tests otherwise saw the real engine and no gap).
    monkeypatch.setitem(sys.modules, "woof.correct_mode", None)
    package = sys.modules.get("woof")
    if package is not None:
        monkeypatch.delattr(package, "correct_mode", raising=False)


def test_correct_mode_on_an_engine_without_it_is_a_named_gap(monkeypatch, capsys):
    _no_engine(monkeypatch)
    with pytest.warns(correct_mode.CorrectModeGapWarning, match="WB4, WB5"):
        resolved = correct_mode.engine_settings("correct")
    assert resolved["settings"] == {}
    assert resolved["engine_takes_mode"] is False
    receipt = resolved["receipt"]
    assert receipt["mode"] == "correct"
    assert receipt["applied"] == [] and receipt["skipped"] == []
    assert [row["finding"] for row in receipt["gaps"]] == ["WB4", "WB5"]
    assert all(row["applied"] is False for row in receipt["gaps"])
    assert [row["slot"] for row in receipt["gaps"]] == ["convection", "microphysics"]
    assert all(row["owner"] and row["reason"] for row in receipt["gaps"])
    err = capsys.readouterr().err
    assert "CORRECT MODE: named gap WB4 NOT APPLIED" in err
    assert "CORRECT MODE: named gap WB5 NOT APPLIED" in err


def test_the_run_record_carries_every_gap_once(monkeypatch):
    # The door writes this block into the run receipt after the run (the LTS route persists no backend
    # receipt), so a correct-mode run's gaps reach its receipt, once each however many backends hit them.
    _no_engine(monkeypatch)
    correct_mode.reset_run_record()
    assert correct_mode.run_record("correct")["engine_resolutions"] == 0
    assert "note" in correct_mode.run_record("correct")
    for _ in range(4):  # one backend per rate class
        with pytest.warns(correct_mode.CorrectModeGapWarning):
            correct_mode.engine_settings("correct", active_schemes={"wsm6"})
    record = correct_mode.run_record("correct")
    assert record["engine_resolutions"] == 4 and "note" not in record
    assert record["mode"] == "correct" and record["applied"] == []
    assert [row["finding"] for row in record["gaps"]] == ["WB5"]
    assert record["gaps"][0]["owner"] and record["gaps"][0]["slot"] == "microphysics"
    # a sealing step's silent resolution records nothing
    correct_mode.reset_run_record()
    correct_mode.engine_settings("correct", report=False)
    assert correct_mode.run_record("correct")["engine_resolutions"] == 0
    correct_mode.reset_run_record()


def test_a_gap_names_only_the_schemes_the_run_runs(monkeypatch):
    _no_engine(monkeypatch)
    with pytest.warns(correct_mode.CorrectModeGapWarning):
        resolved = correct_mode.engine_settings("correct", active_schemes={"wsm6", "gf"})
    assert [g.finding for g in resolved["gaps"]] == ["WB5"]
    with pytest.warns(correct_mode.CorrectModeGapWarning):
        skipped = correct_mode.engine_settings("correct", ["WB5"])
    assert [g.finding for g in skipped["gaps"]] == ["WB4"]


def test_a_seam_without_mode_keywords_is_a_named_gap(monkeypatch):
    _fake_engine(monkeypatch)

    def old_engine_init(self, *, n_levels, n_columns):
        pass

    with pytest.warns(correct_mode.CorrectModeGapWarning):
        resolved = correct_mode.engine_settings(
            "correct", seam=old_engine_init, seam_label="MpasColumnBatchPhysics")
    assert resolved["engine_takes_mode"] is False and resolved["settings"] == {}
    assert "MpasColumnBatchPhysics" in resolved["gaps"][0].reason


def test_an_engine_that_carries_the_corrections_has_no_gap(monkeypatch, recwarn):
    _fake_engine(monkeypatch)
    resolved = correct_mode.engine_settings("correct")
    assert resolved["gaps"] == [] and resolved["receipt"]["gaps"] == []
    assert resolved["settings"] == {"ntiedtke_dry_mass_coupling": True,
                                    "wsm6_conservative_surface": True}
    assert not [w for w in recwarn if issubclass(w.category, correct_mode.CorrectModeGapWarning)]


def test_exact_mode_touches_no_engine(monkeypatch):
    _no_engine(monkeypatch)
    assert correct_mode.engine_settings("exact") == {
        "settings": {}, "receipt": None, "gaps": [], "engine_takes_mode": True}


def test_the_arwen_backend_keeps_mode_keywords_out_of_an_old_engine(monkeypatch):
    import types

    from woof.hex import cuda_arwen_physics_v841 as arwen

    _fake_engine(monkeypatch)

    class OldBatch:
        def __init__(self, *, n_levels, n_columns):
            pass

    batch = types.ModuleType("woof.core.mpas_column_batch")
    batch.MpasColumnBatchPhysics = OldBatch
    monkeypatch.setitem(sys.modules, "woof.core.mpas_column_batch", batch)
    # `import woof.core.mpas_column_batch as m` reads the package attribute first: on gpuwm 2.9.0 an
    # earlier test binds the real batch there, whose constructor takes the mode keywords.
    monkeypatch.setattr(sys.modules["woof.core"], "mpas_column_batch", batch, raising=False)
    constructor = types.SimpleNamespace(
        physics_mode="correct", physics_correct_skip=(), microphysics_scheme="wsm6",
        cumulus_scheme="gf")
    with pytest.warns(correct_mode.CorrectModeGapWarning):
        receipt, takes_mode = arwen._engine_correct_mode(constructor)
    assert takes_mode is False
    assert [row["finding"] for row in receipt["gaps"]] == ["WB5"]
    exact = types.SimpleNamespace(physics_mode="exact")
    assert arwen._engine_correct_mode(exact) == (None, True)
    assert correct_mode.without_mode_keywords(
        {"n_levels": 1, "physics_mode": "correct", "physics_correct_skip": ()}) == {"n_levels": 1}


def test_the_door_validates_before_reading_inputs():
    from woof.hex import forecast_door

    fields = forecast_door._physics_mode_fields(argparse.Namespace(
        physics_mode="correct", physics_correct_skip=["WB5"]))
    assert fields == {"physics_mode": "correct", "physics_correct_skip": ("WB5",)}
    # No mode named and no preset row: the shipped default, correct.
    assert forecast_door._physics_mode_fields(argparse.Namespace()) == {
        "physics_mode": "correct", "physics_correct_skip": ()}
    with pytest.raises(Exception, match="cannot switch"):
        forecast_door._physics_mode_fields(argparse.Namespace(
            physics_mode="correct", physics_correct_skip=["WB0"]))


# --------------------------------------------------------------------------
# The generic (registry-resolved) suite entry records the mode and the
# engine's receipt too (repair of the verifier's finding: only the five-slot
# entry and the Arwen backend wrote correct_mode into the receipt).  The
# engine is faked: these checks are about hex's sealing and receipt, and run
# on a host with no engine checkout.

def _fake_engine(monkeypatch):
    import types

    applied_keys = {"WB4": "ntiedtke_dry_mass_coupling", "WB5": "wsm6_conservative_surface"}

    class _Resolution:
        def __init__(self, mode, skip):
            self.mode, self.skipped = mode, tuple(skip)
            self.applied = tuple(f for f in ("WB4", "WB5") if f not in self.skipped)
            self.settings = {applied_keys[f]: True for f in self.applied}

        def receipt(self):
            return {"schema": "gpuwm-correct-mode-v1", "physics.mode": self.mode,
                    "physics.correct_skip": list(self.skipped),
                    "applied_findings": list(self.applied)}

    engine_mode = types.ModuleType("woof.correct_mode")
    engine_mode.resolve = lambda mode, skip=(), *, source="physics.mode": _Resolution(mode, skip)
    suite_module = types.ModuleType("woof.core.mpas_column_suite")
    suite_module.column_physics_plan = lambda physics, **_: types.SimpleNamespace(
        n_soil_levels=4, static_fields=(), species=("qv", "qc", "qr"), identity={"fake": True})
    woof = sys.modules.get("woof") or types.ModuleType("woof")
    core = sys.modules.get("woof.core") or types.ModuleType("woof.core")
    monkeypatch.setitem(sys.modules, "woof", woof)
    monkeypatch.setitem(sys.modules, "woof.core", core)
    monkeypatch.setitem(sys.modules, "woof.correct_mode", engine_mode)
    monkeypatch.setitem(sys.modules, "woof.core.mpas_column_suite", suite_module)
    monkeypatch.setattr(woof, "correct_mode", engine_mode, raising=False)
    monkeypatch.setattr(woof, "core", core, raising=False)
    monkeypatch.setattr(core, "mpas_column_suite", suite_module, raising=False)


def _generic_values(**extra):
    from datetime import datetime

    import numpy as np

    ncol, nlev, nsoil = 3, 6, 4
    f32 = lambda value: np.full(ncol, value, np.float32)  # noqa: E731
    values = {
        "physics": {"mp_physics": 6}, "n_levels": nlev, "n_columns": ncol, "dt": 60.0,
        "radiation_seconds": 1800.0, "surface_pbl_seconds": 60.0, "cumulus_seconds": 60.0,
        "cumulus_scheme": "gf", "microphysics_scheme": "wsm6",
        "start_time": datetime(2026, 10, 13, 12), "p_top_pa": 2500.0, "dx_m": 12000.0,
        "gf_ishallow": 0, "wsm6_hail_opt": 0, "fractional_seaice": None,
        "orographic_statistics": None, "radiation_scheme": "rrtmg_legacy",
        "surface_layer_scheme": "sfclay", "land_surface_scheme": "noah", "pbl_scheme": "ysu",
        "drag_scheme": "none", "latitude_deg": f32(38.0), "longitude_deg": f32(-100.0),
        "terrain_height_m": f32(300.0), "landmask": f32(1.0), "xland": f32(1.0),
        "vegfra": f32(40.0), "tsk": f32(295.0), "tmn": f32(287.0), "xice": f32(0.0),
        "snow": f32(0.0), "snow_depth": f32(0.0),
        "ivgtyp": np.full(ncol, 10, np.int32), "isltyp": np.full(ncol, 6, np.int32),
        "soil_temperature": np.full((nsoil, ncol), 290.0, np.float32),
        "soil_moisture": np.full((nsoil, ncol), 0.25, np.float32),
        "dx_column_m": f32(12000.0), "z_interface_nominal_m": np.linspace(0.0, 20000.0, nlev + 1),
    }
    values.update(extra)
    return values


def test_generic_suite_entry_exact_mode_seals_the_identity_of_before(monkeypatch):
    from woof.hex import cuda_suite_physics_v841 as suite

    _fake_engine(monkeypatch)
    base = suite.seal_constructor(_generic_values())
    explicit = suite.seal_constructor(_generic_values(physics_mode="exact"))
    assert explicit.identity_sha256 == base.identity_sha256
    assert explicit.physics_mode == "exact"
    assert "physics_mode" not in explicit.receipt()
    assert suite.suite_correct_mode_receipt(explicit) == (None, True)


def test_generic_suite_entry_correct_mode_receipt_lists_the_findings(monkeypatch):
    from woof.hex import cuda_suite_physics_v841 as suite

    _fake_engine(monkeypatch)
    sealed = suite.seal_constructor(_generic_values(physics_mode="correct"))
    assert sealed.physics_mode == "correct"
    assert dict(sealed.physics)["wsm6_conservative_surface"] is True
    assert sealed.receipt()["physics_mode"] == "correct"
    receipt, takes_mode = suite.suite_correct_mode_receipt(sealed)
    assert takes_mode is True
    assert receipt is not None and receipt["applied"] == ["WB4", "WB5"]
    assert receipt["engine"]["applied_findings"] == ["WB4", "WB5"]
    assert receipt["gaps"] == []


def test_generic_suite_entry_skipping_every_finding_is_still_correct_mode(monkeypatch):
    # Before the repair the generic entry inferred the mode from the physics
    # keys, so a row skipping both findings read back as exact.
    from woof.hex import cuda_suite_physics_v841 as suite

    _fake_engine(monkeypatch)
    sealed = suite.seal_constructor(_generic_values(
        physics_mode="correct", physics_correct_skip=("WB4", "WB5")))
    assert "wsm6_conservative_surface" not in dict(sealed.physics)
    assert sealed.physics_mode == "correct"
    assert sealed.physics_correct_skip == ("WB4", "WB5")
    receipt, _ = suite.suite_correct_mode_receipt(sealed)
    assert receipt["applied"] == [] and receipt["skipped"] == ["WB4", "WB5"]
    assert receipt["gaps"] == []


def test_generic_suite_entry_on_an_engine_without_correct_mode_names_gaps(monkeypatch):
    from woof.hex import cuda_suite_physics_v841 as suite

    _fake_engine(monkeypatch)
    sealed_with = suite.seal_constructor(_generic_values(physics_mode="correct"))
    monkeypatch.setitem(sys.modules, "woof.correct_mode", None)
    monkeypatch.delattr(sys.modules["woof"], "correct_mode", raising=False)
    sealed = suite.seal_constructor(_generic_values(physics_mode="correct"))
    assert sealed.physics_mode == "correct"
    assert "wsm6_conservative_surface" not in dict(sealed.physics)
    assert sealed.identity_sha256 != sealed_with.identity_sha256
    with pytest.warns(correct_mode.CorrectModeGapWarning):
        receipt, _ = suite.suite_correct_mode_receipt(sealed)
    assert receipt["applied"] == []
    assert [row["finding"] for row in receipt["gaps"]] == ["WB4", "WB5"]

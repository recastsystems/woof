# tests/test_restart.py
"""Restart files: manifest completeness, exact roundtrip, bit-identity.

CPU tier (no device work): the DomainState/PhysicsDriver/scratch manifest
classification is exhaustively cross-checked against the source (a new
model attribute or scratch slot without a manifest entry fails here and in
every ``write_restart``), synthetic host arrays round-trip bit-exactly
through the NPZ format, and config/setup mismatches are rejected loudly.
The DomainState introspection runs on a NumPy-backed cupy shim so no GPU
allocation happens.

GPU tier (controller-run): a short full-physics 20+20-vs-40-step
continuation on the small physics state pins FP32 bit identity for every
serialized field, and the slow-acceptance real74 gate runs 6 h -> restart
-> FRESH PROCESS -> 6 h against the uninterrupted 12 h via the CLI,
comparing the complete 12 h restart serializations (every prognostic,
physics field, accumulator, and held tendency), the hour-7..12 wrfouts,
and the run-summary trackers.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_cupy

from conftest import requires_gpu
from woof.core.model import (ADAPTIVE_TIMESTEP_RUN_FIELDS,
                              publish_declared_experiment)

from woof.config import RunConfig
from woof.core.nssl2_contract import (
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_MODE as NSSL2_DEFAULT_MODE,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT as NSSL2_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as NSSL2_WRF_REFERENCE_VERSION,
)
from woof.io import restart


BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "wrfout_reference" /
         "wrfout_d01_1974-04-03_13_00_00").is_file(),
    reason="WRF_1974_MP55 reference bundle not present",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CPU helpers: a NumPy-backed DomainState (no device allocation).
# ---------------------------------------------------------------------------

class _NumpyCupyShim:
    """The subset of the cupy module surface DomainState allocation uses."""

    ndarray = np.ndarray
    float32 = np.float32

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def ascontiguousarray(value):
        return np.ascontiguousarray(value)

    isfinite = staticmethod(np.isfinite)
    any = staticmethod(np.any)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)


class _DeclaredPhysicsCallable:
    """Minimal public injection-contract callable with explicit identity."""

    def __init__(self, name):
        self.restart_identity = {
            "algorithm": name,
            "above_atmosphere_policy": "declared-custom-policy-v1",
            "assets": {"fixture": {"sha256": "declared-by-caller"}},
        }

    def __call__(self, **kwargs):
        raise AssertionError("identity tests never execute physics")


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=6, ny=4, nz=5, dx=2000.0, dy=2000.0, ztop=8000.0,
                  dt=10.0, run_seconds=0.0)
    values.update(overrides)
    return RunConfig(**values)


def _shim_state(cfg, monkeypatch):
    import woof.core.state as state_module
    # woof.core.dycore binds state's cp at import (``from woof.core.state
    # import cp``) and keeps it.  Load it with the real one before the shim
    # replaces it: a dycore first imported inside this window (building a
    # PhysicsDriver reaches it) kept the shim for the rest of the process
    # (2.9.11 pass 3 R5: test_checkpoint_config_echo, then the batch layout
    # trials raised "_NumpyCupyShim has no attribute 'ElementwiseKernel'").
    import woof.core.dycore  # noqa: F401

    monkeypatch.setattr(state_module, "cp", _NumpyCupyShim)
    return state_module.DomainState(cfg)


def _fill_setup(state) -> None:
    """Deterministic nonzero setup arrays (identical across builds)."""
    # Preserve every pre-v7 fixture word when the inverse identity is added.
    names = tuple(n for n in restart.STATE_SETUP_ARRAYS if n != "msfv_inv") + ("msfv_inv",)
    for index, name in enumerate(names):
        array = getattr(state, name)
        array[...] = np.arange(array.size, dtype=np.float64).reshape(
            array.shape).astype(np.float32) * np.float32(0.01 * (index + 1))


def _fill_serialized(state, seed: int) -> None:
    """Random FP32 payloads plus adversarial bit patterns."""
    rng = np.random.default_rng(seed)
    for name in restart.STATE_SERIALIZED_ATTRS:
        array = getattr(state, name, None)
        if array is None:
            continue
        array[...] = rng.standard_normal(array.shape).astype(np.float32)
        flat = array.reshape(-1)
        flat[0] = np.float32(-0.0)           # signed zero
        flat[-1] = np.float32(1.0e-42)       # denormal
        if flat.size > 2:
            flat[1] = np.float32(np.nan)     # NaN payload survives NPZ


def _replace_restart_member(path, output, key, replacement):
    """Rewrite one test member and keep the archive's self-manifest valid."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    payload[key] = np.asarray(replacement)
    header["array_manifest"][key] = {
        "shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _rewrite_restart_archive(path, output, edit):
    """Apply a test-only archive edit and rebuild its self-manifest."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload.pop(restart._HEADER_KEY))).decode("utf-8"))
    edit(payload, header)
    header["array_manifest"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
    }
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _shim_driver_state(cfg, monkeypatch):
    """NumPy-backed PhysicsDriver with required surface outputs and no callables."""
    import woof.core.physics as physics

    state = _shim_state(cfg, monkeypatch)
    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    fields = ({name: np.zeros(state.mup.shape, dtype=np.float32)
               for name in physics.MYNN_SURFACE_OUTPUTS}
              if cfg.sf_sfclay_physics == physics.MYNN_SFCLAY_SCHEME else {})
    driver = physics.PhysicsDriver(
        state, cfg, fields=fields, sfclay_result=None, noah_params=None)
    state.physics = driver
    return state, driver


def _identity_bound_physics_state(cfg, monkeypatch, *, trace_co2=3.30e-4,
                                  latitude_offset=0.0,
                                  noah_offset=0.0):
    """CPU-only fully configured physics setup for restart-identity tests."""
    from woof.core.kf import KainFritsch
    from woof.core.noah import NoahParams
    from woof.core.rrtmgp import RRTMGPRadiation

    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    shape = state.mup.shape
    latitude = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    radiation = object.__new__(RRTMGPRadiation)
    radiation.start_time = datetime(1974, 4, 3, 12)
    radiation.latitude_deg = latitude + np.float32(latitude_offset)
    radiation.longitude_deg = np.full(shape, -87.0, dtype=np.float32)
    radiation.column_chunk = 17
    radiation.validation_mode = "fused"
    radiation.trace_gas_overrides = {"co2": 3.30e-4}
    radiation.trace_vmr = {"co2": float(trace_co2), "n2o": 3.0e-7}
    radiation._ozone_logp = np.linspace(1.0, 2.0, cfg.nz,
                                       dtype=np.float32)
    radiation._ozone_vmr = np.linspace(3.0e-8, 8.0e-6, cfg.nz,
                                      dtype=np.float32)
    radiation.update_count = 0
    radiation.lw_tables = SimpleNamespace(
        kind="lw", coefficient=np.array([1.0, 2.0], dtype=np.float64))
    radiation.sw_tables = SimpleNamespace(
        kind="sw", coefficient=np.array([3.0, 4.0], dtype=np.float64))
    radiation.lw_cloud_tables = SimpleNamespace(
        kind="lw", extinction=np.array([5.0], dtype=np.float64))
    radiation.sw_cloud_tables = SimpleNamespace(
        kind="sw", extinction=np.array([6.0], dtype=np.float64))
    radiation.chunk_workspace = None
    driver.radiation_callable = radiation
    driver.cumulus_callable = KainFritsch()
    driver.noah_params = NoahParams(
        veg=np.arange(30, dtype=np.float64).reshape(2, 15)
            + float(noah_offset),
        soil=np.arange(20, dtype=np.float64).reshape(2, 10),
        gen=np.arange(16, dtype=np.float64),
        lucats=2, slcats=2, bare=1, natural=2,
        lutype="USGS", sltype="STAS",
    )
    return state, driver


# ---------------------------------------------------------------------------
# Manifest completeness (CPU).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(moist=False),
    dict(moist=True, mp_physics=1),
    dict(moist=True, mp_physics=8),
    dict(moist=True, mp_physics=9),
    dict(moist=True, mp_physics=10),
    dict(moist=True, mp_physics=18),
    dict(moist=True, mp_physics=50),
], ids=["dry", "kessler", "thompson", "milbrandt2", "morrison", "nssl2",
        "p3"])
def test_every_domainstate_attribute_is_classified(monkeypatch, overrides):
    """A DomainState attribute outside the manifest raises; the serialized
    manifest names exactly the allocated cross-step arrays per config."""
    state = _shim_state(_cfg(**overrides), monkeypatch)
    for name in vars(state):
        restart.classify_state_attr(name)  # raises on any unclassified attr

    manifest = restart.state_manifest(state)
    expected = {f"state/{name}" for name in restart.STATE_SERIALIZED_ATTRS
                if getattr(state, name, None) is not None}
    # CHECKPOINT-ONLY carriers ride their OWN namespace, and the
    # separation is essential rather than cosmetic: `state/` is what
    # live_state_sha256 hashes, and therefore what relocate_child
    # compares to assert a parent is never written across a move.  A
    # dycore-workspace view in there makes that assertion fire on
    # workspace churn, which is a false positive on a real safety check.
    expected |= {f"acoustic/{name}"
                 for name in restart.CHECKPOINT_ONLY_STATE
                 if getattr(state, name, None) is not None}
    assert set(manifest) == expected
    assert not any(key.startswith("state/") for key in manifest
                   if key[len("state/"):] in restart.CHECKPOINT_ONLY_STATE)
    if overrides.get("mp_physics") == 10:
        # W6 advisory: h_diabatic is restart state (WRF `rdu`,
        # Registry.EM_COMMON:1389); effective radii feed the next
        # radiation call.
        assert {"state/h_diabatic", "state/effc", "state/ng"} <= set(manifest)
    if overrides.get("mp_physics") == 9:
        # Milbrandt-Yau carries hail mass beside graupel and a number
        # moment for all six categories; every one is a transported
        # prognostic (woof/core/moist.py::MY2_SPECIES), so every one is
        # in the serialized manifest.  The NSSL spellings are a DIFFERENT
        # scheme's package and must not appear.
        assert {f"state/{name}"
                for name in restart.MILBRANDT2_RESTART_STATE} <= set(manifest)
        assert not (set(manifest) & {
            "state/qndrop", "state/qnr", "state/qni", "state/qns",
            "state/qng", "state/qnh", "state/qnn", "state/qvolg",
            "state/qvolh",
        })
    if overrides.get("mp_physics") == 18:
        expected_nssl = {
            *(f"state/{name}" for name in restart.NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}"
              for name in restart.NSSL2_RESTART_AUXILIARY_STATE),
        }
        assert expected_nssl <= set(manifest)
        assert not (set(manifest) & {
            "state/nc", "state/nr", "state/ni", "state/ns", "state/ng",
        })
    if overrides.get("mp_physics") == 50:
        # P3's rime mass/volume and its two cross-step supersaturation
        # carriers are all restart state in WRF (Registry.EM_COMMON:555-558
        # ``i0rhusdf`` and :1598-1599 ``rusd`` -- both carry the ``r``), and
        # the rime pair is transported besides.  Their RK time-t copies are
        # rebuilt, and P3 has no snow/graupel inventory at all.
        assert {"state/qir", "state/qib",
                "state/th_old", "state/qv_old"} <= set(manifest)
        assert not (set(manifest) & {
            "state/qs", "state/qg", "state/effs", "state/qir0",
            "state/qib0",
        })
    if not overrides.get("moist"):
        assert "state/qv" not in manifest


def test_unclassified_state_attribute_fails_the_write(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    state.brand_new_accumulator = np.zeros(3, dtype=np.float32)
    with pytest.raises(restart.RestartManifestError,
                       match="brand_new_accumulator"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)


def test_a_per_follower_window_slot_does_not_fail_the_write(
        monkeypatch, tmp_path):
    """The classifier must stay TOTAL over the slots a run allocates.

    ``CARRIED_SCRATCH_SLOTS`` holds exact names, but a per-domain [follow]
    allocates one window PER independently-cadenced child --
    ``uh_follow_window.dNN`` (woof/runtime.py, via
    ``uh_diag.follow_window_slot``).  Those names match no carried name and
    no rebuilt prefix, so ``_scratch_manifest``'s walk over the live pool
    raised at the FIRST checkpoint any such run tried to write: a run with a
    per-domain follow and ``restart_interval_s > 0`` died at its first
    checkpoint instant, hours in, having produced no checkpoint at all.
    The write path is not covered by the resume-time lifecycle refusal.
    """
    from woof.core import uh_diag

    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    slot = uh_diag.follow_window_slot(4)
    assert slot == "uh_follow_window.d04"
    state.scratch((2, 2), slot)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)
    with np.load(path) as archive:
        # Carried, not serialized: the window is "max since that consumer
        # last looked" and this file cannot say when that was, so it stays
        # out of the stream exactly as its two exact-named siblings do.
        assert f"scratch/{slot}" not in set(archive.files)


def test_per_follower_window_slots_classify_with_the_tracker_windows(
        monkeypatch):
    """One classification for all three, keyed off uh_diag's own predicate.

    ``is_tracker_window_slot`` (woof/core/uh_diag.py) already answers this
    question for the rest of the model by testing the two exact names plus
    the per-follower PREFIX.  The restart classifier answering differently
    for the third form is what made it partial.
    """
    from woof.core import uh_diag

    for gid in (2, 4, 12):
        slot = uh_diag.follow_window_slot(gid)
        assert uh_diag.is_tracker_window_slot(slot)
        assert restart.classify_scratch_slot(slot) == "carry"
    for slot in uh_diag.TRACKER_WINDOW_SLOTS:
        assert restart.classify_scratch_slot(slot) == "carry"
    # Both directions: the prefix admits the generated family and nothing
    # else.  A near-miss under the same stem must still fail closed rather
    # than be silently dropped from every checkpoint that holds it.
    for near_miss in ("uh_follow_window.zz", "uh_follow_window_extra",
                      "uh_follow_windows.d04"):
        assert not uh_diag.is_tracker_window_slot(near_miss)
        with pytest.raises(restart.RestartManifestError):
            restart.classify_scratch_slot(near_miss)


def test_a_per_follower_window_slot_is_a_streaming_carrier(
        monkeypatch, tmp_path):
    """The CARRY class is read by the streaming inventory, so prove it there.

    ``carried_scratch_manifest`` is what
    ``tilestream.physics_inventory.carrier_manifest`` takes, and it walks the
    live pool through the same classifier -- so the partial classifier broke
    the streamed carrier set in the same way it broke the checkpoint write.
    A window that is not a carrier is accumulated separately inside each tile
    buffer, over whatever tiles that buffer served, and the domain then has
    no window anywhere.
    """
    from woof.core import uh_diag

    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    slot = uh_diag.follow_window_slot(4)
    state.scratch((2, 2), slot)
    state.scratch((2, 2), uh_diag.UH_SPAWN_WINDOW_SLOT)
    carried = restart.carried_scratch_manifest(state)
    assert f"scratch/{slot}" in carried
    assert f"scratch/{uh_diag.UH_SPAWN_WINDOW_SLOT}" in carried


def test_physicsdriver_attribute_classification_matches_source():
    """Every ``self.*`` assigned anywhere in PhysicsDriver is classified,
    and every classified name exists in the source (no stale entries)."""
    import woof.core.physics as physics

    tree = ast.parse(Path(physics.__file__).read_text(encoding="utf-8"))
    class_node = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "PhysicsDriver")
    assigned = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    assigned.add(sub.attr)
    serialized = set(restart.DRIVER_SERIALIZED_ATTRS)
    rebuilt = set(restart.DRIVER_REBUILT_ATTRS)
    checkpoint_only = set(restart.DRIVER_CHECKPOINT_ONLY_ATTRS)
    classified = serialized | rebuilt | checkpoint_only
    # THREE classes, and a name belongs to exactly one: the union above
    # would hide a name filed twice, and "rebuilt" and "carried in the
    # checkpoint" are contradictory instructions to the restore path.
    assert not serialized & rebuilt
    assert not serialized & checkpoint_only
    assert not rebuilt & checkpoint_only
    assert "refl_10cm" in restart.DRIVER_REBUILT_ATTRS
    assert "refl_10cm" not in restart.DRIVER_SERIALIZED_ATTRS
    for name in ("_sr_roundoff_upper", "_sr_roundoff_max_ulps",
                 "_wsm6_minor_loops"):
        assert name in restart.DRIVER_REBUILT_ATTRS
        assert name not in restart.DRIVER_SERIALIZED_ATTRS
    assert assigned == classified, {
        "unclassified (add to woof/io/restart.py)":
            sorted(assigned - classified),
        "stale manifest entries": sorted(classified - assigned),
    }


def test_olr_is_carried_by_the_checkpoint_and_not_rebuilt():
    """It was ``rebuild``, and a resumed run therefore published 0 W m-2
    of outgoing longwave into every frame before its first post-restart
    radiation call -- the only variable of 77 that differed across a
    restart, on any configuration.  It is written on RADIATION's cadence,
    which is coarser than the history cadence, so a frame between calls
    publishes the last computed field and there is nothing to rebuild it
    from.

    Its own namespace, like ``acoustic/``: no ``driver/``/``fields/``/
    ``cumulus/`` key set moves, so every checkpoint already on disk stays
    restorable.  That was the stated cost of carrying it, and it is not a
    cost this shape pays.
    """
    assert "olr" in restart.DRIVER_CHECKPOINT_ONLY_ATTRS
    assert "olr" not in restart.DRIVER_REBUILT_ATTRS
    assert "olr" not in restart.DRIVER_SERIALIZED_ATTRS


def _harvest_scratch_slots() -> tuple[set[str], set[str]]:
    """All literal slot names and dynamic-slot prefixes in the package."""
    import woof

    literals: set[str] = set()
    prefixes: set[str] = set()
    for source in Path(woof.__file__).parent.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "scratch"
                    and len(node.args) >= 2):
                continue
            slot = node.args[1]
            if isinstance(slot, ast.Constant) and isinstance(slot.value, str):
                literals.add(slot.value)
            elif isinstance(slot, ast.JoinedStr) and slot.values and \
                    isinstance(slot.values[0], ast.Constant):
                prefixes.add(str(slot.values[0].value))
            elif isinstance(slot, ast.BinOp) and \
                    isinstance(slot.left, ast.Constant):
                prefixes.add(str(slot.left.value))
            # Name-valued slots (dycore/diffusion spec tuples) are string
            # literals elsewhere in the same module; the module-wide
            # literal sweep below picks them up.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and (node.value.startswith(("smag_", "diff6_", "diff_"))
                         )):
                literals.add(node.value)
    return literals, prefixes


def test_every_source_scratch_slot_is_classified():
    """New scratch slots must be classified: every literal slot name in
    the package classifies without raising, and every dynamically built
    slot family resolves to a known classification.

    Tier note (review F6): this static harvest covers direct literals,
    f-string/concatenation prefixes, and the Name-passed slot families via
    the module-wide constant sweep (smag_/diff6_/diff_).  A future
    Name-passed slot under some other prefix escapes THIS test only:
    the enforcement boundary is write_restart's runtime walk over
    ``state._scratch`` (every live slot classifies or the write raises),
    which the GPU gates execute against live full-physics state.  This
    test is the early-warning tier, not the enforcement."""
    literals, prefixes = _harvest_scratch_slots()
    assert "mp_rainnc" in literals and "cu_nca" in literals  # harvest sanity
    assert "smag_ru" in literals and "diff6_m" in literals   # name-passed
    for slot in sorted(literals):
        restart.classify_scratch_slot(slot)   # raises on any new slot
    for prefix in sorted(prefixes):
        if prefix == "mp_":
            # The family executor's dynamic names are bounded by its table.
            # They are serialized accumulators, never rebuilt scratch.
            from woof.core.wsm_family_constants import selectors, precipitation_classes
            for selector in selectors():
                for species in precipitation_classes(selector):
                    for suffix in ("nc", "ncv"):
                        assert restart.classify_scratch_slot(f"mp_{species}{suffix}") == "serialize"
            continue
        assert (prefix.startswith(tuple(restart.REBUILT_SCRATCH_PREFIXES))
                or prefix in ("cu_", "ltng_")), prefix
    # The one dynamic serialized family: the KF stored per-column rates.
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten"):
        assert restart.classify_scratch_slot(f"cu_{name}") == "serialize"


def test_scratch_classification_kinds():
    for slot in ("mp_rainnc", "mp_snownc", "mp_graupelnc", "mp_sr",
                 "mp_kessler_sr", "cu_rainc", "cu_nca", "cu_pratec",
                 "cu_raincv", "cu_rthcuten"):
        assert restart.classify_scratch_slot(slot) == "serialize", slot
    for slot in ("rk_ww", "adv_ru", "smag_rqv", "smag_km", "diff6_m",
                 "diff_u", "acoustic_c2a", "openbc_upp_faces",
                 "moist_pd_q0", "pd_fxl", "morr_theta", "mp_th", "mp_z8w",
                 "nssl2_fused_temperature", "nssl2_primary_ice_target",
                 "nssl2_nucond_ss",
                 "refl_10cm", "physics_qtot", "lbc_relax_u",
                 "lbc_weights_0", "lbc_old_mup_frame_1",
                 "lbc_forcing_tables", "integration_health_partial"):
        assert restart.classify_scratch_slot(slot) == "rebuild", slot
    with pytest.raises(restart.RestartManifestError):
        restart.classify_scratch_slot("totally_new_slot")


def test_serialized_dataclasses_match_their_component_manifests():
    """Review F1: the component tuples are pinned to the dataclasses both
    ways, so a future PhysicsTendencies/MicrophysicsDiagnostics field
    without a manifest update fails here (and every write fails through
    the same helper)."""
    import dataclasses as dc

    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.core.physics import PhysicsTendencies

    assert ({field.name for field in dc.fields(PhysicsTendencies)}
            == set(restart.TENDENCY_COMPONENTS))
    assert ({field.name for field in dc.fields(MicrophysicsDiagnostics)}
            == set(restart.MICROPHYSICS_COMPONENTS))
    # The exact helper the write path runs accepts the real classes.
    restart._require_dataclass_components(
        PhysicsTendencies, restart.TENDENCY_COMPONENTS, "PhysicsTendencies")
    restart._require_dataclass_components(
        MicrophysicsDiagnostics, restart.MICROPHYSICS_COMPONENTS,
        "MicrophysicsDiagnostics")


def test_planted_dataclass_field_fails_the_write_helper():
    """A planted (unclassified) dataclass field must fail the write,
    the negative for review F1, in both directions."""
    import dataclasses as dc

    planted = dc.make_dataclass(
        "PlantedTendencies",
        [(name, object) for name in restart.TENDENCY_COMPONENTS]
        + [("rqg", object)])
    with pytest.raises(restart.RestartManifestError, match="rqg"):
        restart._require_dataclass_components(
            planted, restart.TENDENCY_COMPONENTS, "PhysicsTendencies")
    with pytest.raises(restart.RestartManifestError, match="gone_field"):
        restart._require_dataclass_components(
            planted,
            tuple(restart.TENDENCY_COMPONENTS) + ("rqg", "gone_field"),
            "PhysicsTendencies")


def test_callable_walk_catches_container_hidden_state():
    """Review F2: arrays hidden in dict-valued or object-container
    callable attributes fail the write unless the container is
    explicitly classified."""

    class Bag:
        pass

    scheme = Bag()
    scheme.w0avg = np.zeros((2, 2), np.float32)
    restart._callable_state_check(
        scheme, frozenset({"w0avg"}), frozenset(), "cumulus")

    scheme.rates = {"rthcuten": np.zeros(3, np.float32)}
    with pytest.raises(restart.RestartManifestError, match="rates"):
        restart._callable_state_check(
            scheme, frozenset({"w0avg"}), frozenset(), "cumulus")
    del scheme.rates

    tables = Bag()
    tables.k_major = np.zeros(3, np.float32)
    scheme.tables = tables
    with pytest.raises(restart.RestartManifestError, match="tables"):
        restart._callable_state_check(
            scheme, frozenset({"w0avg"}), frozenset(), "radiation")
    # Explicit container classification accepts it DELIBERATELY.
    restart._callable_state_check(
        scheme, frozenset({"w0avg"}), frozenset({"tables"}), "radiation")

    direct = Bag()
    direct.cache = np.zeros(3, np.float32)
    with pytest.raises(restart.RestartManifestError, match="cache"):
        restart._callable_state_check(
            direct, frozenset(), frozenset(), "radiation")

    scalars = Bag()
    scalars.trace_vmr = {"co2": 330.0e-6}     # scalar dicts pass
    scalars.start_time = datetime(1974, 4, 3, 12)
    scalars.update_count = 3
    restart._callable_state_check(
        scalars, frozenset(), frozenset(), "radiation")


def test_rrtmgp_table_containers_are_explicitly_classified():
    """The RRTMGP gas/cloud table objects (rebuild-on-load lru caches)
    are in the container allowlist by name, not silently invisible."""
    assert restart.RADIATION_CALLABLE_CONTAINERS == frozenset(
        {"lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables",
         "chunk_workspace",
         # legacy-RRTMG adapter containers (all rebuild-on-load; see the
         # classification comments in woof/io/restart.py)
         "_C", "_sw_tables", "_cuda_sw", "_ozone_climo",
         "_night_outputs", "_ozone",
         # the RRTM+Dudhia (1/1) composition's rebuild-on-load
         # sub-adapters and the RRTM coefficient bundle (1.9.1 D3; see
         # woof/io/restart.py and tests/test_radiation_callable_manifest)
         "longwave_adapter", "shortwave_adapter", "_tables"})
    assert restart.CUMULUS_CALLABLE_CONTAINERS == frozenset(
        {"_history_state", "_driver"})


def test_every_cumulus_adapter_attribute_is_classified():
    """Each selectable cumulus scheme's LIVE adapter passes the walk.

    The classification allowlists are written against the adapters, so the
    adapters -- not hand-built stand-ins -- are what has to satisfy them.
    The GF adapter is the reason this test exists: it stores the
    PhysicsDriver back-reference ``bind_driver`` hands it, the driver is an
    array-bearing object container, and the walk refused every checkpoint
    of a ``cu_physics=3`` run until ``_driver`` was classified (the first
    real-case GF trajectory died at its first restart having integrated
    179 outer steps).  Constructing each adapter here needs no device.
    """
    from woof.core.gf import GrellFreitas
    from woof.core.kf import KainFritsch

    def check(adapter):
        restart._callable_state_check(
            adapter, restart.CUMULUS_CALLABLE_ARRAYS,
            restart.CUMULUS_CALLABLE_CONTAINERS, "cumulus")

    check(KainFritsch())
    check(GrellFreitas())

    # Bound, which is the only shape a checkpoint ever sees: PhysicsDriver
    # calls bind_driver on every cumulus step, so a GF adapter that has run
    # once always carries the driver.
    driver_like = SimpleNamespace(
        rthratenlw=np.zeros((2, 2, 2), np.float32),
        rthratensw=np.zeros((2, 2, 2), np.float32))
    bound = GrellFreitas()
    bound.bind_driver(driver_like)
    check(bound)

    # The walk is still armed: an adapter that grows real state of its own
    # must be classified, not absorbed by the container entries above.
    grown = GrellFreitas()
    grown.closure_memory = np.zeros((4, 4), np.float32)
    with pytest.raises(restart.RestartManifestError,
                       match="closure_memory"):
        check(grown)


def test_surface_audit_carry_in_names_live_in_the_fields_inventory():
    """The audit's explicit restart list is covered by whole-dict fields
    serialization: every name is a member of the driver fields inventory
    (SFCLAY outputs + the Noah launch set)."""
    from woof.core.noah import _F2D, _F3D
    from woof.core.sfclay import SFCLAY_OUTPUTS

    inventory = set(_F2D) | set(_F3D) | set(SFCLAY_OUTPUTS) | {"pblh"}
    audit_list = {"ust", "mol", "znt", "qsfc", "hfx", "qfx", "pblh",
                  "sh2o", "snotime", "albedo", "emiss"}
    assert audit_list <= inventory


# ---------------------------------------------------------------------------
# Roundtrip and rejection (CPU).
# ---------------------------------------------------------------------------

def test_physics_storage_aliases_are_lifetime_gated(monkeypatch):
    import woof.core.physics as physics
    from woof.core.microphysics import MicrophysicsDiagnostics

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               ra_physics=90, bldt=0.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)

    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(driver.microphysics, component) is state._scratch[slot]
    shape = state.mup.shape
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=np.full(shape, 2.0, np.float32),
        rainncv=np.full(shape, 0.25, np.float32),
        sr=np.full(shape, 0.5, np.float32)))
    assert driver.microphysics.rainnc is state._scratch["mp_rainnc"]
    np.testing.assert_array_equal(
        driver.microphysics.rainnc,
        np.full_like(driver.microphysics.rainnc, 2.0))
    np.testing.assert_array_equal(
        driver._pending_rainbl, np.full_like(driver._pending_rainbl, 0.25))
    assert driver.tendencies is driver.pbl_tendencies
    assert not physics.physics_retains_ysu_output(cfg)
    assert physics.physics_reuses_pbl_composition(cfg)

    for array in vars(driver.pbl_tendencies).values():
        if array is not None:
            array[...] = np.float32(1.0)
    for array in vars(driver.radiation_tendencies).values():
        if array is not None:
            array[...] = np.float32(2.0)
    for array in vars(driver.cumulus_tendencies).values():
        if array is not None:
            array[...] = np.float32(3.0)
    driver._compose_tendencies(cfg)
    assert driver.tendencies is driver.pbl_tendencies
    # Since d0c23dad0 the cumulus adapter contract carries momentum
    # tendencies and _compose_tendencies adds them into the composed
    # winds (zero face stacks for a scheme that supplies none, so
    # production is inert); with every cumulus array planted at 3.0 the
    # composed ru and rv read the PBL's 1.0 plus 3.0.

    for name in ("ru", "rv"):
        value = getattr(driver.tendencies, name)
        np.testing.assert_array_equal(value, np.full_like(value, 4.0))
    for name in ("rtheta", "rqv", "rqc"):
        value = getattr(driver.tendencies, name)
        np.testing.assert_array_equal(value, np.full_like(value, 6.0))

    held_cfg = replace(cfg, bldt=5.0)
    _held_state, held_driver = _shim_driver_state(held_cfg, monkeypatch)
    assert held_driver.tendencies is not held_driver.pbl_tendencies
    assert physics.physics_retains_ysu_output(held_cfg)
    assert not physics.physics_reuses_pbl_composition(held_cfg)


def test_mixed_phase_kf_held_inventory_is_eager_before_first_due_call(
        monkeypatch):
    """KF phase categories are canonical held state, not lazy work arrays.

    The real74 driver reuses the fresh PBL stack as its composed target at
    ``bldt=0``.  Consequently mixed-phase KF requires QR/QI/QS storage both
    in the raw cumulus family and in that target before the first scheduled
    KF call.  A due call may change bytes, but it must not change the restart
    manifest's array inventory.
    """
    from woof.core.physics import CumulusResult
    from woof.verify.cases.real74_d02 import (
        canonical_state_digest, validate_inventory_growth)

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               cu_physics=1, bldt=0.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    shape = state.p.shape
    clock = SimpleNamespace(dtbc_fp32=np.float32(cfg.dt))

    before = set(restart._driver_manifest(driver))
    before_inventory = canonical_state_digest(
        state, clock)["inventory"]
    for owner in ("pbl_tendencies", "cumulus_tendencies"):
        for component in ("rqr", "rqi", "rqs"):
            assert f"driver/{owner}/{component}" in before

    rates = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate((
            "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten"))
    }
    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**rates)
    driver._run_cumulus({}, state, cfg)
    driver._compose_tendencies(cfg)

    after = set(restart._driver_manifest(driver))
    assert after == before
    after_inventory = canonical_state_digest(state, clock)["inventory"]
    assert after_inventory["sha256"] == before_inventory["sha256"]
    assert validate_inventory_growth(
        before_inventory, after_inventory, domain="d01", ticks=900
    ) == after_inventory


def test_current_restart_rejects_missing_canonical_kf_phase_member(
        monkeypatch, tmp_path):
    """A v5 archive cannot turn eager held QS back into lazy state."""
    from woof.core.kf import KainFritsch

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               cu_physics=1, bldt=0.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    source_driver.cumulus_callable = KainFritsch()
    path = restart.write_restart(tmp_path / "canonical.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    missing = "driver/cumulus_tendencies/rqs"
    del payload[missing]
    del header["array_manifest"][missing]
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    tampered = tmp_path / "missing-rqs.npz"
    with tampered.open("wb") as stream:
        np.savez(stream, **payload)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    live_driver.cumulus_callable = KainFritsch()
    with pytest.raises(restart.RestartMismatchError,
                       match="missing canonical member.*rqs"):
        restart.restore_restart(tampered, live, cfg)


def test_wind_detection_receipt_is_rebuilt_without_hiding_unknown_state(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=10)
    _state, driver = _shim_driver_state(cfg, monkeypatch)
    before = restart._driver_manifest(driver)
    driver.wind_farm_detection = {"enabled": True, "turbine_count": 631}
    after = restart._driver_manifest(driver)
    assert after.keys() == before.keys()
    assert all(after[key] is before[key] for key in before)
    driver.wind_farm_unclassified_state = np.ones(1, dtype=np.float32)
    with pytest.raises(restart.RestartManifestError,
                       match="wind_farm_unclassified_state.*not classified"):
        restart._driver_manifest(driver)


def test_restart_refuses_unfinalized_kf_expiry_transition(monkeypatch):
    """A checkpoint cannot retain an unconsumed transient expiry mask."""
    cfg = _cfg(moist=True, mp_physics=10, cu_physics=1)
    _state, driver = _shim_driver_state(cfg, monkeypatch)

    # The host recovery receipt alone is sufficient to reject a checkpoint:
    # an exception may have occurred before a nonzero mask was published.
    driver._cu_expiry_pending = True
    with pytest.raises(restart.RestartManifestError,
                       match="KF expiry finalization is pending"):
        restart._driver_manifest(driver)
    driver.finish_step()
    assert driver._cu_expiry_pending is False
    restart._driver_manifest(driver)

    # Keep probing the device mask for direct callers and synthetic states
    # that do not update the host receipt.
    driver.cu_expiring[0, 0] = np.float32(1.0)

    with pytest.raises(restart.RestartManifestError,
                       match="KF expiry finalization is pending"):
        restart._driver_manifest(driver)

    driver.finish_step()
    restart._driver_manifest(driver)


def test_restart_v5_serializes_one_microphysics_set_and_reads_bound_v2_layout(
        monkeypatch, tmp_path):
    import woof.core.physics as physics

    cfg = _cfg(moist=True, mp_physics=10)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, (component, slot) in enumerate(
            physics.microphysics_scratch_slots(cfg.mp_physics), start=1):
        state._scratch[slot][...] = np.float32(index) / np.float32(8.0)
        assert getattr(driver.microphysics, component) is state._scratch[slot]

    path = restart.write_restart(tmp_path / "v5.npz", state, cfg)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    assert header["format_version"] == restart.RESTART_FORMAT_VERSION
    assert not any(key.startswith("driver/microphysics/")
                   for key in header["array_manifest"])

    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(fresh_driver.microphysics, component) is \
            fresh._scratch[slot]
        assert fresh._scratch[slot].tobytes() == state._scratch[slot].tobytes()

    # Synthesize the exact v2 dual-key layout from the v5 payload.  The array
    # shim remains testable only with the new mandatory physics-identity
    # header; genuinely old, unbound v2 files are intentionally refused.
    # It accepts the dual layout only while both historical copies have
    # identical bytes.
    header["format_version"] = 2
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        old_key = f"driver/microphysics/{component}"
        source = payload[f"scratch/{slot}"].copy()
        payload[old_key] = source
        header["array_manifest"][old_key] = {
            "shape": list(source.shape), "dtype": str(source.dtype)}
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    v2 = tmp_path / "v2.npz"
    with v2.open("wb") as stream:
        np.savez(stream, **payload)

    legacy, legacy_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(legacy)
    restart.restore_restart(v2, legacy, cfg)
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(legacy_driver.microphysics, component) is \
            legacy._scratch[slot]

    payload["driver/microphysics/rainnc"] = \
        payload["driver/microphysics/rainnc"].copy()
    payload["driver/microphysics/rainnc"].flat[0] += np.float32(1.0)
    mismatch = tmp_path / "v2-mismatch.npz"
    with mismatch.open("wb") as stream:
        np.savez(stream, **payload)
    rejected, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(rejected)
    with pytest.raises(restart.RestartMismatchError,
                       match="differ byte-for-byte"):
        restart.restore_restart(mismatch, rejected, cfg)


def test_wsm6_sr_exact_upper_roundtrips_restart_bits(monkeypatch, tmp_path):
    """The rebuilt guard metadata must not block or alter canonical SR."""
    from woof.core.microphysics import MicrophysicsDiagnostics

    cfg = _cfg(moist=True, mp_physics=6, dt=60.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    shape = source.mup.shape
    upper = np.float32(source_driver._sr_roundoff_upper)
    source_driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=np.ones(shape, np.float32),
        rainncv=np.ones(shape, np.float32),
        sr=np.full(shape, upper, np.float32)))

    path = restart.write_restart(tmp_path / "wsm6-upper.npz", source, cfg)
    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)

    assert fresh_driver._sr_roundoff_max_ulps == 7
    assert fresh_driver._wsm6_minor_loops == 1
    assert fresh_driver.microphysics.sr is fresh._scratch["mp_sr"]
    np.testing.assert_array_equal(
        fresh_driver.microphysics.sr.view(np.uint32),
        np.full(shape, upper, np.float32).view(np.uint32))


@pytest.mark.parametrize("mp_physics", [9, 10, 50],
                         ids=["milbrandt2", "morrison", "p3"])
def test_synthetic_state_roundtrips_bit_exactly(monkeypatch, tmp_path,
                                                mp_physics):
    cfg = _cfg(moist=True, mp_physics=mp_physics)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20260716)
    state.elapsed_seconds = 1234.5
    rng = np.random.default_rng(7)
    # EVERY canonical accumulator this scheme's driver arm binds, not
    # just the rain one: a live driver allocates the whole row in one
    # pass (woof/core/physics.py's micro_slots comprehension), so a
    # state carrying one and not the rest is a state no forecast
    # produces -- and for mp=9 the row is the nine-slot one whose hail
    # pair a WSM6-family assumption drops.
    from woof.core.physics_inventory import microphysics_scratch_slots
    seeded_slots = [slot for _, slot
                    in microphysics_scratch_slots(cfg.mp_physics)]
    for slot in seeded_slots:
        state.scratch((cfg.ny, cfg.nx), slot)[...] = \
            rng.standard_normal((cfg.ny, cfg.nx)).astype(np.float32)

    path = restart.write_restart(
        tmp_path / "rst.npz", state, cfg,
        run_trackers={"nan_free": True, "w_max_ms": 3.5})

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)                       # identical deterministic setup
    info = restart.restore_restart(path, fresh, cfg)

    assert info.elapsed_seconds == 1234.5
    assert fresh.elapsed_seconds == 1234.5
    assert info.run_trackers == {"nan_free": True, "w_max_ms": 3.5}
    for name in restart.STATE_SERIALIZED_ATTRS:
        source = getattr(state, name, None)
        if source is None:
            continue
        target = getattr(fresh, name)
        assert target.dtype == source.dtype, name
        # Byte-level identity: NaN payloads, signed zeros, denormals.
        assert target.tobytes() == source.tobytes(), name
    for slot in seeded_slots:
        assert (fresh._scratch[slot].tobytes()
                == state._scratch[slot].tobytes()), slot
    if mp_physics == 9:
        # Named explicitly for the reason the P3 block below gives: the
        # twelve Milbrandt-Yau moments are what a resumed mp=9 run
        # integrates, and a manifest edit that dropped one would leave
        # this equality trivially true.
        for name in restart.MILBRANDT2_RESTART_STATE:
            source = getattr(state, name)
            assert source.any(), f"{name} was never filled"
            assert getattr(fresh, name).tobytes() == source.tobytes(), name
    if mp_physics == 50:
        # Named explicitly so a later edit cannot restore the equality by
        # dropping P3's carriers from the serialized manifest: the rime
        # pair IS the ice inventory (rho_rime = qir/qib picks the lookup
        # table's density index), and th_old/qv_old are the cross-step
        # supersaturation carriers WRF restart-writes.
        for name in ("qir", "qib", "th_old", "qv_old"):
            source = getattr(state, name)
            assert source.any(), f"{name} was never filled"
            assert getattr(fresh, name).tobytes() == source.tobytes(), name

    header = restart.read_restart_header(path)
    assert header["format_version"] == restart.RESTART_FORMAT_VERSION
    assert header["elapsed_seconds"] == 1234.5
    assert header["config"]["nx"] == cfg.nx
    assert set(header["array_manifest"]) >= {"state/u", "state/h_diabatic",
                                             "scratch/mp_rainnc"}


@pytest.mark.parametrize("dropped", ["nh", "nc", "qh"])
def test_a_milbrandt2_write_missing_a_moment_is_refused(monkeypatch, tmp_path,
                                                       dropped):
    """A build that stopped allocating one moment may not checkpoint.

    The generic writer walks what the state HAS, so a state missing a
    moment writes a file that is internally consistent and that a
    resuming state missing the same moment accepts -- two equally wrong
    endpoints agreeing with each other.  The refusal names the field.
    """
    cfg = _cfg(moist=True, mp_physics=9)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=99)
    setattr(state, dropped, None)

    with pytest.raises(restart.RestartManifestError,
                       match=f"mp_physics=9 restart requires array "
                             f"'state/{dropped}'"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)


def test_a_milbrandt2_restart_missing_a_moment_is_refused(monkeypatch,
                                                          tmp_path):
    """And the file is refused on the way back in, by name."""
    cfg = _cfg(moist=True, mp_physics=9)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=101)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    tampered = _rewrite_restart_archive(
        path, tmp_path / "no-hail-number.npz",
        lambda payload, header: payload.pop("state/nh"))

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="omits canonical mp_physics=9 state"):
        restart.restore_restart(tampered, fresh, cfg)


def test_a_milbrandt2_write_missing_the_hail_accumulators_is_refused(
        monkeypatch, tmp_path):
    """The seven-slot (WSM6-family) assumption is refused, not accepted.

    A partial accumulator row is the shape the mistake takes: everything
    a rain/snow/graupel scheme has, and no hail.  All-absent is the
    pre-first-call state and stays writable, which this pins by writing
    that state successfully first.
    """
    cfg = _cfg(moist=True, mp_physics=9)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=17)
    restart.write_restart(tmp_path / "no-accumulators.npz", state, cfg)

    for slot in restart.MILBRANDT2_RESTART_PRECIPITATION_SLOTS:
        if slot in ("mp_hailnc", "mp_hailncv"):
            continue
        state.scratch((cfg.ny, cfg.nx), slot)[...] = np.float32(1.0)

    with pytest.raises(restart.RestartManifestError,
                       match="missing precipitation accumulators"):
        restart.write_restart(tmp_path / "seven-of-nine.npz", state, cfg)


def test_a_milbrandt2_restart_missing_the_hail_accumulators_is_refused(
        monkeypatch, tmp_path):
    """And a file with half a row is refused on the way back IN.

    Measured before this refusal existed: a checkpoint tampered to drop
    ``scratch/mp_hailnc`` restored successfully, because the generic
    reader answers a missing serialized slot by zero-initializing it with
    a note -- the right answer for a slot added after the file was
    written, and silently wrong here.  The resumed run carried storm-total
    hail of 0.0 beside a rain total of 3.0 and integrated on from there.
    ALL-absent stays readable: that is the pre-first-call state, and the
    write side accepts it for the same reason.
    """
    cfg = _cfg(moist=True, mp_physics=9)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=23)
    for slot in restart.MILBRANDT2_RESTART_PRECIPITATION_SLOTS:
        state.scratch((cfg.ny, cfg.nx), slot)[...] = np.float32(3.0)
    path = restart.write_restart(tmp_path / "all-nine.npz", state, cfg)

    tampered = _rewrite_restart_archive(
        path, tmp_path / "no-hail-total.npz",
        lambda payload, header: payload.pop("scratch/mp_hailnc"))

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="omits \\['scratch/mp_hailnc'\\]"):
        restart.restore_restart(tampered, fresh, cfg)


def test_a_milbrandt2_restart_with_no_accumulators_at_all_restores(
        monkeypatch, tmp_path):
    """The other half of the same rule, so the refusal is not a blanket.

    A checkpoint written before the scheme's first call carries none of
    the nine, and must still restore -- refusing it would turn a legal
    early checkpoint into a lost run.
    """
    cfg = _cfg(moist=True, mp_physics=9)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=29)
    path = restart.write_restart(tmp_path / "pre-first-call.npz", state, cfg)
    with np.load(path, allow_pickle=False) as stored:
        assert not [key for key in stored.files
                    if key[len("scratch/"):]
                    in restart.MILBRANDT2_RESTART_PRECIPITATION_SLOTS]

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)


def _grell_freitas_shim_state(cfg, monkeypatch):
    """A host-backed ``cu_physics=3`` state with its real GF adapter bound.

    ``write_restart`` refuses an active cumulus scheme with no attached
    driver (the identity block), and the GF identity resolves the REAL
    adapter class -- so the checkpoint these tests write is the one a
    Grell-Freitas run writes, not a stand-in with a different key set.
    """
    import woof.core.physics as physics

    from woof.core.gf import GrellFreitas

    state = _shim_state(cfg, monkeypatch)
    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    driver = physics.PhysicsDriver(
        state, cfg, fields={}, sfclay_result=None, noah_params=None,
        cumulus=GrellFreitas())
    state.physics = driver
    return state, driver


def test_the_advective_forcing_pair_roundtrips_and_only_where_it_exists(
        monkeypatch, tmp_path):
    """WRF RTHFTEN/RQVFTEN cross a resume, and cost nobody else a key.

    THE BREAKAGE.  The pair is written by RK stage 1 of step N and read by
    the cumulus call at the TOP of step N+1, so between a resume and its
    first cumulus call there is no producer -- exactly h_diabatic's
    lifecycle.  Classified REBUILT it would resume from a zeroed buffer
    and hand Grell-Freitas one step of hard zeros in the middle of a
    trajectory, silently and finitely.  So it is SERIALIZED, and this cell
    is the byte-level proof.

    The second half is the price: a cu_physics without an advective
    consumer must not gain two archive members, or every checkpoint on
    disk for every other configuration would stop resuming.
    """
    cfg = _cfg(moist=True, mp_physics=10, cu_physics=3, cudt_minutes=0.0)
    state, _ = _grell_freitas_shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20260820)
    state.elapsed_seconds = 600.0

    manifest = restart.state_manifest(state)
    assert {"state/rthften", "state/rqvften"} <= set(manifest)

    path = restart.write_restart(tmp_path / "gf.npz", state, cfg)
    fresh, _ = _grell_freitas_shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name in restart.ADVECTIVE_FORCING_STATE:
        source = getattr(state, name)
        assert source.any(), f"{name} was never filled"
        assert getattr(fresh, name).tobytes() == source.tobytes(), name

    # Every other cumulus selection keeps its exact archive inventory.
    for cu_physics in (0, 1):
        other_cfg = _cfg(moist=True, mp_physics=10, cu_physics=cu_physics,
                         cudt_minutes=0.0)
        other = _shim_state(other_cfg, monkeypatch)
        other_manifest = restart.state_manifest(other)
        assert not (set(other_manifest)
                    & {"state/rthften", "state/rqvften"}), cu_physics


def test_a_checkpoint_predating_the_advective_export_is_refused_by_name(
        monkeypatch, tmp_path):
    """The refusal names the change and the remedy, not two sorted lists.

    A Grell-Freitas checkpoint written before the dycore exported the pair
    is two state members short.  The key-set check already refused it, but
    with a diff of two long sorted lists the operator had to spot the
    difference by eye and then guess whether it was recoverable.  Refusals
    name the breakage AND the remedy.
    """
    cfg = _cfg(moist=True, mp_physics=10, cu_physics=3, cudt_minutes=0.0)
    state, _ = _grell_freitas_shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20260820)
    path = restart.write_restart(tmp_path / "gf.npz", state, cfg)

    def _drop_the_pair(payload, header):
        for name in restart.ADVECTIVE_FORCING_STATE:
            payload.pop(f"state/{name}")

    legacy = _rewrite_restart_archive(
        path, tmp_path / "legacy.npz", _drop_the_pair)

    fresh, _ = _grell_freitas_shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError) as excinfo:
        restart.restore_restart(legacy, fresh, cfg)
    message = str(excinfo.value)
    assert "predates the dycore's advective forcing export" in message
    assert "rthften" in message and "rqvften" in message
    assert "RTHFTEN/RQVFTEN" in message
    assert "Remedy:" in message


def test_restore_normalizes_spec_zone_ring_microphysics(monkeypatch,
                                                        tmp_path):
    """v5 migration pin: a checkpoint seeded with nonzero spec-zone ring
    MP accumulators/*NCV/SR/refl and ring h_diabatic (producible only by
    the pre-ring-exclusion whole-field microphysics; no WRF-valid
    trajectory contains them -- allocator zero init
    frame/module_domain.F:770-777 + clipped tiles solve_em.F:3631-3639)
    restores with those rings zeroed while every interior byte
    round-trips exactly.  The same seeding under a periodic config
    restores byte-identically: the normalization is
    specified/nested-scoped.  Without it the ring guard's
    capture/restore would carry the stale ring forever, and ring
    h_diabatic would feed every RK stage.  The boundary-forced arm uses
    the nested-child flag (where d02/d03 ring precipitation actually
    accumulated pre-fix; the normalization path is identical for
    specified, but restoring a specified root requires the full
    attach_lateral_boundaries preparation this shim harness omits)."""
    surface_slots = ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                     "mp_graupelnc", "mp_graupelncv", "mp_sr")
    for boundary_forced in (True, False):
        cfg = _cfg(nx=8, ny=7, moist=True, mp_physics=10,
                   nested=boundary_forced)
        state = _shim_state(cfg, monkeypatch)
        _fill_setup(state)
        _fill_serialized(state, seed=20260727)
        rng = np.random.default_rng(31)
        for slot in surface_slots:
            state.scratch((cfg.ny, cfg.nx), slot)[...] = 0.5 + np.abs(
                rng.standard_normal((cfg.ny, cfg.nx))).astype(np.float32)
        state.scratch((cfg.nz, cfg.ny, cfg.nx), "refl_10cm")[...] = (
            rng.standard_normal(
                (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32))
        hd = rng.standard_normal(
            (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32)
        state.h_diabatic[...] = hd
        path = restart.write_restart(
            tmp_path / f"ring-migration-{boundary_forced}.npz", state, cfg)

        fresh = _shim_state(cfg, monkeypatch)
        _fill_setup(fresh)
        restart.restore_restart(path, fresh, cfg)

        sz = cfg.spec_zone
        ring = np.ones((cfg.ny, cfg.nx), dtype=bool)
        ring[sz:cfg.ny - sz, sz:cfg.nx - sz] = False
        for slot in surface_slots:
            got = fresh._scratch[slot]
            src = state._scratch[slot]
            np.testing.assert_array_equal(got[~ring], src[~ring],
                                          err_msg=slot)
            if boundary_forced:
                assert (got[ring] == 0.0).all(), slot
            else:
                np.testing.assert_array_equal(got, src, err_msg=slot)
        # refl_10cm is rebuild-classified (REBUILT_SCRATCH_PREFIXES
        # "refl_"): checkpoints never serialize the stash, so it cannot
        # carry pre-fix ring bytes across a restart -- the restored state
        # has no refl_10cm slot at all.  h_diabatic IS serialized and is
        # the dangerous carrier (it feeds every RK stage).
        assert "refl_10cm" not in fresh._scratch
        got, src = fresh.h_diabatic, hd
        np.testing.assert_array_equal(got[:, ~ring], src[:, ~ring],
                                      err_msg="h_diabatic")
        if boundary_forced:
            assert (got[:, ring] == 0.0).all(), "h_diabatic"
        else:
            np.testing.assert_array_equal(got, src, err_msg="h_diabatic")


def _ring_seeded_state(monkeypatch, cfg, *, seed):
    """A state whose spec-zone ring carries nonzero MP accumulators.

    Pre-fix whole-field microphysics is one way to get there; a
    RELOCATION is the other, and the two are indistinguishable in the
    bytes -- which is the whole reason the restore has to be told which
    it is looking at.
    """
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=seed)
    rng = np.random.default_rng(seed)
    for slot in _RING_CARRIED_SLOTS:
        state.scratch((cfg.ny, cfg.nx), slot)[...] = 0.5 + np.abs(
            rng.standard_normal((cfg.ny, cfg.nx))).astype(np.float32)
    state.h_diabatic[...] = rng.standard_normal(
        (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32)
    return state


#: The accumulators a relocation shifts in index space AND the ring guard
#: excludes -- the exact intersection the exemption covers.  Spelled here
#: so the test states its own subject; the production rule reads it from
#: ``physics_continuation.continuation_slots()`` instead of a list.
_RING_CARRIED_SLOTS = ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                       "mp_graupelnc", "mp_graupelncv", "mp_sr")


def _ring_mask_2d(cfg):
    ring = np.ones((cfg.ny, cfg.nx), dtype=bool)
    sz = cfg.spec_zone
    ring[sz:cfg.ny - sz, sz:cfg.nx - sz] = False
    return ring


@pytest.mark.parametrize(
    ("moved_grid_ids", "ring_survives"),
    [([1], True), ([2, 3], False), (None, True)],
    ids=["this-grid-moved", "another-grid-moved", "no-list-recorded"])
def test_a_relocated_domain_keeps_the_ring_its_move_shifted_into_it(
        monkeypatch, tmp_path, moved_grid_ids, ring_survives):
    """A move puts REAL accumulation in the ring, and the migration must
    not eat it.

    ``shift_continuation`` moves the carried accumulators in INDEX space,
    so one eastward step makes the column that spent the run accumulating
    at interior ``i = 1`` into ring ``i = 0`` -- same ground, same rain
    total, and it becomes interior again on the next westward step.  The
    v5 ring migration was written when only pre-fix whole-field
    microphysics could leave a ring nonzero, and its idempotence claim
    fails on exactly this file: MEASURED on a 1 km nest 6 h into a 24 h
    forecast, a resume came back with its whole west column of RAINNC
    zeroed (up to 3.77 mm) and frozen there, the single variable of 77
    that was not byte-identical across the seam.

    The gate is per-grid, so a domain the run never moved keeps the
    migration unchanged, and a checkpoint that records the relocation
    without the per-grid list is read as "moved": its block's mere
    presence already proves a build newer than the ring exclusion, so it
    has nothing to migrate, while normalizing it anyway is the data loss.
    """
    cfg = _cfg(nx=8, ny=7, moist=True, mp_physics=10, nested=True)
    assert int(cfg.grid_id) == 1, "the parametrization names grid 1"
    state = _ring_seeded_state(monkeypatch, cfg, seed=20260825)
    written = restart.write_restart(tmp_path / "moved.npz", state, cfg)

    def stamp(payload, header):
        block = {"moves": 7, "segment_id": "0" * 64}
        if moved_grid_ids is not None:
            block["moved_grid_ids"] = moved_grid_ids
        header["relocation"] = block

    path = _rewrite_restart_archive(written, tmp_path / "stamped.npz", stamp)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)

    ring = _ring_mask_2d(cfg)
    for slot in _RING_CARRIED_SLOTS:
        got, src = fresh._scratch[slot], state._scratch[slot]
        np.testing.assert_array_equal(got[~ring], src[~ring], err_msg=slot)
        if ring_survives:
            np.testing.assert_array_equal(got[ring], src[ring], err_msg=slot)
            assert (got[ring] != 0.0).all(), (
                f"{slot}: the fixture must seed a NONZERO ring or this "
                "test passes vacuously")
        else:
            assert (got[ring] == 0.0).all(), slot
    # h_diabatic is NOT relocation-carried and its WRF ring value is an
    # unconditional zero (the ring guard re-pins it on every microphysics
    # call), so the migration keeps it in both arms.  Conflating the two
    # classes is what produced the defect.
    assert (fresh.h_diabatic[:, ring] == 0.0).all()


def test_grid_relocated_reads_the_checkpoint_header_per_grid():
    """The gate's truth table, stated once, away from any state."""
    cfg = _cfg()
    assert restart._grid_relocated({"grid_id": 3}, cfg) is False
    assert restart._grid_relocated(
        {"grid_id": 3, "relocation": None}, cfg) is False
    assert restart._grid_relocated(
        {"grid_id": 3, "relocation": {"moved_grid_ids": [2, 3]}}, cfg) is True
    assert restart._grid_relocated(
        {"grid_id": 1, "relocation": {"moved_grid_ids": [2, 3]}}, cfg) is False
    # No list recorded: read as moved (the block's presence already dates
    # the file past the ring exclusion).
    assert restart._grid_relocated(
        {"grid_id": 1, "relocation": {"moves": 4}}, cfg) is True
    # No grid_id in the header: the live config answers instead.
    assert restart._grid_relocated(
        {"relocation": {"moved_grid_ids": [int(cfg.grid_id)]}}, cfg) is True
    assert restart._grid_relocated(
        {"relocation": {"moved_grid_ids": [int(cfg.grid_id) + 1]}},
        cfg) is False


def _nssl2_restart_fixture(monkeypatch, *, seed: int = 1):
    cfg = _cfg(moist=True, mp_physics=18)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, name in enumerate((
            *restart.NSSL2_RESTART_PROGNOSTICS,
            *restart.NSSL2_RESTART_AUXILIARY_STATE), start=1):
        value = np.float32(seed * 0.01 + index * 1.0e-4)
        getattr(state, name)[...] = value
    for index, slot in enumerate(
            restart.NSSL2_RESTART_PRECIPITATION_SLOTS, start=1):
        state._scratch[slot][...] = np.float32(seed + index * 0.125)
    driver._pending_rainbl[...] = np.float32(seed * 0.25)
    driver.microphysics_updates = 0
    state.elapsed_seconds = 0.0
    return cfg, state, driver


def _advance_nssl2_restart_fixture(state, driver, cfg, steps: int) -> None:
    """Deterministic FP32 continuation seam for restart inventory tests."""
    for _ in range(steps):
        update = driver.microphysics_updates
        first_call = update == 0
        scale = np.float32(update + 1)
        for index, name in enumerate(
                restart.NSSL2_RESTART_PROGNOSTICS, start=1):
            increment = (
                np.float32(index) * np.float32(2.5e-6) * scale)
            getattr(state, name)[...] += increment
        if first_call:
            # Mirrors the one-time calcnfromq gate's dependence on persistent
            # first-call state: repeating this after restore must diverge.
            state.qnn[...] += np.float32(17.0)
        state.h_diabatic[...] = (
            state.qv * np.float32(0.125)
            + np.float32(update) * np.float32(1.0e-5))
        for index, slot in enumerate(
                restart.NSSL2_RESTART_PRECIPITATION_SLOTS, start=1):
            state._scratch[slot][...] += (
                np.float32(index) * np.float32(0.03125) * scale)
        driver._pending_rainbl[...] += \
            state._scratch["mp_rainncv"] * np.float32(0.01)
        driver.microphysics_updates += 1
        state.elapsed_seconds += float(cfg.dt)


def _nssl2_restart_bytes(state) -> dict[str, bytes]:
    manifest = {}
    manifest.update(restart.state_manifest(state))
    manifest.update(restart._scratch_manifest(state))
    manifest.update(restart._driver_manifest(state.physics))
    return {
        key: restart._host(value).tobytes()
        for key, value in sorted(manifest.items())
    }


def test_nssl2_restart_contract_pins_exact_canonical_inventory(
        monkeypatch, tmp_path):
    cfg, state, driver = _nssl2_restart_fixture(monkeypatch)
    _advance_nssl2_restart_fixture(state, driver, cfg, 2)
    path = restart.write_restart(tmp_path / "nssl2.npz", state, cfg)
    header = restart.read_restart_header(path)

    assert header["format_version"] == restart.RESTART_FORMAT_VERSION
    contract = header["physics_setup"]["microphysics"]["restart_contract"]
    assert contract == {
        "schema_version": restart.NSSL2_RESTART_CONTRACT_VERSION,
        "physics_contract_id": NSSL2_CONTRACT_ID,
        "wrf_reference": {
            "version": NSSL2_WRF_REFERENCE_VERSION,
            "commit": NSSL2_WRF_REFERENCE_COMMIT,
        },
        "resolved_mode": asdict(NSSL2_DEFAULT_MODE),
        "transported_fields": list(NSSL2_DEFAULT_MODE.transported_fields),
        "absent_fields": [],
        "resolved_wrf_namelist_defaults": dict(NSSL2_WRF_NAMELIST_DEFAULTS),
        "state_members": [
            *(f"state/{name}" for name in
              restart.NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}" for name in
              restart.NSSL2_RESTART_AUXILIARY_STATE),
        ],
        "precipitation_members": [
            f"scratch/{slot}"
            for slot in restart.NSSL2_RESTART_PRECIPITATION_SLOTS
        ],
        "first_call_authority": "driver.microphysics_updates == 0",
        "clock_authority": "header.elapsed_seconds",
        "continuation_policy": "bitwise",
    }
    manifest = set(header["array_manifest"])
    assert set(contract["state_members"]) <= manifest
    assert set(contract["precipitation_members"]) <= manifest
    assert not (manifest & restart.NSSL2_LEGACY_RESTART_ALIASES)
    assert header["driver"]["microphysics_updates"] == 2
    assert header["elapsed_seconds"] == 2 * cfg.dt


def test_nssl2_split_run_restart_continuation_is_bitwise(
        monkeypatch, tmp_path):
    cfg, straight, straight_driver = _nssl2_restart_fixture(
        monkeypatch, seed=7)
    _advance_nssl2_restart_fixture(straight, straight_driver, cfg, 8)

    _, split, split_driver = _nssl2_restart_fixture(monkeypatch, seed=7)
    _advance_nssl2_restart_fixture(split, split_driver, cfg, 3)
    boundary_bytes = _nssl2_restart_bytes(split)
    path = restart.write_restart(tmp_path / "split.npz", split, cfg)
    header = restart.read_restart_header(path)
    assert not any("nssl2_driver_" in name
                   or "nssl2_fused_" in name
                   or "nssl2_primary_ice_" in name
                   or "nssl2_nucond_" in name
                   for name in header["array_manifest"])

    _, resumed, resumed_driver = _nssl2_restart_fixture(
        monkeypatch, seed=99)
    rebuilt_binding = resumed_driver.nssl2_binding
    rebuilt_ids = tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    ))
    assert rebuilt_binding is not split_driver.nssl2_binding
    restart.restore_restart(path, resumed, cfg)
    assert resumed_driver.nssl2_binding is rebuilt_binding
    assert tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    )) == rebuilt_ids
    rebuilt_binding.validate(resumed, cfg.dt)
    assert _nssl2_restart_bytes(resumed) == boundary_bytes
    assert resumed_driver.microphysics_updates == 3
    assert resumed.elapsed_seconds == 3 * cfg.dt

    _advance_nssl2_restart_fixture(resumed, resumed_driver, cfg, 5)
    assert _nssl2_restart_bytes(resumed) == _nssl2_restart_bytes(straight)
    for name in (
            *restart.NSSL2_RESTART_PROGNOSTICS,
            *restart.NSSL2_RESTART_AUXILIARY_STATE):
        assert getattr(resumed, name).tobytes() == \
            getattr(straight, name).tobytes(), name
    for slot in restart.NSSL2_RESTART_PRECIPITATION_SLOTS:
        assert resumed._scratch[slot].tobytes() == \
            straight._scratch[slot].tobytes(), slot
    assert resumed_driver.microphysics_updates == \
        straight_driver.microphysics_updates == 8
    assert resumed.elapsed_seconds == straight.elapsed_seconds == 8 * cfg.dt


@pytest.mark.parametrize(("key", "corrupt"), [
    ("state/qnr", "shape"),
    ("state/qvolh", "dtype"),
    ("scratch/mp_hailncv", "shape"),
    ("scratch/mp_sr", "dtype"),
])
def test_nssl2_restore_rejects_shape_and_dtype_before_mutation(
        monkeypatch, tmp_path, key, corrupt):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    with np.load(path, allow_pickle=False) as data:
        original = data[key]
    replacement = (original.reshape(-1)[:-1]
                   if corrupt == "shape" else original.astype(np.float64))
    tampered = _replace_restart_member(
        path, tmp_path / f"{key.replace('/', '-')}-{corrupt}.npz",
        key, replacement)

    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=91)
    before = _nssl2_restart_bytes(live)
    with pytest.raises(restart.RestartMismatchError,
                       match=f"{key}.*(shape|dtype)"):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("edit_name", [
    "missing", "extra", "generic-alias", "fortran-alias", "driver-alias",
])
def test_nssl2_restore_rejects_missing_extra_and_legacy_names_atomically(
        monkeypatch, tmp_path, edit_name):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def edit(payload, header):
        if edit_name == "missing":
            payload.pop("state/qvolh")
        elif edit_name == "extra":
            payload["state/qnr_legacy"] = payload["state/qnr"].copy()
        elif edit_name == "generic-alias":
            payload["state/nc"] = payload.pop("state/qndrop")
        elif edit_name == "fortran-alias":
            payload["state/ccw"] = payload.pop("state/qndrop")
        else:
            payload["driver/microphysics/rainnc"] = \
                payload.pop("scratch/mp_rainnc")

    tampered = _rewrite_restart_archive(
        path, tmp_path / f"{edit_name}.npz", edit)
    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=92)
    before = _nssl2_restart_bytes(live)
    match = "legacy" if "alias" in edit_name else "inventory"
    with pytest.raises(restart.RestartMismatchError, match=match):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("edit_name", [
    "missing-contract", "wrong-contract-version", "extended-contract",
    "wrong-outer-version",
])
def test_nssl2_restore_rejects_unknown_contract_and_outer_versions(
        monkeypatch, tmp_path, edit_name):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def edit(payload, header):
        if edit_name == "wrong-outer-version":
            header["format_version"] = 4
            return
        microphysics = header["physics_setup"]["microphysics"]
        if edit_name == "missing-contract":
            microphysics.pop("restart_contract")
        elif edit_name == "wrong-contract-version":
            microphysics["restart_contract"]["schema_version"] = 0
        else:
            microphysics["restart_contract"]["legacy_permitted"] = True
        header["physics_setup_fingerprint"] = restart._json_sha256(
            header["physics_setup"])

    tampered = _rewrite_restart_archive(
        path, tmp_path / f"{edit_name}.npz", edit)
    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=93)
    before = _nssl2_restart_bytes(live)
    match = ("format version" if edit_name == "wrong-outer-version"
             else "MP18 restart contract")
    with pytest.raises(restart.RestartMismatchError, match=match):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("corrupt", [
    "missing-driver", "missing-slot", "wrong-dtype", "bad-counter",
])
def test_nssl2_write_refuses_incomplete_live_state(
        monkeypatch, tmp_path, corrupt):
    cfg, state, driver = _nssl2_restart_fixture(monkeypatch)
    if corrupt == "missing-driver":
        state.physics = None
    elif corrupt == "missing-slot":
        state._scratch.pop("mp_hailncv")
    elif corrupt == "wrong-dtype":
        state.qnh = state.qnh.astype(np.float64)
    else:
        driver.microphysics_updates = -1

    with pytest.raises(restart.RestartManifestError, match="MP18 restart"):
        restart.write_restart(tmp_path / f"{corrupt}.npz", state, cfg)


def test_restore_rejects_config_mismatch(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError, match="dt"):
        restart.restore_restart(path, fresh, replace(cfg, dt=30.0))
    with pytest.raises(restart.RestartMismatchError, match="mp_physics"):
        restart.restore_restart(path, fresh, replace(cfg, mp_physics=10))
    # Run-length knobs are the sanctioned difference between the writing
    # and the resuming run.
    restart.restore_restart(
        path, fresh, replace(cfg, run_seconds=999.0, output_interval_s=50.0,
                             restart_interval_s=100.0))


def test_restore_rejects_setup_fingerprint_mismatch(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh.thb[0] += np.float32(0.5)          # a different base state
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, fresh, cfg)


def test_thompson_restart_identity_binds_implementation_and_table_bytes(
        monkeypatch, tmp_path):
    """Experimental mp8 cannot cross a table or implementation boundary."""
    cfg = _cfg(moist=True, mp_physics=8)
    monkeypatch.setenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", "/fixture/thompson")
    generation = {"sha256": "a" * 64}

    def table_identity(path):
        assert path == "/fixture/thompson"
        return {
            "schema": 1,
            "table_set": "fixture-classic-thompson",
            "wrf_version": "v4.6.1",
            "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
            "assets": [{
                "filename": "fixture.dat", "bytes": 16,
                "sha256": generation["sha256"],
            }],
        }

    monkeypatch.setattr(restart, "_thompson_table_identity", table_identity)
    source, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    identity = restart.physics_setup_identity(source, cfg)
    assert identity["algorithms"]["microphysics"] == \
        restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8]
    thompson = identity["microphysics"]["thompson"]
    assert thompson["tables"]["assets"][0]["sha256"] == "a" * 64
    assert thompson["graupel_number_policy"].startswith("wrf-private")
    assert thompson["reflectivity_policy"].endswith("output-only-v1")
    assert "png-scw-held-number" in thompson["snow_rime_conversion_policy"]
    assert "vts-boost" in thompson["snow_fall_speed_policy"]

    path = restart.write_restart(tmp_path / "thompson.npz", source, cfg)
    same, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(same)
    restart.restore_restart(path, same, cfg)

    generation["sha256"] = "b" * 64
    changed, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(changed)
    before = changed.qg.tobytes()
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup"):
        restart.restore_restart(path, changed, cfg)
    assert changed.qg.tobytes() == before


def test_thompson_restart_identity_resolves_packaged_root_without_env(
        monkeypatch, tmp_path):
    """The identity binds the RESOLVED root: packaged default, env override.

    Until the mp8 promotion (product/v1 packaging lane 2026-07-28) this
    test asserted that identity construction refused to run without
    WOOF_EXPERIMENTAL_THOMPSON_MP8=1 and WOOF_THOMPSON_TABLE_ROOT.  The
    guard is retired and the table root defaults to the packaged
    packaged classic-table directory; what this test now pins is that
    the restart identity resolves exactly the root the forecast adapter
    loads, records the promoted admission token instead of the retired
    guard, and still honors the override.

    THE USER-LEVEL CACHE IS PINNED TO AN EMPTY DIRECTORY, and that is
    the whole reason this takes ``tmp_path``.  ``thompson_table_root``
    answers env override, then a COMPLETE packaged root, then a COMPLETE
    staged root under ``~/.woof/tables/thompson``.  A wheel install
    externalizes the packaged assets, so on any box where someone has
    run ``woof fetch-tables`` the third branch answers and this test
    read the staged path -- a red that says nothing about the product
    and everything about whose machine ran it.  Three lanes proved it
    identical on pristine tips.  Pointing the staged root at an empty
    directory makes the assertion about resolution ORDER, which is what
    it is for, on a box with a cache and on a box without one.
    """
    from woof import physics_compat
    from woof.physics_compat import packaged_thompson_table_root

    staged = tmp_path / "user-cache" / "thompson"
    staged.mkdir(parents=True)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: staged)

    cfg = _cfg(moist=True, mp_physics=8)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    seen = []

    def table_identity(path):
        seen.append(path)
        return {"schema": 1, "table_set": "fixture", "assets": []}

    monkeypatch.setattr(restart, "_thompson_table_identity", table_identity)

    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    identity = restart.physics_setup_identity(state, cfg)
    thompson = identity["microphysics"]["thompson"]
    assert seen == [str(packaged_thompson_table_root())]
    assert thompson["admission"] == "first-class-mp8-packaged-tables-v1"
    assert "implementation_guard" not in thompson

    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", "/override/thompson")
    restart.physics_setup_identity(state, cfg)
    assert seen[-1] == "/override/thompson"


def test_thompson_identity_reads_a_complete_staged_root_over_a_short_one(
        monkeypatch, tmp_path):
    """The OTHER direction of the isolation above, pinned once.

    The test before this one asserts the packaged root answers when the
    staged root cannot.  That assertion is only worth something if the
    staged root CAN answer when it is complete -- otherwise pinning an
    empty directory would pass for the wrong reason on every box, and
    the resolution order would be untested in the direction users
    actually hit after ``woof fetch-tables``.

    Both roots are pinned here rather than only the staged one, so the
    branch under test is the same on a git clone (complete packaged root)
    and on a wheel install (externalized, short).
    """
    from woof import physics_compat
    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS

    packaged = tmp_path / "packaged-short" / "tables"
    packaged.mkdir(parents=True)
    staged = tmp_path / "user-cache" / "thompson"
    staged.mkdir(parents=True)
    for asset in CLASSIC_TABLE_ASSETS:
        (staged / asset.filename).write_bytes(b"")
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root",
                        lambda: packaged)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: staged)

    cfg = _cfg(moist=True, mp_physics=8)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    seen = []
    monkeypatch.setattr(
        restart, "_thompson_table_identity",
        lambda path: seen.append(path) or {
            "schema": 1, "table_set": "fixture", "assets": []})
    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)

    restart.physics_setup_identity(state, cfg)
    assert seen == [str(staged)]


def test_restart_header_binds_resolved_physics_and_active_assets(
        monkeypatch, tmp_path):
    """The identity is an auditable resolved setup, not only scheme IDs.

    In particular, ra_physics=4 must say which radiation algorithm and
    above-atmosphere policy produced the held heating/flux state, and must
    bind every packaged coefficient/climatology file it actually consumes.
    """
    cfg = _cfg(
        moist=True, mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    state, driver = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"test-sha256:{Path(path).name}")

    path = restart.write_restart(tmp_path / "physics-identity.npz", state,
                                 cfg)
    header = restart.read_restart_header(path)
    identity = header["physics_setup"]

    assert header["format_version"] == restart.RESTART_FORMAT_VERSION
    assert header["physics_setup_fingerprint"] == \
        restart.physics_setup_fingerprint(state, cfg)
    assert identity["schema_version"] == restart.PHYSICS_SETUP_SCHEMA_VERSION
    assert identity["algorithms"]["radiation"] == \
        restart.RADIATION_ALGORITHM_IDENTITIES[4]
    radiation = identity["radiation"]
    assert radiation["algorithm"] == restart.RADIATION_ALGORITHM_IDENTITIES[4]
    assert radiation["above_atmosphere_policy"] == \
        restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES[4]
    assert radiation["start_time"] == "1974-04-03T12:00:00"
    assert radiation["trace_vmr"] == {"co2": 3.30e-4, "n2o": 3.0e-7}
    assert radiation["trace_gas_overrides"] == {"co2": 3.30e-4}
    assert radiation["latitude"]["shape"] == [cfg.ny, cfg.nx]
    assert radiation["ozone_vmr"]["dtype"] == "float32"

    assert set(identity["assets"]) == {
        "rrtmgp_gas_lw", "rrtmgp_gas_sw",
        "rrtmgp_cloud_lw", "rrtmgp_cloud_sw", "rrtmgp_rfmip",
        "noah_vegparm", "noah_soilparm", "noah_genparm", "noah_landuse",
        "kf_lutab",
    }
    for asset in identity["assets"].values():
        assert asset["path"].startswith("data/")
        assert asset["sha256"] == \
            f"test-sha256:{Path(asset['path']).name}"
    assert identity["land_surface"]["parameters"]["arrays"]["veg"][
        "shape"] == [2, 15]
    assert identity["microphysics"]["morrison_rimed_ice"] == {
        "selection": 1, "ag": 114.5, "bg": 0.5,
        "rhog": 900.0, "cg": pytest.approx(150.0 * np.pi),
    }

    fresh, _ = _identity_bound_physics_state(cfg, monkeypatch)
    restart.restore_restart(path, fresh, cfg)

    # A diagnostic call counter does not define setup and must not make a
    # freshly constructed adapter incompatible with its own restart.
    before = restart.physics_setup_fingerprint(state, cfg)
    driver.radiation_callable.update_count = 123
    assert restart.physics_setup_fingerprint(state, cfg) == before


def test_stock_analytic_radiation_identity_roundtrips_without_gpu(
        monkeypatch, tmp_path):
    from woof.core import analytic_radiation

    cfg = _cfg(moist=True, ra_physics=90)
    monkeypatch.setattr(analytic_radiation, "cp", _NumpyCupyShim)

    def prepared(latitude_offset=0.0):
        state, driver = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(state)
        shape = state.mup.shape
        driver.radiation_callable = \
            analytic_radiation.AnalyticClearSkyRadiation(
                datetime(1974, 4, 3, 12),
                np.full(shape, 39.0 + latitude_offset, dtype=np.float32),
                np.full(shape, -87.0, dtype=np.float32))
        return state

    source = prepared()
    path = restart.write_restart(tmp_path / "analytic.npz", source, cfg)
    header = restart.read_restart_header(path)
    radiation = header["physics_setup"]["radiation"]
    assert radiation["callable"] == {
        "class": "woof.core.analytic_radiation.AnalyticClearSkyRadiation",
        "implementation": "stock",
    }
    assert radiation["above_atmosphere_policy"] == \
        restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES[90]
    assert radiation["constants"]["clear_sky_transmissivity"] == 0.75
    assert header["physics_setup"]["assets"] == {}

    restart.restore_restart(path, prepared(), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup"):
        restart.restore_restart(path, prepared(latitude_offset=0.5), cfg)


def test_active_surface_identity_binds_packaged_landuse_coefficients(
        monkeypatch):
    cfg = _cfg(
        moist=True, sf_sfclay_physics=1, sf_surface_physics=2,
        bl_pbl_physics=1,
    )
    state, _ = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"test-sha256:{Path(path).name}")

    landuse = restart.physics_setup_identity(state, cfg)["assets"][
        "noah_landuse"]

    assert landuse == {
        "path": "data/noah_tables/LANDUSE.TBL",
        "bytes": 41236,
        "sha256": "test-sha256:LANDUSE.TBL",
    }


def test_stock_rrtmgp_and_kf_identities_bind_live_resolved_objects(
        monkeypatch):
    from woof.core import kf

    cfg = _cfg(
        moist=True, mp_physics=10, sf_sfclay_physics=1,
        sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    state, driver = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(restart, "_asset_sha256", lambda path: "disk-bytes")
    table = kf.KFTable(
        temperature=np.array([[1.0, 2.0]], dtype=np.float32),
        qsat=np.array([[3.0, 4.0]], dtype=np.float32),
        thetae_base=np.array([5.0], dtype=np.float32),
        log_ratio=np.array([6.0], dtype=np.float32),
        pressure_top=10000.0, pressure_reciprocal=1.0,
        thetae_reciprocal=2.0,
    )
    monkeypatch.setattr(kf, "load_kf_table", lambda: table)

    baseline = restart.physics_setup_fingerprint(state, cfg)
    driver.radiation_callable.lw_tables.coefficient[0] += 0.25
    changed_rrtmgp = restart.physics_setup_fingerprint(state, cfg)
    assert changed_rrtmgp != baseline
    driver.radiation_callable.lw_tables.coefficient[0] -= 0.25

    driver.radiation_callable.chunk_workspace = SimpleNamespace(
        nz=cfg.nz, column_chunk=17, p_top=10000.0, nbytes=4096,
        _phase_layouts={"lw": {"tau": ((17, cfg.nz), 4)}},
    )
    changed_workspace = restart.physics_setup_fingerprint(state, cfg)
    assert changed_workspace != baseline
    driver.radiation_callable.chunk_workspace.p_top = 10001.0
    changed_workspace_top = restart.physics_setup_fingerprint(state, cfg)
    assert changed_workspace_top != changed_workspace
    driver.radiation_callable.chunk_workspace = None

    table.temperature[0, 0] += np.float32(0.5)
    changed_kf = restart.physics_setup_fingerprint(state, cfg)
    assert changed_kf != baseline


def test_declared_custom_physics_identity_does_not_require_stock_internals(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True, ra_physics=4, cu_physics=1)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    driver.radiation_callable = _DeclaredPhysicsCallable("custom-radiation")
    driver.cumulus_callable = _DeclaredPhysicsCallable("custom-cumulus")
    driver.cumulus_callable.w0avg = None

    identity = restart.physics_setup_identity(state, cfg)

    assert identity["radiation"]["callable"]["declared_identity"][
        "algorithm"] == "custom-radiation"
    assert identity["cumulus"]["callable"]["declared_identity"][
        "algorithm"] == "custom-cumulus"
    assert not set(identity["assets"]) & {
        "rrtmgp_gas_lw", "rrtmgp_gas_sw", "rrtmgp_cloud_lw",
        "rrtmgp_cloud_sw", "rrtmgp_rfmip", "kf_lutab",
    }

    path = restart.write_restart(tmp_path / "custom-physics.npz", state, cfg)
    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh_driver.radiation_callable = \
        _DeclaredPhysicsCallable("custom-radiation")
    fresh_driver.cumulus_callable = _DeclaredPhysicsCallable("custom-cumulus")
    fresh_driver.cumulus_callable.w0avg = None
    restart.restore_restart(path, fresh, cfg)


@pytest.mark.parametrize(
    "mismatch",
    ["radiation-geometry", "resolved-trace-gas", "noah-parameters",
     "asset-bytes", "radiation-algorithm", "above-atmosphere-policy",
     "land-surface-algorithm", "resolved-driver-switch"],
)
def test_restore_rejects_changed_physics_setup_before_mutation(
        monkeypatch, tmp_path, mismatch):
    """Every trajectory-changing resolved setup input is a refusal gate."""
    cfg = _cfg(
        moist=True, mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    asset_generation = {"value": "writer"}
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"{asset_generation['value']}:{Path(path).name}")
    source, _ = _identity_bound_physics_state(cfg, monkeypatch)
    path = restart.write_restart(tmp_path / f"{mismatch}.npz", source, cfg)

    kwargs = {}
    if mismatch == "radiation-geometry":
        kwargs["latitude_offset"] = 0.25
    elif mismatch == "resolved-trace-gas":
        kwargs["trace_co2"] = 3.31e-4
    elif mismatch == "noah-parameters":
        kwargs["noah_offset"] = 0.5
    live, live_driver = _identity_bound_physics_state(
        cfg, monkeypatch, **kwargs)
    _fill_serialized(live, seed=20260718)
    live.elapsed_seconds = 47.0
    before = {
        name: getattr(live, name).tobytes()
        for name in restart.STATE_SERIALIZED_ATTRS
        if getattr(live, name, None) is not None}

    if mismatch == "asset-bytes":
        asset_generation["value"] = "reader"
    elif mismatch == "radiation-algorithm":
        monkeypatch.setitem(
            restart.RADIATION_ALGORITHM_IDENTITIES, 4,
            "rte-rrtmgp-deliberately-different-test")
    elif mismatch == "above-atmosphere-policy":
        monkeypatch.setitem(
            restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES, 4,
            "explicit-upper-atmosphere-test-policy")
    elif mismatch == "land-surface-algorithm":
        monkeypatch.setitem(
            restart.LAND_SURFACE_ALGORITHM_IDENTITIES, 2,
            "pre-chs2-and-source-water-lake-skin-test-identity")
    elif mismatch == "resolved-driver-switch":
        live_driver.surface_enabled = False

    # A moved algorithm identity is a change of build, refused before the
    # configuration walk with the scheme named; the rest are the physics
    # gate's resolved-setup refusals.
    expected = {
        "radiation-algorithm": (
            "scheme implementations this build does not run: "
            "ra_lw_physics=4 \\(radiation lw\\): rte-rrtmgp-v1 -> "
            "rte-rrtmgp-deliberately-different-test"),
        "land-surface-algorithm": (
            "scheme implementations this build does not run: "
            "sf_surface_physics=2 \\(land surface\\): .* -> "
            "pre-chs2-and-source-water-lake-skin-test-identity"),
    }.get(mismatch, "physics setup")
    with pytest.raises(restart.RestartMismatchError, match=expected):
        restart.restore_restart(path, live, cfg)

    assert live.elapsed_seconds == 47.0
    assert before == {
        name: getattr(live, name).tobytes() for name in before}


def test_restart_rejects_missing_or_self_inconsistent_physics_identity(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "source.npz", state, cfg)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))

    missing = dict(header)
    missing.pop("physics_setup_fingerprint")
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(missing).encode("utf-8"), dtype=np.uint8)
    missing_path = tmp_path / "missing-identity.npz"
    with missing_path.open("wb") as stream:
        np.savez(stream, **payload)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics_setup_fingerprint"):
        restart.restore_restart(missing_path, fresh, cfg)

    header["physics_setup"]["algorithms"]["microphysics"] = \
        "silently-tampered"
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    inconsistent = tmp_path / "inconsistent-identity.npz"
    with inconsistent.open("wb") as stream:
        np.savez(stream, **payload)
    other = _shim_state(cfg, monkeypatch)
    _fill_setup(other)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup fingerprint"):
        restart.restore_restart(inconsistent, other, cfg)


def test_restore_rejects_live_w0avg_when_archive_has_no_history(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    source_driver.cumulus_callable = SimpleNamespace(w0avg=None)
    path = restart.write_restart(tmp_path / "no-w0avg.npz", source, cfg)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    live_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.full(live.p.shape, 17.0, dtype=np.float32))
    before = live.u.tobytes()
    with pytest.raises(restart.RestartMismatchError, match="W0AVG"):
        restart.restore_restart(path, live, cfg)
    assert live.u.tobytes() == before
    np.testing.assert_array_equal(live_driver.cumulus_callable.w0avg, 17.0)


@pytest.mark.parametrize(
    ("key", "corrupt"), [
        ("driver/pbl_tendencies/ru", "shape"),
        ("driver/pbl_tendencies/ru", "dtype"),
        ("cumulus/w0avg", "shape"),
        ("cumulus/w0avg", "dtype"),
    ],
    ids=["held-shape", "held-dtype", "w0avg-shape", "w0avg-dtype"],
)
def test_restore_prevalidates_held_and_w0avg_arrays_atomically(
        monkeypatch, tmp_path, key, corrupt):
    """All driver payload contracts are checked before one live byte moves."""
    cfg = _cfg(moist=True)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=1)
    source_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.zeros(source.mup.shape, dtype=np.float32))
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    with np.load(path, allow_pickle=False) as data:
        original = data[key]
    replacement = (original.reshape(-1)[:-1]
                   if corrupt == "shape"
                   else original.astype(np.float64))
    tampered = _replace_restart_member(
        path, tmp_path / f"{key.replace('/', '-')}-{corrupt}.npz",
        key, replacement)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    _fill_serialized(live, seed=2)
    live_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.full(live.mup.shape, 7.0, dtype=np.float32))
    before_state = {
        name: getattr(live, name).tobytes()
        for name in restart.STATE_SERIALIZED_ATTRS
        if getattr(live, name, None) is not None}
    before_driver = {
        f"{tend}/{component}": getattr(getattr(live_driver, tend), component).tobytes()
        for tend in restart.DRIVER_TENDENCY_ATTRS
        for component in restart.TENDENCY_REQUIRED_COMPONENTS}
    before_w0avg = live_driver.cumulus_callable.w0avg.tobytes()

    with pytest.raises(restart.RestartMismatchError,
                       match=f"{key}.*(shape|dtype)"):
        restart.restore_restart(tampered, live, cfg)

    assert before_state == {
        name: getattr(live, name).tobytes() for name in before_state}
    assert before_driver == {
        f"{tend}/{component}": getattr(getattr(live_driver, tend), component).tobytes()
        for tend in restart.DRIVER_TENDENCY_ATTRS
        for component in restart.TENDENCY_REQUIRED_COMPONENTS}
    assert live_driver.cumulus_callable.w0avg.tobytes() == before_w0avg


def test_restore_rejects_tampered_member_set(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    removed = next(key for key in payload if key.startswith("state/"))
    payload.pop(removed)
    tampered = tmp_path / "tampered.npz"
    with tampered.open("wb") as stream:
        np.savez(stream, **payload)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError, match="manifest"):
        restart.restore_restart(tampered, fresh, cfg)


def test_lbc_forcing_tables_are_pinned_by_the_fingerprint(monkeypatch,
                                                          tmp_path):
    """Review F3: a same-config resume against modified boundary forcing
    (a replaced/edited reference bundle) must be rejected, the setup
    fingerprint digests every interval's side tables byte-level."""
    from woof.ingest.lateral_bc import build_lateral_boundaries

    cfg = _cfg(moist=True, mp_physics=1)

    def boundaries(seed):
        rng = np.random.default_rng(seed)
        snapshots = [{"u": rng.standard_normal((8, 8))} for _ in range(2)]
        return build_lateral_boundaries(
            snapshots, [0.0, 3600.0], spec_bdy_width=3, spec_zone=1,
            relax_zone=2)

    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.lateral_boundaries = boundaries(1)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    # An identically rebuilt forcing set restores.
    same = _shim_state(cfg, monkeypatch)
    _fill_setup(same)
    same.lateral_boundaries = boundaries(1)
    restart.restore_restart(path, same, cfg)

    # Different forcing tables under the same config are refused.
    other = _shim_state(cfg, monkeypatch)
    _fill_setup(other)
    other.lateral_boundaries = boundaries(2)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, other, cfg)

    # Forcing presence itself is fingerprinted: attached-vs-None differs.
    none_state = _shim_state(cfg, monkeypatch)
    _fill_setup(none_state)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, none_state, cfg)


def _historical_setup_fingerprint(state, *, inverse=True, lateral=True) -> str:
    """The pre-extension digest algorithm, copied here as a freeze oracle."""
    digest = hashlib.sha256()

    def array(name, value):
        host = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(host.shape).encode())
        digest.update(str(host.dtype).encode())
        digest.update(host.tobytes())

    for name in restart.STATE_SETUP_ARRAYS:
        if name != "msfv_inv" or inverse:
            array(name, getattr(state, name))
    for name in restart.STATE_SETUP_SCALARS:
        value = getattr(state, name)
        if value is not None and not isinstance(value, bool):
            value = float(value)
        digest.update(f"{name}={value!r};".encode())
    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        assert nest_class == "REBUILT"
        digest.update(b"nest_tables=REBUILT;")
    elif lateral:
        boundaries = getattr(state, "lateral_boundaries", None)
        if boundaries is None:
            digest.update(b"lateral_boundaries=None;")
        else:
            digest.update(
                f"lbc:width={boundaries.spec_bdy_width};"
                f"spec={boundaries.spec_zone};"
                f"relax={boundaries.relax_zone};"
                f"intervals={len(boundaries.intervals)};".encode())
            for interval in boundaries.intervals:
                digest.update(
                    f"[{interval.start_seconds!r},"
                    f"{interval.end_seconds!r}]".encode())
                for name in sorted(interval.fields):
                    boundary = interval.fields[name]
                    for side_name in ("west", "east", "south", "north"):
                        side = getattr(boundary, side_name)
                        array(f"{name}/{side_name}/value", side.value)
                        array(f"{name}/{side_name}/tendency", side.tendency)
    return digest.hexdigest()


def _extension_boundaries(count: int, *, seed=11, seconds=None,
                          spec_bdy_width=3, spec_zone=1, relax_zone=2,
                          extra_field=False):
    from woof.ingest.lateral_bc import build_lateral_boundaries

    rng = np.random.default_rng(seed)
    snapshots = []
    for _ in range(count):
        row = {"u": rng.standard_normal((8, 8))}
        if extra_field:
            row["v"] = rng.standard_normal((8, 8))
        snapshots.append(row)
    if seconds is None:
        seconds = [float(index * 3600) for index in range(count)]
    return build_lateral_boundaries(
        snapshots, seconds, spec_bdy_width=spec_bdy_width,
        spec_zone=spec_zone, relax_zone=relax_zone)


def _extension_state(cfg, monkeypatch, *, count: int, seed=11,
                     seconds=None, spec_bdy_width=3, spec_zone=1,
                     relax_zone=2, extra_field=False):
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.lateral_boundaries = _extension_boundaries(
        count, seed=seed, seconds=seconds,
        spec_bdy_width=spec_bdy_width, spec_zone=spec_zone,
        relax_zone=relax_zone, extra_field=extra_field)
    return state


def test_setup_fingerprint_refactor_preserves_legacy_bytes(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=1)
    root = _extension_state(cfg, monkeypatch, count=3)
    assert restart.setup_fingerprint(root) == \
        _historical_setup_fingerprint(root)

    child = _shim_state(replace(cfg, grid_id=2, nested=True), monkeypatch)
    _fill_setup(child)
    child._nest_restart_classification = "REBUILT"
    assert restart.setup_fingerprint(child) == \
        _historical_setup_fingerprint(child)


def test_exact_restart_still_refuses_forcing_extension(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(tmp_path / "exact.npz", source, cfg)
    live = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, live, cfg)


def test_sealed_extension_requires_writer_opt_in_and_a_suffix(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    ordinary = restart.write_restart(tmp_path / "ordinary.npz", source, cfg)
    extended = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="not intentionally"):
        restart._validate_restart(
            ordinary, extended, cfg, sealed_forcing_extension=True)

    sealed = restart.write_restart(
        tmp_path / "sealed.npz", source, cfg,
        sealed_forcing_extension=True)
    same = _extension_state(cfg, monkeypatch, count=2)
    with pytest.raises(restart.RestartMismatchError,
                       match="append at least one"):
        restart._validate_restart(
            sealed, same, cfg, sealed_forcing_extension=True)


@pytest.mark.parametrize("mutation", [
    "old-bytes", "bounds", "controls", "fields", "gap",
])
def test_sealed_extension_rejects_any_non_append_change(
        monkeypatch, tmp_path, mutation):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / f"sealed-{mutation}.npz", source, cfg,
        sealed_forcing_extension=True)
    kwargs = {}
    if mutation == "bounds":
        kwargs["seconds"] = [0.0, 3590.0, 7200.0]
    elif mutation == "controls":
        kwargs["spec_bdy_width"] = 4
        kwargs["relax_zone"] = 3
    elif mutation == "fields":
        kwargs["extra_field"] = True
    elif mutation == "old-bytes":
        kwargs["seed"] = 12
    live = _extension_state(cfg, monkeypatch, count=3, **kwargs)
    if mutation == "gap":
        original = restart.lateral_boundary_prefix_identity

        def identity(state):
            result = original(state)
            if state is live:
                result = json.loads(json.dumps(result))
                result["intervals"][1]["start_seconds"] = 3700.0
            return result

        monkeypatch.setattr(restart, "lateral_boundary_prefix_identity", identity)
    with pytest.raises(restart.RestartMismatchError,
                       match="forcing|boundary"):
        restart._validate_restart(
            path, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_rejects_changed_suffix_seam(
        monkeypatch, tmp_path):
    from woof.ingest.lateral_bc import (
        BoundaryInterval, LateralBoundaries, SideBoundary,
        record_built_end_frame,
    )

    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "sealed-seam.npz", source, cfg,
        sealed_forcing_extension=True)
    live = _extension_state(cfg, monkeypatch, count=3)
    intervals = list(live.lateral_boundaries.intervals)
    suffix = intervals[1]
    fields = dict(suffix.fields)
    boundary = fields["u"]
    changed = np.array(boundary.west.value, copy=True)
    changed.flat[0] += 123.0
    fields["u"] = replace(
        boundary,
        west=SideBoundary(changed, boundary.west.tendency))
    # The suffix keeps the end frame its builder recorded and starts from
    # a changed frame: a hand-built interval records none, and a live
    # series that does not record every end frame is refused against a
    # checkpoint that does before its seam is read (A140b).
    intervals[1] = record_built_end_frame(BoundaryInterval(
        suffix.start_seconds, suffix.end_seconds, fields),
        suffix.end_frame_sha256)
    live.lateral_boundaries = LateralBoundaries(
        tuple(intervals), live.lateral_boundaries.spec_bdy_width,
        live.lateral_boundaries.spec_zone,
        live.lateral_boundaries.relax_zone)

    with pytest.raises(restart.RestartMismatchError,
                       match="discontinuous live forcing frame"):
        restart._validate_restart(
            path, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_binds_checkpoint_endpoint_setup_config_and_physics(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3500.0
    endpoint = tmp_path / "endpoint.npz"
    with pytest.raises(restart.RestartMismatchError, match="not sealed"):
        restart.write_restart(
            endpoint, source, cfg, sealed_forcing_extension=True)
    assert not endpoint.exists()

    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "identity.npz", source, cfg,
        sealed_forcing_extension=True)
    live = _extension_state(cfg, monkeypatch, count=3)
    changed_setup = _extension_state(cfg, monkeypatch, count=3)
    changed_setup.thb.flat[0] += np.float32(0.5)
    with pytest.raises(restart.RestartMismatchError, match="immutable setup"):
        restart._validate_restart(
            path, changed_setup, cfg, sealed_forcing_extension=True)
    with pytest.raises(restart.RestartMismatchError, match="dt"):
        restart._validate_restart(
            path, live, replace(cfg, dt=cfg.dt * 2),
            sealed_forcing_extension=True)

    tampered = _rewrite_header(
        path, tmp_path / "physics.npz",
        lambda header: header.__setitem__(
            "physics_setup_fingerprint", "0" * 64))
    with pytest.raises(restart.RestartMismatchError, match="physics setup"):
        restart._validate_restart(
            tampered, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_header_tamper_and_rebuilt_child_contract(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "root.npz", source, cfg,
        sealed_forcing_extension=True)

    def alter(header):
        header["lateral_boundary_prefix"]["intervals"][0]["sha256"] = \
            "0" * 64

    tampered = _rewrite_header(path, tmp_path / "tampered.npz", alter)
    live = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="sealed interval"):
        restart._validate_restart(
            tampered, live, cfg, sealed_forcing_extension=True)

    changed_prefix = _extension_state(cfg, monkeypatch, count=3, seed=12)
    forged_setup = _rewrite_header(
        path, tmp_path / "forged-setup.npz",
        lambda header: header.__setitem__(
            "setup_fingerprint",
            restart.setup_fingerprint(changed_prefix)))
    with pytest.raises(restart.RestartMismatchError, match="sealed interval"):
        restart._validate_restart(
            forged_setup, changed_prefix, cfg,
            sealed_forcing_extension=True)

    extra_key = _rewrite_header(
        path, tmp_path / "extra-prefix-key.npz",
        lambda header: header["lateral_boundary_prefix"].__setitem__(
            "unrecognized", True))
    with pytest.raises(restart.RestartMismatchError,
                       match="malformed sealed forcing document"):
        restart._validate_restart(
            extra_key, live, cfg, sealed_forcing_extension=True)

    missing_root_prefix = _rewrite_header(
        path, tmp_path / "missing-root-prefix.npz",
        lambda header: header.__setitem__("lateral_boundary_prefix", None))
    before = live.u.copy()
    before_elapsed = live.elapsed_seconds
    with pytest.raises(restart.RestartMismatchError,
                       match="specified root has no sealed"):
        restart._validate_restart(
            missing_root_prefix, live, cfg,
            sealed_forcing_extension=True)
    assert np.array_equal(live.u, before, equal_nan=True)
    assert live.elapsed_seconds == before_elapsed

    child_cfg = replace(cfg, grid_id=2, specified=False, nested=True)
    child = _shim_state(child_cfg, monkeypatch)
    _fill_setup(child)
    child._nest_restart_classification = "REBUILT"
    child_path = restart.write_restart(
        tmp_path / "child.npz", child, child_cfg,
        sealed_forcing_extension=True)
    fresh = _shim_state(child_cfg, monkeypatch)
    _fill_setup(fresh)
    fresh._nest_restart_classification = "REBUILT"
    restart._validate_restart(
        child_path, fresh, child_cfg, sealed_forcing_extension=True)
    fresh.thb.flat[0] += np.float32(1.0)
    with pytest.raises(restart.RestartMismatchError, match="child/nest"):
        restart._validate_restart(
            child_path, fresh, child_cfg, sealed_forcing_extension=True)

    unexpected_child_prefix = _rewrite_header(
        child_path, tmp_path / "child-prefix.npz",
        lambda header: header.__setitem__(
            "lateral_boundary_prefix",
            restart.lateral_boundary_prefix_identity(source)))
    with pytest.raises(restart.RestartMismatchError,
                       match="nested child unexpectedly"):
        restart._validate_restart(
            unexpected_child_prefix, child, child_cfg,
            sealed_forcing_extension=True)


# Only the explicitly added inverse binding is unwound for old byte pins.
# New fields still enter the historical algorithm and therefore fail its pins.
_V6_SETUP_HASHES = {}


def _remember_v6_setup(state):
    _V6_SETUP_HASHES[restart.setup_fingerprint(state)] = _historical_setup_fingerprint(state, inverse=False)
    _V6_SETUP_HASHES[restart.setup_core_fingerprint(state)] = _historical_setup_fingerprint(state, inverse=False, lateral=False)


def _unwind_v7_inverse_binding(header):
    for name in ("setup_fingerprint", "setup_core_fingerprint"):
        if name in header:
            header[name] = _V6_SETUP_HASHES.get(header[name], header[name])


def _sealed_tree_fixture(monkeypatch, *, forcing_count: int,
                         run_seconds: float, payload_seed: int,
                         nwp_diagnostics: int = 0, run_overrides=None):
    from woof.core.model import ModelRuntimeStatus

    overrides = dict(run_overrides or {})
    nwp_diagnostics = overrides.pop("nwp_diagnostics", nwp_diagnostics)
    root_cfg = _cfg(
        grid_id=1, moist=True, mp_physics=1,
        run_seconds=run_seconds, restart_interval_s=3600.0,
        nwp_diagnostics=nwp_diagnostics,
        specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2,
        **overrides)
    child_cfg = _cfg(
        grid_id=2, nested=True, moist=True, mp_physics=1,
        nwp_diagnostics=nwp_diagnostics,
        run_seconds=run_seconds, restart_interval_s=3600.0, **overrides)
    root_state = _extension_state(
        root_cfg, monkeypatch, count=forcing_count)
    child_state = _shim_state(child_cfg, monkeypatch)
    _fill_setup(child_state)
    child_state._nest_restart_classification = "REBUILT"
    _fill_serialized(root_state, payload_seed)
    _fill_serialized(child_state, payload_seed + 1)
    _remember_v6_setup(root_state)
    _remember_v6_setup(child_state)

    start = datetime(2026, 7, 30)

    def clock(grid_id):
        return SimpleNamespace(
            ticks=3600, tick_den=1, step_count=3600,
            dtbc_fp32=np.float32(0.0),
            spec=SimpleNamespace(
                grid_id=grid_id, start_ticks=0, step_ticks=1),
            history_due=lambda: False)

    root_clock = clock(1)
    # CPU fixture analogue of attach + bind: validation only needs the
    # resident mirror's semantic identity; the host forcing inventory above
    # supplies the exact prefix bytes.
    root_state._lateral_boundary_device = SimpleNamespace(
        rolling=False, clock=root_clock)
    root = SimpleNamespace(
        cfg=SimpleNamespace(
            grid_id=1, parent_id=0, run=root_cfg, start_time=None,
            i_parent_start=1, j_parent_start=1),
        state=root_state, clock=root_clock, parent=None, children=[],
        coupler=None, _started=True)

    class Coupler:
        valid = True

        def invalidate(self):
            self.valid = False

    child = SimpleNamespace(
        cfg=SimpleNamespace(
            grid_id=2, parent_id=1, run=child_cfg, start_time=None,
            i_parent_start=20, j_parent_start=18),
        state=child_state, clock=clock(2), parent=root, children=[],
        coupler=Coupler(), _started=True)
    root.children.append(child)
    nodes = {1: root, 2: child}
    model = SimpleNamespace(
        root=root,
        experiment_fingerprint="sealed-tree-fixture-v1",
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(
                tick_den=1, run_ticks=int(run_seconds), start_time=start)),
        walk_parent_first=lambda: iter((root, child)),
        nodes_by_grid_id=nodes,
        node=lambda gid: nodes[int(gid)],
        _runtime_status=ModelRuntimeStatus(),
        _io_manager=None,
        _last_checkpoint=None,
        _resumed=False,
        _resume_committed_history_grid_ids=frozenset())
    return model, start


def test_sealed_tree_restart_extends_root_and_restores_rebuilt_child(
        monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        sealed_forcing_extension=True)
    assert restart.read_restart_header(root_path)[
        "root_external_lbc_clock"] == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200.0,
        payload_seed=91)
    info = restart.restore_tree_restart(
        root_path, resumed, sealed_forcing_extension=True)
    assert info.elapsed_ticks == 3600
    assert np.array_equal(resumed.root.state.u, source.root.state.u,
                          equal_nan=True)
    resumed_child = tuple(resumed.walk_parent_first())[1]
    source_child = tuple(source.walk_parent_first())[1]
    assert np.array_equal(resumed_child.state.u, source_child.state.u,
                          equal_nan=True)
    assert resumed_child.coupler.valid is False


def test_tree_restart_at_exactly_the_stop_tick_reports_a_complete_run(
        monkeypatch, tmp_path):
    """A restore point that IS the stop tick means the run finished.

    The 24 h four-domain reproduction integrated every tick, wrote every
    history frame and its final checkpoint, and then the supervisor
    relaunched from that final checkpoint: each fresh worker restored at
    ``elapsed_ticks == run_ticks`` and raised, so a scientifically
    complete run reported failure with rc 1.  Nothing is broken at that
    point -- the state is exactly the state the run stopped on -- so the
    restore reports completion and the route finalizes.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=91)
    info = restart.restore_tree_restart(root_path, resumed)

    assert info.already_complete is True
    assert info.elapsed_ticks == 3600
    assert resumed.root.clock.ticks == 3600
    # The state still restored: completion is not a skipped restore.
    assert np.array_equal(resumed.root.state.u, source.root.state.u,
                          equal_nan=True)


def test_tree_restart_past_the_stop_tick_refuses_and_names_the_remedy(
        monkeypatch, tmp_path):
    """A checkpoint BEYOND the stop is still a genuine mismatch.

    The completion case above must not swallow this one: a checkpoint
    later than the configured stop means the run this checkpoint came
    from is longer than the run being asked for, and integrating it
    backwards is not a thing.  The refusal names that and names the key
    to change, which the bare ``nothing left before stop`` sentence did
    not.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=1800.0,
        payload_seed=91)
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.restore_tree_restart(root_path, resumed)

    message = str(caught.value)
    assert "3600" in message and "1800" in message
    assert "run_seconds" in message


# ---------------------------------------------------------------------------
# The nest-lifecycle block on a tree checkpoint (the write side)
# ---------------------------------------------------------------------------
#
# Compatibility is the essential half here.  A run that declares no
# dormant nest and no follower must keep writing the checkpoint it always
# wrote, byte for byte, because every checkpoint on disk was written by
# such a run and a moved header key is a checkpoint that refuses to
# restore.  The digest below is that claim, made as a number.

#: Header fields a SECOND write of the same state legitimately moves:
#: a wall-clock stamp, the installed release, the publish UUID, and the
#: memory road the writer was on.
#:
#: ``written_mode`` is in this list for the same reason ``producer`` is,
#: and for one stronger one.  It describes the WRITER, not the state:
#: ``woof.io.restart.written_mode_note`` records which memory road the
#: file came off, and the whole restart x memory mode contract is that
#: the two roads produce the same checkpoint for the same weather;
#: ``woof/state_digest.py`` requires them to agree BIT FOR BIT.  A
#: provenance stamp inside this digest would make that contract false by
#: construction, so it is excluded here, and the digests below keep the
#: values they were pinned at before it existed.  Nothing else about the
#: header moved with it.
_VOLATILE_CHECKPOINT_HEADER = ("created", "producer", "checkpoint_set_id",
                               "written_mode")

#: The terrain clock every digest below was pinned under.  From 2.8.8 the
#: default is "local_face", which the echo and the restart identity bind
#: (a run under it can take a longer step than the checkpoint's), so the
#: digest fixtures name the clock they were recorded with.  That the
#: default moves the header and nothing else is
#: test_the_local_face_default_moves_only_the_clock_binding below.
#: It also names the solar generation they were recorded with: from 2.9.0
#: the default is WRF 4.8's equation of time ("wrf_480"), which the echo
#: and the restart identity bind (COSZEN and HRANG change).  That the
#: default moves the solar binding and nothing else is
#: test_the_wrf480_solar_default_moves_only_the_solar_binding below.
_PINNED_UNDER_MEASURED_CLOCK = {"terrain_clock": "measured",
                                "solar_generation": "wrf_461"}

#: The format stamp every historical digest in this file and its two
#: per-change siblings was harvested under.  v6 declared the 2.7.0 break
#: (ENG-010 / ENG-011: adaptive + eta_levels echo, fields/ustm, held/gf_*),
#: so the stamp is normalised back to 5 before hashing: these digests
#: attribute WHAT ELSE moved, and the stamp is the one header field whose
#: move is declared rather than attributed.  The live stamp is asserted
#: separately (test_the_live_format_stamp_is_the_declared_v7_and_v5_is_named).
_HISTORICAL_FORMAT_VERSION = 5


def _canonical_member_digest(path) -> str:
    """One checkpoint member's bytes, minus the volatile header fields."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))
        for name in _VOLATILE_CHECKPOINT_HEADER:
            header.pop(name, None)
        header["format_version"] = _HISTORICAL_FORMAT_VERSION
        _unwind_v7_inverse_binding(header)
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        for name in sorted(data.files):
            if name == restart._HEADER_KEY:
                continue
            host = data[name]
            digest.update(name.encode("utf-8"))
            digest.update(str(host.dtype).encode("utf-8"))
            digest.update(str(host.shape).encode("utf-8"))
            digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _member_names(path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return sorted(name for name in data.files
                      if name != restart._HEADER_KEY)


#: The lifecycle-free two-domain checkpoint, as one number.  It moves only
#: when the checkpoint FORMAT moves; a nest-lifecycle feature that touches
#: it has broken every checkpoint written before it.
#:
#: Re-pinned when the relocation carry landed: every tree member gained
#: the ``placement`` header field and the ``acoustic/ww_pp`` carrier,
#: both absent-tolerant on restore -- a checkpoint written before the
#: move still restores; these pins state the format going FORWARD.
#:
#: Re-pinned at the mp=28 WIF-climatology config addition: ``RunConfig``
#: gained ``wif_climatology_path`` at its inert "" default and
#: ``mp28_aerosol_source`` at "auto", so the header's CONFIG ECHO grew by
#: two keys and the two hashes derived from it --
#: ``physics_setup.configuration_sha256`` and
#: ``physics_setup_fingerprint`` -- moved with it.  Those are the WHOLE
#: diff, and that is MEASURED rather than asserted: the reconstruction
#: below drops the two keys from a live checkpoint's echo, recomputes the
#: two hashes exactly as the writer computes them, and recovers BOTH
#: pre-lane digests byte for byte.  A second change riding in under this
#: re-pin fails there instead of being absorbed here.
#:
#: What this re-pin does NOT claim: that a pre-lane checkpoint still
#: resumes.  It does not.  ``configuration_sha256`` hashes the whole
#: RunConfig, so ANY field addition invalidates every checkpoint written
#: before it; that is this tree's existing behaviour for every
#: config-surface addition and it is not what these pins measure.
#:
#: RE-PINNED 2026-08-29 by the P3 CUDA port, which appended ONE RunConfig
#: field (`p3_backend`).  The attribution is not this comment: it is
#: test_the_checkpoint_pins_moved_for_the_two_wif_keys_and_nothing_else
#: below, which removes the appended config keys, recomputes the two hashes
#: the writer derives from the echo, and requires the result to be the
#: pre-lane digest EXACTLY.  It does, so nothing but the config echo moved:
#: no header key, no array member name, no dtype, no shape, no byte of any
#: array.  A checkpoint-format change riding in behind a config addition
#: would leave that reconstruction short and would fail there.
#:
#: RE-PINNED on lane/release-reds-266 for d0c23dad0 (2.6.4, the New
#: Tiedtke port), which appended ONE RunConfig field
#: (`ntiedtke_tiedtke_closure`) and re-pinned nothing.  Same attribution,
#: same gate: the key joined _WIF_CONFIG_KEYS below, and the
#: reconstruction recovers both pre-lane digests exactly -- on the
#: Windows desktop and on the Linux node alike -- so the config echo is
#: the whole diff once more.
#:
#: RE-PINNED AGAIN on lane/adaptive-timestep, which appended TWELVE
#: RunConfig fields (the adaptive time step's namelist surface,
#: woof/core/adaptive_timestep.ADAPTIVE_TIMESTEP_RUN_FIELDS).  Same
#: attribution and the same gate one link further along:
#:
#:   2.6.0 / 2.6.1     5431237076a4 / 0b9e45449cba
#:     + ntiedtke_tiedtke_closure (1 key)
#:   2.6.5 line        c85c921d720d / fb448bfd5719
#:     + the adaptive-timestep surface (12 keys)
#:   this lane         the pair below
#:
#: Each arrow is a gate rather than a claim: the first is
#: tests/test_ntiedtke_closure_checkpoint.py, the second is
#: tests/test_adaptive_timestep_checkpoint.py, and the reconstruction
#: below unwinds all thirteen keys at once and must still land on the
#: pre-WIF pair.  Only the config echo moved: 14 arrays either side with
#: identical names, dtypes, shapes and bytes, an unchanged header key set
#: and array_manifest.
#:
#: RE-PINNED on the per-domain vertical lane, which appended ONE RunConfig
#: field (`eta_levels`, a domain's own eta ladder).  Same attribution and the
#: same gate one link further along: the key joined _WIF_CONFIG_KEYS below,
#: and the reconstruction unwinds all fourteen keys at once and must still
#: land on the pre-WIF pair.  It does, so the config echo is the whole diff
#: once more: 14 arrays either side with identical names, dtypes, shapes and
#: bytes, an unchanged header key set and array_manifest.  The field's
#: default is None -- "inherit the ladder the source carries", which is
#: exactly what every checkpoint written before it describes -- so no
#: prepared state moved with it.
#:
#: RE-PINNED for TWO appended RunConfig fields (`relax_timescale_s`,
#: `relax_w`, the downscaled child's lateral zone).  Same attribution and
#: the same gate: both keys joined _WIF_CONFIG_KEYS below, and the
#: reconstruction unwinds all sixteen keys at once and lands on the
#: pre-WIF pair exactly, root and child.  The config echo is the whole
#: diff.  Their defaults (0.0, False) are WRF's own relaxation, which is
#: what every checkpoint written before them ran.
#:
#: RE-PINNED for ONE appended RunConfig field (`min_time_step_sound`, the
#: adaptive clock's floor under its derived acoustic substep count), from
#: 53602b734854 / b770a432fe7a.  Same gate: the key is one of
#: ADAPTIVE_TIMESTEP_RUN_FIELDS, which _WIF_CONFIG_KEYS imports, and the
#: reconstruction unwinds it with the rest and lands on the pre-WIF pair
#: exactly, root and child, so the config echo is the whole diff.  Its
#: default 0 leaves WRF's derived count, and nothing reads it under the
#: fixed clock this fixture runs.
#:
#: RE-PINNED for THREE appended RunConfig fields (`sf_urban_physics`,
#: `use_wudapt_lcz`, `num_urban_hi`, lane/urban-infra), from
#: 0cd69d00f667 / db0cff6b5e09.  Same gate: the three keys joined
#: _WIF_CONFIG_KEYS below and the reconstruction lands on the pre-WIF pair
#: exactly, root and child, so the config ECHO is the whole diff.  The
#: configuration digest does not move: with no urban model the three keys
#: are dropped from it (restart._configuration_digest_values), so every
#: checkpoint written before them still resumes.
#:
#: RE-PINNED for the moist_cq default flip (e13fa45c0 / 59f7e280f).
#: The header echoes True instead of False and its two derived physics
#: hashes move.  The dedicated default-move reconstruction below changes
#: only that value back and recovers both previous canonical digests,
#: retaining every array byte and every other header value.  Older ledger
#: reconstructions first unwind the same default before removing their keys.
#:
#: RE-PINNED for the Kessler identity (2.8.8, gfix/288-mp-clamp): the
#: shared microphysics finish clamps at WRF's REAL product mp_tend_lim*dt,
#: so this mp=1 fixture's physics_setup names "kessler-warm-rain-v2-real-
#: clamp-product" and the fingerprint derived from it moves.  The
#: attribution is test_the_checkpoint_pins_moved_for_the_real_clamp_
#: identity_and_nothing_else below: naming the v1 identity again and
#: recomputing the fingerprint recovers the previous pair exactly, and
#: every older reconstruction unwinds the same name
#: (_PRE_REAL_CLAMP_ALGORITHMS).
#:
#: RE-PINNED for HAILCAST (lane/cap-hail, 97e9a3ca061): hailcast_opt = -1
#: runs the diagnostic at dx <= 4 km with microphysics on
#: (woof.core.hail.enabled), so this mp=1, 2 km fixture now carries the
#: eleven persistent HAILCAST slots (_HAILCAST_PERSISTENT_MEMBERS: running
#: maximum, updraft mask and duration, the embryo and window diameters) as
#: scratch members with their manifest rows, which a resume needs to
#: continue the running maxima exactly.  The attribution is
#: test_the_checkpoint_pins_moved_for_the_hailcast_slots_and_nothing_else:
#: dropping exactly those members and rows recovers the previous pair, and
#: every older reconstruction unwinds the same members
#: (_digest_without_config_keys unwind_hailcast).
#:
#: RE-PINNED for tc/nest's fixed-frame nest accumulation archive (2.9.7):
#: a nested domain's echo carries nest_accumulation_archive at its default
#: True, so the child's header and its two derived physics hashes move; the
#: root is not nested and keeps its digest.  The attribution is
#: test_the_checkpoint_pins_moved_for_the_nest_archive_and_nothing_else:
#: dropping that key recovers the previous child digest
#: (_PRE_NEST_ARCHIVE_CHILD_DIGEST), and every older reconstruction unwinds it
#: (_digest_without_config_keys unwind_nest_archive).
_LIFECYCLE_FREE_ROOT_DIGEST = \
    "72c1bf64128f7686fb82e04d6430d3140a899faa3c6aba319bfe73e1f01ef5e8"
_LIFECYCLE_FREE_CHILD_DIGEST = \
    "c63b17c9cb1bb4966f81f9f9ca15ca448629c483356ad828f3f824596d8c62d3"
_PRE_NEST_ARCHIVE_CHILD_DIGEST = \
    "cce75c2551272396fa9cdbbb69701612922d73ce5e006054cb2a510546d05354"

#: The pair above before the HAILCAST slots joined the checkpoint.
_PRE_HAILCAST_ROOT_DIGEST = \
    "b2ce004e590de75a7d37596fd6437f8aa33de0ffd40b86a8e459bb626217089e"
_PRE_HAILCAST_CHILD_DIGEST = \
    "e03f5ad776fb76735fa796e555813e379e2d874d9641fe3f4f11574bbde690e0"

#: lane/cap-hail's persistent slots at default (mp_physics = 1 has no PSD
#: row, so the four hail_* PSD slots are absent).
_HAILCAST_PERSISTENT_MEMBERS = (
    "scratch/hailcast_dhail1", "scratch/hailcast_dhail2",
    "scratch/hailcast_dhail3", "scratch/hailcast_dhail4",
    "scratch/hailcast_dhail5", "scratch/hailcast_diam_max",
    "scratch/hailcast_diam_mean", "scratch/hailcast_diam_std",
    "scratch/hailcast_runmax", "scratch/hailcast_wdur",
    "scratch/hailcast_wup_mask")


def _unwind_hailcast_members(header: dict, names) -> list[str]:
    """Drop the HAILCAST manifest rows; return the member names to keep."""
    for name in _HAILCAST_PERSISTENT_MEMBERS:
        assert name in names, name
        del header["array_manifest"][name]
    return [name for name in names if name not in _HAILCAST_PERSISTENT_MEMBERS]

#: The pair above before the Kessler identity moved, and the identity every
#: digest older than 2.8.8's clamp fix was harvested under.
_PRE_REAL_CLAMP_ROOT_DIGEST = \
    "efca0135a058d4aedb3a120ee729b769ed13d08352446eaa350f7cf5276fc76a"
_PRE_REAL_CLAMP_CHILD_DIGEST = \
    "e35a7d91043c4096d8b7087e66eea09a7ca87bea63f3ad4cb8f47b3d5cce48c7"
_PRE_EXACT_CHAIN_ALGORITHMS = {
    "physics_driver": "gpuwm-physics-driver-v3-kf-phase-energy-pre-mp-expiry",
}
_PRE_REAL_CLAMP_ALGORITHMS = {
    "microphysics": "kessler-warm-rain-v1",
    **_PRE_EXACT_CHAIN_ALGORITHMS,
}

_PRE_CQ_DEFAULT_ROOT_DIGEST = \
    "9ead8250634097dc3d514611b5b2aed7aefec327008f36c8b2897683a7325b93"
_PRE_CQ_DEFAULT_CHILD_DIGEST = \
    "231ea033d29f5c3fabea95e4ce46bf4ad0a83907eaf2688d33453d1a3b53d4da"


#: The values these pins carried immediately before the two mp=28 aerosol
#: config keys landed, kept so the reconstruction has something to
#: reconstruct.
_PRE_WIF_ROOT_DIGEST = \
    "194bd6a0613231ab32c41bdb3fb19dd99c5357f08b2b6e052d7ab5adc1704a98"
_PRE_WIF_CHILD_DIGEST = \
    "6c8b3f3a6f73961d2b8dcddd6f4c93ec0873dd05039c37716d27fef3dcc2c8f4"

#: The RunConfig keys added SINCE the pre-lane digests below were taken
#: (the first two by the mp=28 aerosol lanes, which named this tuple).
#: Removing them and recomputing the two derived hashes is what separates
#: "a config field was added" from "the checkpoint FORMAT moved" -- the
#: second must still fail, and it will, because the reconstruction only
#: removes these names.  ``p3_backend`` joined with the P3 CUDA port
#: (2026-08-29); it is scheme-scoped to mp=50 in the restart identity, but
#: the header CONFIG ECHO carries every RunConfig field regardless, which
#: is exactly why this reconstruction exists.  ``ntiedtke_tiedtke_closure``
#: joined with the New Tiedtke port (d0c23dad0, 2.6.4), unscoped, and
#: the echo grew by that one key: the reconstruction below still recovers
#: both pre-lane digests with it removed, so the io/restart.py changes in
#: that commit (a cu=16 algorithm identity and an expected-class row,
#: neither reached by this cu=0 fixture) moved no header key, no member,
#: no dtype and no byte of any array.
#: ``ADAPTIVE_TIMESTEP_RUN_FIELDS`` is IMPORTED rather than retyped: the
#: module that owns those twelve names is the only thing that should be
#: able to change them, and a copy here would quietly stop matching the
#: day it gains a thirteenth.
#: ``eta_levels`` joined with the per-domain vertical ladder (the offline
#: child's own eta grid).  Unscoped, and its default is ``None`` -- "inherit
#: the ladder the source carries", which is what every checkpoint written
#: before it describes -- so the echo grew by exactly one key whose value is
#: null in this fixture.
_WIF_CONFIG_KEYS = ("mp28_aerosol_source", "wif_climatology_path",
                    "p3_backend", "ntiedtke_tiedtke_closure", "eta_levels",
                    "relax_timescale_s", "relax_w",
                    # The urban canopy selector and its two companions
                    # (lane/urban-infra), appended last to RunConfig.
                    "sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
                    ) + ADAPTIVE_TIMESTEP_RUN_FIELDS



def _digest_without_config_keys(path, keys, *, config_overrides=None,
                                algorithm_overrides=None,
                                unwind_hailcast=True,
                                unwind_nest_archive=True) -> str:
    """The canonical member digest as it would read WITHOUT ``keys``.

    Same construction as :func:`_canonical_member_digest`, with exactly
    three substitutions: the named keys are dropped from the config echo,
    and the two hashes the writer derives from that echo --
    ``physics_setup.configuration_sha256`` and
    ``physics_setup_fingerprint`` -- are recomputed from the trimmed one
    using the writer's OWN helpers rather than a second implementation of
    them.  Every array member is hashed unchanged, so anything that moved
    outside the config echo survives into the result and the caller's
    comparison fails.

    ``config_overrides`` reconstructs a declared default-value move without
    dropping its identity-bound key.  The writer's own derived hashes are
    recomputed after the named replacements; every other header value and
    every array byte remain bound.

    ``unwind_hailcast`` (default on: every digest this reconstructs
    predates lane/cap-hail) drops the eleven HAILCAST persistent members and
    their manifest rows, and asserts each was there, so a missing or extra
    slot still leaves the reconstruction short.

    ``unwind_nest_archive`` (default on: every digest this reconstructs
    predates tc/nest) drops ``nest_accumulation_archive`` from the echo, which
    a nested domain writes at its default True.

    ``algorithm_overrides`` reconstructs a declared scheme identity move
    (``physics_setup.algorithms``) the same way: the named components take
    their previous identity and the fingerprint is recomputed; nothing
    else is touched.

    Taking ``keys`` as an argument is what lets one construction serve
    both the cumulative reconstruction below and the per-change
    attribution in ``tests/test_ntiedtke_closure_checkpoint.py``; the
    alternative was a third copy of these twenty lines.
    """
    import copy

    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))
        for name in _VOLATILE_CHECKPOINT_HEADER:
            header.pop(name, None)
        header["format_version"] = _HISTORICAL_FORMAT_VERSION
        _unwind_v7_inverse_binding(header)
        for key in keys:
            header["config"].pop(key, None)
        if unwind_nest_archive:
            header["config"].pop("nest_accumulation_archive", None)
        for key, value in (config_overrides or {}).items():
            assert key in header["config"], key
            header["config"][key] = value
        values = restart._configuration_digest_values(header["config"])
        setup = copy.deepcopy(header["physics_setup"])
        setup["configuration_sha256"] = restart._json_sha256(
            restart._json_value(values, "RunConfig"))
        for component, identity in (algorithm_overrides or {}).items():
            assert component in setup["algorithms"], component
            setup["algorithms"][component] = identity
        header["physics_setup"] = setup
        header["physics_setup_fingerprint"] = restart._json_sha256(setup)
        names = [name for name in sorted(data.files)
                 if name != restart._HEADER_KEY]
        if unwind_hailcast:
            names = _unwind_hailcast_members(header, names)
        digest = hashlib.sha256()
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        for name in names:
            host = data[name]
            digest.update(name.encode("utf-8"))
            digest.update(str(host.dtype).encode("utf-8"))
            digest.update(str(host.shape).encode("utf-8"))
            digest.update(host.tobytes(order="C"))
        return digest.hexdigest()


def _digest_without_the_wif_config_keys(path) -> str:
    """Unwind later config additions and the declared moisture default move."""
    # e13fa45c0 / 59f7e280f turned moist_cq on by default.  These pre-WIF
    # anchors were recorded with it off; restore that value, retain the key.
    # They also predate 2.8.8's Kessler identity move: name v1 again.
    return _digest_without_config_keys(
        path, _WIF_CONFIG_KEYS, config_overrides={"moist_cq": False},
        algorithm_overrides=_PRE_REAL_CLAMP_ALGORITHMS)


def test_the_checkpoint_pins_moved_for_the_two_wif_keys_and_nothing_else(
        monkeypatch, tmp_path):
    """The attribution for the re-pin above, as a gate rather than a claim.

    THE CONCRETE BREAKAGE THIS PREVENTS: a checkpoint-FORMAT change -- a
    moved header key, a renamed array member, a dtype that widened --
    riding into the tree inside a config-addition re-pin.  Both digests
    above would move, both would be updated, and the format change would
    be invisible, because "we added a config field" explains a moved
    digest.  Here it does not: the appended keys are removed and the two
    derived hashes recomputed, so anything ELSE that moved leaves the
    reconstruction short of the pre-lane value and this fails.

    The name says "two" because two is what ``_WIF_CONFIG_KEYS`` held when
    it was written; it is four now (``p3_backend`` 2026-08-29,
    ``ntiedtke_tiedtke_closure`` 2026-09-02).  This is the CUMULATIVE arm
    -- it anchors on the pre-WIF pair and absorbs everything appended
    since.  The per-change arm, which anchors each re-pin on the pair it
    replaced and so names WHICH key moved it, is a file per change:
    ``tests/test_ntiedtke_closure_checkpoint.py``,
    ``tests/test_adaptive_timestep_checkpoint.py``.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))

    assert (_digest_without_the_wif_config_keys(root_path)
            == _PRE_WIF_ROOT_DIGEST)
    assert (_digest_without_the_wif_config_keys(child_path)
            == _PRE_WIF_CHILD_DIGEST)

    # And the two keys really are in the echo, so the reconstruction is
    # removing something rather than succeeding vacuously.
    with np.load(root_path, allow_pickle=False) as data:
        echo = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))["config"]
    for key in _WIF_CONFIG_KEYS:
        assert key in echo, key


def _written_sealed_tree_members(monkeypatch, directory, overrides):
    directory.mkdir()
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=31, run_overrides=overrides)
    restart.write_tree_restart(
        directory, source, start + timedelta(seconds=3600))
    members = {}
    for path in sorted(directory.glob("gpuwmrst_d0*_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(bytes(bytearray(
                data[restart._HEADER_KEY])).decode("utf-8"))
            arrays = {name: (str(data[name].dtype), data[name].shape,
                             data[name].tobytes(order="C"))
                      for name in data.files
                      if name != restart._HEADER_KEY}
        members[path.name.split("_")[1]] = (header, arrays)
    return members


def test_the_local_face_default_moves_only_the_clock_binding(
        monkeypatch, tmp_path):
    """From 2.8.8 the default terrain clock is "local_face" and it binds.

    THE CONCRETE BREAKAGE THIS PREVENTS: a checkpoint-format change riding
    in under the default flip, hidden because the digests above are pinned
    under "measured".  The same tree written under the default and under
    "measured" must differ only in the config echo's terrain_clock and the
    two hashes the writer derives from the echo; every array member is
    byte for byte the same.
    """
    default = _written_sealed_tree_members(monkeypatch, tmp_path / "default", None)
    measured = _written_sealed_tree_members(
        monkeypatch, tmp_path / "measured", {"terrain_clock": "measured"})
    assert sorted(default) == sorted(measured) == ["d01", "d02"]
    for member in default:
        live, live_arrays = default[member]
        old, old_arrays = measured[member]
        assert live_arrays == old_arrays
        assert live["config"]["terrain_clock"] == "local_face"
        assert "terrain_clock" not in old["config"]
        moved = {key for key in set(live) | set(old)
                 if key not in _VOLATILE_CHECKPOINT_HEADER
                 and live.get(key) != old.get(key)}
        assert moved == {"config", "physics_setup",
                         "physics_setup_fingerprint"}
        assert {key for key in set(live["config"]) | set(old["config"])
                if live["config"].get(key) != old["config"].get(key)} \
            == {"terrain_clock"}
        assert {key for key in set(live["physics_setup"])
                | set(old["physics_setup"])
                if live["physics_setup"].get(key)
                != old["physics_setup"].get(key)} == {"configuration_sha256"}


def test_the_wrf480_solar_default_moves_only_the_solar_binding(
        monkeypatch, tmp_path):
    """From 2.9.0 the default solar generation is "wrf_480" and it binds.

    THE CONCRETE BREAKAGE THIS PREVENTS: a checkpoint-format change riding
    in under the WRF 4.8 equation-of-time default (golden/wrf-fixes), hidden
    because the digests above are pinned under "wrf_461".  The same tree
    written under the default and under "wrf_461" must differ only in the
    config echo's solar_generation, the "solar" algorithm identity and the
    hashes the writer derives from them; every array member is byte for
    byte the same.  (The YSU, Noah-MP snow-aging and urban water-balance
    selectors are scheme-scoped and absent from this fixture's echo.)
    """
    default = _written_sealed_tree_members(monkeypatch, tmp_path / "default", None)
    legacy = _written_sealed_tree_members(
        monkeypatch, tmp_path / "legacy", {"solar_generation": "wrf_461"})
    assert sorted(default) == sorted(legacy) == ["d01", "d02"]
    for member in default:
        live, live_arrays = default[member]
        old, old_arrays = legacy[member]
        assert live_arrays == old_arrays
        assert live["config"]["solar_generation"] == "wrf_480"
        assert "solar_generation" not in old["config"]
        for scoped in ("ysu_version", "noahmp_snow_age",
                       "urban_water_balance"):
            assert scoped not in live["config"], scoped
        moved = {key for key in set(live) | set(old)
                 if key not in _VOLATILE_CHECKPOINT_HEADER
                 and live.get(key) != old.get(key)}
        assert moved == {"config", "physics_setup",
                         "physics_setup_fingerprint"}
        assert {key for key in set(live["config"]) | set(old["config"])
                if live["config"].get(key) != old["config"].get(key)}             == {"solar_generation"}
        assert {key for key in set(live["physics_setup"])
                | set(old["physics_setup"])
                if live["physics_setup"].get(key)
                != old["physics_setup"].get(key)}             == {"configuration_sha256", "algorithms"}
        assert {key for key in set(live["physics_setup"]["algorithms"])
                | set(old["physics_setup"]["algorithms"])
                if live["physics_setup"]["algorithms"].get(key)
                != old["physics_setup"]["algorithms"].get(key)} == {"solar"}


def test_exact_chain_checkpoint_pin_moves_only_the_driver_identity(monkeypatch, tmp_path):
    """Reconstruct the base digest without changing an array or config word."""
    from tools.release.refresh_checkpoint_identity_pins import BASE_DIGESTS
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600., payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    write = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    child = next(tmp_path.glob("gpuwmrst_d02_*.npz"))
    for grid, path in (("d01", write), ("d02", child)):
        assert _digest_without_config_keys(
            path, (), algorithm_overrides=_PRE_EXACT_CHAIN_ALGORITHMS) == BASE_DIGESTS[grid]


def test_a_lifecycle_free_tree_checkpoint_is_byte_identical(
        monkeypatch, tmp_path):
    """Ran green BEFORE the lifecycle block existed and green after."""
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))

    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST
    assert _canonical_member_digest(child_path) == \
        _LIFECYCLE_FREE_CHILD_DIGEST


def test_the_checkpoint_pins_moved_for_the_nest_archive_and_nothing_else(
        monkeypatch, tmp_path):
    """tc/nest's archive key moved the nested child's echo, not the format."""
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    assert restart.read_restart_header(child_path)["config"][
        "nest_accumulation_archive"] is True
    assert _canonical_member_digest(child_path) == _LIFECYCLE_FREE_CHILD_DIGEST
    assert _LIFECYCLE_FREE_CHILD_DIGEST != _PRE_NEST_ARCHIVE_CHILD_DIGEST
    assert _digest_without_config_keys(
        child_path, (), unwind_hailcast=False) == _PRE_NEST_ARCHIVE_CHILD_DIGEST


def test_the_checkpoint_pins_moved_for_the_moist_cq_default_and_nothing_else(
        monkeypatch, tmp_path):
    """e13fa45c0 / 59f7e280f changed one bound default, not the format."""
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    for path, old_digest, current_digest in (
            (root_path, _PRE_CQ_DEFAULT_ROOT_DIGEST,
             _PRE_REAL_CLAMP_ROOT_DIGEST),
            (child_path, _PRE_CQ_DEFAULT_CHILD_DIGEST,
             _PRE_REAL_CLAMP_CHILD_DIGEST)):
        assert restart.read_restart_header(path)["config"]["moist_cq"] is True
        # The CQ flip's own "after" pair predates the Kessler identity
        # move, so both sides name v1 (_PRE_REAL_CLAMP_ALGORITHMS).
        assert _digest_without_config_keys(
            path, (), algorithm_overrides=_PRE_REAL_CLAMP_ALGORITHMS
        ) == current_digest
        assert current_digest != old_digest
        assert _digest_without_config_keys(
            path, (), config_overrides={"moist_cq": False},
            algorithm_overrides=_PRE_REAL_CLAMP_ALGORITHMS) == old_digest


def test_the_checkpoint_pins_moved_for_the_real_clamp_identity_and_nothing_else(
        monkeypatch, tmp_path):
    """gfix/288-mp-clamp moved one scheme identity, not the format.

    THE CONCRETE BREAKAGE THIS PREVENTS: a checkpoint-format change riding
    in under the Kessler identity re-pin.  Naming the v1 identity again and
    recomputing the fingerprint must land on the previous pair exactly;
    anything else that moved leaves the reconstruction short.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    for path, old_digest, current_digest in (
            (root_path, _PRE_REAL_CLAMP_ROOT_DIGEST,
             _LIFECYCLE_FREE_ROOT_DIGEST),
            (child_path, _PRE_REAL_CLAMP_CHILD_DIGEST,
             _LIFECYCLE_FREE_CHILD_DIGEST)):
        algorithms = restart.read_restart_header(path)["physics_setup"][
            "algorithms"]
        assert algorithms["microphysics"] == (
            restart.MICROPHYSICS_ALGORITHM_IDENTITIES[1])
        assert algorithms["microphysics"] != (
            _PRE_REAL_CLAMP_ALGORITHMS["microphysics"])
        assert _canonical_member_digest(path) == current_digest
        assert current_digest != old_digest
        assert _digest_without_config_keys(
            path, (), algorithm_overrides=_PRE_REAL_CLAMP_ALGORITHMS
        ) == old_digest


def test_the_checkpoint_pins_moved_for_the_hailcast_slots_and_nothing_else(
        monkeypatch, tmp_path):
    """lane/cap-hail added eleven persistent members, not a format change.

    THE CONCRETE BREAKAGE THIS PREVENTS: a checkpoint-format change riding
    in under the HAILCAST re-pin.  Dropping exactly the eleven members and
    their manifest rows must land on the previous pair; anything else that
    moved leaves the reconstruction short.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    for path, old_digest, current_digest in (
            (root_path, _PRE_HAILCAST_ROOT_DIGEST,
             _LIFECYCLE_FREE_ROOT_DIGEST),
            (child_path, _PRE_HAILCAST_CHILD_DIGEST,
             _LIFECYCLE_FREE_CHILD_DIGEST)):
        assert set(_HAILCAST_PERSISTENT_MEMBERS) <= set(_member_names(path))
        assert _canonical_member_digest(path) == current_digest
        assert _digest_without_config_keys(
            path, (), unwind_hailcast=False,
            unwind_nest_archive=False) == current_digest
        assert current_digest != old_digest
        assert _digest_without_config_keys(path, ()) == old_digest


def test_the_live_format_stamp_is_the_declared_v7_and_v5_is_named(
        monkeypatch, tmp_path):
    """The break is DECLARED: a real 2.6.5-shaped member is refused by name.

    The member is this build's own root checkpoint re-stamped 5 with the
    thirteen 2.7.0 config keys removed -- the header a 2.6.5 build wrote --
    so the refusal is exercised on a real file through the tree reader, not
    on a dict.  It must name the release, the format and the remedy, and
    it must fire BEFORE the field-by-field identity walk that would read
    as "your configuration changed".
    """
    from woof.core.model import ADAPTIVE_TIMESTEP_RUN_FIELDS
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    assert restart.read_restart_header(root_path)["format_version"] == restart.RESTART_FORMAT_VERSION
    assert restart.RESTART_FORMAT_VERSION == 7
    assert 5 in restart.RETIRED_RESTART_FORMAT_VERSIONS
    for member in tmp_path.glob("gpuwmrst_d0*_*.npz"):
        with np.load(member, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        header = json.loads(bytes(bytearray(
            payload[restart._HEADER_KEY])).decode("utf-8"))
        header["format_version"] = 5
        for key in ADAPTIVE_TIMESTEP_RUN_FIELDS + ("eta_levels",):
            header["config"].pop(key, None)
        payload[restart._HEADER_KEY] = np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8).copy()
        np.savez(member, **payload)
    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    with pytest.raises(restart.RestartMismatchError) as refused:
        restart.restore_tree_restart(root_path, resumed)
    message = str(refused.value)
    assert "2.6.5 checkpoint format 5" in message
    assert "fields/ustm" in message and "held/gf_rthblten" in message
    assert "restart from the run's initial conditions" in message
    assert "absent from the restart file" not in message, (
        "the identity walk spoke before the version gate")


def test_a_lifecycle_free_tree_checkpoint_names_no_lifecycle_key(
        monkeypatch, tmp_path):
    """The digest above says the bytes are the same; this says WHICH keys,
    so a failure names the addition instead of reporting a hash."""
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    header = restart.read_restart_header(root_path)

    # Literal names, so this runs on a build that predates the constants.
    assert "nest_lifecycle" not in header
    assert "relocation" not in header
    assert sorted(header) == [
        "array_manifest", "case", "checkpoint_set_id", "config", "created",
        "domain_ids", "domain_lifecycle", "domain_start_ticks",
        "domain_start_time", "driver", "dtbc_fp32_bits", "elapsed_seconds",
        "elapsed_ticks", "experiment_fingerprint", "format_version",
        "grid_id", "nest_tables", "parent_id", "phase", "physics_setup",
        "physics_setup_fingerprint", "placement", "producer",
        "root_external_lbc_clock",
        "run_trackers", "setup_fingerprint", "tick_den",
        # Provenance, not identity: which memory road WROTE the file.
        # Listed here because this assertion is the guard that names an
        # addition instead of reporting a hash, and this addition is
        # deliberate (woof.io.restart.written_mode_note).
        "written_mode",
    ]
    # The resident writer's stamp, beside the producer stamp it belongs
    # with: both describe the writer rather than the weather, and both
    # are excluded from the pinned digests above for that reason.
    assert header["written_mode"]["mode"] == "resident"
    keys = list(header)
    assert keys[keys.index("producer") + 1] == "written_mode"
    assert keys[keys.index("written_mode") + 1] == "elapsed_seconds"
    assert not any(name.startswith("scratch/uh_")
                   for name in _member_names(root_path))


def _lifecycle_experiment(*, follow: bool):
    """The DECLARED experiment behind the lifecycle fixture.

    d02 is dormant on a time trigger, placed at (20, 18) by the config.
    A per-domain ``[follow]`` table is the one fact that separates a
    ``per-domain`` follower entry from a ``legacy`` one, and no runner
    can see it -- which is why the writer fills that key.
    """
    from dataclasses import replace as _replace

    from woof.core.nest_lifecycle import DomainFollowConfig
    from woof.core.nest_spawn import SpawnConfig
    from woof.core.storm_tracking import FollowConfig
    from test_nest_spawn_init import _experiment

    exp = _experiment()
    child = _replace(exp.domains[1],
                     spawn=SpawnConfig(trigger="time", at_s=120.0))
    if follow:
        child = _replace(child, follow=DomainFollowConfig(
            tracker=FollowConfig(
                field="uh", threshold=60.0, search_margin_cells=4,
                min_shift_cells=1, max_shift_cells=4,
                cooldown_seconds=600.0, fallback_threshold=35.0),
            cadence_seconds=360.0, max_move_parent_cells=4,
            min_overlap_fraction=0.25))
    return _replace(exp, domains=(exp.domains[0], child))


#: The spawn state the fixture's runner is seeded with: d02 fired at its
#: declared (20, 18) in episode 1 and has since been carried to (24, 21)
#: by its follower, which is exactly the fired-vs-current split the block
#: has to keep separate.
_FIXTURE_SPAWN_BLOCK = {
    "watches": {"2": {"fired": True, "closed": False}},
    "spawned": {"2": {"fired": [20, 18], "current": [24, 21],
                      "episode": 1, "born_t": 120.0}},
    "retired": {},
    "episodes": {"2": 1},
    "quiet_since": {},
    "spawns_executed": 1,
}


def _lifecycle_tree_fixture(monkeypatch, *, follow: bool = True,
                            payload_seed: int = 31, spawn: bool = True):
    """A two-domain CPU tree carrying LIVE lifecycle runners.

    The runners are the real ones (``SpawnRunner``, ``RelocationRunner``),
    seeded through their own restore seams, because the whole point of the
    write side is that it calls their ``state_json`` rather than
    reconstructing the same facts a second way.
    """
    from dataclasses import replace as _replace

    from woof.core import uh_diag
    from woof.core.relocation_runner import (RelocationRunner,
                                              RelocationRunnerCollection)
    from woof.core.spawn_runner import SpawnRunner
    from woof.core.storm_tracking import StormTracker
    from woof.experiment import RelocationConfig

    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=payload_seed, nwp_diagnostics=1)
    exp = _lifecycle_experiment(follow=follow)
    # Published the way every route publishes it, and NOT through the
    # case-data route's build-time context: a tree assembled from a
    # prepared cache has no such context, and a checkpoint is not a
    # property of how the tree was built.
    publish_declared_experiment(model, exp)
    model._spawn_leg_seconds = 360.0
    root, child = tuple(model.walk_parent_first())
    # The follower has carried d02 off its declared footprint; only
    # node.cfg knows that, which is why the writer fills the key.
    child.cfg.i_parent_start = 24
    child.cfg.j_parent_start = 21

    if spawn:
        runner = SpawnRunner.from_experiment(
            exp, on_child_built=lambda *_a: None, array_module=np)
        runner.restore_state(_FIXTURE_SPAWN_BLOCK)
        model._spawn_runner = runner
    else:
        model._spawn_runner = None

    shape = (int(root.cfg.run.ny), int(root.cfg.run.nx))
    if follow:
        # Exactly the view build_real_relocation_runners assembles for a
        # per-domain [follow]: its own window slot on the parent, its own
        # tracker, and a tree-level config narrowed to this child.
        tracker = exp.domains[1].follow.tracker
        slot = uh_diag.follow_window_slot(2)
        root.state.scratch(shape, slot)
        provider = StormTracker(tracker, uh_slot=slot)
    else:
        tracker = _legacy_follow_config()
        provider = StormTracker(tracker)
    config = RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=360.0, follow=tracker)
    follower = RelocationRunner(
        config=config, provider=provider,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        on_child_built=lambda *_a: None, staging="host")
    # A tracker that has proposed once but not moved: the two cooldown
    # anchors are the follower state a resume most easily loses.
    provider._last_proposal_t = 60.0
    provider._last_move_t = None
    model._relocation_runner = (RelocationRunnerCollection((follower,))
                                if follow else follower)
    return model, start


def _legacy_follow_config():
    """The tree-level ``[relocation.follow]`` tracker block."""
    from woof.core.storm_tracking import FollowConfig

    return FollowConfig(
        field="uh", threshold=60.0, search_margin_cells=4,
        min_shift_cells=1, max_shift_cells=4, cooldown_seconds=600.0,
        fallback_threshold=35.0)


def test_the_declaration_is_published_on_the_tree_not_read_off_a_route(
        monkeypatch, tmp_path):
    """Checkpointing a run with a live follower is a GENERIC capability.

    The entry has to say where the experiment DECLARED a domain, so that
    a resume can tell a nest the follower moved from a nest someone
    reconfigured.  That fact belongs to the configuration.  Reading it
    out of ``_activation_context`` -- the case-data route's build-time
    bundle of case data, forcing calendar and radiation workspace --
    made the capability a property of how the tree was built: a tree
    restored from a prepared cache has no such bundle, so its first
    checkpoint refused however the tree was forced.
    """
    model, start = _lifecycle_tree_fixture(monkeypatch)
    assert getattr(model, "_activation_context", None) is None
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    entry = restart.read_restart_header(root_path)[
        restart.NEST_LIFECYCLE_HEADER_KEY]["followers"]["2"]
    assert entry["kind"] == "per-domain"
    assert entry["declared_placement"] == [20, 18]
    assert entry["current_placement"] == [24, 21]


def test_a_tree_that_publishes_no_declaration_still_refuses_by_name(
        monkeypatch, tmp_path):
    """The refusal is kept, and it is kept NAMED.

    A tree that states no declaration anywhere really cannot say whether
    the placement it is about to write is the declared one or one the
    follower moved to, and a checkpoint that cannot say is a resume that
    reads a moved nest as a reconfigured one.
    """
    model, start = _lifecycle_tree_fixture(monkeypatch)
    model._declared_experiment = None

    with pytest.raises(restart.RestartManifestError,
                       match="no declared experiment"):
        restart.write_tree_restart(
            tmp_path, model, start + timedelta(seconds=3600))


def test_a_follower_resumes_off_a_checkpoint_no_route_context_wrote(
        monkeypatch, tmp_path):
    """The other half: the block written from the published declaration
    seeds a live follower through the ordinary resume seam.

    The cooldown anchors are the state a resume most easily loses -- a
    tracker that forgets them proposes at the resumed run's first cadence
    boundary, moving the nest early and spinning up the strip it just
    filled -- so they are what this asserts landed.
    """
    from woof.runtime import restore_nest_followers

    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    resumed, _start = _lifecycle_tree_fixture(monkeypatch, payload_seed=17)
    provider = restart.lifecycle_followers(resumed)[2].provider
    provider._last_proposal_t = None            # a cold resumed tracker
    provider._last_move_t = None

    peek = restart.read_tree_lifecycle_header(root_path, resumed)
    assert restore_nest_followers(resumed, peek) == [2]
    assert provider._last_proposal_t == 60.0
    assert provider._last_move_t is None
    assert restart.lifecycle_followers(resumed)[2].moves_executed == 0


def test_the_window_planes_are_written_as_members_of_their_owner(
        monkeypatch, tmp_path):
    """The consumer windows are a max-fold since that consumer last
    looked.  Restarting them empty under-reads the next spawn/retire/
    follow decision, so a lifecycle run serializes them per domain."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))

    assert {name for name in _member_names(root_path)
            if name.startswith("scratch/uh_")} == {
        "scratch/uh_follow_window", "scratch/uh_follow_window.d02",
        "scratch/uh_spawn_window"}
    audit = restart.read_restart_header(root_path)[
        restart.NEST_LIFECYCLE_HEADER_KEY]["window_slots"]
    assert audit["1"] == ["uh_follow_window", "uh_follow_window.d02",
                          "uh_spawn_window"]
    assert audit["2"] == ["uh_follow_window", "uh_spawn_window"]
    assert {name for name in _member_names(child_path)
            if name.startswith("scratch/uh_")} == {
        "scratch/uh_follow_window", "scratch/uh_spawn_window"}


class _FakeStreamedDomain:
    """The store half of a streamed domain, for the publish hook alone.

    ``StreamedDomain`` leaves itself on the state it took over
    (``state._streamed_domain``); that marker is the door every reader
    that holds a NODE rather than a stepper already uses
    (woof/io/wrfout.py), and the checkpoint writer holds nodes.
    """

    def __init__(self, store):
        self.store = dict(store)
        self.published: list[tuple[str, ...]] = []

    def attach_to(self, state):
        state._streamed_domain = self
        self._state = state
        return self

    def impose_clock(self, seconds):
        self._state.elapsed_seconds = float(seconds)

    def write_restart(self, path, cfg, *, run_trackers=None, tree_header=None,
                      extra_scratch_slots=()):
        # This fixture spies only on optional lifecycle-window publication.
        # The complete store payload transport is tested independently.
        return SimpleNamespace(path=restart.write_restart(
            path, self._state, cfg, run_trackers=run_trackers,
            tree_header=tree_header, extra_scratch_slots=extra_scratch_slots))

    def publish(self, names):
        names = tuple(names)
        self.published.append(names)
        for name in names:
            slot = name[len("scratch/"):]
            self._state.existing_scratch(slot)[...] = self.store[name]
        return names


def test_a_streamed_parents_window_is_published_before_it_is_serialized(
        monkeypatch, tmp_path):
    """A streamed domain's arrays live in its store and its DomainState
    stopped changing at attach.  Serializing the state's copy would
    checkpoint the attach-time zeros under the name of a live window."""
    model, start = _lifecycle_tree_fixture(monkeypatch, follow=False)
    root = model.root
    shape = (int(root.cfg.run.ny), int(root.cfg.run.nx))
    live = np.arange(shape[0] * shape[1],
                     dtype=np.float32).reshape(shape) * np.float32(3.5)
    stream = _FakeStreamedDomain({
        "scratch/uh_spawn_window": live,
        "scratch/uh_follow_window": live * np.float32(2.0),
    }).attach_to(root.state)
    assert not root.state.existing_scratch("uh_spawn_window").any()

    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    assert stream.published == [("scratch/uh_follow_window",
                                "scratch/uh_spawn_window")]
    with np.load(root_path, allow_pickle=False) as data:
        assert np.array_equal(data["scratch/uh_spawn_window"], live)


def test_a_streamed_parent_publishes_its_generated_follower_window(
        monkeypatch, tmp_path):
    """The live generated carrier is checkpointed, never its stale state copy."""
    model, start = _lifecycle_tree_fixture(monkeypatch, follow=True)
    root = model.root
    shape = (int(root.cfg.run.ny), int(root.cfg.run.nx))
    live = np.full(shape, 73.25, dtype=np.float32)
    stream = _FakeStreamedDomain({
        "scratch/uh_spawn_window": np.zeros(shape, dtype=np.float32),
        "scratch/uh_follow_window": np.zeros(shape, dtype=np.float32),
        "scratch/uh_follow_window.d02": live,
    }).attach_to(root.state)
    root.state.existing_scratch("uh_follow_window.d02").fill(-99)
    path = restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    assert "scratch/uh_follow_window.d02" in stream.published[0]
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["scratch/uh_follow_window.d02"], live)


def test_a_checkpoint_written_after_a_move_attaches_the_posture_block(
        monkeypatch, tmp_path):
    """D3: the posture block was COMPUTED and never attached, so the
    crossed-move warning and the posture branch of
    ``tree_fingerprint_mismatch_reason`` could never fire from this
    writer -- a restart across a move refused as an unexplained hash."""
    from woof.core.nest_relocation import RESTART_ACROSS_MOVE_POSTURE

    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    model._relocation_receipts = [
        {"event": "relocated", "record_sha256": "a" * 64,
         "segment_id": "seg-1", "grid_id": 2},
    ]
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))

    for path in (root_path, child_path):
        moved = restart.read_restart_header(path)["relocation"]
        assert moved["moves"] == 1
        assert moved["record_sha256"] == ["a" * 64]
        assert moved["segment_id"] == "seg-1"
        assert moved["posture"] == RESTART_ACROSS_MOVE_POSTURE
    reason = restart.tree_fingerprint_mismatch_reason(
        2, restart.read_restart_header(root_path), model)
    assert "1 nest relocation(s)" in reason and "seg-1" in reason


def _bind_tree_identity(model):
    """Publish the named identity components the tree routes bind.

    The experiment component is ``restart_identity_payload`` of a real
    ExperimentConfig carrying this fixture's own RunConfigs, which is what
    both tree routes store beside the fingerprint.
    """
    from woof.core.model import restart_identity_payload
    from test_nest_spawn_init import _experiment

    exp = _experiment()
    exp = replace(exp, domains=tuple(
        replace(domain, run=model.node(domain.grid_id).cfg.run)
        for domain in exp.domains))
    components = {"experiment_identity": restart_identity_payload(exp),
                  "inputs": "same-prepared-bytes"}
    model._experiment_fingerprint_components = components
    model.experiment_fingerprint = hashlib.sha256(json.dumps(
        components, sort_keys=True, default=str).encode()).hexdigest()


def _diagnostic_default(name):
    return RunConfig.__dataclass_fields__[name].default


def _diagnostic_on(name):
    return True if isinstance(_diagnostic_default(name), bool) else 1


@pytest.mark.parametrize("name", sorted(restart.CONFIG_DIAGNOSTIC_FIELDS))
@pytest.mark.parametrize("switched_on", [True, False])
def test_a_tree_resumes_with_an_output_only_switch_changed(
        monkeypatch, tmp_path, name, switched_on):
    """The tree accepts what each member's config walk accepts.

    The member walk skips CONFIG_DIAGNOSTIC_FIELDS, but the tree compared
    its outer experiment identity raw, so switching nwp_diagnostics (or
    any output-only toggle) at a restart was refused as
    "experiment_identity differ(s) from the checkpoint".
    """
    default = _diagnostic_default(name)
    chosen = _diagnostic_on(name)
    before, after = (default, chosen) if switched_on else (chosen, default)
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=31,
        run_overrides={name: before})
    _bind_tree_identity(source)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=61,
        run_overrides={name: after})
    _bind_tree_identity(resumed)
    from woof.core.model import RESTART_TOLERATED_RUN_FIELDS
    written = restart.read_restart_header(root_path)["experiment_fingerprint"]
    if name in RESTART_TOLERATED_RUN_FIELDS:
        # Tolerated outright (surface_energy_diag, lane/sf-landsurface): it
        # never enters the identity payload, so the fingerprints agree.
        assert resumed.experiment_fingerprint == written
    else:
        assert resumed.experiment_fingerprint != written

    info = restart.restore_tree_restart(root_path, resumed)

    assert info.elapsed_ticks == 3600
    for gid in (1, 2):
        assert resumed.node(gid).clock.ticks == 3600
        assert np.array_equal(resumed.node(gid).state.thp,
                              source.node(gid).state.thp, equal_nan=True)


@pytest.mark.parametrize("name", sorted(restart.CONFIG_DIAGNOSTIC_FIELDS))
def test_the_configuration_digest_ignores_an_output_only_switch(name):
    """Both values of a switch give one digest, and a configuration that
    leaves the switch off keeps the digest it always had, so no checkpoint
    byte pin moves for the table growing."""
    off = _cfg(grid_id=1)
    on = replace(off, **{name: _diagnostic_on(name)})
    assert (restart._configuration_fingerprint(on)
            == restart._configuration_fingerprint(off))
    moved = replace(off, dt=off.dt * 2)
    assert (restart._configuration_fingerprint(moved)
            != restart._configuration_fingerprint(off))


@pytest.mark.parametrize("change", ["dt", "nest_position", "inputs"])
def test_an_output_only_switch_does_not_hide_a_tree_identity_change(
        monkeypatch, tmp_path, change):
    """Beside a switched diagnostic, a real change still refuses by name."""
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=31)
    _bind_tree_identity(source)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))

    overrides = {"nwp_diagnostics": 1, "hmix_k_diag": True}
    if change == "dt":
        overrides["dt"] = 20.0
    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=61,
        run_overrides=overrides)
    _bind_tree_identity(resumed)
    components = json.loads(json.dumps(
        resumed._experiment_fingerprint_components))
    if change == "nest_position":
        components["experiment_identity"]["domains"][1]["i_parent_start"] += 1
    elif change == "inputs":
        components["inputs"] = "different-prepared-bytes"
    resumed._experiment_fingerprint_components = components
    resumed.experiment_fingerprint = hashlib.sha256(json.dumps(
        components, sort_keys=True).encode()).hexdigest()
    before = resumed.root.state.thp.copy()

    with pytest.raises(restart.RestartMismatchError) as refused:
        restart.restore_tree_restart(root_path, resumed)

    message = str(refused.value)
    if change == "dt":
        # The member's own config walk names the field first.
        assert "dt: restart=10.0 run=20.0" in message
    else:
        named = ("experiment_identity" if change == "nest_position"
                 else "inputs")
        assert f"{named} differ(s) from the checkpoint" in message
        assert "output-only diagnostic switches" in message
    assert np.array_equal(resumed.root.state.thp, before, equal_nan=True)


# ---------------------------------------------------------------------------
# Window bit-identity: fold half, checkpoint, restore, fold the rest
# ---------------------------------------------------------------------------
#
# The claim that lets a window be persisted at all: its content is a
# max-fold since the consumer's last reset, and max is associative,
# commutative and rounding-free on float32, so folding the persisted
# window and then the remaining steps is BITWISE the unbroken fold.  A
# window that came back merely close would move the next spawn/retire/
# follow decision, which is a different forecast, not a rounding note.


def _uh_fold_steps(shape, seed: int, count: int):
    """Deterministic ``(uh, use)`` pairs, one per model step.

    Signed zero and a denormal in every plane, because the claim under
    test is about BITS and float32 has values that survive arithmetic and
    not serialization if anything in the path is careless.
    """
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(count):
        uh = rng.standard_normal(shape).astype(np.float32) * np.float32(140.0)
        flat = uh.reshape(-1)
        flat[0] = np.float32(-0.0)
        flat[-1] = np.float32(1.0e-42)
        steps.append((uh, rng.random(shape) > 0.2))
    return steps


def _fold_window(window, steps) -> None:
    """The model's own fold: one 9-point smooth into a running max."""
    from woof.core import uh_diag

    for uh, use in steps:
        uh_diag.apply_uh_smoother_max_np(uh, use, window)


def _window_shape(node):
    return (int(node.cfg.run.ny), int(node.cfg.run.nx))


def test_a_split_window_fold_is_bitwise_the_unbroken_fold(
        monkeypatch, tmp_path):
    """Every window a lifecycle checkpoint carries, both families."""
    model, start = _lifecycle_tree_fixture(monkeypatch, payload_seed=31)
    root = model.root
    shape = _window_shape(root)
    slots = restart.lifecycle_window_slots(root.state)
    assert slots == ("uh_follow_window", "uh_follow_window.d02",
                     "uh_spawn_window")

    # A different fold per slot, so a test that crossed two planes would
    # notice rather than compare a window against its own twin.
    steps = {slot: _uh_fold_steps(shape, 4703 + index, 8)
             for index, slot in enumerate(slots)}
    unbroken = {}
    for slot in slots:
        plane = np.zeros(shape, dtype=np.float32)
        _fold_window(plane, steps[slot])
        unbroken[slot] = plane
        assert plane.any(), "an all-zero fold proves nothing"

    for slot in slots:
        _fold_window(root.state.existing_scratch(slot), steps[slot][:4])
    half = {slot: root.state.existing_scratch(slot).copy() for slot in slots}
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    resumed, _ = _lifecycle_tree_fixture(monkeypatch, payload_seed=91)
    for slot in slots:
        assert not resumed.root.state.existing_scratch(slot).any(), \
            "preparation must leave the window empty, or the restore is " \
            "not what carried the number across"
    restart.restore_tree_restart(root_path, resumed)

    for slot in slots:
        plane = resumed.root.state.existing_scratch(slot)
        assert plane.tobytes() == half[slot].tobytes(), \
            f"{slot} did not round-trip through the checkpoint bit for bit"
        _fold_window(plane, steps[slot][4:])
        assert plane.tobytes() == unbroken[slot].tobytes(), \
            f"{slot}: half + half is not the unbroken fold"


def test_a_child_domains_windows_round_trip_on_their_own_member(
        monkeypatch, tmp_path):
    """Each domain's windows ride ITS member, so two domains folding
    different weather cannot be restored into each other."""
    model, start = _lifecycle_tree_fixture(monkeypatch, payload_seed=31)
    root, child = tuple(model.walk_parent_first())
    shape = _window_shape(child)
    steps = _uh_fold_steps(shape, 8821, 6)
    _fold_window(child.state.existing_scratch("uh_spawn_window"), steps[:3])
    _fold_window(root.state.existing_scratch("uh_spawn_window"),
                 _uh_fold_steps(shape, 9932, 6)[:3])
    unbroken = np.zeros(shape, dtype=np.float32)
    _fold_window(unbroken, steps)

    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    resumed, _ = _lifecycle_tree_fixture(monkeypatch, payload_seed=91)
    restart.restore_tree_restart(root_path, resumed)

    _, resumed_child = tuple(resumed.walk_parent_first())
    plane = resumed_child.state.existing_scratch("uh_spawn_window")
    _fold_window(plane, steps[3:])
    assert plane.tobytes() == unbroken.tobytes()


def test_a_streamed_parents_published_window_is_the_one_that_round_trips(
        monkeypatch, tmp_path):
    """The whole point of the publish hook: the plane that reaches the
    file is the STORE's, so the fold a streamed run actually performed is
    the fold the resumed run continues."""
    model, start = _lifecycle_tree_fixture(
        monkeypatch, follow=False, payload_seed=31)
    root = model.root
    shape = _window_shape(root)
    steps = _uh_fold_steps(shape, 5150, 8)
    unbroken = np.zeros(shape, dtype=np.float32)
    _fold_window(unbroken, steps)

    # The sweep folded into the STORE; the DomainState stopped changing
    # at attach and still holds preparation's zeros.
    store_plane = np.zeros(shape, dtype=np.float32)
    _fold_window(store_plane, steps[:4])
    _FakeStreamedDomain({
        "scratch/uh_spawn_window": store_plane,
        "scratch/uh_follow_window": np.zeros(shape, dtype=np.float32),
    }).attach_to(root.state)
    assert not root.state.existing_scratch("uh_spawn_window").any()

    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    resumed, _ = _lifecycle_tree_fixture(
        monkeypatch, follow=False, payload_seed=91)
    restart.restore_tree_restart(root_path, resumed)
    plane = resumed.root.state.existing_scratch("uh_spawn_window")
    assert plane.tobytes() == store_plane.tobytes()
    _fold_window(plane, steps[4:])
    assert plane.tobytes() == unbroken.tobytes()


def test_a_window_folded_in_three_pieces_is_still_the_unbroken_fold(
        monkeypatch, tmp_path):
    """N segments, not two: the associativity claim has to hold for the
    restart cadence a real run uses, where a window spans many files."""
    steps = None
    unbroken = None
    carried = None
    for index, (lo, hi) in enumerate(((0, 3), (3, 7), (7, 12))):
        model, start = _lifecycle_tree_fixture(
            monkeypatch, follow=False, payload_seed=31 + index)
        plane = model.root.state.existing_scratch("uh_spawn_window")
        if steps is None:
            steps = _uh_fold_steps(_window_shape(model.root), 6161, 12)
            unbroken = np.zeros(_window_shape(model.root), dtype=np.float32)
            _fold_window(unbroken, steps)
        if carried is not None:
            restart.restore_tree_restart(carried, model)
        _fold_window(plane, steps[lo:hi])
        carried = restart.write_tree_restart(
            tmp_path / f"segment-{index}", model,
            start + timedelta(seconds=3600))

    assert plane.tobytes() == unbroken.tobytes()


# ---------------------------------------------------------------------------
# The nest-lifecycle block on a tree checkpoint (the read side: the peek)
# ---------------------------------------------------------------------------
#
# The peek is what a resume does BEFORE it rebuilds a single domain: one
# JSON member off the root, validated against the experiment that is about
# to be built.  Every refusal here names the concrete breakage, because
# each one is a run that would otherwise integrate from a policy state it
# only half understands.


def _lifecycle_free_model(monkeypatch):
    """A tree with no spawn runner and no follower: the compat baseline."""
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    return model, start


def _retune_lifecycle_block(root_path, out_dir, mutate):
    """Rewrite the root member's lifecycle block, keeping its filename."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def edit(payload, header):
        del payload
        mutate(header[restart.NEST_LIFECYCLE_HEADER_KEY])

    return _rewrite_restart_archive(
        root_path, out_dir / root_path.name, edit)


def test_the_peek_walks_to_the_root_when_handed_a_child_member(
        monkeypatch, tmp_path):
    """A supervisor hands whichever member it found.  The block is a
    whole-tree fact and rides the root alone, so the reader walks."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))

    peek = restart.read_tree_lifecycle_header(child_path, model)
    assert peek.root_path == root_path
    assert peek.spawn["spawns_executed"] == 1


def test_a_checkpoint_without_the_block_refuses_a_lifecycle_run(
        monkeypatch, tmp_path):
    """The named conversion of today's SILENT divergence: a spawn-only
    checkpoint written before this block existed cannot say which slots
    fired, so restoring it re-fires them at the next boundary."""
    free, start = _lifecycle_free_model(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, free, start + timedelta(seconds=3600))
    live, _ = _lifecycle_tree_fixture(monkeypatch)

    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(root_path, live)
    message = str(caught.value)
    assert "predates lifecycle persistence" in message
    assert "which slots fired" in message
    assert "t=0" in message


def test_a_lifecycle_checkpoint_refuses_a_lifecycle_free_run(
        monkeypatch, tmp_path):
    """The other direction is just as wrong: the checkpoint carries policy
    state this run has nowhere to put, so the slots it names would never
    be consulted again."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    free, _ = _lifecycle_free_model(monkeypatch)

    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(root_path, free)
    assert "declares no dormant nest and no follower" in str(caught.value)


def test_a_lifecycle_free_checkpoint_into_a_lifecycle_free_run_peeks_none(
        monkeypatch, tmp_path):
    """The compatibility contract, from the reader's side."""
    free, start = _lifecycle_free_model(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, free, start + timedelta(seconds=3600))

    peek = restart.read_tree_lifecycle_header(root_path, free)
    assert peek.block is None
    assert peek.spawn is None and peek.followers == {}
    assert peek.relocation_records == ()


def test_the_reader_and_the_writer_share_one_compatibility_predicate():
    """Two spellings of "does this run have a lifecycle" drift, and the
    drift is a checkpoint one side writes and the other refuses."""
    assert restart._declares_nest_lifecycle is restart.declares_nest_lifecycle


def test_an_unknown_contract_refuses_by_name(monkeypatch, tmp_path):
    """A block half-understood is a slot that silently re-fires."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    def bump(block):
        block["contract"] = "gpuwm-nest-lifecycle-restart.v99"

    tampered = _retune_lifecycle_block(root_path, tmp_path / "v99", bump)
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(tampered, model)
    message = str(caught.value)
    assert "v99" in message and restart.NEST_LIFECYCLE_CONTRACT in message


def test_a_block_with_an_unknown_key_refuses(monkeypatch, tmp_path):
    """Read in part is not read: the entry is honored whole or refused."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    tampered = _retune_lifecycle_block(
        root_path, tmp_path / "extra",
        lambda block: block.update({"hysteresis": {}}))
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(tampered, model)
    assert "hysteresis" in str(caught.value)


def test_a_live_episode_with_no_member_refuses(monkeypatch, tmp_path):
    """The block says d03 is a live episode; the checkpoint carries no
    d03 member, so its arrays exist nowhere and the rebuilt nest would
    integrate from the parent interpolation for the rest of the run."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    def rename(block):
        block["spawn"]["spawned"]["3"] = block["spawn"]["spawned"].pop("2")

    tampered = _retune_lifecycle_block(root_path, tmp_path / "ghost", rename)
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(tampered, model)
    message = str(caught.value)
    assert "d03" in message and "member set" in message


def test_a_retired_slot_that_is_still_in_the_member_set_refuses(
        monkeypatch, tmp_path):
    """Retirement is leg-boundary schedule surgery: a retired subtree is
    detached before any later checkpoint, so a retired slot that still
    owns a member means the two halves describe different trees."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    def retire_the_live_one(block):
        block["spawn"]["spawned"].pop("2")
        block["spawn"]["retired"]["2"] = {"retired_t": 900.0, "episode": 1}

    tampered = _retune_lifecycle_block(
        root_path, tmp_path / "retired", retire_the_live_one)
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(tampered, model)
    message = str(caught.value)
    assert "d02" in message and "retired" in message


def test_a_window_audit_naming_an_absent_domain_refuses(
        monkeypatch, tmp_path):
    """The audit half of the window opt-in has to address the member set
    it was written beside, or "the window was written and it is empty" is
    indistinguishable from "no window was written"."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    tampered = _retune_lifecycle_block(
        root_path, tmp_path / "audit",
        lambda block: block["window_slots"].update({"7": ["uh_spawn_window"]}))
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(tampered, model)
    assert "d07" in str(caught.value)


def test_a_leg_cadence_mismatch_refuses_and_names_both(
        monkeypatch, tmp_path):
    """Every lifecycle decision is taken at a leg boundary, so two runs on
    different lattices take them at different instants -- and the resumed
    run's trajectory is then no continuation of the one on disk."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    live, _ = _lifecycle_tree_fixture(monkeypatch)
    live._spawn_leg_seconds = 900.0

    with pytest.raises(restart.RestartMismatchError) as caught:
        restart.read_tree_lifecycle_header(root_path, live)
    message = str(caught.value)
    assert "360" in message and "900" in message


def test_a_leg_cadence_absent_on_either_side_is_tolerated(
        monkeypatch, tmp_path):
    """``None`` says a route ran no leg walk and therefore states no
    lattice.  There is nothing to compare, so nothing is refused; the
    writer already refused a non-finite or zero cadence."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    model._spawn_leg_seconds = None
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    live, _ = _lifecycle_tree_fixture(monkeypatch)

    assert restart.read_tree_lifecycle_header(
        root_path, live).block["leg_seconds"] is None
    live._spawn_leg_seconds = None
    assert restart.read_tree_lifecycle_header(root_path, live) is not None


def test_the_peek_carries_the_relocation_record_chain(monkeypatch, tmp_path):
    """The fingerprint a moved run's checkpoint is keyed to is the chain
    folded over these digests; the resume needs them before it builds."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    model._relocation_receipts = (
        {"event": "relocated", "record_sha256": "a" * 64,
         "segment_id": "seg-1", "grid_id": 2},
        {"event": "relocated", "record_sha256": "b" * 64,
         "segment_id": "seg-1", "grid_id": 2},
    )
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    peek = restart.read_tree_lifecycle_header(root_path, model)
    assert peek.relocation_records == ("a" * 64, "b" * 64)
    assert peek.relocation["moves"] == 2


def test_a_resumed_runs_checkpoint_carries_the_WHOLE_move_chain(
        monkeypatch, tmp_path):
    """The chain is the RUN's, not the segment's.

    A resumed run's live fingerprint is the fresh build folded over every
    move the whole run has made.  A checkpoint that listed only this
    segment's moves could not be resumed from at all: folding a short
    chain reproduces a different value, so the third segment of a moving
    run would refuse its own predecessor."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    # What restore_tree_restart left behind for this run's own writer.
    model._restart_crossed_relocation = {
        "moves": 2, "record_sha256": ["a" * 64, "b" * 64],
        "segment_id": "seg-1", "grid_id": 2, "posture": "..."}
    model._relocation_receipts = (
        {"event": "relocated", "record_sha256": "c" * 64,
         "segment_id": "seg-2", "grid_id": 2},)
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    peek = restart.read_tree_lifecycle_header(root_path, model)
    assert peek.relocation_records == ("a" * 64, "b" * 64, "c" * 64)
    assert peek.relocation["moves"] == 3
    # The newest executed move names the segment, not the inherited one.
    assert peek.relocation["segment_id"] == "seg-2"


def test_a_resumed_run_that_has_not_moved_still_states_its_inheritance(
        monkeypatch, tmp_path):
    """A segment that executes no move of its own still carries the chain
    its fingerprint is folded over, or its own checkpoints are keyed to a
    history the next segment cannot reproduce."""
    model, start = _lifecycle_tree_fixture(monkeypatch)
    model._restart_crossed_relocation = {
        "moves": 1, "record_sha256": ["a" * 64], "segment_id": "seg-1",
        "grid_id": 2, "posture": "..."}
    root_path = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    peek = restart.read_tree_lifecycle_header(root_path, model)
    assert peek.relocation_records == ("a" * 64,)
    assert peek.relocation["segment_id"] == "seg-1"


def _grell_freitas_tree(monkeypatch, *, seed=61):
    """Two ``cu_physics = 3`` domains on the CPU shim, legal for
    :func:`write_tree_restart`.  No device: the manifest walk that refused
    the real run is pure Python."""
    from woof.core.gf import GrellFreitas
    from woof.core.model import ModelRuntimeStatus

    def build(grid_id, nested, payload):
        cfg = _cfg(grid_id=grid_id, nested=nested, moist=True,
                   cu_physics=3, cudt_minutes=0.0, run_seconds=60.0)
        state, driver = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(state)
        _fill_serialized(state, payload)
        state._nest_restart_classification = "REBUILT"
        adapter = GrellFreitas()
        # Exactly what PhysicsDriver._run_cumulus does on every cumulus
        # step, and therefore the state every real checkpoint sees.
        adapter.bind_driver(driver)
        driver.cumulus_callable = adapter
        return cfg, state

    def clock(dtbc):
        return SimpleNamespace(
            ticks=0, tick_den=1, step_count=0,
            dtbc_fp32=np.float32(dtbc),
            spec=SimpleNamespace(start_ticks=0, step_ticks=1))

    root_cfg, root_state = build(1, False, seed)
    child_cfg, child_state = build(2, True, seed + 1)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, run=root_cfg,
                            start_time=None),
        state=root_state, clock=clock(7.0), parent=None, children=[],
        coupler=None, _started=True)
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, run=child_cfg,
                            start_time=None),
        state=child_state, clock=clock(9.0), parent=root, children=[],
        coupler=None, _started=True)
    root.children.append(child)
    return SimpleNamespace(
        root=root, experiment_fingerprint="gf-tree-fixture-v1",
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(tick_den=1, run_ticks=60,
                                  start_time=datetime(1974, 4, 3, 12))),
        walk_parent_first=lambda: iter((root, child)),
        _runtime_status=ModelRuntimeStatus(), _io_manager=None,
        _last_checkpoint=None, _resumed=False,
        _resume_committed_history_grid_ids=frozenset())


def test_tree_checkpoint_publishes_with_grell_freitas_on_every_domain(
        monkeypatch, tmp_path):
    """The path the real run actually died on, reproduced without a device.

    The production failure was a TREE checkpoint of two GF domains --
    ``write_tree_restart`` -> ``write_restart`` -> ``_driver_manifest`` ->
    ``_callable_state_check`` -- and the tree writer publishes nothing
    unless every member writes, so one unclassified attribute on one
    domain refused the entire generation and unlinked what it had already
    written.  The single-domain bitwise gate cannot see that: it never
    calls the tree writer.

    Both adapters are bound the way ``_run_cumulus`` binds them, because
    an unbound adapter is exactly the state that WOULD have passed.
    """
    model = _grell_freitas_tree(monkeypatch)
    for node in model.walk_parent_first():
        assert node.state.physics.cumulus_callable._driver is \
            node.state.physics

    root_path = restart.write_tree_restart(
        tmp_path, model, datetime(1974, 4, 3, 15))

    published = sorted(path.name for path in tmp_path.glob("gpuwmrst_*.npz"))
    assert len(published) == 2, published
    assert root_path.exists()
    for path in tmp_path.glob("gpuwmrst_*.npz"):
        header = restart.read_restart_header(path)
        assert header["physics_setup"]["cumulus"]["callable"] == {
            "class": "woof.core.gf.GrellFreitas",
            "implementation": "stock"}
        # GF ships no coefficient table; the KF asset must not be bound
        # into a GF manifest.
        assert header["physics_setup"]["cumulus"]["coefficient_table"] is None
        assert "cumulus/w0avg" not in header["array_manifest"]

    # And the negative, on the same tree: restore the pre-fix
    # classification and the whole generation is refused, publishing
    # nothing -- the exact production failure, message included.
    refused = _grell_freitas_tree(monkeypatch, seed=71)
    monkeypatch.setattr(restart, "CUMULUS_CALLABLE_CONTAINERS",
                        frozenset({"_history_state"}))
    empty = tmp_path / "refused"
    with pytest.raises(restart.RestartManifestError,
                       match=r"'_driver' is an object container"):
        restart.write_tree_restart(empty, refused, datetime(1974, 4, 3, 15))
    assert not list(empty.glob("gpuwmrst_*.npz"))


def test_sealed_tree_writer_preflights_all_members_before_publication(
        monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=51)
    for node in source.walk_parent_first():
        node.clock.ticks = 3500

    with pytest.raises(restart.RestartMismatchError, match="not sealed"):
        restart.write_tree_restart(
            tmp_path, source, start + timedelta(seconds=3600),
            sealed_forcing_extension=True)

    assert not list(tmp_path.glob("gpuwmrst_*.npz"))


def _rewrite_header(path, output, mutate) -> Path:
    """Rewrite the JSON header only, leaving every array member intact."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    mutate(header)
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _external_mirror(*, bound: bool):
    """The binding-status surface the restart identity rule consults.

    ``attach_lateral_boundaries`` leaves ``clock=None`` (unbound legacy
    consumption); the production tree build assigns the root DomainClock
    (bound WRF post-increment consumption)."""
    return SimpleNamespace(rolling=False,
                           clock=object() if bound else None)


def _external_boundaries(seed: int):
    from woof.ingest.lateral_bc import build_lateral_boundaries

    rng = np.random.default_rng(seed)
    snapshots = [{"u": rng.standard_normal((8, 8))} for _ in range(2)]
    return build_lateral_boundaries(
        snapshots, [0.0, 3600.0], spec_bdy_width=3, spec_zone=1,
        relax_zone=2)


def test_specified_restart_header_carries_root_lbc_clock_identity(
        monkeypatch, tmp_path):
    """Davies-bind restart epoch (dossier section 6): a specified domain's
    checkpoint records WHICH root external-LBC clock semantic integrated
    its trajectory.  Binding is invisible to the config echo, the setup
    fingerprint, and the physics identity, so without this key an old
    elapsed-based (pre-bind) checkpoint would silently resume under
    WRF-post-increment semantics -- a different trajectory."""
    cfg = _cfg(moist=True, mp_physics=1, specified=True)

    bound = _shim_state(cfg, monkeypatch)
    _fill_setup(bound)
    bound.lateral_boundaries = _external_boundaries(1)
    bound._lateral_boundary_device = _external_mirror(bound=True)
    bound_path = restart.write_restart(tmp_path / "bound.npz", bound, cfg)
    header = restart.read_restart_header(bound_path)
    assert header.get("root_external_lbc_clock") == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY
    assert restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY == "wrf-postincrement-v1"

    unbound = _shim_state(cfg, monkeypatch)
    _fill_setup(unbound)
    unbound.lateral_boundaries = _external_boundaries(1)
    unbound._lateral_boundary_device = _external_mirror(bound=False)
    legacy_path = restart.write_restart(
        tmp_path / "legacy.npz", unbound, cfg)
    legacy_header = restart.read_restart_header(legacy_path)
    assert legacy_header.get("root_external_lbc_clock") == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY
    assert restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY == "legacy-elapsed-v0"

    # Non-specified domains have no external Davies consumer: no key.
    plain_cfg = _cfg(moist=True, mp_physics=1)
    plain = _shim_state(plain_cfg, monkeypatch)
    _fill_setup(plain)
    plain_path = restart.write_restart(
        tmp_path / "plain.npz", plain, plain_cfg)
    assert "root_external_lbc_clock" not in restart.read_restart_header(
        plain_path)


def test_restore_refuses_root_lbc_clock_semantic_mismatch(
        monkeypatch, tmp_path):
    """Old unbound checkpoints fail closed under bound production code,
    and post-bind checkpoints fail closed on legacy unbound paths.  A
    header with NO identity key is a pre-epoch file and is treated as
    legacy-elapsed (the semantics every pre-bind writer integrated)."""
    cfg = _cfg(moist=True, mp_physics=1, specified=True)

    def specified_state(*, bound: bool):
        state = _shim_state(cfg, monkeypatch)
        _fill_setup(state)
        state.lateral_boundaries = _external_boundaries(1)
        state._lateral_boundary_device = _external_mirror(bound=bound)
        return state

    bound_path = restart.write_restart(
        tmp_path / "bound.npz", specified_state(bound=True), cfg)
    legacy_path = restart.write_restart(
        tmp_path / "legacy.npz", specified_state(bound=False), cfg)
    pre_epoch = _rewrite_header(
        bound_path, tmp_path / "pre-epoch.npz",
        lambda header: header.pop("root_external_lbc_clock"))

    # Matching semantics restore: bound file -> bound resuming state.
    # (The shim mirror satisfies the attach requirement; the resident
    # device tables themselves are rebuilt by preparation, not restored.)
    restart.restore_restart(bound_path, specified_state(bound=True), cfg)
    # Legacy file -> legacy resuming state also remains valid.
    restart.restore_restart(legacy_path, specified_state(bound=False), cfg)

    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            legacy_path, specified_state(bound=True), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            pre_epoch, specified_state(bound=True), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            bound_path, specified_state(bound=False), cfg)


def test_truncated_restart_file_is_rejected_loudly(monkeypatch, tmp_path):
    """Review F4: corrupt/truncated archives raise a gpuwm-branded error
    naming the file and the likely cause, and writes publish atomically
    (no .tmp residue, never a truncated file under the final name)."""
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)
    assert not list(tmp_path.glob("*.tmp"))

    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(path.read_bytes()[:256])
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.read_restart_header(truncated)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.restore_restart(truncated, fresh, cfg)


def test_tree_restore_corrupt_later_member_leaves_every_domain_untouched(
        monkeypatch, tmp_path):
    """Shadow #2: d01 may be valid while a d02 payload member has a bad
    CRC.  Its header remains readable, but complete-set validation must refuse
    before applying even one d01 byte or changing any clock/coupler metadata.
    """
    cfg1 = _cfg(grid_id=1, moist=False, run_seconds=60.0)
    cfg2 = _cfg(grid_id=2, nested=True, moist=False, run_seconds=60.0)

    def state(cfg, seed):
        value = _shim_state(cfg, monkeypatch)
        value._nest_restart_classification = "REBUILT"
        _fill_setup(value)
        _fill_serialized(value, seed)
        return value

    source1, source2 = state(cfg1, 1), state(cfg2, 2)
    common = {
        "experiment_fingerprint": "complete-set-fixture",
        "checkpoint_set_id": "one-set",
        "domain_ids": [1, 2],
        "elapsed_ticks": 0,
        "tick_den": 1,
        "phase": "PERIOD_BEGIN",
        "nest_tables": "REBUILT",
        "dtbc_fp32_bits": int(np.float32(0.0).view(np.uint32)),
    }
    path1 = restart.write_restart(
        tmp_path / "gpuwmrst_d01_1982-05-20_00_00_00.npz",
        source1, cfg1,
        tree_header={**common, "grid_id": 1, "parent_id": 0})
    path2 = restart.write_restart(
        tmp_path / "gpuwmrst_d02_1982-05-20_00_00_00.npz",
        source2, cfg2,
        tree_header={**common, "grid_id": 2, "parent_id": 1})

    # Flip one stored member byte without rewriting its ZIP CRC. np.load can
    # still read the independent JSON header member, but loading state/u fails.
    with zipfile.ZipFile(path2) as archive:
        member = archive.getinfo("state/u.npy")
    raw = bytearray(path2.read_bytes())
    local = raw[member.header_offset:member.header_offset + 30]
    fields = struct.unpack("<IHHHHHIIIHH", local)
    payload_start = member.header_offset + 30 + fields[-2] + fields[-1]
    raw[payload_start + member.file_size // 2] ^= 0x01
    path2.write_bytes(raw)
    assert restart.read_restart_header(path2)["grid_id"] == 2

    live1, live2 = state(cfg1, 11), state(cfg2, 12)

    class Coupler:
        valid = True

        def invalidate(self):
            self.valid = False

    root_cfg = SimpleNamespace(grid_id=1, parent_id=0, run=cfg1)
    child_cfg = SimpleNamespace(grid_id=2, parent_id=1, run=cfg2)
    clock1 = SimpleNamespace(
        ticks=0, tick_den=1, step_count=0, dtbc_fp32=np.float32(7.0),
        spec=SimpleNamespace(step_ticks=1))
    clock2 = SimpleNamespace(
        ticks=0, tick_den=1, step_count=0, dtbc_fp32=np.float32(9.0),
        spec=SimpleNamespace(step_ticks=1))
    root = SimpleNamespace(
        cfg=root_cfg, state=live1, clock=clock1, parent=None, coupler=None)
    child = SimpleNamespace(
        cfg=child_cfg, state=live2, clock=clock2, parent=root,
        coupler=Coupler())

    model = SimpleNamespace(
        experiment_fingerprint="complete-set-fixture", root=root,
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(tick_den=1, run_ticks=60)),
        walk_parent_first=lambda: iter((root, child)))

    before = {
        gid: {name: getattr(value, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(value, name, None) is not None}
        for gid, value in ((1, live1), (2, live2))}
    metadata = (clock1.ticks, clock1.step_count, clock1.dtbc_fp32.tobytes(),
                clock2.ticks, clock2.step_count, clock2.dtbc_fp32.tobytes(),
                child.coupler.valid, live1.elapsed_seconds,
                live2.elapsed_seconds)
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.restore_tree_restart(path1, model)

    after = {
        gid: {name: getattr(value, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(value, name, None) is not None}
        for gid, value in ((1, live1), (2, live2))}
    assert after == before
    assert metadata == (
        clock1.ticks, clock1.step_count, clock1.dtbc_fp32.tobytes(),
        clock2.ticks, clock2.step_count, clock2.dtbc_fp32.tobytes(),
        child.coupler.valid, live1.elapsed_seconds, live2.elapsed_seconds)


def test_failed_write_leaves_no_residue(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)

    def explode(path, *args, **kwargs):
        Path(path).write_bytes(b"PK\x03\x04 partial")     # a half-written archive
        raise OSError("disk full")

    monkeypatch.setattr(restart, "write_npz", explode)
    with pytest.raises(OSError, match="disk full"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)
    assert list(tmp_path.iterdir()) == []


def test_read_header_rejects_non_restart_npz(tmp_path):
    path = tmp_path / "plain.npz"
    with path.open("wb") as stream:
        np.savez(stream, some_array=np.zeros(3))
    with pytest.raises(restart.RestartMismatchError, match="header"):
        restart.read_restart_header(path)


def test_restart_filename_is_wrfrst_style():
    assert (restart.restart_filename(datetime(1974, 4, 3, 18))
            == "gpuwmrst_d01_1974-04-03_18_00_00.npz")


# ---------------------------------------------------------------------------
# Config knob and CLI plumbing (CPU).
# ---------------------------------------------------------------------------

def test_restart_interval_validation():
    from woof.config import load_config
    from woof.verify.cases.real74_d01 import (_restart_outer_steps,
                                               phase3_config)

    cfg = phase3_config()
    assert _restart_outer_steps(cfg) is None                    # default off
    assert _restart_outer_steps(
        replace(cfg, restart_interval_s=21600.0)) == 360
    with pytest.raises(ValueError, match="restart_interval_s"):
        _restart_outer_steps(replace(cfg, restart_interval_s=90.5))

    import tempfile
    text = (REPO_ROOT / "configs" / "real74_d01.toml").read_text(
        encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.toml"
        bad.write_text(text + "\nrestart_interval_s = -5.0\n",
                       encoding="utf-8")
        with pytest.raises(ValueError, match="restart_interval_s"):
            load_config(bad)
        good = Path(tmp) / "good.toml"
        good.write_text(text + "\nrestart_interval_s = 21600.0\n",
                        encoding="utf-8")
        assert load_config(good).restart_interval_s == 21600.0


# NEEDS CUPY INSTALLED, and opens no device: this test runs `woof run
# --restart`; without cupy the door refuses and returns 2 before the restart
# argument is read.
@requires_cupy
def test_cli_run_passes_restart_through(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    import woof.cli as cli

    calls = []
    cfg = SimpleNamespace(case="real74_d01")
    fake = SimpleNamespace(
        run_config=lambda loaded, outdir, restart=None: calls.append(
            (loaded, outdir, restart)) or SimpleNamespace(
                wrfout_paths=(), completed_seconds=43200.0, nan_free=True))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setitem(cli._REAL_CASES, "real74_d01", fake)
    restart_file = tmp_path / "gpuwmrst_d01_1974-04-03_18_00_00.npz"
    # The config path must EXIST even though `load_config` is mocked:
    # a4417efb refuses a path that is not a readable regular file before
    # anything opens it, so a name that was never written now exits 2
    # ("does not exist; pass the experiment .toml that `woof domain`
    # wrote") without reaching the loader this test replaces.  A legacy
    # [run]-shaped body keeps the CLI on the registered-case branch the
    # rest of the test asserts on.
    config = tmp_path / "cfg.toml"
    config.write_text("[run]\ncase = \"real74_d01\"\n", encoding="utf-8")

    assert cli.main(["run", str(config),
                     "--outdir", str(tmp_path / "run"),
                     "--restart", str(restart_file)]) == 0
    assert calls == [(cfg, tmp_path / "run", restart_file)]
    assert "'restarted': True" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# GPU: live manifest coverage and the cheap bit-identity gate.
# ---------------------------------------------------------------------------

def _physics_state(cp, **cfg_overrides):
    """Small full-physics state: Morrison + analytic radiation + KF +
    SFCLAY/Noah/YSU, mixed land/water, with a condensate layer so the
    microphysics accumulators and h_diabatic are non-trivial.

    ``cp`` is the caller's cupy module, passed in rather than imported here.
    See :func:`_grell_freitas_state` for why that matters.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    values = dict(nx=6, ny=4, nz=16, dx=2000.0, dy=2000.0,
                  ztop=8000.0, dt=10.0, run_seconds=0.0,
                  time_step_sound=4, moist=True, mp_physics=10,
                  sf_sfclay_physics=1, sf_surface_physics=2,
                  bl_pbl_physics=1, ra_physics=90, cu_physics=1,
                  radt_minutes=1.0, cudt_minutes=1.0, bldt=0.0)
    values.update(cfg_overrides)
    cfg = RunConfig(**values)
    coord = make_vertical_coord(cfg.nz)
    def theta(z):
        return 298.0 + 0.004 * np.asarray(z, np.float64)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.011 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    state.u[...] = cp.float32(6.0)
    state.v[...] = cp.float32(0.5)
    state.qc[2:6] = cp.float32(4.0e-4)       # cloud layer -> rain/heating

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 299.0)
    tsk[:, -1] = 296.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.31)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=np.where(landmask, 10, 17),
        isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=286.0,
        swdown=450.0, glw=310.0, pblh=700.0,
        radiation_start_time=datetime(1974, 4, 3, 12),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 39.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.0))
    return state, cfg, driver


def _grell_freitas_state(cp):
    """Small state in which ``cu_physics=3`` actually convects.

    Deliberately NOT ``_physics_state`` with an override: that state's 2 km
    grid and 8 km lid are a configuration Grell-Freitas declines (the
    ``sig = (1-frh)^2`` scale taper, and no room above the LFC), and a
    restart gate for a scheme that never triggers would compare nothing but
    zeros on the axis it exists to test.  This is the production family --
    12 km, RTE+RRTMGP, Morrison, YSU, ``cudt_minutes = 0`` -- over a warm,
    conditionally unstable, deep-moist column that GF fires on within the
    first ten steps and is still firing at step 40.  The 14 km lid is the
    RRTMGP temperature-range floor, not a physics choice.

    ``cp`` is the caller's cupy module.  It is a parameter, not an ``import
    cupy`` in this body, and that is essential: conftest marks a module
    ``gpu`` *in its entirety* when any non-test function imports cupy,
    because a helper's callers are not decidable from the AST.  These three
    state builders are called only from the five explicitly
    ``@pytest.mark.gpu`` tests below, but the coarse rule cannot know that,
    so importing cupy here swept the module's other 91 device-free tests --
    every restart-manifest classification gate, including the two covering
    the Grell-Freitas manifest fix -- out of ``-m "not gpu"`` and made them
    unrunnable without a card.  Keeping the import in the callers puts the
    device contact where it actually happens.  Do not "tidy" it back.
    """
    from woof.config import validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = validate_run_config(RunConfig(
        nx=8, ny=6, nz=40, dx=12000.0, dy=12000.0, ztop=14000.0,
        dt=60.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        # radt 6 min on a 60 s step -> stepra 6, so radiation is due at
        # itimestep % 6 == 1 and the step-20 checkpoint is OFF its calendar.
        ra_physics=4, radt_minutes=6.0, bldt=0.0,
        cu_physics=3, cudt_minutes=0.0, ishallow=1))
    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        # Well-mixed to 2 km, then conditionally unstable.
        z = np.asarray(z, np.float64)
        return 300.0 + 0.0025 * np.maximum(z - 2000.0, 0.0)

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.019 * np.exp(-np.asarray(z, np.float64) / 5000.0))
    state.u[...] = cp.float32(4.0)
    state.v[...] = cp.float32(1.0)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0                      # one water column: XLAND varies
    tsk = np.full((cfg.ny, cfg.nx), 308.0)
    tsk[:, -1] = 303.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.35)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=np.where(landmask, 10, 17),
        isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=292.0,
        swdown=850.0, glw=380.0, pblh=1200.0,
        radiation_start_time=datetime(1974, 4, 3, 18),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 39.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.0))
    return state, cfg, driver


def _thompson_restart_state(cp):
    """Small guarded-mp8 state with all classic hydrometeor groups active.

    ``cp`` is the caller's cupy module.  See :func:`_grell_freitas_state`.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.core.state import init_theta_perturbation

    cfg = RunConfig(
        nx=16, ny=12, nz=24, dx=1000.0, dy=1000.0, ztop=12000.0,
        dt=2.0, run_seconds=40.0, time_step_sound=4,
        moist=True, mp_physics=8)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, dtype=float),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_theta_perturbation(
        cfg, coord, base,
        lambda x, z: 0.20 * np.exp(-(np.asarray(z) / 4500.0) ** 2)[
            :, None, None])

    oracle = (REPO_ROOT / "woof" / "data" / "thompson" / "oracle" /
              "mixed-column.csv")
    with oracle.open(newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["phase"] == "before"]

    def profile(name):
        return np.asarray([float(row[name]) for row in rows], np.float32)

    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        values = profile(name)[:, None, None]
        getattr(state, name)[...] = cp.asarray(
            np.broadcast_to(values, (cfg.nz, cfg.ny, cfg.nx)))
    state.nr[...] = cp.asarray(np.broadcast_to(
        profile("nr_per_kg")[:, None, None], state.nr.shape))
    state.ni[...] = cp.asarray(np.broadcast_to(
        profile("ni_per_kg")[:, None, None], state.ni.shape))
    state.u[...] = cp.float32(6.0)
    initialize_physics(state, cfg)
    return state, cfg


def _advance_thompson_with_refl(state, cfg, steps):
    """Advance and consume the output-only handoff every fifth step."""
    from woof.core.dycore import step
    from woof.core.refl import consume_refl_10cm

    for _ in range(steps):
        step_index = int(round(state.elapsed_seconds / cfg.dt)) + 1
        due = step_index % 5 == 0
        step(state, cfg, refl_10cm_due=due)
        if due:
            refl = consume_refl_10cm(state)
            assert bool(np.isfinite(refl.get()).all())


def _assert_restart_equal(path_a, path_b, *, compare_trackers=False):
    """Bit-exact equality of two restart files (arrays + relevant header)."""
    with np.load(path_a, allow_pickle=False) as data:
        arrays_a = {key: data[key] for key in data.files}
    with np.load(path_b, allow_pickle=False) as data:
        arrays_b = {key: data[key] for key in data.files}
    header_key = "__gpuwm_restart_header__"
    header_a = json.loads(bytes(bytearray(arrays_a.pop(header_key))))
    header_b = json.loads(bytes(bytearray(arrays_b.pop(header_key))))

    assert set(arrays_a) == set(arrays_b), (
        sorted(set(arrays_a) ^ set(arrays_b)))
    for key in sorted(arrays_a):
        assert arrays_a[key].dtype == arrays_b[key].dtype, key
        np.testing.assert_array_equal(arrays_a[key], arrays_b[key],
                                      err_msg=key)
        assert arrays_a[key].tobytes() == arrays_b[key].tobytes(), key

    assert header_a["elapsed_seconds"] == header_b["elapsed_seconds"]
    assert header_a["setup_fingerprint"] == header_b["setup_fingerprint"]
    assert header_a["physics_setup_fingerprint"] == \
        header_b["physics_setup_fingerprint"]
    assert header_a["physics_setup"] == header_b["physics_setup"]
    assert header_a["driver"] == header_b["driver"]
    assert header_a["array_manifest"] == header_b["array_manifest"]
    for key, value in header_a["config"].items():
        if key in restart.CONFIG_RUN_LENGTH_FIELDS:
            continue
        assert header_b["config"][key] == value, key
    if compare_trackers:
        assert header_a["run_trackers"] == header_b["run_trackers"]


@requires_gpu
@pytest.mark.gpu
def test_live_full_state_write_covers_every_mechanism(tmp_path):
    """write_restart's internal completeness walk passes on a live
    full-physics state, and the manifest carries every audited mechanism."""
    import cupy as cp

    from woof.core.dycore import run_steps

    state, cfg, driver = _physics_state(cp)
    run_steps(state, cfg, 6)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)
    header = restart.read_restart_header(path)
    keys = set(header["array_manifest"])
    for expected in (
            "state/u", "state/thp", "state/php", "state/mup",
            "state/h_diabatic", "state/effc", "state/ng",
            "scratch/mp_rainnc", "scratch/mp_sr",
            "scratch/cu_rainc", "scratch/cu_nca", "scratch/cu_pratec",
            "scratch/cu_raincv", "scratch/cu_rthcuten",
            "driver/rthratenlw", "driver/rthratensw",
            "driver/pending_rainbl",
            "driver/pbl_tendencies/ru", "driver/radiation_tendencies/rtheta",
            "driver/cumulus_tendencies/rqv",
            "fields/tsk", "fields/ust", "fields/mol", "fields/znt",
            "fields/qsfc", "fields/hfx", "fields/qfx", "fields/pblh",
            "fields/sh2o", "fields/snotime", "fields/albedo",
            "fields/emiss", "fields/rainbl",
            "cumulus/w0avg"):
        assert expected in keys, expected
    assert not any(key.startswith("driver/microphysics/") for key in keys)
    assert header["driver"]["microphysics_updates"] == 6
    assert header["driver"]["call_counts"]["radiation"] >= 1


@requires_gpu
@pytest.mark.gpu
def test_short_full_physics_restart_is_bit_identical(tmp_path):
    """Cheap suite-resident gate: 20 steps + restart + 20 steps == 40
    steps, FP32-bit-exact on every serialized field.

    The 20-step boundary is deliberately OFF both physics calendars
    (radiation due at itimestep % 6 == 1 -> 19/25; cumulus due at
    itimestep % 6 == 0 -> 18/24), so the held radiation/cumulus
    tendencies, the KF NCA hold, W0AVG, and _pending_rainbl all span the
    restart boundary and must survive serialization, not recomputation.
    """
    import cupy as cp

    from woof.core.dycore import run_steps

    state_a, cfg_a, _ = _physics_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, _ = _physics_state(cp)
    run_steps(state_b, cfg_b, 20)
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _physics_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)

    _assert_restart_equal(resumed, reference)
    # The comparison must be non-trivial: microphysics ran on every step
    # and produced retained heating; the surface cycle ran.
    assert state_c.physics.microphysics_updates == 40
    assert bool(cp.any(state_c.h_diabatic != 0.0))
    assert state_c.physics.call_counts["radiation"] == 7   # 1, 7, ..., 37
    assert state_c.physics.call_counts["cumulus"] == 7     # 1, 6, ..., 36


def _milbrandt2_state(cp):
    """Small mp=9 state in which Milbrandt-Yau actually carries moments.

    ``_physics_state(cp, mp_physics=9)`` alone is not enough, and the
    reason is the scheme rather than the harness: that state's cloud
    layer carries mass with a droplet NUMBER of zero, and a two-moment
    scheme handed a zero moment under nonzero mass evaporates the mass
    within a few steps -- after 20 steps every one of the twelve moments
    is exactly zero, and a bit comparison over an all-zero inventory
    proves nothing.  So this seeds a mid-forecast state instead of an
    initial one: a droplet-bearing warm layer under an ice layer, in a
    14 km column deep enough for the cold categories to exist.  Both
    numbers are ordinary continental values (1e8 droplets and 5e4 ice
    crystals per kg).

    ``cp`` is the caller's cupy module, a parameter for the reason
    :func:`_grell_freitas_state` gives at length.
    """
    state, cfg, driver = _physics_state(cp, mp_physics=9, nz=24,
                                        ztop=14000.0)
    state.qc[4:12] = cp.float32(1.5e-3)
    state.nc[4:12] = cp.float32(1.0e8)
    state.qi[12:18] = cp.float32(4.0e-4)
    state.ni[12:18] = cp.float32(5.0e4)
    return state, cfg, driver


@requires_gpu
@pytest.mark.gpu
def test_short_milbrandt2_restart_is_bit_identical(tmp_path):
    """mp_physics=9, on the same terms: 20 + restart + 20 == 40 steps,
    FP32-bit-exact on every serialized field.

    THE GATE THE SCHEME SHIPPED WITHOUT.  Until this ran, no mp=9
    checkpoint had ever been written: the identity lookup refused the
    write, so the first forecast to reach a restart interval died there
    (2026-09-10, single-domain ERA5, 59 minutes in).  Adding the identity
    row alone would only prove the write no longer raises; this proves
    the checkpoint is COMPLETE -- the masses and number moments the
    scheme is carrying at the boundary, the precipitation accumulators,
    and the held physics all cross it as bytes rather than being
    recomputed by the resume.  A two-moment scheme is exactly where a
    lost moment hides: mass and number enter the size distribution as a
    ratio, and a moment rebuilt from the scheme's own bounds stays
    finite, so nothing short of a bit comparison would notice.
    """
    import cupy as cp

    from woof.core.dycore import run_steps

    state_a, cfg_a, _ = _milbrandt2_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, _ = _milbrandt2_state(cp)
    run_steps(state_b, cfg_b, 20)
    # THE state this test is about: at the checkpoint the scheme is
    # carrying condensate mass with the number moments that belong to it,
    # in both phases, and has accumulated surface rain.  Asserted before
    # the checkpoint is written, so a state that stopped being a real
    # mixed-phase one fails here instead of passing an empty comparison.
    for name in ("qs", "qr", "nr", "ni", "ns"):
        assert bool(cp.any(getattr(state_b, name) != 0.0)), name
    assert float(cp.abs(state_b._scratch["mp_rainnc"]).max()) > 0.0
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _milbrandt2_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)

    _assert_restart_equal(resumed, reference)

    # Non-triviality: the scheme ran on every step and left real numbers.
    assert state_c.physics.microphysics_updates == 40
    assert bool(cp.any(state_c.h_diabatic != 0.0))

    # COMPLETENESS, by name: every Milbrandt-Yau moment and every one of
    # the nine accumulators its driver arm binds is IN the file.  The
    # equality above is only as strong as the inventory it compares.
    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for name in restart.MILBRANDT2_RESTART_STATE:
        assert f"state/{name}" in keys, name
    for slot in restart.MILBRANDT2_RESTART_PRECIPITATION_SLOTS:
        assert f"scratch/{slot}" in keys, slot
    # NSSL spells its hail number qnh and mp=9 spells it nh; a checkpoint
    # carrying the other scheme's package would mean the two ports had
    # been crossed.
    assert not (keys & {"state/qnh", "state/qndrop", "state/qvolh"})


@requires_gpu
@pytest.mark.gpu
def test_short_grell_freitas_restart_is_bit_identical(tmp_path):
    """cu_physics=3, on the same terms: 20 + restart + 20 == 40 steps,
    FP32-bit-exact on every serialized field.

    The gate the GF selectable wiring shipped without, and the one that
    would have caught the manifest refusal that killed the first real-case
    GF trajectory 179 outer steps in.  GF's adapter is stateless, so
    everything that must cross the boundary crosses it on the DRIVER: the
    held ``cu_rates``, the RAINC/RAINCV/PRATEC accumulators, and --
    GF-specific -- the retained radiative heating that ``bind_driver``
    feeds back into GFDRV's forced state.  ``cudt_minutes=0`` puts GF on
    every step (its pinned WRF configuration), so the step-21 call is a
    real cumulus call reading restored inputs; and the step-20 boundary is
    off the radiation calendar (due at ``itimestep % 6 == 1`` -> 19/25), so
    that call must consume a DESERIALIZED rthraten rather than one the
    resume happened to recompute.
    """
    import cupy as cp

    from woof.core.dycore import run_steps
    from woof.core.gf import GrellFreitas

    state_a, cfg_a, driver_a = _grell_freitas_state(cp)
    assert isinstance(driver_a.cumulus_callable, GrellFreitas)
    run_steps(state_a, cfg_a, 40)
    # The adapter is bound by the time any checkpoint is taken -- which is
    # exactly the shape the manifest walk has to accept.
    assert driver_a.cumulus_callable._driver is driver_a
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _grell_freitas_state(cp)
    run_steps(state_b, cfg_b, 20)
    # THE state the fix is about: at the checkpoint GF is convecting, so
    # held rates and a RAINC total both have to survive serialization.
    assert bool(cp.any(driver_b.cu_rates["rthcuten"] != 0.0))
    assert float(cp.abs(driver_b.rainc).max()) > 0.0
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _grell_freitas_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 1200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)

    _assert_restart_equal(resumed, reference)

    # Non-triviality: GF ran on every one of the 40 steps and left real
    # numbers behind, and the radiation it reads spans the boundary.
    assert state_c.physics.call_counts["cumulus"] == 40
    assert state_c.physics.call_counts["radiation"] == 7   # 1, 7, ..., 37
    assert float(cp.abs(state_c.physics.rainc).max()) > 0.0
    assert bool(cp.any(state_c.h_diabatic != 0.0))
    # The exact sum bind_driver hands GFDRV as its radiative forcing: if it
    # were zero the boundary-crossing this test is about would be vacuous.
    held_rthraten = state_c.physics.rthratenlw + state_c.physics.rthratensw
    assert bool(cp.any(held_rthraten != 0.0))

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for expected in ("scratch/cu_rainc", "scratch/cu_raincv",
                     "scratch/cu_pratec", "scratch/cu_rthcuten",
                     "scratch/cu_rqvcuten"):
        assert expected in keys, expected
    # GF carries no trigger history: the KF-only W0AVG member must not
    # appear, and its absence must not be mistaken for a lost field.
    assert "cumulus/w0avg" not in keys


@requires_gpu
@pytest.mark.gpu
def test_short_thompson_refl_restart_is_bit_identical(tmp_path):
    """Guarded mp8: 10 + restart + 10 equals uninterrupted 20 exactly.

    REFL_10CM is due/consumed four times.  Its private graupel-number shadow
    is rebuild-only, while all prognostics, held heating, and precipitation
    accumulators must cross the step-10 checkpoint without one changed bit.
    """
    if not os.environ.get("WOOF_THOMPSON_TABLE_ROOT"):
        pytest.skip("WOOF_THOMPSON_TABLE_ROOT is required for mp8 GPU gate")
    if os.environ.get("WOOF_EXPERIMENTAL_THOMPSON_MP8") != "1":
        pytest.skip("WOOF_EXPERIMENTAL_THOMPSON_MP8=1 is required")

    import cupy as cp

    control, control_cfg = _thompson_restart_state(cp)
    _advance_thompson_with_refl(control, control_cfg, 20)
    reference = restart.write_restart(
        tmp_path / "thompson-reference.npz", control, control_cfg)

    split, split_cfg = _thompson_restart_state(cp)
    _advance_thompson_with_refl(split, split_cfg, 10)
    mid = restart.write_restart(
        tmp_path / "thompson-mid.npz", split, split_cfg)

    resumed, resumed_cfg = _thompson_restart_state(cp)
    info = restart.restore_restart(mid, resumed, resumed_cfg)
    assert info.elapsed_seconds == 20.0
    _advance_thompson_with_refl(resumed, resumed_cfg, 10)
    resumed_path = restart.write_restart(
        tmp_path / "thompson-resumed.npz", resumed, resumed_cfg)

    _assert_restart_equal(resumed_path, reference)
    assert resumed.physics.refl_10cm is None
    assert resumed.physics.microphysics_updates == 20


# ---------------------------------------------------------------------------
# GPU + slow_acceptance: THE Task 8 gate (fresh-process real74 6h+6h == 12h).
# ---------------------------------------------------------------------------

def _write_case_config(tmp_path, name, run_seconds, restart_interval):
    text = (REPO_ROOT / "configs" / "real74_d01.toml").read_text(
        encoding="utf-8")
    text = text.replace("run_seconds = 43200.0",
                        f"run_seconds = {run_seconds}")
    text += f"restart_interval_s = {restart_interval}\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _cli_run(config_path, outdir, restart_file=None):
    command = [sys.executable, "-m", "woof.cli", "run", str(config_path),
               "--outdir", str(outdir)]
    if restart_file is not None:
        command += ["--restart", str(restart_file)]
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"{' '.join(command)}\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}")
    return result


def _assert_wrfouts_equal(path_a, path_b):
    import netCDF4

    with netCDF4.Dataset(path_a) as a, netCDF4.Dataset(path_b) as b:
        assert set(a.variables) == set(b.variables), path_a
        for name in sorted(a.variables):
            va = np.asarray(np.ma.filled(a.variables[name][:], np.nan))
            vb = np.asarray(np.ma.filled(b.variables[name][:], np.nan))
            np.testing.assert_array_equal(va, vb,
                                          err_msg=f"{path_a.name}:{name}")


@requires_bundle
@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow_acceptance
def test_real74_restart_continuation_is_bit_identical(tmp_path):
    """Plan Task 8 gate: real74 6 h -> write restart -> FRESH PROCESS ->
    6 h more == uninterrupted 12 h, FP32-bit-exact on every state field
    and accumulator, with identical run-summary trackers.

    Three CLI subprocesses (each a fresh Python/CUDA process):
      A: run_seconds=21600, restart_interval_s=21600 -> gpuwmrst @ 18Z.
      C: run_seconds=43200, restart_interval_s=21600 -> gpuwmrst @ 18Z
         and @ 00Z (the uninterrupted control).
      B: run_seconds=43200, restart_interval_s=43200, --restart A@18Z ->
         gpuwmrst @ 00Z (the resumed run).
    The restart files ARE the complete state serialization, so comparing
    them bit-for-bit compares every prognostic, every physics/soil/snow
    field, every accumulator, every held tendency, W0AVG/NCA, and the
    clock; the write-time A@18Z == C@18Z check additionally pins that
    writing a restart never perturbs the running state.
    """
    rst_6h = restart.restart_filename(datetime(1974, 4, 3, 18))
    rst_12h = restart.restart_filename(datetime(1974, 4, 4, 0))

    cfg_a = _write_case_config(tmp_path, "real74-a.toml", 21600.0, 21600.0)
    cfg_b = _write_case_config(tmp_path, "real74-b.toml", 43200.0, 43200.0)
    cfg_c = _write_case_config(tmp_path, "real74-c.toml", 43200.0, 21600.0)
    out_a = tmp_path / "run-a"
    out_b = tmp_path / "run-b"
    out_c = tmp_path / "run-c"

    _cli_run(cfg_a, out_a)
    _cli_run(cfg_c, out_c)
    result_b = _cli_run(cfg_b, out_b, restart_file=out_a / rst_6h)
    assert "'restarted': True" in result_b.stdout

    # Write-time identity: the 6 h state is independent of run_seconds and
    # of having written a restart mid-run.
    _assert_restart_equal(out_a / rst_6h, out_c / rst_6h,
                          compare_trackers=True)
    # THE continuation gate: fresh-process 6h+6h == uninterrupted 12h,
    # bit-exact on every serialized array, with identical gate trackers.
    _assert_restart_equal(out_b / rst_12h, out_c / rst_12h,
                          compare_trackers=True)

    # The resumed run's hourly products match the uninterrupted run's.
    for hour in range(19, 24):
        name = f"wrfout_d01_1974-04-03_{hour:02d}_00_00"
        _assert_wrfouts_equal(out_b / name, out_c / name)
    name = "wrfout_d01_1974-04-04_00_00_00"
    _assert_wrfouts_equal(out_b / name, out_c / name)
    # The resumed run does not rewrite the 12Z cold-start frame.
    assert not (out_b / "wrfout_d01_1974-04-03_12_00_00").exists()
    assert (out_c / "wrfout_d01_1974-04-03_12_00_00").exists()


def test_olr_is_written_into_the_checkpoint_and_restored(monkeypatch,
                                                        tmp_path):
    """The carry itself: a driver whose longwave scheme publishes a TOA
    flux writes ``diag/olr``, and a resume lands on that field rather
    than on the zeros a fresh driver allocates."""
    cfg = _cfg()
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, 7)
    shape = state.mup.shape
    driver.olr = np.full(shape, 301.5, dtype=np.float32)
    target = tmp_path / "ck.npz"
    restart.write_restart(target, state, cfg)
    with np.load(target, allow_pickle=False) as data:
        assert "diag/olr" in data.files
        np.testing.assert_array_equal(data["diag/olr"], driver.olr)

    state2, driver2 = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state2)
    driver2.olr = np.zeros(shape, dtype=np.float32)
    restart.restore_restart(target, state2, cfg)
    np.testing.assert_array_equal(driver2.olr, np.full(shape, 301.5,
                                                       dtype=np.float32))


def test_a_checkpoint_without_the_diag_namespace_still_restores(monkeypatch,
                                                               tmp_path):
    """The cost that used to be the argument against carrying OLR was
    that it "would reject every checkpoint already on disk".  It does
    not: the key has its own ``diag/`` namespace, absent-tolerant on
    restore, so an older checkpoint resumes with the zeros its own build
    resumed with anyway.

    Proven by STRIPPING the key from a checkpoint this build wrote --
    which is exactly what an older one looks like -- rather than by
    keeping a stale fixture that would rot.
    """
    cfg = _cfg()
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, 11)
    shape = state.mup.shape
    driver.olr = np.full(shape, 288.0, dtype=np.float32)
    written = tmp_path / "ck.npz"
    restart.write_restart(written, state, cfg)

    def strip_diag(payload, header):
        removed = [k for k in list(payload) if k.startswith("diag/")]
        assert removed, ("nothing to strip -- this build wrote no diag/ "
                         "keys, so the test would pass vacuously")
        for key in removed:
            payload.pop(key)

    older = _rewrite_restart_archive(written, tmp_path / "older.npz",
                                     strip_diag)

    state2, driver2 = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state2)
    driver2.olr = np.zeros(shape, dtype=np.float32)
    restart.restore_restart(older, state2, cfg)          # must not raise
    assert not driver2.olr.any()


def test_a_resume_whose_scheme_publishes_no_olr_ignores_the_key(monkeypatch,
                                                                tmp_path):
    """Tolerant in the OTHER direction too, and it is a real
    configuration rather than a defensive shrug: the buffer exists only
    when the attached longwave scheme declares a TOA flux, so a resume
    onto a scheme that does not has nowhere to put the field."""
    cfg = _cfg()
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, 13)
    driver.olr = np.full(state.mup.shape, 275.0, dtype=np.float32)
    target = tmp_path / "ck.npz"
    restart.write_restart(target, state, cfg)

    state2, driver2 = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state2)
    driver2.olr = None                       # no TOA-flux producer
    restart.restore_restart(target, state2, cfg)         # must not raise
    assert driver2.olr is None


class _FakeStreamedDomainForSealedWrite:
    """A streamed domain that records the gather the sealed route must make."""

    def __init__(self, directory):
        self.directory = directory
        self.calls: list = []

    def attach_to(self, state):
        state._streamed_domain = self
        self._state = state
        return self

    def impose_clock(self, seconds):
        self.calls.append(("impose_clock", float(seconds)))

    def refresh_state(self, state=None):
        # What members exist at the moment of the gather: must be none.
        self.calls.append(("refresh_state",
                           sorted(p.name for p in self.directory.glob("*.npz"))))
        return 1

    def write_restart(self, *args, **kwargs):
        raise AssertionError(
            "the store writer has no sealed route and must not be used for a "
            "sealed forcing-extension checkpoint")


def test_a_streamed_domain_is_gathered_before_a_sealed_forcing_extension_checkpoint(
        monkeypatch, tmp_path):
    """The sealed route wrote a streamed domain's STALE resident state.

    A streamed domain's arrays live in its store and its DomainState stopped
    changing at attach; the sealed route has no store writer and fell
    through to the resident writer, so a horizon-extension checkpoint of a
    tiled tree was the analysis-era state under a sealed header, restored
    into the canonical store as if it were the forecast (ENG-012).  The
    store is gathered onto the state (refresh_state) with the exact clock
    imposed BEFORE any validation or member write.
    """
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    stream = _FakeStreamedDomainForSealedWrite(tmp_path).attach_to(
        source.root.state)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        sealed_forcing_extension=True)
    assert ("impose_clock", 3600.0) in stream.calls
    refreshes = [call for call in stream.calls if call[0] == "refresh_state"]
    assert len(refreshes) == 1
    assert refreshes[0][1] == [], "the gather must precede every member write"
    assert stream.calls.index(("impose_clock", 3600.0)) < stream.calls.index(refreshes[0])
    header = restart.read_restart_header(root_path)
    assert header["forcing_extension_mode"] == restart.SEALED_FORCING_EXTENSION_MODE


def test_v7_binds_independent_map_inverse_and_refuses_v6(monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(monkeypatch, forcing_count=2,
        run_seconds=3600., payload_seed=31)
    path=restart.write_tree_restart(tmp_path/'first', source, start+timedelta(seconds=3600))
    first=restart.read_restart_header(path)
    assert first['format_version']==7
    assert first['setup_fingerprint']==restart.setup_fingerprint(source.root.state)
    assert first['setup_fingerprint']!=_historical_setup_fingerprint(source.root.state,inverse=False)
    source.root.state.msfv_inv.flat[0]=np.nextafter(source.root.state.msfv_inv.flat[0],np.float32(1.))
    second_path=restart.write_tree_restart(tmp_path/'second', source,start+timedelta(seconds=3600))
    second=restart.read_restart_header(second_path)
    assert first['setup_fingerprint']!=second['setup_fingerprint']
    legacy=_rewrite_header(path,path,lambda h:h.__setitem__('format_version',6))
    with pytest.raises(restart.RestartMismatchError,match='MF_VX_INV'):
        restart.restore_tree_restart(legacy,source)

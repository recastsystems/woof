"""Memory-preflight estimator, scratch registry, nest manifest, N0 gates.

Phase-5 Task 11 (architecture section E).  Everything here except the
``gpu``-marked N0 allocation runs is pure CPU shape-formula arithmetic.

The golden byte pins are DELIBERATE: any change to the DomainState /
PhysicsDriver / scratch allocation surface must update the preflight
manifests AND these pins in the same diff, keeping the estimator an
enforced upper bound instead of a stale note.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof.config import RunConfig, load_config
from woof.core import preflight as pf
from woof.case_data import load_experiment_case
from woof.experiment import build_experiment, experiment_from_run_config
from woof.io import restart
from conftest import requires_grib1_bridge, requires_case_inputs

ROOT = Path(__file__).resolve().parents[1]

#: These tests load configs/real74_4dom.toml with its declared inputs
#: required, so they run only where the WRF 1974 reference bundle its
#: [case_data] names is on disk, and skip naming the absent file elsewhere.
requires_4dom_inputs = requires_case_inputs(
    Path(__file__).resolve().parents[1] / "configs" / "real74_4dom.toml")
CONFIG_4DOM = ROOT / "configs" / "real74_4dom.toml"
CONFIG_D01 = ROOT / "configs" / "real74_d01.toml"

# The four-domain configs below declare their inputs from the 1974
# reference bundle under the case-data root: `source_orography` from its
# met_em files, the forcing from its ERA5 GRIB.  A test that loads one of
# them with the inputs required reads those files, and on a box that has
# not staged the bundle (the Linux release node: proof/node-reds-276) the
# loader refuses with "... declared in [case_data] of ... does not exist"
# before the test's subject is reached.  The same root the configs
# resolve ${GPUWM_CASE_DATA_ROOT} against, so bundle and config relocate
# together; the idiom is tests/test_case_data.py's.
from woof.case_data import case_data_root  # noqa: E402

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))

BUNDLE = case_data_root() / "WRF_1974_MP55_reference_bundle"
_BUNDLE_INPUTS = (BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc",
                  BUNDLE / "era5_grib" / "era5_19740403.grb")
requires_reference_bundle = pytest.mark.skipif(
    not all(path.is_file() for path in _BUNDLE_INPUTS),
    reason=("the 1974 reference bundle is not staged under the case-data "
            f"root: {BUNDLE} must carry met_em/met_em.d01.1974-04-03_12_00_00.nc "
            "and era5_grib/era5_19740403.grb, which the four-domain configs "
            "this test loads declare as their inputs"))

GIB = pf.GIB

_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)


@pytest.fixture(scope="module")
def d01_cfg() -> RunConfig:
    return load_config(CONFIG_D01)


@pytest.fixture(scope="module")
def exp1(d01_cfg):
    return experiment_from_run_config(d01_cfg, datetime(1974, 4, 3, 12))


@pytest.fixture(scope="module")
def exp4():
    return load_experiment_case(CONFIG_4DOM)[0]


@pytest.fixture(scope="module")
def est4(exp4):
    return pf.estimate_experiment(exp4)


# ---------------------------------------------------------------------------
# (a) DomainState shape formulas
# ---------------------------------------------------------------------------

def test_state_manifest_matches_restart_classification(d01_cfg):
    """The state shape manifest and the restart manifest classify the SAME
    attribute set: a DomainState field added to state.py without updating
    both fails here (full-featured config exercises every conditional).

    The union must span every scheme that allocates something no other
    scheme does, because it is asserted as an EQUALITY in both directions.
    ``mp_physics=18`` covers the NSSL scalars; ``mp_physics=28`` covers
    Thompson aerosol-aware's nwfa/nifa/nc0/nwfa0/nifa0 and the two 2-D
    surface emission fields.  Widening this union is the sanctioned way to
    admit a new scheme; TRIMMING the classification sets is not -- an
    attribute dropped from the restart manifest is a field that silently
    vanishes across a checkpoint.
    """
    from woof.config import CCPP_TKE_EDMF_PBL_SCHEME, SASE_PBL_SCHEME

    nssl_cfg = dataclasses.replace(d01_cfg, mp_physics=18)
    # km_opt=2 is the only configuration that allocates the prognostic-TKE
    # carrier, so the union needs an LES-closure arm or tke/tke0 would be
    # classified in the restart manifest and unaccounted for in the VRAM
    # projection (validate_run_config is deliberately not run here -- this
    # is a shape manifest, not an admissible run).
    les_cfg = dataclasses.replace(
        d01_cfg, km_opt=2, bl_pbl_physics=0, khdif=0.0, kvdif=0.0)
    aerosol_cfg = dataclasses.replace(d01_cfg, mp_physics=28)
    # The SASE closure owns one conditional prognostic that no other
    # configuration allocates, so it joins the union for the same reason
    # NSSL does: this test's whole claim is that every conditional is
    # exercised.
    sase_cfg = dataclasses.replace(d01_cfg,
                                   bl_pbl_physics=SASE_PBL_SCHEME,
                                   km_opt=0, khdif=0.0, kvdif=0.0,
                                   bldt=0.0)
    # P3 (mp=50) owns four names no other scheme allocates -- the rime
    # mass/volume pair and the two cross-step supersaturation carriers --
    # plus their RK time-t copies.  Without this arm they would be
    # classified in the restart manifest and unaccounted for in the VRAM
    # projection, which is exactly the drift this equality exists to catch.
    p3_cfg = dataclasses.replace(d01_cfg, mp_physics=50)
    # Milbrandt-Yau owns nh/nh0 and WDM6 owns nn/nn0; both joined the
    # classification sets at 1.9.1 (D1's class), so both join the union
    # for the same reason NSSL and P3 do.  The per-scheme closure itself
    # is gated by tests/test_mp_accepted_builds.py.
    my2_cfg = dataclasses.replace(d01_cfg, mp_physics=9)
    wdm6_cfg = dataclasses.replace(d01_cfg, mp_physics=16)
    # FAST_SBM owns the 142 spectral bins, six number moments and their
    # time-t copies (woof/core/state.py mp==30 arm), classified in the
    # restart manifest, so it joins the union.
    fsbm_cfg = dataclasses.replace(d01_cfg, mp_physics=30)
    # Grell-Freitas owns the two names in
    # woof.config.CUMULUS_ADVECTIVE_FORCING_SCHEMES' allocation -- the
    # dycore's exported advective forcing pair (WRF RTHFTEN/RQVFTEN) --
    # and no other configuration allocates them, so the arm joins for the
    # same reason NSSL, P3, MY2 and WDM6 do.
    gf_cfg = dataclasses.replace(d01_cfg, cu_physics=3)
    # TEMPO (a named scheme) owns the graupel volume pair (volg/volg0) no
    # ported WRF scheme allocates, so its arm joins the union too.
    tempo_cfg = dataclasses.replace(d01_cfg, mp_physics=900)
    # MYNN with bl_mynn_tkeadvect owns the advected TKE carrier (WRF's
    # qke_adv scalar) and its RK time-t copy; no other configuration
    # allocates them, so the arm joins the union (2.9.0).
    tkeadv_cfg = dataclasses.replace(d01_cfg, bl_pbl_physics=5,
                                     sf_sfclay_physics=5,
                                     bl_mynn_tkeadvect=True)
    # The carried CCPP energy and its time-t copy belong to this full union.
    ccpp_cfg = dataclasses.replace(d01_cfg, bl_pbl_physics=CCPP_TKE_EDMF_PBL_SCHEME)
    names = (set(pf.state_array_shapes(d01_cfg))
             | set(pf.state_array_shapes(tkeadv_cfg))
             | set(pf.state_array_shapes(nssl_cfg))
             | set(pf.state_array_shapes(les_cfg))
             | set(pf.state_array_shapes(aerosol_cfg))
             | set(pf.state_array_shapes(sase_cfg))
             | set(pf.state_array_shapes(p3_cfg))
             | set(pf.state_array_shapes(my2_cfg))
             | set(pf.state_array_shapes(wdm6_cfg))
             | set(pf.state_array_shapes(fsbm_cfg))
             | set(pf.state_array_shapes(gf_cfg))
             | set(pf.state_array_shapes(tempo_cfg))
             | set(pf.state_array_shapes(ccpp_cfg)))
    # Strict WRF arithmetic attaches its own carriers at first write, outside
    # DomainState's allocation list (c501f4d34, lane/wrf-exact-start); they
    # are classified, so they join the union by name.
    assert restart.STRICT_LAZY_STATE_ATTRS <= restart.STATE_REBUILT_ATTRS
    names |= set(restart.STRICT_LAZY_STATE_ATTRS)
    # aer_opt = 1 (golden/ports-rad) attaches the monthly aerosol climatology
    # ``aerodm`` at its first radiation call, outside DomainState's allocation
    # list: it is classified as derived setup (rebuilt, never in a checkpoint)
    # and priced among the radiation arrays, so it joins the union by name
    # once that pricing is shown.
    aer1_cfg = dataclasses.replace(d01_cfg, aer_opt=1)
    assert "radiation/aerodm" in pf.physics_array_shapes(aer1_cfg)
    assert "radiation/aerodm" not in pf.physics_array_shapes(d01_cfg)
    names.add("aerodm")
    # Analysis nudging (labs/nudging, 2.9.11) attaches its held FDDA
    # tendencies at setup (woof.spectral_nudging.prepared
    # .allocate_held_tendencies), outside DomainState's allocation list, and
    # serializes them so a resume keeps them: they join the union by name.
    names |= {"nudging_tendency_u", "nudging_tendency_v",
              "nudging_tendency_temperature", "nudging_tendency_geopotential",
              "nudging_tendency_qv"}
    # Added packages join the allocation census; old configurations remain.
    for mp in (5,):
        names |= set(pf.state_array_shapes(dataclasses.replace(d01_cfg, mp_physics=mp)))
    classified = (set(restart.STATE_SERIALIZED_ATTRS)
                  | set(restart.STATE_REBUILT_ATTRS)
                  | set(restart.CHECKPOINT_ONLY_STATE)
                  | set(restart.STATE_SETUP_ARRAYS)
                  | set(restart.STATE_DERIVED_SETUP_ARRAYS))
    assert names == classified
    # The derived setup arrays are priced like every other allocation and
    # classified like every other attribute, but they are deliberately
    # OUTSIDE the fingerprint's byte stream: each is a function of an
    # array already in it, so hashing them would reject every earlier
    # checkpoint to hash the same numbers twice.
    for name in restart.STATE_DERIVED_SETUP_ARRAYS:
        assert name not in restart.STATE_SETUP_ARRAYS
        assert restart.classify_state_attr(name) == "derived_setup"
    # And the specific names this port added, so a later edit cannot make
    # the equality hold again by deleting them from BOTH sides.
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert name in restart.STATE_SERIALIZED_ATTRS
    # The advected TKE carrier is cross-step state; its time-t copy is not.
    assert "qke_adv" in restart.STATE_SERIALIZED_ATTRS
    assert "qke_adv0" in restart.STATE_REBUILT_ATTRS
    for name in ("nc0", "nwfa0", "nifa0"):
        assert name in restart.STATE_REBUILT_ATTRS
    # Same guard for P3: the rime pair and the supersaturation carriers are
    # cross-step state WRF restart-carries, their time-t copies are not.
    for name in ("qir", "qib", "th_old", "qv_old"):
        assert name in restart.STATE_SERIALIZED_ATTRS
    for name in ("qir0", "qib0"):
        assert name in restart.STATE_REBUILT_ATTRS
    # Same guard for the advective forcing pair: it is cross-step state the
    # producer cannot rebuild before its first post-resume consumer, so it
    # is SERIALIZED and must never be demoted to rebuilt to dodge the
    # checkpoint layout change.
    for name in restart.ADVECTIVE_FORCING_STATE:
        assert name in restart.STATE_SERIALIZED_ATTRS
        assert name not in restart.STATE_REBUILT_ATTRS


def test_state_shape_formulas_staggering():
    cfg = RunConfig(**_TINY)
    shapes = pf.state_array_shapes(cfg)
    assert shapes["u"] == (4, 6, 9)
    assert shapes["v"] == (4, 7, 8)
    assert shapes["w"] == (5, 6, 8)
    assert shapes["php"] == (5, 6, 8)
    assert shapes["mup"] == (6, 8)
    assert shapes["msfu"] == (6, 9)
    # Flat terrain keeps 1-D base profiles (state.py:148-152).
    assert shapes["thb"] == (4,)
    assert shapes["phb"] == (5,)
    assert "qv" not in shapes and "h_diabatic" not in shapes
    terrain = pf.state_array_shapes(
        RunConfig(**_TINY, terrain_opt=1, moist=True, mp_physics=10))
    assert terrain["phb"] == (5, 6, 8)
    assert terrain["qv"] == (4, 6, 8)
    for name in ("qi", "ng0", "effs", "nc", "h_diabatic"):
        assert terrain[name] == (4, 6, 8)
    # Kessler moist: no Morrison moments, no nc.
    kessler = pf.state_array_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=1))
    assert "qv" in kessler and "qi" not in kessler and "nc" not in kessler


@requires_4dom_inputs
def test_shared_dycore_state_symbols_are_restart_rebuilt_source(exp4):
    """The sharing registry is the restart REBUILT authority, less the one
    rebuilt symbol a live strict run carries between steps."""
    assert pf.shared_dycore_state_symbols() == (
        restart.STATE_REBUILT_ATTRS - {"p_perturbation"})
    active = set().union(*(
        pf.state_array_shapes(dc.run).keys() for dc in exp4.domains))
    assert set(pf.shared_dycore_state_workspace_shapes(exp4.domains)) == (
        (set(restart.STATE_REBUILT_ATTRS) - {"p_perturbation"}) & active)


def test_strict_pressure_perturbation_is_never_shared_between_domains():
    """Strict rk_step 1 and every history P read the previous step's
    p_perturbation (core/dycore.py:_wrf_reuse_diagnostics).  Shared, a
    two-domain strict run read the other domain's pressure: the UAE pair's
    d01 history P differed from wrf.exe 4.6.1 in 527,901 of 528,000 words
    at 45 s."""
    assert "p_perturbation" in restart.STATE_REBUILT_ATTRS
    assert "p_perturbation" not in pf.shared_dycore_state_symbols()


def test_shared_dycore_state_workspace_binds_contiguous_prefixes(monkeypatch):
    """Every rebuilt field keeps its allocation shape/dtype/strides.

    Two differently sized Morrison domains bind each symbol to a C-contiguous
    prefix of the same per-symbol maximum backing.  The ordinary constructor
    remains independent for single-domain callers.
    """
    import types

    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    cfg_small = RunConfig(
        **_TINY, moist=True, mp_physics=10)
    cfg_large = RunConfig(
        **{**_TINY, "nx": 11, "ny": 7, "nz": 5},
        moist=True, mp_physics=10)
    domains = (types.SimpleNamespace(run=cfg_small),
               types.SimpleNamespace(run=cfg_large))
    workspace = state_mod.build_shared_dycore_state_workspace(domains)

    active = (set(pf.state_array_shapes(cfg_small))
              | set(pf.state_array_shapes(cfg_large)))
    assert workspace.symbols == restart.STATE_REBUILT_ATTRS & active
    assert workspace.nbytes == pf.shared_dycore_state_workspace_bytes(domains)
    small = state_mod.DomainState(
        cfg_small, dycore_state_workspace=workspace)
    large = state_mod.DomainState(
        cfg_large, dycore_state_workspace=workspace)
    small_shapes = pf.state_array_shapes(cfg_small)
    large_shapes = pf.state_array_shapes(cfg_large)

    for name in sorted(workspace.symbols):
        small_value = getattr(small, name)
        large_value = getattr(large, name)
        assert small_value.shape == small_shapes[name]
        assert large_value.shape == large_shapes[name]
        assert small_value.dtype == large_value.dtype == np.dtype(np.float32)
        assert small_value.flags.c_contiguous
        assert large_value.flags.c_contiguous
        assert small_value.strides == np.empty(
            small_shapes[name], dtype=np.float32).strides
        assert large_value.strides == np.empty(
            large_shapes[name], dtype=np.float32).strides
        assert np.shares_memory(small_value, large_value)
        assert np.shares_memory(small_value, workspace.backing(name))

    default_a = state_mod.DomainState(cfg_small)
    default_b = state_mod.DomainState(cfg_small)
    assert not np.shares_memory(default_a.u0, default_b.u0)


def test_shared_dycore_state_workspace_rejects_concurrent_owners(
        monkeypatch):
    """A second domain turn cannot acquire the shared arrays concurrently."""
    import types

    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    cfg = RunConfig(**_TINY, moist=True, mp_physics=10)
    workspace = state_mod.build_shared_dycore_state_workspace(
        (types.SimpleNamespace(run=cfg),))

    with workspace.acquire(("STEP", 1)):
        assert workspace.owner == ("STEP", 1)
        with pytest.raises(RuntimeError, match="owned.*STEP.*1"):
            with workspace.acquire(("FORCE", 2, 1)):
                pass
    assert workspace.owner is None


# ---------------------------------------------------------------------------
# (b) PhysicsDriver persistents
# ---------------------------------------------------------------------------

@requires_4dom_inputs
def test_physics_shapes_scheme_selection(d01_cfg, exp4):
    full = pf.physics_array_shapes(d01_cfg)
    nzs = (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx)
    s2 = (d01_cfg.ny, d01_cfg.nx)
    # KF persistence + rqr growth are d01-only (cu_physics=1).
    assert full["cumulus/w0avg"] == nzs
    assert full["cumulus_tendencies/rqr"] == nzs
    assert full["pbl_tendencies/rqr"] == nzs
    assert not any(name.startswith("tendencies/") for name in full)
    # Morrison + YSU carries rqi through the bldt=0 in-place composition.
    assert full["pbl_tendencies/rqi"] == nzs
    assert full["radiation/latitude_deg"] == s2
    assert not any(name.startswith("microphysics/") for name in full)
    assert full["fields/smois"] == (4, d01_cfg.ny, d01_cfg.nx)
    # At bldt=0 the raw YSU dict is transient, not driver-persistent.
    assert not any(name.startswith("last_ysu/") for name in full)
    transient = pf.ysu_output_transient_shapes(d01_cfg)
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                 "exch_h", "exch_m"):
        assert transient[f"ysu_output/{name}"] == nzs
    for name in ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum",
                 "wstar3_2", "cloudflg"):
        assert transient[f"ysu_output/{name}"] == s2

    # Positive cadence retains raw rates once, shared with the diagnostic dict.
    held = pf.physics_array_shapes(dataclasses.replace(d01_cfg, bldt=5.0))
    assert held["pbl_raw_rates/du"] == nzs
    assert "last_ysu/du" not in held
    assert held["last_ysu/cloudflg"] == s2
    assert held["tendencies/rqr"] == nzs
    assert held["tendencies/rqi"] == nzs
    # Once-per-process KF device LUT, counted on the cumulus domain.
    assert sum(4 * math.prod(shape) for name, shape in full.items()
               if "kf_lut" in name) == 441680
    # RRTMGP ozone climatology profiles (rrtmgp.py:1076-1077).
    assert sum(4 * math.prod(shape) for name, shape in full.items()
               if "_ozone" in name) == 480

    child = pf.physics_array_shapes(exp4.domain(2).run)
    assert "cumulus/w0avg" not in child
    assert "cumulus_tendencies/rqr" not in child
    assert not any("kf_lut" in name for name in child)
    assert "pbl_tendencies/rqi" in child
    assert not any(name.startswith(("last_ysu/", "tendencies/"))
                   for name in child)

    assert pf.physics_array_shapes(RunConfig(**_TINY)) == {}


def test_physics_manifest_groups_cover_restart_driver_attrs(d01_cfg):
    """Cross-pin against the restart driver manifest (review F4/F2): every
    array-bearing serialized driver attribute has a manifest group, and
    conditional/rebuilt driver arrays are cross-pinned to their lifetime
    decisions.  Active Morrison diagnostics are absent here because the
    driver aliases the separately counted serialized ``mp_*`` scratch set."""
    # Conditional raw-rate owners need a representative positive-cadence
    # GF configuration as well as the default every-step KF configuration.
    groups = {name.split("/")[0]
              for cfg in (d01_cfg, dataclasses.replace(d01_cfg, cu_physics=3, bldt=2.),
                          # SFIRE's coupler and its held tendencies.
                          dataclasses.replace(d01_cfg, ifire=2, sr_x=4, sr_y=4, moist=True))
              for name in pf.physics_array_shapes(cfg)}
    scalar_attrs = {"microphysics_updates", "call_counts",
                    "ysu_nan_guard_fires",
                    # The surface-radiation carrier contract: two scalars
                    # per carrier (source, last producer model time) in the
                    # checkpoint HEADER, no array of its own.  The carrier
                    # FIELDS ride the serialized surface inventory, which
                    # is where their shapes are already accounted for.
                    "carriers"}
    ozone = pf.physics_array_shapes(d01_cfg, cam_ozone=True)
    assert ozone["radiation/o33d_grid"] == (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx)
    groups.add("o3rad")  # direct owner retains the historical radiation/o33d_grid key
    # Hydro's network-sized backing roster cannot be inferred from RunConfig.
    # It is priced by the table-selected provider in the forecast envelope,
    # including checkpoint work. The functional nonzero allocation and actual
    # restart-array cross-checks live in test_hydro_memory_contract.
    dynamic_attrs = {"hydro"}
    assert dynamic_attrs <= set(restart.DRIVER_SERIALIZED_ATTRS)
    assert "hydro_memory" in {f.name for f in dataclasses.fields(pf.ExperimentMemoryEstimate)}
    assert set(restart.DRIVER_SERIALIZED_ATTRS) - scalar_attrs - dynamic_attrs <= groups
    assert "last_ysu" not in {name.split("/")[0]
                               for name in pf.physics_array_shapes(d01_cfg)}
    assert "last_ysu" in restart.DRIVER_REBUILT_ATTRS
    assert "microphysics" not in restart.DRIVER_SERIALIZED_ATTRS
    assert "microphysics" in restart.DRIVER_REBUILT_ATTRS


def test_physics_lifetime_audit_is_exact_name_closed_world(d01_cfg):
    names = [name for row in pf.PHYSICS_ARRAY_LIFETIME_AUDIT
             for name in row.names]
    assert len(names) == len(set(names)) == 69  # four optional rw names + raw dw
    assert {row.disposition for row in pf.PHYSICS_ARRAY_LIFETIME_AUDIT} == {
        "transient_when_bldt_zero", "aliases_serialized_scratch",
        "aliases_fresh_pbl_at_bldt_zero", "retained_family_state"}
    for prefix, components in {
            "last_ysu": {"du", "dv", "dtheta", "dqv", "dqc", "dqi",
                         "exch_h", "exch_m", "hpbl", "kpbl", "wstar",
                         "delta", "topdown_radsum", "wstar3_2", "cloudflg"},
            "microphysics": set(restart.MICROPHYSICS_COMPONENTS),
            "tendencies": set(restart.TENDENCY_COMPONENTS)}.items():
        assert {name.split("/", 1)[1] for name in names
                if name.startswith(prefix + "/")} == components
    for name in names:
        assert pf.physics_array_lifetime(name) is not None
    assert pf.physics_array_lifetime("last_ysu/future_component") is None


@pytest.mark.parametrize("mp_physics", (0, 6, 18, 50))
def test_dry_map_coupling_uses_registered_scratch(monkeypatch, mp_physics):
    """The allowlisted helper may allocate only the four priced dry rows."""
    from woof.core import dycore, state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    cfg = RunConfig(**_TINY, moist=bool(mp_physics),
                    mp_physics=mp_physics, km_opt=4)
    state = state_mod.DomainState(cfg)
    state.has_msf = True
    state.msft[...] = state.msfu[...] = state.msfv[...] = 1.25
    requested = {}
    scratch = state.scratch

    def tracked(shape, slot, dtype=None):
        array = scratch(shape, slot, dtype=dtype)
        requested[slot] = (array.shape, array.nbytes)
        return array

    monkeypatch.setattr(state, "scratch", tracked)
    specs = dycore._smag2d_specs(state, None, None, time_t=True)
    dycore._couple_dry_mixing_map_factor(state, specs)
    expected = {
        "smag_ru": (cfg.nz, cfg.ny, cfg.nx + 1),
        "smag_rv": (cfg.nz, cfg.ny + 1, cfg.nx),
        "smag_rw": (cfg.nz + 1, cfg.ny, cfg.nx),
        "smag_rth": (cfg.nz, cfg.ny, cfg.nx),
    }
    registry = pf.scratch_slot_registry(cfg)
    assert set(requested) == set(expected)
    for name, shape in expected.items():
        assert requested[name] == (shape, 4 * math.prod(shape))
        assert registry[name] == shape


def test_physics_fields_union_covers_sources():
    from woof.core.noah import _F2D
    from woof.core.sfclay import SFCLAY_OUTPUTS

    names = set(pf.physics_field_names_2d())
    assert set(SFCLAY_OUTPUTS) <= names
    assert set(_F2D) <= names
    assert {"ebal", "kpbl", "landmask", "xland", "lakemask"} <= names


# ---------------------------------------------------------------------------
# (c) Scratch-slot registry + completeness over call sites
# ---------------------------------------------------------------------------

def test_scratch_registry_feature_matrix(d01_cfg):
    d01 = pf.scratch_slot_registry(d01_cfg, n_lbc_intervals=2)
    m = (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx)
    assert d01["cu_rthcuten"] == m
    assert d01["morr_z8w"] == (d01_cfg.nz + 1, d01_cfg.ny, d01_cfg.nx)
    assert d01["pd_fxl"] == (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx + 1)
    assert d01["lbc_qv_held"] == m
    assert d01["smag_rqi"] == m and d01["smag_rng"] == m
    assert d01["diff6_m"] == m
    assert d01["acoustic_mudf"] == (d01_cfg.ny, d01_cfg.nx)  # emdiv=0.01
    assert d01["integration_health_partial"] == (256, 9)
    assert d01["integration_health_field_ptr"] == (2048,)
    assert d01["integration_health_aux_ptr"] == (2048,)
    assert d01["integration_health_field_size"] == (2048,)
    assert d01["integration_health_bounds"] == (1024, 2)
    assert d01["integration_health_flags"] == (1024,)
    assert d01["integration_health_planes"] == (1024,)
    assert d01["integration_health_status_bits"] == (2048,)
    assert d01["integration_health_rim"] == (1024, 3)
    assert d01["integration_health_validation"] == (4,)
    assert d01["physics_validation_status"] == (1,)
    assert d01["lbc_old_mup_frame_1"] == (
        pf._perimeter_count(d01_cfg.ny, d01_cfg.nx, 1),)
    assert d01["lbc_forcing_tables"] == (2 * pf.lbc_interval_values(d01_cfg),)
    assert "mp_th" not in d01 and "openbc_upp_faces" not in d01

    kessler = pf.scratch_slot_registry(RunConfig(
        **_TINY, moist=True, mp_physics=1, open_x=True, open_y=True,
        khdif=1.0, kvdif=1.0))
    assert kessler["mp_kessler_sr"] == (6, 8)
    assert kessler["openbc_upp_faces"] == (4, 6, 2)
    assert kessler["openbc_vpp_faces"] == (4, 2, 8)
    # km_opt = 1 with khdif/kvdif runs WRF's diff_opt = 2 package, so its
    # slots are priced (the retired constant-K operator's diff_* were
    # priced in their place, and the package's were missing).
    assert "diff_u" not in kessler
    assert kessler["smag_km"] == kessler["smag_kmv"] == (4, 6, 8)
    assert kessler["smag_ru"] == (4, 6, 9) and kessler["smag_rqv"] == (4, 6, 8)
    assert kessler["diff6_x"] == (4, 6, 9) and kessler["diff6_y"] == (4, 7, 8)
    assert kessler["physics_validation_status"] == (1,)
    # Open boundaries: PD final stage disabled -> no pd_* slots; not
    # specified -> no LBC residents.
    assert "pd_fxl" not in kessler and "lbc_relax_u" not in kessler
    assert "cu_nca" not in kessler

    dry_phys = pf.scratch_slot_registry(
        RunConfig(**_TINY, sf_sfclay_physics=1))
    assert dry_phys["physics_dry_qv"] == (4, 6, 8)
    assert dry_phys["physics_qi"] == (4, 6, 8)
    assert dry_phys["physics_qs"] == (4, 6, 8)
    assert "physics_validation_status" not in dry_phys

    kf_only = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, cu_physics=1))
    assert kf_only["physics_validation_status"] == (1,)
    assert pf._CUMULUS_KERNEL_MODULES[1] == ("kf", "kf_validation")


def test_nwp_diagnostics_prices_exactly_the_uh_planes(d01_cfg):
    """The UP_HELI_MAX lane costs FIVE (ny, nx) FP32 planes and nothing
    else; the flagship (nwp_diagnostics = 0) registry is untouched.

    Three were the diagnostic's own (the accumulator plus two per-launch
    work planes).  The other two are the consumer-owned tracking windows
    added 2026-08-07: same running-max operator, folded in the same pass,
    but reset by the consumer that reads them instead of by the history
    writer, so a storm-following nest's placement stopped depending on
    the output cadence.  They are priced on this gate because that is the
    gate that allocates them.
    """
    base = pf.scratch_slot_registry(d01_cfg, n_lbc_intervals=2)
    on_cfg = dataclasses.replace(d01_cfg, nwp_diagnostics=1)
    on = pf.scratch_slot_registry(on_cfg, n_lbc_intervals=2)
    added = {"up_heli_max", "uh_diag_col", "uh_diag_use",
             "uh_follow_window", "uh_spawn_window"}
    assert set(on) - set(base) == added
    assert not added & set(base)
    for slot in added:
        assert on[slot] == (d01_cfg.ny, d01_cfg.nx)


def test_scratch_registry_classifiable_by_restart_manifest(d01_cfg):
    """Every registry slot must already be classified by the restart
    manifest (serialize or rebuild) -- one namespace, two manifests, no
    drift.  ``nest_*`` slots are excluded: their REBUILT classification
    is Task 14's restart.py edit (architecture 'WRF deviations')."""
    union: set[str] = set()
    for cfg in (d01_cfg,
                RunConfig(**_TINY, moist=True, mp_physics=1, open_x=True,
                          open_y=True, khdif=1.0, kvdif=1.0, emdiv=0.01),
                RunConfig(**_TINY, km_opt=4, diff_opt=1),
                RunConfig(**_TINY, sf_sfclay_physics=1)):
        union |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    for slot in union:
        assert restart.classify_scratch_slot(slot) in ("serialize", "rebuild")


@requires_4dom_inputs
def test_scratch_lifetime_audit_covers_registry_and_manifest(d01_cfg, exp4):
    """Architecture-E lever-2 admission is closed-world and reviewed.

    Every possible registry slot exercised by the feature matrix, plus every
    frozen F4 ``nest_*`` manifest slot, maps to exactly one committed audit
    row. Only explicit write-before-read rows may enter the arena.
    """
    configs = [dc.run for dc in exp4.domains]
    configs += [
        d01_cfg,
        RunConfig(**_TINY, km_opt=4, diff_opt=1),
        RunConfig(**_TINY, moist=True, mp_physics=1, open_x=True,
                  open_y=True, khdif=1.0, kvdif=1.0, emdiv=0.01),
        RunConfig(**_TINY, sf_sfclay_physics=1),
        # Every microphysics scheme with slots of its own, so a new family
        # cannot land unaudited.  km_opt=4 turns on the smag_r* held
        # tendencies, which is where the mp=28 number/aerosol rows live.
        RunConfig(**_TINY, moist=True, mp_physics=8, km_opt=4),
        RunConfig(**_TINY, moist=True, mp_physics=18, km_opt=4),
        RunConfig(**_TINY, moist=True, mp_physics=28, km_opt=4),
        RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=50,
                  km_opt=4),
    ]
    slots = set()
    for cfg in configs:
        slots |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    for manifest in pf.nest_allocation_manifest(exp4).values():
        slots |= set(manifest)

    assert slots
    for slot in slots:
        row = pf.scratch_slot_lifetime(slot)
        assert row is not None, slot
        assert row.kind in {"write_before_read", "carrying",
                            "excluded_unproven"}
        assert bool(pf.scratch_slot_uses_arena(slot)) == (
            row.kind == "write_before_read")
        if row.arena_eligible:
            assert row.evidence and row.rationale

    # High-risk exclusions are pins, not prefix accidents.
    for slot in ("mp_rainnc", "cu_rthcuten", "refl_10cm",
                 "physics_dry_qv", "lbc_forcing_tables", "nest_u_bxs"):
        assert not pf.scratch_slot_uses_arena(slot)
    for slot in ("rk_ru", "acoustic_c2a", "smag_rqi", "pd_fxl",
                 "morr_theta", "lbc_relax_u"):
        assert pf.scratch_slot_uses_arena(slot)


def test_shared_scratch_arena_aliases_views_and_default_does_not(monkeypatch):
    """Two different domain shapes share an admitted slot's max backing.

    The no-arena constructor retains the original independent, zero-filled
    per-state allocation path.
    """
    import types

    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    # CQ is enabled by default; state it explicitly because this test
    # exercises the CQ-to-advection aliases.
    cfg_small = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=1)
    cfg_large = RunConfig(**{**_TINY, "nx": 11, "ny": 7, "nz": 5},
                          moist=True, moist_cq=True, mp_physics=1)
    domains = (types.SimpleNamespace(run=cfg_small),
               types.SimpleNamespace(run=cfg_large))
    arena = state_mod.build_shared_scratch_arena(domains)
    expected_shape = (cfg_large.nz + 1, cfg_large.ny, cfg_large.nx)
    assert arena.slot_shapes["rk_ww"] == expected_shape

    small = state_mod.DomainState(cfg_small, scratch_arena=arena)
    large = state_mod.DomainState(cfg_large, scratch_arena=arena)
    small_ww = small.scratch(
        (cfg_small.nz + 1, cfg_small.ny, cfg_small.nx), "rk_ww")
    large_ww = large.scratch(expected_shape, "rk_ww")
    assert np.shares_memory(small_ww, large_ww)
    assert np.count_nonzero(large_ww) == 0
    small_ww.reshape(-1)[0] = np.float32(7.0)
    assert large_ww.reshape(-1)[0] == np.float32(7.0)

    # The three WRF cq faces add no physical arena backings: the acoustic
    # and standalone advection-only paths are mutually exclusive, and each
    # cq array is completely overwritten before its first stage read.
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert aliases["acoustic_cqu"] == "adv_ru"
    assert aliases["acoustic_cqv"] == "adv_rv"
    assert aliases["acoustic_cqw"] == "adv_rw"
    for cq, adv in (("acoustic_cqu", "adv_ru"),
                    ("acoustic_cqv", "adv_rv"),
                    ("acoustic_cqw", "adv_rw")):
        cq_view = large.scratch(arena.slot_shapes[cq], cq)
        adv_view = large.scratch(arena.slot_shapes[adv], adv)
        assert np.shares_memory(cq_view, adv_view)

    default_a = state_mod.DomainState(cfg_small)
    default_b = state_mod.DomainState(cfg_small)
    a = default_a.scratch((2, 3), "unit_default")
    b = default_b.scratch((2, 3), "unit_default")
    assert not np.shares_memory(a, b)
    assert np.count_nonzero(a) == np.count_nonzero(b) == 0


@pytest.mark.parametrize('parent_shape', [(12, 10), (6, 4)])
def test_streamed_root_resident_child_arena_shapes_aliases_and_bytes_agree(
        monkeypatch, parent_shape):
    """A resident child's force buffers retain its streamed parent's size."""
    from types import SimpleNamespace
    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, 'cp', np)
    root_cfg = RunConfig(**{**_TINY, 'nx': parent_shape[0], 'ny': parent_shape[1]})
    child_cfg = RunConfig(**_TINY, nested=True)
    root = SimpleNamespace(grid_id=1, parent_id=0, run=root_cfg)
    child = SimpleNamespace(grid_id=2, parent_id=1, run=child_cfg)
    whole = (root, child)
    resident = (child,)
    shapes = pf.shared_scratch_arena_shapes(resident, whole)
    aliases = pf.shared_scratch_arena_aliases(resident, tree_domains=whole)
    assert shapes['nest_parent_field'] == pf._full_field_capacity(root_cfg)
    assert shapes['nest_child_field'] == pf._full_field_capacity(child_cfg)
    assert all(math.prod(shapes[slot]) <= math.prod(shapes[target])
               for slot, target in aliases.items())

    arena = state_mod.build_shared_scratch_arena(resident, whole)
    assert arena.slot_shapes == shapes
    assert arena.nbytes == pf.shared_scratch_arena_bytes(resident, whole)
    for slot, target in aliases.items():
        assert np.shares_memory(arena.view(shapes[slot], slot),
                                arena.view(shapes[target], target))
    parent = arena.view(shapes['nest_parent_field'], 'nest_parent_field')
    child = arena.view(shapes['nest_child_field'], 'nest_child_field')
    assert not np.shares_memory(parent, child)
    parent.fill(np.float32(7.))
    child.fill(np.float32(13.))
    assert np.all(parent == 7.) and np.all(child == 13.)

    # Supplying the full tree preserves the ordinary all-resident arena.
    ordinary = state_mod.build_shared_scratch_arena(whole)
    explicit = state_mod.build_shared_scratch_arena(whole, whole)
    assert ordinary.slot_shapes == explicit.slot_shapes
    assert ordinary.nbytes == explicit.nbytes == pf.shared_scratch_arena_bytes(whole)
    assert pf.shared_scratch_arena_aliases(whole) == pf.shared_scratch_arena_aliases(whole, whole)


def test_diff6_tendencies_alias_lifetime_safe_backings(monkeypatch):
    """Diff6-only uses one backing; every Smag path keeps x/y distinct."""
    import types

    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    tiny = RunConfig(**_TINY, moist=True, mp_physics=10,
                     diff_6th_opt=2)
    domains = (types.SimpleNamespace(run=tiny),)
    shapes = pf.shared_scratch_arena_shapes(domains)
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert {slot: aliases[slot]
            for slot in ("diff6_x", "diff6_y", "diff6_m")} == {
                "diff6_x": "diff6_z",
                "diff6_y": "diff6_z",
                "diff6_m": "diff6_z",
            }
    assert all(math.prod(shapes[slot]) <= math.prod(shapes["diff6_z"])
               for slot in ("diff6_x", "diff6_y", "diff6_m"))

    arena = state_mod.build_shared_scratch_arena(domains)
    z = arena.view(shapes["diff6_z"], "diff6_z")
    for slot in ("diff6_x", "diff6_y", "diff6_m"):
        assert np.shares_memory(arena.view(shapes[slot], slot), z)

    smag = RunConfig(**_TINY, moist=True, mp_physics=10, km_opt=4,
                     bl_pbl_physics=1)
    # A Smag-only domain has no z/m requests itself, but a shared arena may
    # acquire them from a different diff6 domain.  The global lifetime rule
    # must still keep the Smag x/y face buffers distinct.
    mixed_domains = (types.SimpleNamespace(run=smag), *domains)
    smag_aliases = pf.shared_scratch_arena_aliases(mixed_domains)
    assert smag_aliases["diff6_x"] == "diff6_z"
    assert smag_aliases["diff6_m"] == "diff6_z"
    assert "diff6_y" not in smag_aliases

    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data")
    flagship = build_experiment(raw, source=str(CONFIG_4DOM))
    flagship_shapes = pf.shared_scratch_arena_shapes(flagship.domains)
    old_bytes = sum(
        4 * math.prod(flagship_shapes[slot])
        for slot in ("diff6_x", "diff6_y", "diff6_z", "diff6_m"))
    new_bytes = 4 * math.prod(flagship_shapes["diff6_z"])
    assert old_bytes == 283_915_200
    flagship_aliases = pf.shared_scratch_arena_aliases(flagship.domains)
    assert flagship_aliases["diff6_x"] == "diff6_z"
    assert flagship_aliases["diff6_m"] == "diff6_z"
    assert "diff6_y" not in flagship_aliases
    new_bytes += 4 * math.prod(flagship_shapes["diff6_y"])
    assert new_bytes == 142_677_600
    assert old_bytes - new_bytes == 141_237_600


#: Strict WRF mode's own scratch: slot -> shape on the _TINY grid (nz=4,
#: ny=6, nx=8).  The two mass planes live on state.mup's (ny, nx) plane.
_STRICT_SLOTS = {
    "acoustic_wrf_muts": (6, 8),
    "wrf_exact_muts": (6, 8),
    "wrf_exact_mu_halo": (6, 8),
    "rk_ww_ru": (4, 6, 9),
    "rk_ww_rv": (4, 7, 8),
    "moist_sc_t": (4, 6, 8),
}


def test_strict_mode_scratch_is_priced_and_builds_a_multi_domain_arena(
        monkeypatch):
    """Strict WRF mode's scratch has lifetime rows and is priced per domain.

    The breakage this prevents: scratch_slot_registry registered
    ``acoustic_wrf_muts`` under GPUWM_WRF_EXACT=1 (9190ec5f2) with no
    SCRATCH_SLOT_LIFETIME_AUDIT row, so every MULTI-domain strict run raised
    ``KeyError: scratch slot 'acoustic_wrf_muts' has no lifetime audit row``
    while woof/core/model.py priced and built its shared scratch arena
    (a single-domain run builds no arena, which is why the identity proofs
    never met it).  The five other strict allocations -- the step-carried
    mass planes, calc_ww_cp's face fluxes and the held species source --
    were priced nowhere, so a strict estimate under-read every domain.
    """
    import types

    # Price strict shapes without importing a launcher under the temporary
    # flag: acoustic captures ENABLED at import and outlives monkeypatch.
    import woof.core.acoustic  # noqa: F401
    import woof.core.state as state_mod
    import woof.wrf_exact as wrf_exact

    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    root_cfg = RunConfig(**_TINY, moist=True, mp_physics=8, specified=True)
    child_cfg = RunConfig(**_TINY, moist=True, mp_physics=8, nested=True)
    registry = pf.scratch_slot_registry(root_cfg, n_lbc_intervals=2)
    assert {slot: tuple(registry[slot]) for slot in _STRICT_SLOTS} == \
        _STRICT_SLOTS
    for slot in _STRICT_SLOTS:
        assert pf.scratch_slot_lifetime(slot) is not None, slot
        assert not pf.scratch_slot_uses_arena(slot), slot
    dry = pf.scratch_slot_registry(RunConfig(**_TINY), n_lbc_intervals=2)
    assert "moist_sc_t" not in dry and "rk_ww_ru" in dry

    tree = (types.SimpleNamespace(grid_id=1, parent_id=0, run=root_cfg),
            types.SimpleNamespace(grid_id=2, parent_id=1, run=child_cfg))
    arena = state_mod.build_shared_scratch_arena(tree)
    assert arena.nbytes == pf.shared_scratch_arena_bytes(tree)
    assert not set(_STRICT_SLOTS) & set(arena.slot_shapes)

    # The flagship tree, inputs not required: the strict estimate prices
    # exactly the six slots on every domain, and nothing else moves in
    # residency.
    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data")
    flagship = build_experiment(raw, source=str(CONFIG_4DOM))
    strict = pf.estimate_experiment(flagship)
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    default = pf.estimate_experiment(flagship)
    for dc, s_dom, d_dom in zip(flagship.domains, strict.domains,
                                default.domains):
        nz, ny, nx = dc.run.nz, dc.run.ny, dc.run.nx
        expected = 4 * (3 * ny * nx + nz * ny * (nx + 1)
                        + nz * (ny + 1) * nx + nz * ny * nx)
        # A strict process also carries WRF's 3-D PBL and cumulus tendency
        # history rows (2.9.1, physics_inventory.history_tendency_rows); a
        # default process does not.  Each is one (nz, ny, nx) FP32 plane.
        from woof.core.physics_inventory import (CUMULUS_HISTORY_RATES,
                                                  PBL_HISTORY_RATES)
        tendency_names = set(PBL_HISTORY_RATES) | {"rqiblten"} | set(CUMULUS_HISTORY_RATES)
        extra = {it.name for it in s_dom.items} - {it.name for it in d_dom.items}
        tendency = {name for name in extra
                    if name.split("/", 1)[-1] in tendency_names}
        assert tendency, "a strict process carries the tendency rows"
        expected += 4 * nz * ny * nx * len(tendency)
        # The six 0-d SCM forcing scalars are strict-only too
        # (physics_inventory.scm_forcing_fields): 4 B each.
        from woof.core.physics_inventory import SCM_SURFACE_FORCING_FIELDS
        scm = {name for name in extra
               if name.split("/", 1)[-1] in SCM_SURFACE_FORCING_FIELDS}
        assert len(scm) == len(SCM_SURFACE_FORCING_FIELDS)
        expected += 4 * len(scm)
        assert s_dom.resident_bytes - d_dom.resident_bytes == expected
        assert extra - tendency - scm == set(_STRICT_SLOTS)


def test_smag_coefficients_alias_acoustic_backings(monkeypatch):
    """Pre-RK K_m/K_h retire before acoustic alpha/gamma are prepared."""
    import types

    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    tiny = RunConfig(**{**_TINY, "nx": 12, "ny": 10, "nz": 8},
                     moist=True, mp_physics=10, km_opt=4)
    domains = (types.SimpleNamespace(run=tiny),)
    shapes = pf.shared_scratch_arena_shapes(domains)
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert aliases["smag_km"] == "acoustic_alpha"
    assert aliases["smag_kh"] == "acoustic_gamma"
    assert math.prod(shapes["smag_km"]) <= math.prod(
        shapes["acoustic_alpha"])
    assert math.prod(shapes["smag_kh"]) <= math.prod(
        shapes["acoustic_gamma"])

    arena = state_mod.build_shared_scratch_arena(domains)
    for slot, target in (("smag_km", "acoustic_alpha"),
                         ("smag_kh", "acoustic_gamma")):
        assert np.shares_memory(
            arena.view(shapes[slot], slot),
            arena.view(shapes[target], target))

    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data")
    flagship = build_experiment(raw, source=str(CONFIG_4DOM))
    flagship_shapes = pf.shared_scratch_arena_shapes(flagship.domains)
    removed = sum(
        4 * math.prod(flagship_shapes[slot])
        for slot in ("smag_km", "smag_kh"))
    assert removed == 141_120_000


@pytest.mark.parametrize("strict", [False, True])
def test_strict_nest_larger_than_its_parent_gets_its_own_relaxation_backing(
        monkeypatch, strict):
    """The two-domain strict UAE case (2026-10-28): d01 120x100, d02 210x150,
    44 levels.  Strict mode folds a nest's lateral relaxation through the
    lbc_relax_* slots too (dycore._wrf_relaxation_fold), so the shared
    arena must size them by the nest; sized by the specified parent alone,
    the first nest step raised "scratch slot 'lbc_relax_u' arena capacity is
    532400 values ((44, 100, 121)), requested 1392600 values
    ((44, 150, 211))"."""
    import types
    import woof.core.state as state_mod
    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(pf, "WRF_EXACT", strict)
    base = {**_TINY, "nz": 44}
    parent = RunConfig(**{**base, "nx": 120, "ny": 100}, specified=True)
    nest = RunConfig(**{**base, "nx": 210, "ny": 150}, nested=True)
    domains = (types.SimpleNamespace(grid_id=1, parent_id=0, run=parent),
               types.SimpleNamespace(grid_id=2, parent_id=1, run=nest))
    shapes = pf.shared_scratch_arena_shapes(domains)
    arena = state_mod.build_shared_scratch_arena(domains)
    nest_rows = {"lbc_relax_u": (44, 150, 211), "lbc_relax_v": (44, 151, 210),
                 "lbc_relax_theta": (44, 150, 210),
                 "lbc_relax_phi": (45, 150, 210), "lbc_relax_w": (45, 150, 210)}
    if strict:
        for slot, shape in nest_rows.items():
            assert shapes[slot] == shape
            arena.view(shape, slot)[...] = np.float32(1)
    else:
        # The default route recomputes a nest's held increments in
        # lbc_nested_relax and never asks a nest for these slots.
        assert shapes["lbc_relax_u"] == (44, 100, 121)
        assert "lbc_relax_w" not in shapes


@pytest.mark.parametrize("strict,diff_opt,shared", [(True,1,False),(False,1,True),(True,2,True)])
def test_coordinate_scalar_coefficient_survives_the_acoustic_loop(monkeypatch,strict,diff_opt,shared):
    """Acoustic gamma must not overwrite the deferred coordinate Kh."""
    import types
    import woof.core.state as state_mod
    monkeypatch.setattr(state_mod,"cp",np)
    monkeypatch.setattr(pf,"WRF_EXACT",strict)
    cfg=RunConfig(**{**_TINY,"nx":12,"ny":10,"nz":8},
                  moist=True,mp_physics=10,km_opt=4,diff_opt=diff_opt)
    domains=(types.SimpleNamespace(run=cfg),)
    shapes=pf.shared_scratch_arena_shapes(domains)
    arena=state_mod.build_shared_scratch_arena(domains)
    kh=arena.view(shapes["smag_kh"],"smag_kh")
    gamma=arena.view(shapes["acoustic_gamma"],"acoustic_gamma")
    kh[...] = np.float32(.125)
    gamma[...] = np.float32(9)
    assert np.shares_memory(kh,gamma) == shared
    np.testing.assert_array_equal(kh,np.float32(9 if shared else .125))


def _scan_scratch_tree(tree, rel):
    """(kind, payload, file, function) for every scratch(...) call site.

    Hardened per the review fix round: keyword-form slots
    (``scratch(shape, slot=...)`` / ``scratch(shape=..., slot=...)``) are
    inspected via ``Call.keywords``; a bare ``.scratch`` attribute LOAD
    that is not immediately called (method aliasing) and any
    ``getattr(x, "scratch")`` lookup are recorded as ``alias``/
    ``getattr`` sites so the completeness gate can reject them -- both
    were silent-skip bypasses (review F3 / shadow F4).
    """
    sites = []
    call_func_ids = {id(node.func) for node in ast.walk(tree)
                     if isinstance(node, ast.Call)}
    parents = {id(child): node for node in ast.walk(tree)
               for child in ast.iter_child_nodes(node)}

    def data_use(node):
        """Record concrete data operations separately from callable aliases."""
        parent = parents.get(id(node))
        if (isinstance(parent, ast.Subscript) and parent.value is node
                and id(parent) not in call_func_ids):
            return "indexed"
        if (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
                and parent.left is node):
            return "path"
        if (isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)
                and len(parent.values) == 2 and parent.values[0] is node
                and isinstance(parent.values[1], ast.Dict)
                and not parent.values[1].keys):
            consumer = parents.get(id(parent))
            if (isinstance(consumer, ast.Call)
                    and isinstance(consumer.func, ast.Name)
                    and consumer.func.id == "dict"):
                return "mapping"
            if (isinstance(consumer, ast.Attribute)
                    and consumer.attr in ("get", "values")
                    and id(consumer) in call_func_ids):
                return "mapping"
        return None

    def slot_node(call):
        """(slot_node_or_None, is_scratch_call)."""
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "scratch":
            pos = 1
        elif isinstance(func, ast.Name) and func.id == "scratch":
            pos = 1
        elif isinstance(func, ast.Name) and func.id == "_lbc_scratch":
            pos = 2
        else:
            return None, False
        if len(call.args) > pos:
            return call.args[pos], True
        for kw in call.keywords:
            if kw.arg == "slot":
                return kw.value, True
        return None, True

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            name = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
            if isinstance(child, ast.Call):
                slot, is_scratch = slot_node(child)
                if is_scratch:
                    if slot is None:
                        sites.append(("no_slot", None, rel, func))
                    elif isinstance(slot, ast.Constant) and isinstance(
                            slot.value, str):
                        sites.append(("literal", slot.value, rel, func))
                    elif (isinstance(slot, ast.JoinedStr) and slot.values
                          and isinstance(slot.values[0], ast.Constant)
                          and isinstance(slot.values[0].value, str)):
                        sites.append(("prefix", slot.values[0].value,
                                      rel, func))
                    elif (isinstance(slot, ast.BinOp)
                          and isinstance(slot.left, ast.Constant)
                          and isinstance(slot.left.value, str)):
                        sites.append(("prefix", slot.left.value, rel, func))
                    else:
                        sites.append(("variable", None, rel, func))
                if (isinstance(child.func, ast.Name)
                        and child.func.id == "getattr"
                        and len(child.args) >= 2
                        and isinstance(child.args[1], ast.Constant)
                        and child.args[1].value == "scratch"):
                    sites.append(("getattr", None, rel, func))
            if (isinstance(child, ast.Attribute) and child.attr == "scratch"
                    and isinstance(child.ctx, ast.Load)
                    and id(child) not in call_func_ids):
                # `sc = state.scratch` style method alias: the later
                # calls are invisible to the scanner, so the alias itself
                # is the violation.
                usage = data_use(child)
                sites.append(("data" if usage else "alias", usage, rel, func))
            visit(child, name)

    visit(tree, "<module>")
    return sites


def _scratch_call_sites():
    sites = []
    for path in sorted((ROOT / "woof").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sites += _scan_scratch_tree(tree, path.relative_to(ROOT).as_posix())
    return sites


@requires_4dom_inputs
def test_every_scratch_call_site_is_classified(d01_cfg, monkeypatch):
    """The plan's completeness gate: every ``scratch(...)`` call site in
    gpuwm/ resolves against the static registry -- literal slots must be
    registry names (or F4 manifest names for ``nest_*``), dynamic slots
    must use a registered prefix family, and variable-slot sites are
    pinned to an explicit allowlist.  An unclassified slot is an error.

    This cell skips on every machine without the 1974 bundle, so the
    2.9.0 scheme ports, lightning and strict WRF mode landed without arms
    here; it was red from at least 2.8.9 (cd336e62d) to the 2.9.0 lane tip
    and green at 8e170fd66.  The arms below cover them, and every
    generated-name site added with them is checked against the registry by
    the names it actually generates, not only allowlisted."""
    known: set[str] = set()
    for cfg in (d01_cfg,
                RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=1,
                          open_x=True, open_y=True, khdif=1.0, kvdif=1.0,
                          emdiv=0.01),
                RunConfig(**_TINY, moist=True, mp_physics=6),
                RunConfig(**_TINY, moist=True, mp_physics=8),
                RunConfig(**_TINY, moist=True, mp_physics=9),
                RunConfig(**_TINY, moist=True, moist_cq=True,
                          mp_physics=16),
                RunConfig(**_TINY, moist=True, mp_physics=18),
                RunConfig(**_TINY, moist=True, mp_physics=28),
                RunConfig(**_TINY, moist=True, mp_physics=28,
                          specified=True, aer_init_opt=1, wif_input_opt=1),
                # P3 one-category owns the only three ice-diagnostic slots
                # in the tree (p3_vmi/p3_di/p3_rhopo); without this arm they
                # are invisible to the completeness gate.  moist_cq and
                # km_opt=4 are on because P3 owns two more slot families no
                # other scheme reaches: the shared zero plane calc_cq needs
                # for its absent snow/graupel, and the smag_rqir/smag_rqib
                # held tendencies for its transported rime pair.
                RunConfig(**_TINY, moist=True, moist_cq=True,
                          mp_physics=50, km_opt=4),
                # The MPAS column-batch seam (mpas_column_batch.py:469-481)
                # builds its own ny == 1 WSM6 config and is the only
                # allocator of the ny-keyed adapter pair; without this arm
                # physics_column_alt/php are invisible to this gate.
                RunConfig(**{**_TINY, "ny": 1}, moist=True, mp_physics=6),
                RunConfig(**_TINY, sf_sfclay_physics=1),
                RunConfig(**_TINY, nwp_diagnostics=1),
                # LES closures: km_opt=3 owns the vertical exchange-
                # coefficient pair, km_opt=2 adds the prognostic-TKE
                # carrying slots and (with the toggle on) the report-only
                # budget family.  Without these arms the whole LES slot
                # family is invisible to this completeness gate.
                RunConfig(**_TINY, km_opt=3, bl_pbl_physics=0),
                RunConfig(**_TINY, km_opt=2, bl_pbl_physics=0,
                          tke_budget=1),
                RunConfig(**_TINY, km_opt=4, diff_opt=1),
                # The NOAA WRFV3.9 fork's edge-to-edge sixth-order filter
                # (the HRRR recipe default) on a specified domain with the
                # slope taper owns the four diff6_edge_* slots; without
                # this arm their call sites in dycore._diff6_edge_work are
                # invisible to this completeness gate.
                RunConfig(**_TINY, diff_6th_opt=2, specified=True,
                          diff_6th_form="noaa_wrf39", diff_6th_slopeopt=1),
                RunConfig(**_TINY, moist=True, mp_physics=1,
                          zadvect_implicit=1),
                # The UW moist-turbulence PBL owns its zero plane
                # (uwpbl_zero); without this arm its call site in
                # _run_uwpbl is invisible to this completeness gate.
                RunConfig(**_TINY, moist=True, mp_physics=8,
                          bl_pbl_physics=9, sf_sfclay_physics=1),
                # The 2.9.0 microphysics ports: Purdue Lin, WSM3, the WSM5
                # column scheme, Ferrier-Aligo, WSM7, fast SBM, WSM6R and
                # TEMPO each own preparation, precipitation and reflectivity
                # slots no other arm reaches.
                RunConfig(**_TINY, moist=True, mp_physics=2),
                RunConfig(**_TINY, moist=True, mp_physics=3),
                RunConfig(**_TINY, moist=True, mp_physics=4),
                RunConfig(**_TINY, moist=True, mp_physics=5),
                RunConfig(**_TINY, moist=True, mp_physics=24),
                RunConfig(**_TINY, moist=True, mp_physics=30),
                RunConfig(**_TINY, moist=True, mp_physics=106),
                RunConfig(**_TINY, moist=True, mp_physics=900),
                # The NOAA WRFV3.9 Thompson fork's aerosol arm holds the
                # entry ice number (mp_thompson_aero_ni_entry).
                RunConfig(**_TINY, moist=True, mp_physics=28,
                          specified=True, aer_init_opt=1, wif_input_opt=1,
                          thompson_version="wrf_39_noaa"),
                # Lightning: the flash family, its clock and calendar, and
                # option 11's cumulus cloud-top plane (ltng_ktop_deep).
                RunConfig(**_TINY, moist=True, mp_physics=8, cu_physics=5,
                          lightning_option=11),
                RunConfig(**_TINY, moist=True, mp_physics=8,
                          lightning_option=3)):
        known |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    # Strict WRF mode (GPUWM_WRF_EXACT=1, read once at import) owns the
    # step-carried mass planes, calc_ww_cp's face fluxes and the held
    # species source; without this arm they are invisible to this gate.
    import woof.wrf_exact as wrf_exact
    with monkeypatch.context() as strict:
        strict.setattr(wrf_exact, "ENABLED", True)
        for cfg in (RunConfig(**_TINY, moist=True, mp_physics=8,
                              specified=True, diff_6th_opt=2),
                    RunConfig(**_TINY, moist=True, mp_physics=8,
                              nested=True, km_opt=1, diff_opt=2)):
            known |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    exp = load_experiment_case(CONFIG_4DOM)[0]
    for dc in exp.domains:
        known |= set(pf.scratch_slot_registry(dc.run, n_lbc_intervals=2))
    manifest_names: set[str] = set()
    for slots in pf.nest_allocation_manifest(exp).values():
        manifest_names |= set(slots)

    known_prefixes = {"cu_", "smag_r", "lbc_weights_",
                      "lbc_old_mup_frame_", "lbc_relax_",
                      # spec-zone ring-guard snapshot family: registry
                      # shapes from microphysics.spec_zone_ring_save_slots,
                      # lifetime row "mp_ring_save_*" (excluded_unproven),
                      # restart REBUILT_SCRATCH_PREFIXES entry.
                      "mp_ring_save_"}
    allowed_variable_sites = {
        ("woof/core/dycore.py", "add_smag2d_tendencies"),
        ("woof/core/dycore.py", "_compute_wrf_smag_tendencies"),
        ("woof/core/dycore.py", "prepare_fixed_tendencies"),
        # The real spec producer feeds this helper the four dry carrying
        # slots. test_dry_map_coupling_uses_registered_scratch checks the
        # requests and their byte sizes through DomainState.scratch.
        ("woof/core/dycore.py", "_couple_dry_mixing_map_factor"),
        ("woof/core/dycore.py", "add_fixed_dry_tendencies"),
        ("woof/core/dycore.py", "apply_diff6"),
        ("woof/core/physics.py", "__init__"),
        ("woof/io/restart.py", "_apply_validated_restart"),
        ("woof/io/restart.py", "_restore_driver"),
        ("woof/ingest/lateral_bc.py", "_lbc_scratch"),
        ("woof/ingest/lateral_bc.py", "_resident_weights"),
        ("woof/ingest/lateral_bc.py", "attach_lateral_boundaries"),
        ("woof/ingest/lateral_bc.py", "attach_streaming_lateral_boundaries"),
        # Actual immutable forcing prices this slot; test_boundary_time_law
        # checks registered shape and byte equality in both directions.
        ("woof/ingest/lateral_bc.py", "_allocate_evaluated_interval"),
        ("woof/core/preflight.py", "run_alloc_preflight"),
        ("woof/core/nest.py", "_scratch"),
        # MYNN draws its whole declared workspace in two loops over the same
        # shape functions the registry calls, so the slot expressions are
        # variables by construction.  Allowlisting them here would be a hole
        # on its own; what closes it is
        # tests/test_mynn_pbl_scratch.py::test_the_registry_prices_exactly_
        # the_slots_the_solver_asks_for, which runs a real MYNN forecast with
        # DomainState.scratch instrumented and requires the requested slot
        # set to equal preflight.mynn_pbl_scratch_slots(cfg) exactly -- both
        # directions, so neither an unpriced slot nor a stale registry row
        # survives.
        ("woof/core/mynn_pbl_scratch.py", "from_state"),
        # P3's CUDA adapter draws its whole device working set through one
        # allocator handed to p3_device.make_workspace, plus two dict
        # comprehensions over the diagnostic and surface slot maps, so the
        # slot expressions are variables by construction -- the MYNN case
        # exactly.  What closes the hole is
        # tests/test_p3_cuda.py::test_the_registry_prices_exactly_the_slots_
        # p3_asks_for, which runs a real mp=50 step with DomainState.scratch
        # instrumented and requires the requested slot set to equal
        # preflight.scratch_slot_registry's P3 rows in BOTH directions, so
        # neither an unpriced slot nor a stale registry row survives.
        ("woof/core/p3.py", "apply"),
        ("woof/core/mynn_pbl_runtime.py", "mynn_pbl_step"),
        # The column-batch precipitation report loops over a literal tuple
        # of three mp_* names in the same statement -- every one a registry
        # row of the seam's own mp=6 config -- so the slot expression is a
        # variable only to the scanner.
        ("woof/core/mpas_column_batch.py", "accumulated_precipitation"),
        # The relocation restore iterates continuation_slots(), which IS
        # woof/io/restart.py's SERIALIZED_SCRATCH_SLOTS: the checkpoint
        # registry and the relocation inventory are one list, so an
        # unregistered slot cannot enter this loop without first failing
        # the restart manifest classification.
        ("woof/core/physics_continuation.py", "restore_continuation"),
        # A per-domain [follow] gives each independently-cadenced child its
        # own UH window on the parent, and the slot is NAMED BY GRID ID --
        # uh_diag.follow_window_slot(gid) -> "uh_follow_window.dNN" -- so
        # the expression is a variable by construction: which names exist
        # depends on which children the experiment declares, and no literal
        # can stand in for a family whose membership the config decides.
        # (Two children must not share one plane; whichever consumer read
        # first would blind the other.)
        #
        # The completeness guarantee is preserved by the PREFIX rule rather
        # than by a literal slot name, and the concrete breakage it prevents
        # is the one this family already caused: woof/io/restart.py's
        # classify_scratch_slot is total by construction -- it raises on any
        # name it does not recognise, so a new slot cannot be silently
        # dropped from every checkpoint -- and _scratch_manifest walks the
        # LIVE scratch pool through it.  While these names matched no class,
        # a run with a per-domain follow and restart_interval_s > 0 died at
        # its first checkpoint instant with no checkpoint written, and
        # carried_scratch_manifest broke the streamed carrier set the same
        # way.  CARRIED_SCRATCH_PREFIXES now classifies the whole family
        # carry, off uh_diag.UH_FOLLOW_WINDOW_PREFIX itself, which is the
        # same prefix uh_diag.is_tracker_window_slot recognises -- so the
        # allocator here and the classifier cannot drift apart, and a
        # near-miss under the same stem still fails closed on both sides.
        # tests/test_restart.py pins that in both directions.
        ("woof/core/uh_diag.py", "allocate_declared_follower_windows"),
        # Tile buffers allocate the same classified tracker slots as their
        # source state, resized to the compute window. The two-follower
        # transport controls prove exact inventory, pricing and CUDA folding.
        ("woof/core/streaming.py", "make"),
        # Cold prepared-store slabs reserve the same declared follower
        # windows before their carrier inventory freezes. The focused test
        # below executes this source's slot producer and allocation loop,
        # pinning the entire requested set, shapes, and restart classes.
        ("woof/prepared_domain_tree_forecast.py", "physics"),
        # Chemistry uses declared row slots and the fixed MONO slot tuple.
        # test_chem_variable_requests_match_priced_shapes below checks their
        # actual allocation requests against the registry with NumPy storage.
        ("woof/core/chem_transport.py", "chem_fixed_tendencies"),
        ("woof/core/chem_transport.py", "advance_chem_stage"),
        # The same fixed-tendency specs used by diff_opt=2 feed coordinate
        # mixing; test_diff_opt1 prices its additional theta pair separately.
        ("woof/core/dycore.py", "_compute_coordinate_tendencies"),
        # The birth helper only requests existing STRIP_ACCUMULATORS; the
        # relocation continuation test executes that retained-buffer path.
        ("woof/core/physics_continuation.py", "seed_birth_accumulations"),
        # These wrappers forward to BatchDomainState's admitted scratch
        # registry, which refuses new names/shapes before allocation. The
        # batch-state test checks real independent NumPy member backings.
        ("woof/ensemble/batch_acoustic.py", "scratch"),
        ("woof/ensemble/batch_boundaries.py", "scratch"),
        ("woof/ensemble/batch_moist.py", "prepare_scalars_stage"),
        # PackedThompsonState.scratch cannot allocate: its complete plan is
        # validated before launching, with wrong name/shape/dtype refusals
        # pinned by test_packed_thompson_scratch_cannot_grow below.
        ("woof/ensemble/batch_physics.py", "prepare_thompson_column_batch"),
        # The packed MYNN binder borrows exactly the original owner's priced
        # chunk slots, then rechecks their pointers only during its all-member
        # rendezvous. The witness below executes both source loops and proves
        # name/shape/dtype equality, separate member rates, exact plan bytes,
        # stable backings and refusal outside that ownership interval.
        ("woof/ensemble/batch_mynn.py", "prepare_mynn_pbl_column_batch"),
        ("woof/ensemble/batch_mynn.py", "launch"),
        # CHEM_BOUNDARY_SLOTS is the priced fixed eight-table family.
        # test_chem_cams_boundary exercises its actual packed NumPy tables.
        ("woof/ingest/lateral_bc.py", "chem_boundary_rows"),
        # --- Sites that landed while this cell skipped (see the docstring).
        # Every name each one generates is checked against the registry
        # after the scan, so these entries admit no unpriced slot.
        # Strict WRF mode names its slots by module constant
        # (acoustic.WRF_ACOUSTIC_MUTS_SLOT, dycore.WRF_MUTS_SLOT and
        # dycore.WRF_MU_HALO_SLOT).
        ("woof/core/acoustic.py", "wrf_acoustic_muts"),
        ("woof/core/dycore.py", "_capture_wrf_muts"),
        ("woof/core/dycore.py", "_wrf_step_epilogue"),
        # Strict mode's WRF-order folds walk the same fixed-tendency specs
        # rows as prepare_fixed_tendencies above (the registered smag_r
        # family), and fold_lateral_relaxation walks _FOLD_ROWS' held dry
        # slots.
        ("woof/core/dycore.py", "_compute_wrf_constant_k_tendencies"),
        ("woof/core/dycore.py", "_fold_physics_wrf"),
        ("woof/core/dycore.py", "_accumulate_diff6_wrf_order"),
        ("woof/core/dycore.py", "fold_lateral_relaxation"),
        # Scheme adapters that take their names from their own tables:
        # fast SBM's surface map, the column-scheme row's surface tuple, the
        # MPAS column suite's microphysics_scratch_slots, the generated
        # Thompson column's literal preparation tuple, WSM6R's prefix and
        # the WSM family row's name.
        ("woof/core/fast_sbm_runtime.py", "apply"),
        ("woof/core/microphysics.py", "_apply_column_scheme"),
        ("woof/core/mpas_column_suite.py", "accumulated_precipitation"),
        ("woof/core/thompson_column.py", "apply_generated"),
        ("woof/core/wsm6.py", "_wsm6_prepare"),
        ("woof/core/wsm_family.py", "apply"),
        # Correct mode's Thompson cloud booking (WB1, 2.9.0 c4a312f04) names
        # its two surface totals by module constant
        # (thompson_cloud_booking.CLOUDNC_SLOT / CLOUDNCV_SLOT).
        ("woof/core/thompson_cloud_booking.py", "book_settled_cloud"),
    }
    # A dynamic prefix one adapter generates from its own table, pinned BY
    # SITE rather than added to known_prefixes, so it admits nothing
    # anywhere else (``mp_`` and ``physics_`` are far too wide to admit
    # globally).  The generated names are checked after the scan.
    allowed_prefix_sites = {
        ("woof/core/lightning.py", "initialize_lightning", "ltng_"),
        ("woof/core/lightning.py", "update_lightning", "ltng_"),
        ("woof/core/fast_sbm_runtime.py", "apply", "fsbm_frame_"),
        ("woof/core/mpas_column_suite.py", "run_phase1", "physics_"),
        ("woof/core/wsm_family.py", "apply", "mp_"),
    }
    # The one sanctioned hand-off of the bound allocator: fast_sbm_runtime
    # .apply passes state.scratch to fast_sbm_cuda.launch, whose own literal
    # scratch(...) calls this scanner sees and classifies (asserted after
    # the scan), so no call escapes the gate.
    allowed_alias_sites = {("woof/core/fast_sbm_runtime.py", "apply")}
    # Data named scratch is still inventoried, but is not a bound allocator
    # method. These exact uses are tied to concrete tuple, prepared mapping,
    # and Path owners; a new data use or a callable alias still fails closed.
    known_data_sites = {
        ("woof/core/chem_advect_mono.py", "launch_mono_renorm_apply", "indexed"),
        ("woof/ensemble/batch_state.py", "_validate_prepared", "mapping"),
        ("woof/ensemble/batch_state.py", "from_prepared", "mapping"),
        ("woof/ensemble/prepared_batch.py", "nbytes", "mapping"),
        ("woof/ensemble/prepared_batch.py", "prepare_native_member_batch", "mapping"),
        ("woof/gfs_direct.py", "_decode_batch", "path"),
    }
    # The one sanctioned getattr(state, "scratch") lookup: lateral_bc's
    # duck-type guard, whose resulting Name call the scanner classifies.
    allowed_getattr_sites = {
        ("woof/ingest/lateral_bc.py", "_lbc_scratch"),
    }

    sites = _scratch_call_sites()
    assert sites, "AST scan found no scratch call sites -- scanner broken"
    problems = []
    seen_variable_sites = set()
    seen_data_sites = set()
    seen_prefix_sites = set()
    seen_alias_sites = set()
    for kind, payload, rel, func in sites:
        if kind == "literal":
            if payload.startswith("nest_"):
                if payload not in manifest_names:
                    problems.append(f"{rel}::{func}: nest slot {payload!r} "
                                    "is not in the F4 allocation manifest")
            elif payload not in known:
                problems.append(f"{rel}::{func}: slot {payload!r} is not in "
                                "the scratch registry")
        elif kind == "prefix":
            if (rel, func, payload) in allowed_prefix_sites:
                seen_prefix_sites.add((rel, func, payload))
            elif payload not in known_prefixes:
                problems.append(f"{rel}::{func}: dynamic slot prefix "
                                f"{payload!r} is not a registered family")
        elif kind == "alias" and (rel, func) in allowed_alias_sites:
            seen_alias_sites.add((rel, func))
        elif kind == "variable":
            seen_variable_sites.add((rel, func))
            if (rel, func) not in allowed_variable_sites:
                problems.append(f"{rel}::{func}: variable slot expression "
                                "is not in the pinned allowlist")
        elif kind == "getattr":
            if (rel, func) not in allowed_getattr_sites:
                problems.append(f"{rel}::{func}: getattr(..., 'scratch') "
                                "lookup escapes the completeness gate")
        elif kind == "data":
            seen_data_sites.add((rel, func, payload))
            if (rel, func, payload) not in known_data_sites:
                problems.append(f"{rel}::{func}: {payload} scratch data "
                                "has no recorded owner")
        else:  # "alias" / "no_slot": never legitimate
            problems.append(f"{rel}::{func}: {kind} scratch usage escapes "
                            "the completeness gate")
    assert not problems, "\n".join(problems)
    # The allowlist may not silently rot either.
    assert seen_variable_sites == allowed_variable_sites
    assert seen_data_sites == known_data_sites
    assert seen_prefix_sites == allowed_prefix_sites
    assert seen_alias_sites == allowed_alias_sites
    # The callee the fast-SBM hand-off reaches is itself scanned: its six
    # literal slots were classified above.
    assert {payload for kind, payload, rel, func in sites
            if (kind, rel, func) == ("literal", "woof/core/fast_sbm_cuda.py",
                                     "launch")} == {
        "fsbm_dims", "fsbm_pointers", "fsbm_work", "fsbm_cloud_base",
        "fsbm_tally", "fsbm_status"}

    # Every name the sites admitted above by site actually generate, from
    # the same tables the sites read, is a registry slot.
    from woof.core import wsm_family
    from woof.core.acoustic import WRF_ACOUSTIC_MUTS_SLOT
    from woof.core.column_schemes import COLUMN_SCHEMES
    from woof.core.fast_sbm_contract import STATE_FIELDS as FSBM_FIELDS
    from woof.core.fast_sbm_runtime import _SURFACE_SLOTS
    from woof.core.physics_inventory import microphysics_scratch_slots
    from woof.core.thompson_cloud_booking import CLOUDNC_SLOT, CLOUDNCV_SLOT
    from woof.lightning_contract import LIGHTNING_OPTIONS

    # woof.core.dycore imports CuPy at module scope, which would turn this
    # CPU gate into a skip; its three constants are read from the source.
    dycore_constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse((ROOT / "woof/core/dycore.py").read_text(
            encoding="utf-8")).body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("WRF_MUTS_SLOT", "WRF_MU_HALO_SLOT",
                                   "_FOLD_ROWS")}
    generated = {WRF_ACOUSTIC_MUTS_SLOT, dycore_constants["WRF_MUTS_SLOT"],
                 dycore_constants["WRF_MU_HALO_SLOT"]}
    generated |= {slot for slot, _msf in
                  dycore_constants["_FOLD_ROWS"].values()}
    for mp in wsm_family.family_selectors():
        generated |= set(wsm_family.preparation_slots(mp))
        generated |= {f"mp_{cls}{suffix}"
                      for cls in wsm_family.row(mp).precip
                      for suffix in ("nc", "ncv")}
    generated |= {f"wsm6r_{field}"
                  for field in ("theta", "rho", "pii", "dz", "z8w")}
    for row in COLUMN_SCHEMES.values():
        generated |= set(row.surface)
    generated |= {"fsbm_frame_" + name for name in FSBM_FIELDS}
    generated |= set(_SURFACE_SLOTS.values())
    generated |= {"ltng_" + name for option in LIGHTNING_OPTIONS.values()
                  for name in option["fields"]}
    generated |= {"ltng_" + name for name in
                  ("iccg_in_num", "iccg_in_den", "lat", "lon")}
    generated |= {"physics_qi", "physics_qs"}
    generated |= {CLOUDNC_SLOT, CLOUDNCV_SLOT}
    generated |= {"mp_th", "mp_pii", "mp_thompson_temperature", "mp_dz8w"}
    for mp in _INTAKE_MP:
        if mp:
            generated |= {slot for _component, slot in
                          microphysics_scratch_slots(mp)}
    assert generated <= known, sorted(generated - known)


def test_prepared_store_follower_slots_match_declared_registry():
    """Bind the scanner exemption to the actual producer and allocation loop."""
    from types import SimpleNamespace
    from woof.core.uh_diag import declared_follower_slots

    path = ROOT / "woof/prepared_domain_tree_forecast.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    restore = next(node for node in ast.walk(source)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "restore_store_domain")
    physics = next(node for node in restore.body
                   if isinstance(node, ast.FunctionDef) and node.name == "physics")
    producer = [node for node in restore.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "slots"
                        for target in node.targets)]
    loops = [node for node in physics.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Name) and node.iter.id == "slots"]
    assert len(producer) == len(loops) == 1
    sites = _scan_scratch_tree(ast.Module(body=[physics], type_ignores=[]),
                              path.relative_to(ROOT).as_posix())
    assert sites == [("variable", None, "woof/prepared_domain_tree_forecast.py", "physics")]
    code = compile(ast.Module(body=[producer[0], loops[0]], type_ignores=[]), str(path), "exec")
    domains = [SimpleNamespace(grid_id=gid, parent_id=parent, follow=follow)
               for gid, parent, follow in ((1, 0, None), (2, 1, object()),
                                           (3, 1, object()), (4, 2, object()),
                                           (5, 2, None))]
    expected = {1: ("uh_follow_window.d02", "uh_follow_window.d03"),
                2: ("uh_follow_window.d04",), 3: (), 4: (), 5: ()}
    cfg = SimpleNamespace(ny=7, nx=11)
    for domain in domains:
        requests = []

        def scratch(shape, slot):
            requests.append((shape, slot))

        exec(code, {"declared_follower_slots": declared_follower_slots,
                    "exp": SimpleNamespace(domains=domains), "domain": domain,
                    "result": SimpleNamespace(state=SimpleNamespace(scratch=scratch)),
                    "cfg": cfg})
        assert requests == [((7, 11), slot) for slot in expected[domain.grid_id]]
        assert all(restart.classify_scratch_slot(slot) == "carry" for _, slot in requests)


def test_experimental_thompson_scratch_registry_is_complete():
    cfg = RunConfig(**_TINY, moist=True, mp_physics=8)
    slots = pf.scratch_slot_registry(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    assert {
        "mp_th": mass,
        "mp_pii": mass,
        "mp_dz8w": mass,
        "mp_z8w": (cfg.nz + 1, cfg.ny, cfg.nx),
        "mp_thompson_temperature": mass,
        "mp_thompson_frozen_reference_density": mass,
        "mp_thompson_frozen_reference_temperature": mass,
        "mp_thompson_rain_reference_density": mass,
        "mp_thompson_snow_melt_marker": mass,
        "mp_thompson_graupel_melt_marker": mass,
        "mp_thompson_snow_velocity_boost": mass,
        "mp_thompson_graupel_number_shadow": mass,
        # WRF's per-column no_micro flag (:1646, :2020), repair G.
        "mp_thompson_micro_columns": surface,
        "mp_rainnc": surface,
        "mp_rainncv": surface,
        "mp_snownc": surface,
        "mp_snowncv": surface,
        "mp_graupelnc": surface,
        "mp_graupelncv": surface,
        "mp_sr": surface,
        "refl_t": mass,
        "refl_10cm": mass,
    }.items() <= slots.items()
    for slot in (
            "mp_thompson_temperature",
            "mp_thompson_frozen_reference_density",
            "mp_thompson_frozen_reference_temperature",
            "mp_thompson_rain_reference_density",
            "mp_thompson_snow_melt_marker",
            "mp_thompson_graupel_melt_marker",
            "mp_thompson_snow_velocity_boost",
            "mp_thompson_graupel_number_shadow",
            "mp_thompson_micro_columns"):
        assert pf.scratch_slot_uses_arena(slot)



def test_strict_classic_thompson_scratch_registry_is_complete(monkeypatch):
    """Under GPUWM_WRF_EXACT=1, mp=8's priced scratch is exactly what
    apply_classic draws (a default process prices 2.9.1's adapter, above).

    Breakage prevented: a slot the adapter draws that the registry does not
    price is memory the fit gate never charged; a slot priced but not drawn
    reserves memory the run never uses.  The classic unit's working set is
    plane-stacked (planes*levels, ny, nx): eleven tendencies and eighteen
    working fields per level, eleven column profiles per interface level.
    """
    from woof.core import thompson_classic as tcl
    monkeypatch.setattr(pf, "WRF_EXACT", True)
    cfg = RunConfig(**_TINY, moist=True, mp_physics=8)
    slots = pf.scratch_slot_registry(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    expected = {
        "mp_th": mass,
        "mp_pii": mass,
        "mp_dz8w": mass,
        "mp_thompson_temperature": mass,
        "mp_thompson_saved_theta": mass,
        "mp_thompson_graupel_number_shadow": mass,
        "mp_thompson_classic_ten": (tcl.NTEN * cfg.nz, cfg.ny, cfg.nx),
        "mp_thompson_classic_work": (tcl.NWORK * cfg.nz, cfg.ny, cfg.nx),
        "mp_thompson_classic_colbuf": (tcl.NCOLPLANES * (cfg.nz + 1),
                                       cfg.ny, cfg.nx),
        "mp_thompson_classic_flags": mass,
        "mp_thompson_classic_micro": surface,
        "mp_thompson_classic_kmelt": surface,
        "mp_rainnc": surface,
        "mp_rainncv": surface,
        "mp_snownc": surface,
        "mp_snowncv": surface,
        "mp_graupelnc": surface,
        "mp_graupelncv": surface,
        "mp_sr": surface,
        "refl_10cm": mass,
    }
    assert expected.items() <= slots.items()
    # the old adapter's slots are gone from mp=8's price
    for gone in ("mp_z8w", "refl_t", "mp_thompson_micro_columns",
                 "mp_thompson_frozen_reference_density",
                 "mp_thompson_snow_velocity_boost"):
        assert gone not in slots, gone
    for slot in expected:
        if slot.startswith("mp_thompson_"):
            assert pf.scratch_slot_uses_arena(slot), slot


# ---------------------------------------------------------------------------
# mp_physics=28 -- Thompson aerosol-aware.  These pin the state, scratch,
# nest and pricing inventories the rest of the port builds on, and they pin
# the two places a mistake would be invisible: the transport discriminator
# and the mp=8/mp=10 non-interference.
# ---------------------------------------------------------------------------

_MP28 = dict(moist=True, moist_cq=True, mp_physics=28)


def test_mp28_state_allocation_inventory():
    cfg = RunConfig(**_TINY, **_MP28)
    shapes = pf.state_array_shapes(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    # Prognostic scalars + their RK time-t copies.
    for name in ("nc", "nr", "ni", "nwfa", "nifa",
                 "nc0", "nr0", "ni0", "nwfa0", "nifa0",
                 "qi", "qs", "qg", "qi0", "qs0", "qg0",
                 "effc", "effi", "effs"):
        assert shapes[name] == mass, name
    # Surface emission tendencies, 2-D and cross-step constant.
    assert shapes["nwfa2d"] == surface
    assert shapes["nifa2d"] == surface
    # Thompson has no effr; Morrison/NSSL-only fields must not appear.
    for name in ("effr", "ns", "ng", "ns0", "ng0", "qh", "qnn"):
        assert name not in shapes, name


def test_mp28_state_allocation_matches_the_real_domain_state():
    """The shape manifest is a transcription of state.py; prove it.

    A manifest that drifts from the constructor is worse than no manifest:
    the arena is sized from the manifest and bound to the constructor.
    """
    import types

    import woof.core.state as state_mod

    cfg = RunConfig(**_TINY, **_MP28)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        state = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    declared = pf.state_array_shapes(cfg)
    actual = {name: tuple(value.shape)
              for name, value in vars(state).items()
              if isinstance(value, np.ndarray)}
    assert actual == declared
    assert isinstance(state, types.SimpleNamespace) is False  # sanity


def test_mp28_transports_droplet_and_aerosol_number_but_mp10_does_not():
    """The transport gate.  This is the one mp=28 decision whose mistake is
    both silent and expensive: mp_physics=10 ALREADY allocates ``state.nc``
    and deliberately does not transport it, so a presence-of-``nc`` test
    would start advecting Morrison's diagnostic droplet number through every
    generic dycore consumer and move a validated trajectory.  The
    discriminator must be ``nwfa``, which exactly one scheme allocates.
    """
    import woof.core.state as state_mod
    from woof.core import moist

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        mp8 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=8))
        mp10 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=10))
        mp28 = state_mod.DomainState(RunConfig(**_TINY, **_MP28))
    finally:
        monkey.undo()

    # The frozen receipts (tests/test_mp8_frozen.py R3) restated here so a
    # change to moist.py fails in its own test file too.
    assert moist.extra_moist_species(mp8) == ("qi", "qs", "qg", "nr", "ni")
    assert moist.extra_moist_species(mp10) == (
        "qi", "qs", "qg", "nr", "ni", "ns", "ng")
    assert moist.extra_moist_species(mp28) == (
        "qi", "qs", "qg", "nr", "ni", "nc", "nwfa", "nifa")

    # Morrison really does own an nc that is really not transported.
    assert getattr(mp10, "nc", None) is not None
    assert "nc" not in moist.extra_moist_species(mp10)
    assert getattr(mp10, "nc0", None) is None
    # ... and the discriminator is unique to mp=28.
    assert getattr(mp8, "nwfa", None) is None
    assert getattr(mp10, "nwfa", None) is None
    assert getattr(mp28, "nwfa", None) is not None
    # The generic filter itself must not have been widened.
    assert moist.TRANSPORTED_NUMBER_SPECIES == ("nr", "ni", "ns", "ng")


def test_mp28_scratch_registry_is_complete():
    cfg = RunConfig(**_TINY, **_MP28)
    slots = pf.scratch_slot_registry(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    classic = {
        "mp_th": mass,
        "mp_pii": mass,
        "mp_dz8w": mass,
        "mp_z8w": (cfg.nz + 1, cfg.ny, cfg.nx),
        "mp_thompson_temperature": mass,
        "mp_thompson_frozen_reference_density": mass,
        "mp_thompson_frozen_reference_temperature": mass,
        "mp_thompson_rain_reference_density": mass,
        "mp_thompson_snow_melt_marker": mass,
        "mp_thompson_graupel_melt_marker": mass,
        "mp_thompson_snow_velocity_boost": mass,
        "mp_thompson_graupel_number_shadow": mass,
        # WRF's per-column no_micro flag (:1646, :2020), repair G.
        "mp_thompson_micro_columns": surface,
        "mp_rainnc": surface,
        "mp_rainncv": surface,
        "mp_snownc": surface,
        "mp_snowncv": surface,
        "mp_graupelnc": surface,
        "mp_graupelncv": surface,
        "mp_sr": surface,
        "refl_t": mass,
        "refl_10cm": mass,
    }
    assert classic.items() <= slots.items()
    aerosol = {name: mass for name in (
        "mp_thompson_aero_ncten",
        "mp_thompson_aero_nwfaten",
        "mp_thompson_aero_nifaten",
        "mp_thompson_aero_entry_density",
        "mp_thompson_aero_tau1_density",
        "mp_thompson_aero_nwfa_work_m3",
        "mp_thompson_aero_qcten",
        "mp_thompson_aero_qrten",
        "mp_thompson_aero_nrten",
        "mp_thompson_aero_qiten",
        "mp_thompson_aero_niten",
        "mp_thompson_aero_qvten",
        "mp_thompson_aero_tten",
        "mp_thompson_aero_condensation_rate",
    )}
    assert aerosol.items() <= slots.items()
    # The v4.6.1 generation carries no entry-ice copy; the fork, which keeps
    # its rain and ice in place, carries that copy and no rain or ice
    # accumulator.
    assert "mp_thompson_aero_ni_entry" not in slots
    # All fifteen are arena-eligible, and the audit row that says so is a
    # write_before_read row with real evidence.
    for slot in aerosol:
        assert pf.scratch_slot_uses_arena(slot), slot
        row = pf.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "write_before_read"
        assert row.evidence and row.rationale
    # mp=28 owns qi/qs, so the physics prep must NOT substitute zero planes.
    assert "physics_qi" not in slots and "physics_qs" not in slots
    # The aerosol slots belong to mp=28 alone.
    mp8_slots = set(pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=8)))
    assert not (set(aerosol) & mp8_slots)


def test_mp28_every_scratch_slot_is_classified_for_restart():
    """woof/io/restart.py.  ``classify_scratch_slot`` fails CLOSED, and its
    ``mp_`` rule is exact-names-only precisely so a new accumulator cannot be
    silently dropped from a checkpoint.  Every slot mp=28 can create must
    therefore carry an explicit classification.

    All fifteen aerosol slots are ``rebuild``, and that is the physics: WRF
    zeroes ncten/nwfaten/nifaten at the top of every column call
    (module_mp_thompson.F:1679-1681) and applies them once before returning
    (:3972-4021).  Serializing a tendency that has already been applied would
    apply it a second time on the resumed step.
    """
    from woof.io import restart as restart_mod

    slots = set(pf.scratch_slot_registry(
        RunConfig(**_TINY, **_MP28, km_opt=4, diff_6th_opt=2,
                  specified=True), n_lbc_intervals=2))
    assert slots
    for slot in sorted(slots):
        kind = restart_mod.classify_scratch_slot(slot)
        assert kind in ("serialize", "rebuild"), (slot, kind)
    for slot in sorted(s for s in slots if s.startswith("mp_thompson_aero_")):
        assert restart_mod.classify_scratch_slot(slot) == "rebuild", slot
        assert slot not in restart_mod.SERIALIZED_SCRATCH_SLOTS


def test_mp28_smag_held_tendencies_cover_every_transported_species():
    from woof.core import moist

    cfg = RunConfig(**_TINY, **_MP28, km_opt=4)
    slots = pf.scratch_slot_registry(cfg)
    import woof.core.state as state_mod
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        state = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    for name in moist.moist_species(state):
        assert "smag_r" + name in slots, name
    # Morrison's untransported nc keeps no held tendency.
    mp10 = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, mp_physics=10, km_opt=4))
    assert "smag_rnc" not in mp10


def test_mp50_smag_held_tendencies_are_exactly_p3s_species():
    """The held-tendency row for the one scheme with qi and NO qs/qg.

    The ice-mass admission tuple in ``scratch_slot_registry`` (the
    ``(6, 8, 9, 10, 16, 18, 24, 28)`` arm) prices ``smag_rqi``/``rqs``/``rqg``
    together, because until P3 every scheme with cloud ice also had snow
    and graupel.  mp=50 is deliberately absent from it and takes its own
    arm instead: Registry.EM_COMMON:3038 declares P3's package as
    ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib``, so WRF's moist array has
    no snow or graupel index under P3 and the loop that produces these
    tendencies (``do im = PARAM_FIRST_SCALAR, n_moist``,
    module_diffusion_em.F:3036) never reaches one.

    Asserted in BOTH directions, because each direction is a different
    defect.  Widening the ice-mass tuple to 50 would price two full
    (nz, ny, nx) fields no mp=50 state allocates -- a headroom estimate
    that overstates the run on the card it exists to protect.  Dropping
    P3's own arm would silently stop pricing the transported rime pair,
    which is the estimate understating a run that then meets the arena
    short.  The equality below fails on either.

    ``tests/test_p3_port.py`` makes the same claim, but that whole module
    imports cupy and is therefore marked ``gpu`` and skipped on every
    CPU-only invocation (tests/conftest.py:228-259).  This registry is
    priced on CPU-only installs, so its gate has to run there too.
    """
    from woof.core import moist

    cfg = RunConfig(**_TINY, moist=True, mp_physics=50, km_opt=4)
    slots = pf.scratch_slot_registry(cfg)
    import woof.core.state as state_mod
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        state = state_mod.DomainState(cfg)
    finally:
        monkey.undo()

    # The four dynamics rows of _smag2d_specs carry no species name.
    dynamics = {"smag_ru", "smag_rv", "smag_rw", "smag_rth", "smag_rtke"}
    priced = {slot[len("smag_r"):] for slot in slots
              if slot.startswith("smag_r") and slot not in dynamics}
    assert priced == set(moist.moist_species(state))
    assert moist.extra_moist_species(state) == moist.P3_SPECIES

    # Named rather than left to the set equality: these are the two
    # fields the ice-mass tuple would add if 50 joined it, and P3 has
    # neither on the state at all.
    for absent in ("qs", "qg"):
        assert "smag_r" + absent not in slots, absent
        assert getattr(state, absent, None) is None, absent
    # ...and the rime pair, which only P3's own arm prices.
    for present in ("qi", "qir", "qib"):
        assert "smag_r" + present in slots, present


def test_mp28_nest_field_kinds_and_pricing():
    cfg = RunConfig(**_TINY, **_MP28)
    assert pf.nest_field_kinds(cfg) == (
        "u", "v", "w", "t", "ph", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "nr", "ni", "nc", "nwfa", "nifa")
    # mp=10 stays exactly as ratified -- no nc.
    assert pf.nest_field_kinds(
        RunConfig(**_TINY, moist=True, mp_physics=10)) == (
        "u", "v", "w", "t", "ph", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng")


def test_mp28_is_priced_and_never_falls_through_to_a_guess():
    """``domain_kernel_modules`` fails closed on an unpriced selector; mp=28
    must be a priced row, and it must name the modules the adapter really
    launches -- including the frozen mp=8 ``thompson`` module, whose
    sedimentation launchers mp=28 reuses byte-for-byte."""
    from datetime import datetime as _dt

    from woof.experiment import experiment_from_run_config

    cfg = RunConfig(**_TINY, **_MP28, output_interval_s=1.0)
    exp = experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    for name in ("thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
                 "thompson_aerosol_cold", "thompson_aerosol_warm",
                 "thompson_aerosol_sed"):
        assert name in modules, name
    # The probe translation unit is oracle-only and must never be priced
    # into a forecast's local-memory reservation.
    assert "thompson_aerosol_probe" not in modules
    # Every priced module has a driver-measured frame.
    frames = pf.kernel_local_frame_bytes(exp)
    for name in modules:
        assert name in frames, name
    assert pf.kernel_local_memory_bytes(exp) >= 0


def test_mp9_is_priced_and_never_falls_through_to_a_guess():
    """mp=9 must be a priced row, with the frame its kernels really hold.

    The two halves of pricing a new scheme fail in different directions and
    only one of them is loud.  A missing ``KERNEL_MAX_LOCAL_SIZE_BYTES`` row
    is caught by the driver-measured sweep, but only on a GPU box.  A missing
    ``_MICROPHYSICS_KERNEL_MODULES`` row is caught nowhere at all until a
    user runs ``woof check`` on an mp=9 config, at which point
    ``domain_kernel_modules`` fails closed and the scheme cannot be priced
    or gated -- config admits the selector, preflight refuses it.  So this
    asserts BOTH ends meet.
    """
    from datetime import datetime as _dt

    from woof.experiment import experiment_from_run_config

    cfg = RunConfig(**_TINY, moist=True, mp_physics=9, output_interval_s=1.0)
    exp = experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    assert "milbrandt2" in modules
    assert "microphysics_validation" in modules
    # mp=9 fills the REFL_10CM slot from its own diagnostics kernel and only
    # stashes the array (woof/core/milbrandt2.py:291-296), so the shared
    # refl module is never loaded and must never be priced.
    assert "refl" not in modules
    assert 9 not in pf._REFLECTIVITY_MICROPHYSICS

    frames = pf.kernel_local_frame_bytes(exp)
    for name in modules:
        assert name in frames, name
    # The row is the module CEILING, not this configuration's price: _TINY
    # is nz=4, so the launcher takes milbrandt2_sediment_64 at 512 B while
    # the table carries milbrandt2_sediment_256's 2,048 B.  Over-pricing is
    # the safe direction for a rail gate, and it is what the table's own
    # header says these rows mean.
    assert frames["milbrandt2"] == 2048
    assert pf.kernel_local_memory_bytes(exp) >= 0


def test_every_moist_scheme_has_a_written_reflectivity_rail_decision():
    """Out of ``_REFLECTIVITY_MICROPHYSICS`` must be a RULING, not a gap.

    The set decides one thing: does this selector reserve the shared
    reflectivity translation unit's per-thread frame.  Two answers are
    correct and they fail in opposite directions -- an over-priced rail
    refuses a run that would have fit, an under-priced one lets a run
    breach the budget ``woof check`` cleared it against.  So a moist
    selector that is in neither set is not "excluded", it is undecided,
    and ``domain_kernel_modules`` now refuses it by name.
    """
    priced = set(pf._MICROPHYSICS_KERNEL_MODULES) - {0}
    decided = (set(pf._REFLECTIVITY_MICROPHYSICS)
               | set(pf._SELF_REFLECTIVITY_MICROPHYSICS))
    assert priced - decided == set(), (
        "these moist selectors are priced for microphysics kernels but "
        "have no reflectivity-rail decision: "
        f"{sorted(priced - decided)}")
    assert (set(pf._REFLECTIVITY_MICROPHYSICS)
            & set(pf._SELF_REFLECTIVITY_MICROPHYSICS)) == set(), (
        "a selector cannot both load refl.cu and fill REFL_10CM from its "
        "own kernels")
    for scheme, reason in pf._SELF_REFLECTIVITY_MICROPHYSICS.items():
        assert "stash_refl_10cm" in reason, (
            f"mp={scheme}'s recorded reason must name the seam it uses "
            "instead of refl.cu")


def test_p3_is_out_of_the_reflectivity_rail_by_a_named_decision():
    """mp=50 is CORRECTLY out, and the tree says so instead of omitting it.

    P3 computes REFL_10CM inside ``p3_main``'s own final-checks-and-
    diagnostics loop (phys/module_mp_p3.F:4722-4895 -- ``ze_rain`` from the
    rain gamma moment, ``ze_ice`` from ice lookup-table column 9) and hands
    the finished array to ``stash_refl_10cm`` from both arms
    (woof/core/p3.py:1831-1835 reference, :1982-1984 device), never to
    ``compute_and_stash_refl_10cm``.  It could not use refl.cu even if the
    rail priced it: those kernels transcribe the ``calc_refl10cm`` family,
    whose Rayleigh sums read qs and qg, and P3 is ONE ice category with a
    rime pair (qir/qib) and neither species.

    WHAT ADMITTING IT WOULD COST, measured below rather than asserted: on
    an mp=50 domain ``refl`` is the WIDEST frame in the configuration, so
    pricing a kernel the run never launches moves the local-memory
    reservation off zero and shrinks the envelope every other rail is
    checked against.
    """
    from datetime import datetime as _dt

    from woof.experiment import experiment_from_run_config

    assert 50 not in pf._REFLECTIVITY_MICROPHYSICS
    assert 50 in pf._SELF_REFLECTIVITY_MICROPHYSICS
    reason = pf._SELF_REFLECTIVITY_MICROPHYSICS[50]
    assert "module_mp_p3.F" in reason and "qs and qg" in reason

    big = dict(nx=200, ny=200, nz=50, dx=3000.0, dy=3000.0, ztop=20000.0,
               dt=15.0, run_seconds=3600.0)
    # 97e9a3ca061 (cap-hail): automatic HAILCAST at this 3-km spacing loads
    # the 7,392-B hail frame, wider than refl's 5,184 B, which would hide
    # the rail's cost behind it; the consequence is measured with HAILCAST
    # off, where refl is the widest frame the docstring names.
    cfg = RunConfig(**big, moist=True, mp_physics=50,
                    output_interval_s=900.0, hailcast_opt=0)
    exp = experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    assert "p3_composed" in modules and "hail" not in modules
    assert "refl" not in modules and "wdm6_refl" not in modules
    priced = pf.kernel_local_memory_bytes(exp)

    original = pf._REFLECTIVITY_MICROPHYSICS
    try:
        pf._REFLECTIVITY_MICROPHYSICS = original | {50}
        overpriced = pf.kernel_local_memory_bytes(exp)
    finally:
        pf._REFLECTIVITY_MICROPHYSICS = original
    assert overpriced > priced, (
        "admitting mp=50 must be measurably more expensive, or this "
        "ruling has no consequence to record")


def test_a_moist_scheme_in_neither_reflectivity_set_is_refused_by_name():
    """The rail fails CLOSED, so the next scheme's omission cannot be silent.

    Before this refusal existed, a selector added to
    ``_MICROPHYSICS_KERNEL_MODULES`` and to nothing else priced zero
    reflectivity frame and said nothing: ``woof check`` passed against a
    budget that never counted refl.cu, and the run breached it at the first
    history step.
    """
    from datetime import datetime as _dt

    from woof.experiment import experiment_from_run_config

    cfg = RunConfig(**_TINY, moist=True, mp_physics=50,
                    output_interval_s=1.0)
    exp = experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))
    recorded = dict(pf._SELF_REFLECTIVITY_MICROPHYSICS)
    try:
        pf._SELF_REFLECTIVITY_MICROPHYSICS = {
            k: v for k, v in recorded.items() if k != 50}
        with pytest.raises(ValueError) as excinfo:
            pf.physics_kernel_modules(exp)
    finally:
        pf._SELF_REFLECTIVITY_MICROPHYSICS = recorded
    message = str(excinfo.value)
    assert "mp_physics=50" in message
    assert "_SELF_REFLECTIVITY_MICROPHYSICS" in message
    assert "under-prices" in message


# ---------------------------------------------------------------------------
# mp_physics=28 -- the PhysicsDriver budgets.
#
# Deliberately built from ``_TINY`` rather than from the four-domain flagship
# fixture: the flagship configs reference an external ERA5 forcing file that a
# clean checkout does not carry, so every test keyed on them ERRORS at
# collection and could not gate anything.  These three run anywhere.
# ---------------------------------------------------------------------------

_MP28_PBL = dict(bl_pbl_physics=1, sf_sfclay_physics=1)


def test_mp28_physics_driver_budget_admits_the_pbl_ice_tendency():
    """``pbl_tendencies/rqi`` is priced for mp=28 + YSU.

    BEFORE THIS TEST: ``preflight.py:1400`` read ``(6, 8, 10, 18)`` and an
    mp=28 + YSU domain's ``rqi`` stack was neither budgeted nor materialized,
    so the ``--alloc`` measurement understated that run's persistent driver
    set by one mass-grid array (two with a separate composed target).

    ``Registry/Registry.EM_COMMON:3036`` declares the ``thompsonaero``
    package as ``moist:qv,qc,qr,qi,qs,qg``, which is what makes WRF's
    ``F_QI`` true and ``module_first_rk_step_part1.F:1112``'s
    ``CALL pbl_driver`` pass ``moist(...,P_QI), F_QI=F_QI`` (:1199).
    """
    cfg = RunConfig(**_TINY, **_MP28, **_MP28_PBL)
    shapes = pf.physics_array_shapes(cfg)
    assert shapes["pbl_tendencies/rqi"] == (cfg.nz, cfg.ny, cfg.nx)


def test_the_pbl_rqi_budget_matches_the_runtime_predicate_for_every_scheme():
    """The budget and the runtime must not be able to disagree.

    ``preflight.physics_array_shapes`` restates the membership test that
    ``physics._pbl_optional_tendency_components`` decides at run time, and
    ``preflight._materialize_physics`` restates it a third time.  Two of the
    three were updated for mp=18 in an earlier wave and this file never
    checked the agreement; asserting it over every accepted selector is what
    stops the next scheme landing in two of three places.
    """
    from woof.core.physics import _pbl_optional_tendency_components

    for mp in (0, 1, 6, 8, 10, 18, 28):
        cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=mp,
                        **_MP28_PBL)
        priced = "pbl_tendencies/rqi" in pf.physics_array_shapes(cfg)
        at_runtime = "rqi" in _pbl_optional_tendency_components(cfg)
        assert priced == at_runtime, (
            f"mp_physics={mp}: preflight prices pbl_tendencies/rqi="
            f"{priced} but physics.py composes rqi={at_runtime}")
    # And the PBL-off case prices none of it, for any scheme.
    assert "pbl_tendencies/rqi" not in pf.physics_array_shapes(
        RunConfig(**_TINY, **_MP28))


def test_mp28_driver_aliases_the_scheme_accumulators_instead_of_copies():
    """No private ``microphysics/*`` arrays for a scheme with a slot row.

    BEFORE THIS TEST: ``microphysics_scratch_slots(28)`` returned ``()``, so
    the mp=28 PhysicsDriver allocated three private zero-filled surface
    arrays (``microphysics/rainnc``, ``/rainncv``, ``/sr``) and
    ``accept_microphysics`` copied the scheme's result into them on every
    step, instead of aliasing the seven canonical ``mp_*`` scratch
    accumulators the aerosol adapter writes.

    The mp=0 control keeps its three: with no scheme there is no canonical
    set to alias, which is what the three arrays are for.
    """
    from woof.core.physics import microphysics_scratch_slots

    shapes = pf.physics_array_shapes(RunConfig(**_TINY, **_MP28, **_MP28_PBL))
    assert not any(name.startswith("microphysics/") for name in shapes), (
        "mp=28 still budgets private driver-owned precipitation arrays")

    control = pf.physics_array_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=0, **_MP28_PBL))
    assert {name for name in control if name.startswith("microphysics/")} == {
        "microphysics/rainnc", "microphysics/rainncv", "microphysics/sr"}

    slots = dict(microphysics_scratch_slots(28))
    assert slots == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv",
        "graupelnc": "mp_graupelnc", "graupelncv": "mp_graupelncv"}
    # Every one of those slots is in the mp=28 scratch registry already, so
    # the aliasing adds no allocation anywhere -- it removes three.
    registry = pf.scratch_slot_registry(RunConfig(**_TINY, **_MP28),
                                        n_lbc_intervals=2)
    for slot in slots.values():
        assert slot in registry, slot


def test_mp28_rrtmgp_column_inventory_carries_the_effective_radii():
    """WRF's ``use_mp_re`` table lists THOMPSONAERO; the columns are priced.

    ``phys/module_physics_init.F:1005`` (THOMPSON) and ``:1006``
    (THOMPSONAERO) sit in the same disjunction, and the P3/Jensen-Ishmael
    ``has_reqs = 0`` override at ``:1026-1033`` does not touch either, so all
    three of ``has_reqc``/``has_reqi``/``has_reqs`` are 1 for mp=28.

    BEFORE THIS TEST: ``preflight.py:2708`` read ``(6, 8, 18)`` and an mp=28
    RTE+RRTMGP domain priced no radii columns at all.

    Thompson has no ``effr`` (that is Morrison's), and the legacy-RRTMG 4/4
    variant is priced as one shared call-peak envelope rather than through
    this function -- both asserted so a future edit cannot quietly widen the
    row into either.
    """
    cfg = RunConfig(**_TINY, **_MP28, ra_physics=4)
    shapes = pf.rrtmgp_column_shapes(cfg)
    ncol, nz = cfg.ny * cfg.nx, cfg.nz
    for name in ("effc", "effi", "effs"):
        assert shapes[f"columns/{name}"] == ((ncol, nz), 4), name
    assert "columns/effr" not in shapes
    assert "columns/nc" not in shapes

    # mp=8's row is untouched, and Morrison still gets its four radii.
    assert "columns/effc" in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=8, ra_physics=4))
    assert "columns/effr" in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=10, ra_physics=4))
    # Kessler declares no radii in WRF's table and must not price any.
    assert not any("eff" in name for name in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=1, ra_physics=4)))


def test_p3_prices_its_two_rrtmgp_radius_columns_and_no_third():
    """mp=50's two-radius pricing is a decision the site states and holds.

    ``rrtmgp_column_shapes``'s radii tuple ``(6, 8, 16, 18, 28)`` omitted
    mp=50 with nothing anywhere saying whether that was right.  It is
    right, for a reason that is P3's state inventory rather than an
    accident: ``phys/module_physics_init.F:1027-1033`` overrides
    ``has_reqs = 0`` for the P3 family while leaving
    ``has_reqc = has_reqi = 1``, because ``Registry.EM_COMMON:3038`` gives
    mp=50 ONE ice category with rime mass and rime volume and NO qs and no
    qg, and ``phys/module_mp_p3.F:2280-2282`` initialises diag_effc and
    diag_effi with no diag_effs to initialise.  A three-radius row would
    price a column P3 cannot have.

    THE TRIPWIRE THIS TEST USED TO BE HAS FIRED AND RETIRED (2.6.1).
    Its earlier body pinned "mp=50 has no _MP_CLOUD_OPTICS_SCHEME row,
    prices zero radii, and refuses the moment a row lands" -- and the
    row landed with the RTE+RRTMGP cloud-optics coupling.  Per the
    tripwire's own retirement instruction, the pin flips: mp=50 now
    prices EXACTLY effc and effi, and never effs, so the rail can
    neither under-price the two columns the adapter copies nor price a
    third column the scheme cannot allocate.
    """
    from woof.core.rrtmgp import _MP_CLOUD_OPTICS_SCHEME

    # The premise the old refusal enforced, now landed and enforced in
    # the other direction.
    assert _MP_CLOUD_OPTICS_SCHEME.get(50) == "p3"

    cfg = RunConfig(**_TINY, moist=True, mp_physics=50, ra_physics=4)
    shapes = pf.rrtmgp_column_shapes(cfg)
    assert "columns/play" in shapes
    ncol, nz = cfg.ny * cfg.nx, cfg.nz
    for name in ("effc", "effi"):
        assert shapes[f"columns/{name}"] == ((ncol, nz), 4), name
    assert "columns/effs" not in shapes
    assert "columns/effr" not in shapes

    # P3's state has no effs to copy, so a third radius column could
    # never be the right pricing for it (woof/core/state.py, mp==50).
    state = pf.state_array_shapes(cfg)
    assert "effc" in state and "effi" in state
    assert "effs" not in state and "qs" not in state and "qg" not in state

    # The arm prices its two columns unconditionally, exactly as the
    # adapter's p3 branch copies them unconditionally (P3 seeds valid
    # radii at construction; there is no first-call phase to gate on).
    # The membership assert at the top is what keeps the pair accurate:
    # the row and this pricing landed together and retire together.


# ---------------------------------------------------------------------------
# mp_physics=28 -- the rest of the WP-10 infrastructure surface.
#
# These are not preflight tests.  They live here because tests/test_preflight
# .py is the ONE test file WP-10 owns, and the alternative is shipping the
# state/transport/restart/nesting/pricing work with no regression coverage at
# all.  Each one names the module it actually guards.
# ---------------------------------------------------------------------------

def _host_mp28_state(**overrides):
    """A real ``DomainState`` on numpy, so no device is required."""
    import woof.core.state as state_mod

    cfg = RunConfig(**_TINY, **{**_MP28, **overrides})
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        # array_module=np is the DOCUMENTED host-state interface: it
        # records _host_setup_state so LATER calls (state.scratch during
        # the test body, after this monkeypatch is undone) stay on numpy.
        # The cp shim alone stopped being enough the day scratch() began
        # resolving the array module per call -- on a box with a card the
        # old spelling silently opened the device, and on a CPU leg it
        # died cudaErrorNoDevice.
        return state_mod.DomainState(cfg, array_module=np), cfg
    finally:
        monkey.undo()


def test_mp28_acoustic_cq_sums_six_masses_and_no_number_moment():
    """woof/core/acoustic.py.  WRF's calc_cq sums the Registry ``moist``
    package only; mp=28's qnc/qnwfa/qnifa are ``scalar``.  A droplet number
    of order 1e8 leaking into q_tot would not be subtle, but n_mass is an
    integer passed to a kernel and a wrong value is invisible from Python.
    """
    from woof.core import acoustic

    captured = {}

    def fake_get_kernel(module, func):
        assert (module, func) == ("acoustic", "calc_cq")

        def launch(_grid, _block, args):
            captured["n_mass"] = int(args[10])
            captured["fields"] = args[:7]
        return launch

    state, cfg = _host_mp28_state()
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(acoustic, "get_kernel", fake_get_kernel)
        monkey.setattr(acoustic, "cp", np, raising=False)
        _, _, _, use = acoustic.prepare_moist_cq(state, cfg)
    finally:
        monkey.undo()
    assert use is True
    # Identical to mp=8: qv, qc, qr, qi, qs, qg.
    assert captured["n_mass"] == 6
    # And the seventh slot (qh) is the qv placeholder, never an aerosol.
    assert captured["fields"][6] is state.qv
    for moment in (state.nc, state.nwfa, state.nifa):
        assert not any(moment is arg for arg in captured["fields"])


def test_mp50_is_deliberately_out_of_the_six_mass_shape_package():
    """``state_array_shapes`` -- the mp=50 / six-mass arm split, by name.

    The tuple that prices qi/qs/qg + their RK time-t copies + the three
    effective radii is WRF's SIX-MASS moist package transcribed, not "the
    schemes with ice": every member declares moist:qv,qc,qr,qi,qs,qg in
    Registry.EM_COMMON (:3021 WSM6, :3024 Thompson, :3025 Milbrandt-Yau,
    :3026 Morrison, :3031 WDM6, :3033 NSSL, :3036 Thompson aerosol-aware).
    P3's row is moist:qv,qc,qr,qi with no qs and no qg, and
    state:re_cloud,re_ice with no re_snow (:3038), and WRF's driver binds
    it with N_ICECAT=1 and no QS/QG dummy at all
    (module_microphysics_driver.F:1569-1602).  So mp=50 is out of that
    tuple ON PURPOSE.

    This pins the DECISION so a later reader cannot mistake it for an
    omission and "fix" it.  Adding 50 declares five arrays the state
    builder never allocates, and -- the quieter half -- silently drops the
    shared absent-mass plane from the scratch projection, because that
    slot's predicate is "qi declared and qs NOT declared".  The arena
    would then be one (nz, ny, nx) FP32 plane short of what
    woof/core/moist.py allocates on every P3 step.
    """
    m = (_TINY["nz"], _TINY["ny"], _TINY["nx"])
    six_mass = ("qi", "qs", "qg", "qi0", "qs0", "qg0",
                "effc", "effi", "effs")
    members = (6, 8, 9, 10, 16, 18, 24, 28)

    p3 = pf.state_array_shapes(RunConfig(**_TINY, moist=True,
                                         mp_physics=50))
    # P3's own package is priced by its own arm...
    for name in ("qi", "ni", "nr", "qir", "qib", "effc", "effi",
                 "th_old", "qv_old", "qi0", "ni0", "nr0", "qir0", "qib0"):
        assert p3[name] == m, name
    # ...and the three frozen species it never writes are absent, not
    # zero: allocating them would hand advection, output and the nest
    # transition fields that read as a legitimate zero everywhere.
    for name in ("qs", "qg", "effs", "qs0", "qg0"):
        assert name not in p3, name

    # State what the tuple IS, so the exclusion has something to be an
    # exclusion from: every member really does carry all nine names.
    for mp in members:
        shapes = pf.state_array_shapes(RunConfig(**_TINY, moist=True,
                                                 mp_physics=mp))
        for name in six_mass:
            assert name in shapes, (mp, name)

    # And the consequence the exclusion buys, measured through the real
    # registry: the shared absent-mass plane is priced for the one-ice
    # scheme and for no member of the six-mass package.
    def priced(mp: int) -> bool:
        return "moist_absent_mass" in pf.scratch_slot_registry(
            RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=mp))

    assert priced(50) is True
    for mp in members:
        assert priced(mp) is False, mp

    # The STRUCTURAL half, so the comment at the site is a checkable claim
    # and not decoration: the six-mass arm is chained to the mp==50 arm as
    # an ``elif``, exactly the way woof/core/state.py's allocator spells
    # the same split.  Un-chaining it is what would let a widened tuple
    # hand P3 both packages, so the chain is pinned here rather than left
    # to a reader noticing the keyword.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(pf.state_array_shapes))
    p3_arms = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 50]
    assert len(p3_arms) == 1, "the mp==50 shape arm moved or was duplicated"
    orelse = p3_arms[0].orelse
    assert len(orelse) == 1 and isinstance(orelse[0], ast.If), (
        "the six-mass arm is no longer chained to the mp==50 arm; an "
        "unchained sibling lets a widened tuple price qs/qg/effs on top "
        "of P3's own package")
    # Lin has its own six-mass arm without effective-radius fields.
    lin_arm = orelse[0]
    assert isinstance(lin_arm.test, ast.Compare)
    assert isinstance(lin_arm.test.ops[0], ast.Eq)
    assert lin_arm.test.comparators[0].value == 2
    lin_shapes = pf.state_array_shapes(RunConfig(**_TINY, moist=True, mp_physics=2))
    assert {"qi", "qs", "qg", "qi0", "qs0", "qg0"} <= set(lin_shapes)
    assert not {"effc", "effi", "effs"} & set(lin_shapes)
    assert not priced(2)
    assert len(lin_arm.orelse) == 1
    wsm3_arm = lin_arm.orelse[0]
    assert isinstance(wsm3_arm.test.ops[0], ast.Eq)
    assert wsm3_arm.test.comparators[0].value == 3
    assert len(wsm3_arm.orelse) == 1 and isinstance(wsm3_arm.orelse[0], ast.If)
    orelse = wsm3_arm.orelse
    # Column metadata may own an intervening mass-package arm.
    while (isinstance(orelse[0].test, ast.Compare)
           and isinstance(orelse[0].test.comparators[0], ast.Name)
           and orelse[0].test.comparators[0].id == "COLUMN_SCHEMES"):
        from woof.core.column_schemes import COLUMN_SCHEMES
        assert 50 not in COLUMN_SCHEMES
        orelse = orelse[0].orelse
        assert len(orelse) == 1 and isinstance(orelse[0], ast.If)
    chained = orelse[0].test
    assert isinstance(chained, ast.Compare)
    assert isinstance(chained.ops[0], ast.In)
    assert list(eval(compile(ast.Expression(chained.comparators[0]), "<six-mass-membership>", "eval"), pf.__dict__)) == [6, 8, 9, 10, 16, 18, 24, 28, 106], (  # Lin (2) and FAST_SBM (30) own arms; TEMPO is a named row
        "the six-mass membership tuple moved; see Registry.EM_COMMON "
        ":3021/:3024/:3025/:3026/:3031/:3033/:3036 for its members and "
        ":3038 for why 50 is not one of them")


def test_mp50_acoustic_cq_takes_its_one_ice_category_arm():
    """woof/core/acoustic.py -- the P3 arm of ``prepare_moist_cq``.

    That arm had NO coverage.  ``RunConfig.moist_cq`` and
    ``verify.cases.moist_bubble.default_config()`` both default the flag
    OFF, so every other committed mp=50 test short-circuits at the
    ``use_cq`` gate: deleting the whole arm left the p3 suite green while
    a ``moist_cq=True`` run went straight back to the defect it fixed,
    ``ValueError: unsupported mp_physics=50 for cq``.  The config is the
    one ``test_scratch_lifetime_audit_covers_registry_and_manifest``
    already prices (``moist_cq=True, mp_physics=50, km_opt=4``); what is
    new here is that the cq call is actually MADE.

    What the arm owes: select on qi-present/qs-absent, keep the six-mass
    kernel mode, hand the absent snow and graupel the shared zero plane,
    and leave qir/qib out -- they are Registry ``scalar``, and qir is a
    COMPONENT of qi, so summing it would double count the rimed mass.
    """
    from woof.core import acoustic

    captured = {}

    def fake_get_kernel(module, func):
        assert (module, func) == ("acoustic", "calc_cq")

        def launch(_grid, _block, args):
            captured["fields"] = args[:7]
            captured["faces"] = args[7:10]
            captured["n_mass"] = int(args[10])
        return launch

    import woof.core.state as state_mod

    cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=50,
                    km_opt=4)
    monkey = pytest.MonkeyPatch()
    try:
        # numpy for the whole call, scratch allocations included, so this
        # runs in the CPU shard and never needs a device.
        monkey.setattr(state_mod, "cp", np)
        monkey.setattr(acoustic, "get_kernel", fake_get_kernel)
        monkey.setattr(acoustic, "cp", np, raising=False)
        state = state_mod.DomainState(cfg)
        # The arm is selected on PRESENCE, so state the premise it keys on.
        assert getattr(state, "qi", None) is not None
        assert getattr(state, "qs", None) is None
        # Real mass in every P3 field, so "the absent plane is zero" is a
        # measurement and not a restatement of zero-initialisation.
        for field, value in (("qv", 8.0e-3), ("qc", 1.0e-3), ("qr", 4.0e-4),
                             ("qi", 2.0e-4), ("qir", 5.0e-5),
                             ("qib", 1.0e-7)):
            getattr(state, field)[...] = value
        cqu, cqv, cqw, use = acoustic.prepare_moist_cq(state, cfg)
    finally:
        monkey.undo()

    # It reached the kernel at all -- the ValueError arm is what this pins.
    assert use is True
    assert captured, "prepare_moist_cq never launched calc_cq"
    faces = captured["faces"]
    assert cqu is faces[0] and cqv is faces[1] and cqw is faces[2]
    assert cqu.shape == (cfg.nz, cfg.ny, cfg.nx + 1)
    assert cqv.shape == (cfg.nz, cfg.ny + 1, cfg.nx)
    assert cqw.shape == (cfg.nz + 1, cfg.ny, cfg.nx)

    # Six-mass mode with ONE real ice mass behind it.
    assert captured["n_mass"] == 6
    qv, qc, qr, qi, qs, qg, qh = captured["fields"]
    assert qv is state.qv
    assert qc is state.qc
    assert qr is state.qr
    assert qi is state.qi
    # The absent snow and graupel are the SAME shared plane, and it is
    # zeroed before the read, so the six-mass sum is exact.  Every real
    # field above carries mass, so this is a measurement of the plane and
    # not a restatement of zero-initialisation.
    assert qi.any() and qv.any()
    assert qs is qg
    assert not qs.any()
    assert qh is state.qv          # hail placeholder, never read at mp=50
    # The rime pair is not mass loading and must not enter the sum.
    for extra in (state.qir, state.qib):
        assert not any(extra is arg for arg in captured["fields"])


def test_mp50_prices_the_absent_snow_plane_and_not_the_present_ice_one():
    """woof/core/preflight.py:3812-3815 -- the SPLIT physics_qi/physics_qs
    conditions, pinned by what the physics prep actually asks for.

    Both conditions are NEGATED membership tests, so mp=50's presence in
    one and absence from the other is the OPPOSITE of what a scheme-set
    scan reads off them.  ``_prepare_atmosphere`` substitutes a zero-filled
    scratch plane PER FIELD, only when the state has no field of its own
    (woof/core/physics.py:1541-1550), and P3 is the one scheme that
    allocates exactly one of the pair: woof/core/state.py:464-476 gives an
    mp=50 state ``qi``/``ni``/``nr``/``qir``/``qib`` and NO ``qs``, because
    P3 carries one ice category with a rime mass/volume pair instead of
    splitting snow and graupel.  WRF agrees on both halves:
    Registry.EM_COMMON:3038 registers the mp=50 package as
    ``moist:qv,qc,qr,qi``, and module_microphysics_driver.F:1569-1602 binds
    no snow array at all in the ``mp_p3_wrapper_wrf`` call, because
    one-category ``p3_main`` has no snow mixing ratio to return.  So an
    mp=50 run substitutes ``qs`` and only ``qs``.

    THE BREAKAGE THIS PREVENTS, both ways.  Adding 50 to the ``physics_qs``
    condition -- which is exactly what a mechanical "this scheme set omits
    50" sweep would do -- stops pricing a full nz*ny*nx float32 plane that
    every physics-enabled mp=50 step allocates, turning the preflight
    envelope into an under-estimate on a card with no ECC.  Removing 50
    from the ``physics_qi`` condition prices a second such plane the run
    never asks for.  MEASURED before this test existed: adding 50 to the
    ``physics_qs`` condition left the whole preflight module green.

    The producer side is measured, not assumed: the prep is run on a real
    mp=50 ``DomainState`` and its scratch requests recorded, with mp=28 --
    which owns both fields -- as the control.
    """
    import woof.core.physics as physics_mod
    import woof.core.state as state_mod

    def prep_scratch_requests(mp: int):
        cfg = RunConfig(**_TINY, moist=True, mp_physics=mp)
        monkey = pytest.MonkeyPatch()
        try:
            # numpy for the whole call, so this runs in the CPU shard.
            monkey.setattr(state_mod, "cp", np)
            monkey.setattr(physics_mod, "cp", np)
            state = state_mod.DomainState(cfg)
            requested: list[str] = []
            allocate = state.scratch

            def spy(shape, name):
                requested.append(name)
                return allocate(shape, name)

            state.scratch = spy
            with np.errstate(divide="ignore", invalid="ignore"):
                prepared = physics_mod._prepare_atmosphere(state)
        finally:
            monkey.undo()
        # The seam the substitution exists for: whatever the prep hands the
        # radiation/PBL drivers under "qi"/"qs" must be a real array either
        # way, so a missing field can never reach them as None.
        assert prepared["qi"] is not None and prepared["qs"] is not None
        return requested, state

    p3_requests, p3_state = prep_scratch_requests(50)
    thompson_requests, thompson_state = prep_scratch_requests(28)

    # The premise, stated rather than assumed.
    assert getattr(p3_state, "qi", None) is not None
    assert getattr(p3_state, "qs", None) is None
    assert getattr(thompson_state, "qi", None) is not None
    assert getattr(thompson_state, "qs", None) is not None

    assert "physics_qs" in p3_requests
    assert "physics_qi" not in p3_requests
    assert "physics_qi" not in thompson_requests
    assert "physics_qs" not in thompson_requests

    # And the registry prices exactly that, for a physics-enabled config.
    mass = (_TINY["nz"], _TINY["ny"], _TINY["nx"])
    p3_slots = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, mp_physics=50, sf_sfclay_physics=1))
    assert p3_slots.get("physics_qs") == mass, (
        "mp=50 (P3) has no qs of its own, so woof/core/physics.py "
        "substitutes a zero plane on every physics step; dropping it from "
        "the registry under-prices the run by one full mass field")
    assert "physics_qi" not in p3_slots, (
        "mp=50 (P3) allocates prognostic cloud ice, so no zero plane is "
        "substituted for it and none may be priced")
    thompson_slots = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, mp_physics=28, sf_sfclay_physics=1))
    assert "physics_qi" not in thompson_slots
    assert "physics_qs" not in thompson_slots


def test_mp28_npref_cq_species_match_mp8_exactly():
    """woof/verify/npref.py -- the CPU mirror must make the same choice."""
    from woof.verify import npref

    assert (npref._CQ_MASS_SPECIES_BY_MP[28]
            == npref._CQ_MASS_SPECIES_BY_MP[8])
    moisture = {name: np.full((2, 2, 2), 0.001) for name in
                ("qv", "qc", "qr", "qi", "qs", "qg")}
    # Adding aerosol fields to the dict must not change the answer.
    polluted = dict(moisture, nc=np.full((2, 2, 2), 1.0e8),
                    nwfa=np.full((2, 2, 2), 3.0e8),
                    nifa=np.full((2, 2, 2), 5.0e3))
    for clean, dirty in zip(npref.np_calc_cq(moisture, 28),
                            npref.np_calc_cq(polluted, 28), strict=True):
        np.testing.assert_array_equal(np.asarray(clean), np.asarray(dirty))


def test_mp28_health_rules_cover_the_aerosol_tracers():
    """woof/core/health.py.  An uncovered field is a field the integration
    health gate silently ignores -- the exact failure mode this port is most
    exposed to, since a wrong aerosol number stays finite and bounded."""
    from woof.core import health

    state, _ = _host_mp28_state()
    names = {f.name for f in health.collect_state_fields(state)}
    for name in ("nc", "nwfa", "nifa", "nr", "ni", "qi", "qs", "qg"):
        assert name in names, name
    for leaf in ("nwfa", "nifa"):
        rule = health.rule_for_field(leaf)
        assert rule.status_class == "moment"
        assert rule.lower == 0.0
        # WRF's own terminal ceiling is 9999.E6 with an unclamped surface
        # emission on top (module_mp_thompson.F:3977-3982, :1310-1327), so
        # the rule must sit well above it without being unbounded.
        assert rule.upper >= 9999.0e6
    # The census must not have started covering Morrison's untransported nc
    # differently, and must not have picked up the 2-D emission fields
    # (those are constants, not integration state).
    assert "nwfa2d" not in names and "nifa2d" not in names


def test_mp28_lateral_boundary_allow_lists_accept_the_new_scalars():
    """woof/ingest/lateral_bc.py -- ONE shared coupled-scalar allowlist.

    The three sites used to spell their sets inline, and three
    hand-copied sets is how mp=9's nh, WDM6's nn and P3's rime pair were
    each missing from all three at once (1.9.1 D1).  The mp=28 scalars
    this test originally pinned now live in the shared constant, and
    each site must read that constant rather than re-spelling a set.
    """
    import inspect

    from woof.ingest import lateral_bc

    for name in ("nc", "nwfa", "nifa", "nh", "nn", "qir", "qib"):
        assert name in lateral_bc.COUPLED_SCALAR_STATE_FIELDS, name
    for func in (lateral_bc.apply_specified_relaxation,
                 lateral_bc.couple_nest_field,
                 lateral_bc.uncouple_feedback_field):
        source = inspect.getsource(func)
        assert "COUPLED_SCALAR_STATE_FIELDS" in source, func.__name__


def test_mp28_external_lbc_uses_supplied_aerosols_and_retains_synthetic_behavior():
    from dataclasses import replace
    from woof.core.state import DomainState
    from woof.boundary_fields import external_scalar_fields
    from woof.ingest.lateral_bc import domain_boundary_snapshot
    cfg = RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=1., moist=True, mp_physics=28,
                    specified=True, aer_init_opt=1, wif_input_opt=1)
    for supplied in (False, True):
        selected = cfg if supplied else replace(
            cfg, aer_init_opt=0, wif_input_opt=0, mp28_aerosol_source="synthetic")
        state = DomainState(selected, array_module=np)
        state.c1h[:] = 1.; state.c2h[:] = 0.
        state.c1f[:] = 1.; state.c2f[:] = 0.; state.mub2d[:] = 10000.
        state.nwfa[:] = 123.; state.nifa[:] = 456.
        snapshot = domain_boundary_snapshot(state)
        assert set(snapshot) == {"u", "v", "theta", "phi", "mu", *external_scalar_fields(selected)}
        assert not {"nc", "nr", "ni", "qc", "qr", "qi", "qs", "qg"} & set(snapshot)
        if supplied:
            np.testing.assert_array_equal(snapshot["nwfa"], 1230000.)
            np.testing.assert_array_equal(snapshot["nifa"], 4560000.)


def test_mp28_mixed_nest_edge_resolves_on_every_partner_by_name():
    """woof/core/microphysics_transition.py.

    v1 REFUSED rather than inventing an entry closure for nc/nwfa/nifa
    across a scheme boundary.  Audit R-004 ratified that closure, so what
    is measured now is the same three properties with the verdict turned
    over: the edge must (a) resolve in BOTH directions and for every
    partner scheme, (b) carry a receipt that names mp=28's seeded moments
    and the value each one takes rather than a generic diagnosis, and (c)
    still leave the same-scheme mp28 -> mp28 nest alone.  The name said
    "is refused" long after the assertions said the opposite, which is a
    trap for the next reader running ``-k``.
    """
    import types

    from woof.core import microphysics_transition as mt

    def run(mp, policy=mt.SAME_SCHEME_POLICY):
        return types.SimpleNamespace(
            mp_physics=mp, moist=True, moist_cq=True,
            nest_microphysics_transition=policy,
            morr_rimed_ice=1, wsm6_hail_opt=0)

    same = mt.resolve_microphysics_transition(run(28), run(28))
    assert same.mixed is False
    assert same.policy_id == mt.SAME_SCHEME_POLICY

    # RATIFIED (audit R-004).  Every mp=28 mixed pair now RESOLVES, in
    # both directions, and its receipt names the seeded aerosol values
    # rather than a refusal.  The entry closure is WRF's own
    # non-aerosol-aware fallback set (module_mp_thompson.F:1248-1255) and
    # the exit direction never needed a closure at all: the target's
    # moments come from target mass and the aerosol numbers are dropped.
    partners = [mp for mp in mt.PORTED_MP_PHYSICS if mp != 28]
    assert partners, "PORTED_MP_PHYSICS became empty"
    for other in partners:
        for parent, child in ((other, 28), (28, other)):
            if parent == child:
                continue
            contract = mt.resolve_microphysics_transition(
                run(parent), run(child, mt.EDGE_MATRIX_POLICY))
            assert contract.mixed is True
            rows = {row["target_field"]: row
                    for row in contract.species_actions()
                    if row["action"] == "diagnosed"}
            if child == 28:
                for moment, expected in (("nc", 100.0e6),
                                         ("nwfa", 11.1e6),
                                         ("nifa", 5.0e3)):
                    assert rows[moment]["seeded_value"] == expected, (
                        parent, child, moment)
                assert "thompson" in rows["nwfa"]["reason"]
                note = mt.mixed_edge_entry_note(contract)
                assert "non-aerosol-aware" in note

    # And the ratified MP8 -> MP18 edge codes are untouched.  mp=50's
    # ratification APPENDED its rime pair at 20/21 and mp=9's appended its
    # nc/nh at 22/23 (the discipline the PORTED_MP_PHYSICS comment demands),
    # each moving nothing below it.  The length is the count of DISTINCT
    # field names the ported selectors carry between them, so it is read
    # from those selectors rather than retyped: this assertion was pinned at
    # 22 and went red the moment mp=9 joined, which is a guard describing a
    # tree that no longer exists rather than a guard on mp=28.
    assert mt._EDGE_FIELD_CODES["qvolh"] == 19
    assert mt._EDGE_FIELD_CODES["qir"] == 20
    assert mt._EDGE_FIELD_CODES["qib"] == 21
    assert mt._EDGE_FIELD_CODES["nc"] == 22
    assert mt._EDGE_FIELD_CODES["nh"] == 23
    # mp=16 then mp=28 appended next, again moving nothing below them.
    assert mt._EDGE_FIELD_CODES["nn"] == 24
    assert mt._EDGE_FIELD_CODES["nwfa"] == 25
    assert mt._EDGE_FIELD_CODES["nifa"] == 26
    ported_names = {
        name
        for mp in mt.PORTED_MP_PHYSICS
        for name in mt._MASS_FIELDS[mp] + mt._MOMENT_FIELDS[mp]
    }
    assert len(mt._EDGE_FIELD_CODES) == len(ported_names) == 27
    assert set(mt._EDGE_FIELD_CODES) == ported_names
    assert {16, 28} <= set(mt.PORTED_MP_PHYSICS)


def test_mp28_survives_a_restart_round_trip(tmp_path):
    """woof/io/restart.py + woof/state_serialization_contract.py.

    Every mp=28 field must be classified, written, and restored bit-for-bit
    -- including the two 2-D surface emission constants, which nothing in the
    forecast writes and which would therefore come back as zeros (silently
    switching off surface aerosol emission for the rest of the run) if they
    were merely 'rebuilt'.
    """
    import woof.core.state as state_mod
    from woof.io import restart as restart_mod

    state, cfg = _host_mp28_state()
    rng = np.random.default_rng(28)
    written = {}
    for name in ("nc", "nr", "ni", "nwfa", "nifa", "qc", "qi",
                 "nwfa2d", "nifa2d"):
        array = getattr(state, name)
        array[...] = rng.random(array.shape).astype(np.float32) * 1.0e8
        written[name] = array.copy()

    manifest = restart_mod.state_manifest(state)
    for name in written:
        assert f"state/{name}" in manifest, name
    # No mp=28 attribute may be unclassified -- state_manifest walks every
    # instance attribute through classify_state_attr and raises otherwise,
    # so reaching this line already proves it, but name the new ones.
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert restart_mod.classify_state_attr(name) == "serialize"
    for name in ("nc0", "nwfa0", "nifa0"):
        assert restart_mod.classify_state_attr(name) == "rebuild"

    # Restore into a fresh state and compare.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        restored = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    for name, value in written.items():
        assert not np.array_equal(getattr(restored, name), value)
        getattr(restored, name)[...] = manifest[f"state/{name}"]
        np.testing.assert_array_equal(getattr(restored, name), value)


def test_mp28_nest_init_interpolates_and_seeds_every_new_field():
    """woof/ingest/nest_init.py -- the interpolation list and the RK seed
    list.  A field missing from the seed list starts its first child RK step
    with a zero time-t copy, which is a one-step transient no bound catches.
    """
    import inspect

    from woof.ingest import nest_init

    source = inspect.getsource(nest_init)
    for entry in ('("nwfa", "")', '("nifa", "")',
                  '("nwfa2d", "")', '("nifa2d", "")',
                  '("nc", "nc0")', '("nwfa", "nwfa0")',
                  '("nifa", "nifa0")'):
        assert entry in source, entry

    # The RK seed loop is None-guarded, so a Morrison child (nc present,
    # nc0 absent) must be unaffected by the new ("nc", "nc0") row.
    import woof.core.state as state_mod
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        mp10 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=10))
    finally:
        monkey.undo()
    assert getattr(mp10, "nc", None) is not None
    assert getattr(mp10, "nc0", None) is None


@pytest.mark.gpu
def test_mp28_scalars_actually_advect_on_the_device():
    """The transport claim, run rather than asserted.

    ``extra_moist_species`` returning the right tuple only proves the loop
    would VISIT nc/nwfa/nifa.  This drives the real positive-definite
    advection stage on the GPU with a uniform x-flow at face Courant 0.5 and
    requires that each of the five mp=28 number moments (a) moves, (b) stays
    finite, (c) stays non-negative, and (d) conserves its coupled mass under
    the periodic flux telescope.  A field that silently never entered the
    stage loop would sit unchanged and fail (a).
    """
    cp = pytest.importorskip("cupy")

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import (
        advance_scalars_stage, extra_moist_species, init_moist_balanced)

    nx, ny, nz = 16, 8, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=6000.0,
                    dt=10.0, run_seconds=0.0, moist=True, mp_physics=28)
    vc = make_vertical_coord(nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, vc, base, lambda z: np.full(nz, 1.0e-3))

    assert extra_moist_species(state) == (
        "qi", "qs", "qg", "nr", "ni", "nc", "nwfa", "nifa")

    # Plausible magnitudes, deliberately spanning nine decades so an FP32
    # transport bug in the aerosol fields cannot hide behind qv's scale.
    blobs = {"nc": 1.0e8, "nwfa": 3.0e8, "nifa": 5.0e3,
             "nr": 1.0e4, "ni": 1.0e5}
    for name, amplitude in blobs.items():
        host = np.zeros((nz, ny, nx), dtype=np.float32)
        host[4:8, 3:6, 6:10] = amplitude
        getattr(state, name)[...] = cp.asarray(host)
        getattr(state, name + "0")[...] = getattr(state, name)

    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    dt_eff = cfg.dt
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    ru[...] = 0.5 * (cfg.dx / dt_eff) * chm[:, :, :1]
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    ww = cp.zeros((nz + 1, ny, nx), cp.float32)

    dnw_abs = -state.dnw[:, None, None]

    def coupled_mass(field):
        return float(cp.sum((chm * field * dnw_abs).astype(cp.float64)))

    before = {name: coupled_mass(getattr(state, name)) for name in blobs}
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff, final=True)
    cp.cuda.Stream.null.synchronize()

    for name, amplitude in blobs.items():
        field = getattr(state, name)
        assert bool(cp.isfinite(field).all()), name
        assert float(field.min()) >= 0.0, name
        # (a) it MOVED: the blob's leading edge has advanced in +x.
        moved = float(cp.abs(field - getattr(state, name + "0")).max())
        assert moved > 0.01 * amplitude, (
            f"{name} did not advect -- it is allocated but not transported")
        # (d) coupled mass is conserved by the periodic telescope.
        residual = abs(coupled_mass(field) - before[name]) / before[name]
        assert residual < 1e-5, (name, residual)


def test_scratch_scanner_catches_the_review_bypasses():
    """Regression fixtures: the exact bypass constructions from the p5t11
    reviews (keyword-form slots, one-positional + keyword slot, method
    aliasing, getattr lookup) must be seen -- silent skips were the F3/F4
    MAJOR."""
    def kinds(src):
        return [(kind, payload) for kind, payload, _, _ in
                _scan_scratch_tree(ast.parse(src), "synthetic.py")]

    # Shadow F4 construction: both-keyword form.
    assert kinds("state.scratch(shape=(nz, ny, nx), "
                 "slot='unregistered_resident')") == [
        ("literal", "unregistered_resident")]
    # One positional + keyword slot.
    assert kinds("state.scratch((nz, ny, nx), slot='nest_bogus')") == [
        ("literal", "nest_bogus")]
    # Keyword f-string still classifies as a prefix family.
    assert kinds("state.scratch((2, w), slot=f'lbc_weights_{n}')") == [
        ("prefix", "lbc_weights_")]
    # review F3(2): method alias under another name -- the alias itself
    # is flagged even though the later call is unrecognizable.
    assert ("alias", None) in kinds("sc = state.scratch\nsc((1,), 'x')")
    # review F3(3): getattr lookup, stored or immediately called.
    assert ("getattr", None) in kinds(
        "getattr(state, 'scratch')((1,), 'x')")
    assert ("getattr", None) in kinds(
        "f = getattr(state, 'scratch', None)")
    # A slot the scanner cannot identify at all is a finding, not a skip.
    assert kinds("state.scratch((1,))") == [("no_slot", None)]
    # Ordinary calls stay classified exactly as before.
    assert kinds("state.scratch((1,), 'rk_ww')") == [("literal", "rk_ww")]
    assert kinds("state.scratch(f.shape, slotvar)") == [("variable", None)]
    assert kinds("value = work.scratch[2]") == [("data", "indexed")]
    assert kinds("dict(member.scratch or {})") == [("data", "mapping")]
    assert kinds("(member.scratch or {}).values()") == [("data", "mapping")]
    assert kinds("path = self.scratch / 'batch.tsv'") == [("data", "path")]
    # Installing an allocator is a STORE, not a method alias LOAD. Its
    # later calls must still be classified, and callable fallbacks and
    # immediately called indexed values must not become data exemptions.
    assert kinds("state.scratch = allocate\nstate.scratch((1,), 'rk_ww')") == [
        ("literal", "rk_ww")]
    assert ("alias", None) in kinds("fn = state.scratch or fallback")
    assert ("alias", None) in kinds("state.scratch[0]((1,), 'x')")


def test_chem_variable_requests_match_priced_shapes(monkeypatch):
    """Execute row and MONO allocation loops through a checked real owner."""
    from woof.core import chem_advect_mono, chem_transport
    from woof.core.chem_context import MONO_IMPLICIT_SLOT, MONO_SCRATCH_SLOTS
    from woof.core.state import DomainState

    cfg = RunConfig(**_TINY, moist=True, mp_physics=1, km_opt=4,
                    chem_sets="tracer_test", chem_adv_opt=2)
    state = DomainState(cfg, array_module=np)
    registry = pf.scratch_slot_registry(cfg)
    requested = {}
    allocate = state.scratch

    def checked(shape, slot):
        assert tuple(shape) == registry[slot], slot
        value = allocate(shape, slot)
        assert value.nbytes == 4 * math.prod(registry[slot]), slot
        requested[slot] = value
        return value

    state.scratch = checked
    held = chem_transport.chem_fixed_tendencies(state, cfg)
    held_names = {chem_transport.chem_fixed_slot(row)
                  for row in state.chem.transported}
    assert set(requested) == held_names
    assert set(held) == {row.state_attr for row in state.chem.transported}
    # Launches are outside this CPU allocation witness; the real stage still
    # executes every allocation, source fold and row-routing decision.
    monkeypatch.setattr(chem_advect_mono, "launch_mono_fluxes", lambda *a, **k: None)
    monkeypatch.setattr(chem_advect_mono, "launch_mono_renorm_apply", lambda *a, **k: None)
    monkeypatch.setattr(chem_transport, "_pd_fold_sources", lambda *a, **k: a[3])
    monkeypatch.setattr(chem_transport, "_update_scalar_in_place", lambda *a, **k: None)
    shape = (cfg.nz, cfg.ny, cfg.nx)
    chem_transport.advance_chem_stage(
        state, cfg, np.zeros((cfg.nz, cfg.ny, cfg.nx + 1), np.float32),
        np.zeros((cfg.nz, cfg.ny + 1, cfg.nx), np.float32),
        np.zeros((cfg.nz + 1, cfg.ny, cfg.nx), np.float32), 1.0,
        final=True, fixed_tendencies=held)
    expected = {*held_names, *MONO_SCRATCH_SLOTS, MONO_IMPLICIT_SLOT,
                "moist_rq_t", "pd_fxl", "pd_fxc", "pd_fyl", "pd_fyc",
                "pd_fzl", "pd_fzc"}
    assert set(requested) == expected
    assert all(requested[slot].shape == shape for slot in MONO_SCRATCH_SLOTS)
    assert all(pf.scratch_slot_lifetime(slot) is not None for slot in expected)


def test_packed_thompson_scratch_cannot_grow():
    from woof.ensemble.batch_physics import (
        PackedThompsonState, thompson_column_scratch_fields)

    cfg = RunConfig(**_TINY, moist=True, mp_physics=8)
    registry = pf.scratch_slot_registry(cfg)
    fields = thompson_column_scratch_fields(nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)
    buffers = {field.name: np.empty(field.shape, np.float32) for field in fields}
    owner = PackedThompsonState({}, buffers)
    for field in fields:
        assert field.shape == registry[field.name], field.name
        value = owner.scratch(field.shape, field.name)
        assert value is buffers[field.name]
        assert value.nbytes == 4 * math.prod(registry[field.name])
    with pytest.raises(ValueError, match="not admitted"):
        owner.scratch((1,), "new_unpriced_field")
    for shape, dtype in (((1,), None), (fields[0].shape, np.float64)):
        with pytest.raises(ValueError, match="shape/dtype"):
            owner.scratch(shape, fields[0].name, dtype)


@pytest.mark.parametrize("version", ["wrf_461", "gsd_41"])
@pytest.mark.parametrize("nz,ny,nx,chunk", [(5, 2, 3, 5), (9, 3, 5, 15)])
def test_packed_mynn_variable_scratch_matches_priced_owner(version, nz, ny, nx, chunk):
    """Execute the actual bind/recheck loops with CPU allocator backings.

    This checks allocation and rendezvous ownership, not CUDA arithmetic.
    The six member rates remain live beside the borrowed workspace; the
    workspace is reused sequentially by column pieces, never per member.
    """
    from types import SimpleNamespace
    from woof.core.state import DomainState
    from woof.core.mynn_pbl_scratch import (
        MYNN_PBL_CONSTANT_ZERO_SLOTS, MYNN_PBL_TENDENCY_FIELDS,
        mynn_pbl_scratch_shapes, mynn_pbl_index_shapes, mynn_pbl_flag_shapes,
        mynn_pricing_rank_chunk, bind_mynn_rank_chunk)
    from woof.ensemble.batch_mynn import (
        _DeclaredScratchState, mynn_pbl_output_plan, mynn_pbl_transient_bytes)
    from woof.ensemble.batch_storage import BatchStorage
    from woof.ensemble.packed_production_physics import OriginalWorkspaceReuseScope

    path = ROOT / "woof/ensemble/batch_mynn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "prepare_mynn_pbl_column_batch")
    launch = next(node for node in prepare.body if isinstance(node, ast.FunctionDef)
                  and node.name == "launch")
    assignments = {node.targets[0].id: node for node in prepare.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    loops = [node for node in prepare.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call)
             and isinstance(node.iter.func, ast.Attribute)
             and isinstance(node.iter.func.value, ast.Name)
             and node.iter.func.value.id == "shapes" and node.iter.func.attr == "items"]
    rechecks = [node for node in launch.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "borrowed"]
    assert len(loops) == len(rechecks) == 1
    sites = _scan_scratch_tree(ast.Module(body=[prepare], type_ignores=[]),
                              path.relative_to(ROOT).as_posix())
    assert sites == [("variable", None, "woof/ensemble/batch_mynn.py", name)
                     for name in ("prepare_mynn_pbl_column_batch", "launch")]
    bind = compile(ast.Module(body=[assignments["shapes"], assignments["buffers"],
                   loops[0], assignments["identities"]], type_ignores=[]), str(path), "exec")
    recheck = compile(ast.Module(body=rechecks, type_ignores=[]), str(path), "exec")

    cfg = RunConfig(**{**_TINY, "nz": nz, "ny": ny, "nx": nx}, moist=True,
                    bl_pbl_physics=5, bl_mynn_version=version)
    with mynn_pricing_rank_chunk(chunk):
        registry = pf.mynn_pbl_scratch_slots(cfg, tile_buffer=True)
    outputs = {f"mynn_pbl_out_{name}" for name in MYNN_PBL_TENDENCY_FIELDS}
    expected = {slot: shape for slot, shape in registry.items() if slot not in outputs}
    scratch_owner = DomainState(cfg, array_module=np)
    assert bind_mynn_rank_chunk(scratch_owner, cfg, chunk) == chunk
    allocate = scratch_owner.scratch
    requests = []

    def checked(shape, slot, dtype=np.float32):
        assert tuple(shape) == expected[slot], slot
        value = allocate(shape, slot, dtype=dtype)
        assert value.nbytes == 4 * math.prod(expected[slot])
        requests.append(slot)
        return value

    def resident(array, *, shape, dtype, **unused):
        assert array.shape == shape and array.dtype == np.dtype(dtype)
        assert array.flags.c_contiguous

    def pointer(array):
        return int(array.__array_interface__["data"][0]), array.nbytes

    scratch_owner.scratch = checked
    owner = SimpleNamespace(_closed=False, _active=True,
                            drivers=[SimpleNamespace(state=scratch_owner)])
    scope = OriginalWorkspaceReuseScope(owner)
    scope.kind = "pbl"
    env = dict(np=np, chunk=chunk, nz=nz, options={"bl_mynn_version": version},
               borrowed=True, scratch_owner=scratch_owner, reuse_scope=scope,
               mynn_pbl_scratch_shapes=mynn_pbl_scratch_shapes,
               mynn_pbl_index_shapes=mynn_pbl_index_shapes,
               mynn_pbl_flag_shapes=mynn_pbl_flag_shapes,
               _resident=resident, _pointer=pointer, device=0, array_module=np)
    exec(bind, env)
    assert set(requests) == set(env["buffers"]) == set(expected)
    assert len(requests) == len(expected)
    assert {slot: value.shape for slot, value in env["buffers"].items()} == expected
    assert len({pointer(value)[0] for value in env["buffers"].values()}) == len(expected)
    for slot, value in env["buffers"].items():
        row = pf.scratch_slot_lifetime(slot)
        assert row is not None and row.evidence and row.rationale
        assert row.kind == ("excluded_unproven" if slot in MYNN_PBL_CONSTANT_ZERO_SLOTS
                            or value.dtype == np.dtype("int32") else "write_before_read")

    plans = {borrow: mynn_pbl_output_plan(nz=nz, ny=ny, nx=nx, members=2,
             column_chunk=chunk, bl_mynn_version=version, borrow_scratch=borrow)
             for borrow in (False, True)}
    full = BatchStorage(plans[False], 2, array_module=np, available_bytes=1 << 28)
    borrowed = BatchStorage(plans[True], 2, array_module=np, available_bytes=1 << 28)
    assert {name.removeprefix("mynn_pbl:") for name in full.arrays
            if not name.startswith("mynn_pbl:out_")} == set(expected)
    assert set(borrowed.arrays) == {f"mynn_pbl:out_{name}" for name in MYNN_PBL_TENDENCY_FIELDS}
    workspace_payload = sum(value.nbytes for value in env["buffers"].values())
    assert full.payload_bytes == borrowed.payload_bytes + workspace_payload
    assert borrowed.payload_bytes == 6 * 2 * nz * ny * nx * 4
    workspace_blocks = sum(((value.nbytes + 511) // 512) * 512
                           for value in env["buffers"].values())
    assert plans[False].required_bytes(2) - plans[True].required_bytes(2) == workspace_blocks
    assert plans[True].reserved_bytes == mynn_pbl_transient_bytes(
        nz=nz, column_chunk=chunk, bl_mynn_version=version)
    rates = list(borrowed.arrays.values())
    assert all(not np.shares_memory(left, right) for i, left in enumerate(rates)
               for right in rates[i + 1:] + list(env["buffers"].values()))
    assert all(not np.shares_memory(value[0], value[1]) for value in rates)

    exec(recheck, env)
    assert requests == list(expected) * 2
    assert {slot: pointer(value) for slot, value in env["buffers"].items()} == env["identities"]
    fixed = _DeclaredScratchState(env["buffers"])
    first = next(iter(expected))
    for shape, slot, dtype in (((1,), "unpriced", np.float32),
                               ((1,), first, np.float32),
                               (expected[first], first, np.float64)):
        with pytest.raises(ValueError, match="absent from its allocation plan"):
            fixed.scratch(shape, slot, dtype=dtype)
    before = len(requests)
    owner._active = False
    with pytest.raises(RuntimeError, match="complete active member leaf rendezvous"):
        exec(recheck, env)
    assert len(requests) == before
    owner._active = True
    scratch_owner._scratch[first] = np.zeros_like(scratch_owner._scratch[first])
    with pytest.raises(ValueError, match="original scratch owner changed"):
        exec(recheck, env)


def test_atmosphere_density_transient_has_actual_independent_storage(monkeypatch):
    from woof.core import physics, state as state_module

    cfg = RunConfig(**_TINY, moist=True, mp_physics=18, bl_pbl_physics=1,
                    sf_sfclay_physics=1)
    monkeypatch.setattr(state_module, "cp", np)
    monkeypatch.setattr(physics, "cp", np)
    state = state_module.DomainState(cfg, array_module=np)
    with np.errstate(divide="ignore", invalid="ignore"):
        atmosphere = physics._prepare_atmosphere(state)
    rho = atmosphere["rho"]
    assert rho.shape == pf.atmosphere_transient_shapes(cfg)["atmosphere/rho"]
    assert rho.dtype == np.dtype("float32")
    assert rho.nbytes == 4 * cfg.nz * cfg.ny * cfg.nx
    assert not np.shares_memory(rho, state.alt)


# ---------------------------------------------------------------------------
# (d) LBC residents + the F4 nest allocation manifest
# ---------------------------------------------------------------------------

def test_lbc_interval_values_hand_check(d01_cfg):
    """Independent hand arithmetic for one interval's side tables
    (W=5, nz=49, ny=200, nx=250; value+tendency, 4 sides per field)."""
    per_field = {
        "u": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 251),
        "v": 2 * (2 * 49 * 201 * 5 + 2 * 49 * 5 * 250),
        "theta": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 250),
        "qv": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 250),
        "phi": 2 * (2 * 50 * 200 * 5 + 2 * 50 * 5 * 250),
        "mu": 2 * (2 * 1 * 200 * 5 + 2 * 1 * 5 * 250),
    }
    assert pf.lbc_interval_values(d01_cfg) == sum(per_field.values())
    assert pf.lbc_interval_values(d01_cfg) == 2224960
    assert pf.lbc_intervals(43200.0, 21600.0) == 2
    assert pf.lbc_intervals(43201.0, 21600.0) == 3


@requires_4dom_inputs
def test_nest_field_kinds_by_scheme(exp4):
    dry = RunConfig(**_TINY)
    assert pf.nest_field_kinds(dry) == ("u", "v", "w", "t", "ph", "mu")
    kessler = RunConfig(**_TINY, moist=True, mp_physics=1)
    assert pf.nest_field_kinds(kessler) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr")
    # Morrison: all active species incl. the scalar numbers; nc excluded
    # (no advection copy -- state.py has no nc0).
    assert pf.nest_field_kinds(exp4.domain(2).run) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg", "nr", "ni", "ns", "ng")


@requires_4dom_inputs
def test_p3_nest_forcing_excludes_snow_and_graupel(exp4):
    """mp=50 is OUT of the qi/qs/qg block, and the exclusion is the answer.

    ``nest_field_kinds`` decides which of WRF's boundary-forced Registry
    members a scheme activates, not which schemes have ice.  P3's package
    is ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib``
    (Registry.EM_COMMON:3038) -- one ice mass, no ``qs``, no ``qg`` -- and
    WRF's own ``P3_1CATEGORY`` driver arm passes no snow and no graupel
    array (module_microphysics_driver.F:1557-1602).  Both directions are
    pinned so neither edit can land unargued: folding 50 into the
    three-ice-mass tuple would name ``qi`` twice and price sixteen rolling
    boundary tables for species an mp=50 child never allocates, and
    dropping the rime pair would decouple qir/qib from the ice they
    describe.
    """
    p3 = RunConfig(**_TINY, moist=True, mp_physics=50)
    kinds = pf.nest_field_kinds(p3)
    assert kinds == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "ni", "nr", "qir", "qib")
    assert len(kinds) == len(set(kinds)), f"duplicate nest field kind: {kinds}"
    for absent in ("qs", "qg", "qh"):
        assert absent not in kinds, absent

    # The block itself is unchanged: exactly the schemes whose Registry
    # moist package carries all three ice masses (Registry.EM_COMMON:
    # 3021, 3024-3026, 3031, 3033, 3036).
    for mp in (6, 8, 9, 10, 16, 18, 24, 28):
        block = pf.nest_field_kinds(
            RunConfig(**_TINY, moist=True, mp_physics=mp))
        assert {"qi", "qs", "qg"} <= set(block), mp

    # The manifest consequence, not just the list: no rolling boundary
    # table is priced for a species P3 does not carry, and the rime pair
    # gets the same four sides x value/tendency treatment as the moments.
    d02 = exp4.domain(2)
    child = dataclasses.replace(
        d02, run=dataclasses.replace(d02.run, mp_physics=50))
    slots = pf.nest_slot_shapes(child, exp4.spec_bdy_width, exp4.domain(1))
    assert not [name for name in slots
                if name.startswith(("nest_qs_", "nest_qg_"))]
    for name in ("nest_qir_bxs", "nest_qir_btye", "nest_qib_bys",
                 "nest_qib_btxe"):
        assert name in slots, name
    assert len(slots) == 14 * 4 * 2 + 3 * 6 + 2 == 132


@requires_4dom_inputs
def test_nest_allocation_manifest_inventory(exp4):
    manifest = pf.nest_allocation_manifest(exp4)
    assert sorted(manifest) == [2, 3, 4]  # root never registers nest slots
    for slots in manifest.values():
        # 16 kinds x 4 sides x (value + tendency), six geometry arrays
        # per three staggers, plus simultaneously live arena-audited parent
        # and child full fields.
        assert len(slots) == 16 * 4 * 2 + 3 * 6 + 2 == 148
    d02 = manifest[2]
    # Rolling tables: WRF Registry naming/layout (u_bxs/u_btxs style).
    assert d02["nest_u_bxs"] == (49, 400, 5)
    assert d02["nest_u_btxs"] == (49, 400, 5)
    assert d02["nest_u_btys"] == (49, 5, 501)
    assert d02["nest_v_bxs"] == (49, 401, 5)
    assert d02["nest_w_bxs"] == (50, 400, 5)
    assert d02["nest_ph_bye"] == (50, 5, 500)
    assert d02["nest_mu_bye"] == (1, 5, 500)
    assert d02["nest_ng_bxe"] == (49, 400, 5)
    # F16 retires every donor strip.  One full-parent field is borrowed
    # from the shared force-only arena; d02's parent d01 has 50*200*250
    # full-level w values, the largest parent field on that edge.
    assert not any("donor" in name for name in d02)
    assert d02["nest_parent_field"] == (50 * 200 * 250,)
    assert manifest[3]["nest_parent_field"] == (50 * 400 * 500,)
    assert manifest[4]["nest_parent_field"] == (50 * 501 * 501,)
    # T10 device_tables registry: ci/ip/cj/jp int32 maps and ratio-length
    # xig/xjg float32 coefficients, independently stored per stagger.
    assert d02["nest_sint_ci_m"] == (500,)
    assert d02["nest_sint_ip_x"] == (501,)
    assert d02["nest_sint_cj_y"] == (401,)
    assert d02["nest_sint_jp_y"] == (401,)
    assert d02["nest_sint_xig_m"] == (4,)
    assert manifest[4]["nest_sint_xjg_m"] == (3,)
    dtypes = pf.nest_slot_dtypes(exp4.domain(2), exp4.spec_bdy_width,
                                 exp4.domain(1))
    assert dtypes["nest_sint_ci_m"] == "int32"
    assert dtypes["nest_sint_jp_y"] == "int32"
    assert dtypes["nest_sint_xig_m"] == "float32"
    assert dtypes["nest_u_bxs"] == "float32"

    # Logical-request footprint pins.  The simultaneously live full-parent
    # and full-child fields are counted here even though the physical arena
    # aliases them to distinct dead RK backings when capacities permit.
    totals = {gid: sum(4 * math.prod(shape) for shape in slots.values())
              for gid, slots in manifest.items()}
    assert totals == {2: 103165552, 3: 149390256, 4: 193084928}
    assert sum(totals.values()) < 450 * 1024 ** 2


# ---------------------------------------------------------------------------
# (e) RRTMGP workspace/chunk formula
# ---------------------------------------------------------------------------

def test_gas_table_meta_and_default_chunk():
    meta = pf._gas_table_meta()
    assert meta["ngpt_lw"] == 256 and meta["ngpt_sw"] == 224
    from woof.core.rrtmgp import RRTMGPRadiation
    default = {f.name: f.default
               for f in dataclasses.fields(RRTMGPRadiation)}["column_chunk"]
    assert pf.DEFAULT_COLUMN_CHUNK == default == 3125
    # 354 375 000 B = 337.96 MiB.  The history of this number IS the
    # history of the layout: 1 025 700 000 (978.18 MiB) originally;
    # 774 375 000 (738.50) when the RTE phases stopped carrying slots
    # they never read; 832 375 000 (793.83) when the finalize fused into
    # the solvers and the finalized cubes stopped existing;
    # 359 825 000 when the LW solver derived its own Planck sources
    # and lay_source/lev_source/sfc_source stopped existing; and
    # 354 375 000 when the default model top moved from 100 to 50 hPa
    # (fewer above-model layers; tests/test_ptop_default.py).  All
    # four phases sit within 3% of each other -- there is no dominant
    # phase left to shrink.
    assert pf._workspace_total_bytes(49, default) == 354375000


@requires_4dom_inputs
def test_estimate_uses_experiment_column_chunk(exp4):
    configured = dataclasses.replace(exp4, column_chunk=6250)
    estimate = pf.estimate_experiment(configured)
    assert estimate.column_chunk == 6250
    # Same-configuration equality: the reference is priced at the CASE's
    # own model top, because the estimate honours it (the fix(ptop)
    # landing).  Pinning the library default's number here would assert
    # the estimate IGNORES the config's p_top -- the exact defect that
    # landing removed.
    assert estimate.workspace_bytes == pf._workspace_total_bytes(
        49, 6250, configured.vertical.p_top)


def test_rrtmgp_chunk_loops_and_mcica_seed_are_column_local(monkeypatch):
    """Static pin for A-1's no-chunk-size-arithmetic proof.

    Both solver loops may use their absolute ``start`` only to construct the
    columns a chunk covers and its width.  McICA selects the same column's
    bottom pressures and never folds a local/global column number or chunk
    size into its seeds.
    """
    import inspect
    import textwrap

    from woof.core.rrtmgp import RRTMGPRadiation, _rte_gpt_tile

    source = textwrap.dedent(inspect.getsource(RRTMGPRadiation.__call__))
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree)
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name)
             and node.target.id == "start"]
    assert len(loops) == 2
    parent = {child: node for node in ast.walk(tree)
              for child in ast.iter_child_nodes(node)}

    def statement(node):
        while not isinstance(node, ast.stmt):
            node = parent[node]
        return node

    def loads(loop, name):
        return [node for node in ast.walk(loop)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load) and node.id == name]

    def assigned(node):
        stmt = statement(node)
        assert isinstance(stmt, ast.Assign) and len(stmt.targets) == 1, (
            ast.get_source_segment(source, stmt))
        return stmt.targets[0].id

    # How many reads of ``start`` each loop makes.  LW: two, the start and
    # stop of its ``sl`` slice.  SW: four since 7f4d1a2b3 (the shortwave
    # pass skips dark columns) chunks the COMPACTED daylit list, so the
    # stop becomes its own name and the chunk is either a slice or an
    # index array: ``stop = min(start + chunk, sw_ncol)``,
    # ``slice(start, stop)`` when every column is daylit,
    # ``sw_columns[start:stop]`` when some are dark, and
    # ``chunk_ncol = stop - start``.  Every read is index construction,
    # which is what the next check holds rather than the count: ``start``
    # and ``stop`` are read only by the assignments of ``sl``, ``stop``
    # and ``chunk_ncol``.
    starts = {}
    for loop in loops:
        loop_source = ast.get_source_segment(source, loop)
        band = "sw" if "sw_chunk = " in loop_source else "lw"
        starts[band] = len(loads(loop, "start"))
        for name in ("start", "stop"):
            for node in loads(loop, name):
                assert assigned(node) in {"sl", "stop", "chunk_ncol"}
        # The width sizes the chunk's workspace views and its scatter
        # buffers and picks a cached scratch shape.  It never meets a
        # value: its only readers are ``workspace.phase(kind, width)``,
        # a shape tuple and the full-chunk comparison.
        for node in loads(loop, "chunk_ncol"):
            reader = parent[node]
            if isinstance(reader, ast.Tuple):
                reader = parent[reader]
                assert (isinstance(reader, ast.Call)
                        and ast.unparse(reader.func) == "cp.empty")
            elif isinstance(reader, ast.Compare):
                assert (ast.unparse(reader)
                        == "chunk_ncol == self.column_chunk")
            else:
                assert (isinstance(reader, ast.Call)
                        and ast.unparse(reader.func) == "workspace.phase")
        assert not any(token in loop_source for token in (
            "sum(", "mean(", "cumsum(", "reduce("))
        assert loop_source.count("_prepare_above_model_chunk(") == 1
        assert loop_source.count("columns=sl") == 1
    assert starts == {"lw": 2, "sw": 4}
    # The compacted list is the daylit columns in ascending model order,
    # and a dark column's zero is the allocation's own (zeros_like
    # whenever any column is dark), the +0.0 the full path's where() wrote.
    assert "sw_columns = cp.flatnonzero(daylight.reshape(-1))" in source
    assert ("sw_allocate = cp.empty_like if sw_ncol == ncol "
            "else cp.zeros_like") in source
    # The flux stores (bda1aa071, one fused kernel per pair): LW and full
    # SW chunks write their own ``sl`` views, the full SW path with the
    # daylight merge, and a compacted chunk is scattered back to exactly
    # the columns it gathered.
    assert source.count("_store_model_flux_pair(") == 3
    assert "nz, lw_up[sl], lw_dn[sl], xp=cp)" in source
    assert "daylight=daylight.reshape(-1)[sl], xp=cp)" in source
    assert "sw_up[sl] = chunk_up" in source
    assert "sw_dn[sl] = chunk_dn" in source
    # The surface direct beam BEP+BEM reads (sw_dir_sfc, sf_urban_physics =
    # 3), written to its own chunk's columns only: compacted chunks carry
    # daylit columns alone, full chunks merge daylight like sw_up/sw_dn.
    assert source.count("_model_flux_interfaces(") == 1
    assert "sw_dir_sfc[sl] = (direct if sw_columns is not None else" in source
    # A ragged daylit tail is a chunk width the full path never ran, so the
    # solver's g-point fold must not depend on the width: it partitions an
    # unchanged ascending sum at a fixed 32 (SW) or 128 (LW) whatever ncol.
    for name in ("WOOF_RTE_TILE_WIDTH", "WOOF_RTE_LW_TILE_WIDTH",
                 "WOOF_RTE_SW_TILE_WIDTH"):
        monkeypatch.delenv(name, raising=False)
    for ncol in (1, 3, 255, 4096, 12500):
        assert _rte_gpt_tile(None, ncol, 224, fold=True) == 32
        assert _rte_gpt_tile(None, ncol, 256, fold=True, longwave=True) == 128

    kernel = (ROOT / "woof" / "core" / "kernels" /
              "rrtmgp_mcica.cu").read_text(encoding="utf-8")
    # The seed derivation lives in its own device function since the
    # g-point loop became one thread per g-point with a jump-ahead, so
    # the old slice -- from the state declaration to `for (int g ...)` --
    # no longer brackets it and stopped bracketing anything at all.  The
    # PROPERTY is unchanged, and a named function is a better anchor than
    # two statements that happened to sit either side of it.
    seed_start = kernel.index("__device__ __forceinline__ void mcica_seed(")
    seed_end = kernel.index("\n}", seed_start)
    seed = kernel[seed_start:seed_end]
    assert "play[col * nlay + n]" in seed
    assert "frac * 1.0e9" in seed
    assert "permuteseed" in seed
    assert "blockIdx" not in seed and "chunk" not in seed
    # ...and the caller hands it the ABSOLUTE column index, which is the
    # half of column-locality that lives at the call site: it is what
    # makes a chunked run reproduce an unchunked one bit for bit.  Since
    # ad03ddb33 (the subcolumn masks share each column's seed) the kernel
    # is one block per column, and thread 0 seeds the block's shared state
    # once from that column's own pressures; every g-point thread of the
    # block is the same column, so the seed is still per column.
    flat = " ".join(kernel.split())
    assert "const int col = blockIdx.x;" in flat
    assert ("mcica_seed(play, col, nlay, permuteseed, "
            "seed[0], seed[1], seed[2], seed[3]);") in flat
    from woof.core.rrtmgp import _mcica_cloud_masks
    launch = " ".join(inspect.getsource(_mcica_cloud_masks).split())
    assert "(int(ncol),), (threads,)," in launch


def test_workspace_is_the_phase_maximum_simultaneous_set():
    """Shadow F2 fix: the workspace bound is the max over the four solver
    phases' EXACT live sets (col_dry included).  At the 50 hPa default
    model top (tests/test_ptop_default.py) WRF's cap construction adds 13
    LW and one SW layer, so the maximum is now SW OPTICS; with the
    finalize fused into the solvers and the LW solver deriving its own
    Planck sources, the finalized optics cubes and all three Planck source
    arrays exist in no phase.  (At the previous 100 hPa default the cap
    added 25 LW layers and LW OPTICS led, 1 439 300 000 B, all four
    phases within 3%.)"""
    phases = pf.rrtmgp_workspace_phases(49, 12500)

    def total(items):
        return sum(math.prod(shape) * size
                   for shape, size in items.values())

    assert {name: total(items) for name, items in phases.items()} == {
        "lw_optics": 1205900000,
        "lw_rte": 1185100000,
        "sw_optics": 1417500000,
        "sw_rte": 1397550000,
    }
    # A carried slot is only carried if it lands on the SAME BYTES the
    # optics phase produced it in, and `phase()` walks the layout in order
    # -- so this is the property the tightening actually rests on.
    def offsets(items):
        walked, offset = {}, 0
        for name, (shape, size) in items.items():
            walked[name] = offset
            offset += math.prod(shape) * size
        return walked

    for optics, rte in (("lw_optics", "lw_rte"), ("sw_optics", "sw_rte")):
        produced, consumed = offsets(phases[optics]), offsets(phases[rte])
        carried = set(phases[optics]) & set(phases[rte])
        assert carried
        for name in carried:
            assert produced[name] == consumed[name], (rte, name)
    # An OPTICS phase carries col_dry (rrtmgp.py:1615, retained by the gas
    # optics result).  An RTE phase does not: nothing in it reads col_dry,
    # so its bytes hold RTE output.  With the finalize fused into the
    # solvers, what an RTE phase DOES carry is exactly what the fused
    # solver reads -- the gas cube(s), the band cloud cubes it consumes and
    # the McICA mask -- and the finalized optics cubes exist in no phase.
    assert phases["lw_optics"]["col_dry"] == ((12500, 62), 4)
    assert phases["sw_optics"]["col_dry"] == ((12500, 50), 4)
    for dead in ("cld_asy", "col_dry", "optics_tau"):
        assert dead not in phases["lw_rte"]
    for dead in ("vmr", "col_dry", "optics_tau", "optics_ssa", "optics_g"):
        assert dead not in phases["sw_rte"]
    for phase in phases.values():
        assert not any(name.startswith("optics_") for name in phase)
    # LW Planck reads the VMR back; SW builds no Planck source and does not.
    assert phases["lw_rte"]["vmr"] == ((12500, 62, 20), 4)
    # The later sw_rte phase enumerates mu0 + the three flux arrays.
    assert phases["sw_rte"]["mu0"] == ((12500, 50), 4)
    assert phases["sw_rte"]["flux_dir"] == ((12500, 51), 4)
    # The albedo/incidence arrays exist only in the RTE phase; the mask is
    # a CARRIED slot now -- the fused solver reads it during the RTE phase.
    assert "albedo_gpt" not in phases["sw_optics"]
    assert phases["sw_rte"]["mcica_mask"] == ((12500, 50, 224), 1)

    full = pf._workspace_total_bytes(49, 12500)
    assert full == 1417500000
    assert pf._workspace_total_bytes(49, 6250) * 2 == full
    assert pf._workspace_total_bytes(49, 3125) * 4 == full
    # SW OPTICS is the maximum phase at the 50 hPa default (LW led at
    # the previous 100 hPa default, where its 25 above-model layers
    # outweighed SW's one).
    shapes = pf.rrtmgp_workspace_shapes(49, 12500)
    assert all(name.startswith("sw_optics/") for name in shapes)
    assert shapes["sw_optics/gas_tau"] == ((12500, 50, 224), 4)
    assert shapes["sw_optics/mcica_mask"] == ((12500, 50, 224), 1)
    # The Planck source arrays exist in no phase: the LW solver derives
    # them in registers (rrtmgp_planck_common.cuh).
    for gone in ("lay_source", "lev_source", "sfc_source"):
        assert all(gone not in items for items in phases.values())
    toa_column = pf.rrtmgp_workspace_phases(49, 2, p_top=0.0)
    assert toa_column["lw_optics"]["gas_tau"] == ((2, 49, 256), 4)
    assert toa_column["sw_optics"]["gas_tau"] == ((2, 49, 224), 4)


def test_shared_rrtmgp_workspace_is_one_real_allocation_with_full_audit():
    import inspect

    from woof.core.model import SharedRRTMGPChunkWorkspace
    from woof.core.rrtmgp import (RRTMGP_WORKSPACE_LIFETIME_AUDIT,
                                   RRTMGPRadiation)

    layouts = pf.rrtmgp_workspace_phases(3, 2)
    workspace = SharedRRTMGPChunkWorkspace(
        nz=3, column_chunk=2, _array_module=np,
        _phase_layouts_input=layouts)
    assert workspace.p_top == 5000.0
    assert workspace.nbytes == pf._workspace_total_bytes(3, 2)
    for phase, items in layouts.items():
        views = workspace.phase(phase, 2)
        assert set(views) == set(items)
        assert all(np.shares_memory(value, workspace.storage)
                   for value in views.values())
        assert set(RRTMGP_WORKSPACE_LIFETIME_AUDIT[phase]) == set(items)
        assert all(RRTMGP_WORKSPACE_LIFETIME_AUDIT[phase].values())

    # Common optics live values retain the same address in the immediately
    # following RTE layout; only dead tail storage is repurposed.
    for kind in ("lw", "sw"):
        optics = workspace.phase(f"{kind}_optics", 2)
        rte = workspace.phase(f"{kind}_rte", 2)
        for name in set(optics) & set(rte):
            assert optics[name].ctypes.data == rte[name].ctypes.data

    # Distinct per-domain adapters consume the same allocated workspace, not
    # merely equal capacity tokens.
    first = object.__new__(RRTMGPRadiation)
    second = object.__new__(RRTMGPRadiation)
    first.chunk_workspace = second.chunk_workspace = workspace
    assert first.chunk_workspace.storage is second.chunk_workspace.storage
    driver_source = inspect.getsource(RRTMGPRadiation.__call__)
    for phase in layouts:
        assert f'workspace.phase("{phase}"' in driver_source
    for producer in ("_gas_optics", "_cloud_optics", "_mcica_cloud_masks",
                     "_finalize_cloud_optics", "_planck_sources",
                     "_lw_rte", "_sw_rte"):
        assert producer in driver_source


def test_rrtmgp_column_transients(d01_cfg):
    cols = pf.rrtmgp_column_shapes(d01_cfg)
    ncol = d01_cfg.ny * d01_cfg.nx
    chunk = pf.DEFAULT_COLUMN_CHUNK
    assert cols["columns/play"] == ((ncol, 49), 4)
    assert cols["columns/plev"] == ((ncol, 50), 4)
    assert cols["columns/effs"] == ((ncol, 49), 4)  # Morrison extras
    # 62 = 49 model + 13 above-model LW layers at the 50 hPa default
    # (74 = 49 + 25 at the previous 100 hPa default).
    assert cols["columns/metadata_jt"] == ((chunk, 62), 4)
    assert cols["columns/upper_peak_play"] == ((chunk, 62), 4)
    assert cols["columns/upper_peak_plev"] == ((chunk, 63), 4)
    assert pf.rrtmgp_column_shapes(
        d01_cfg, column_chunk=17)["columns/metadata_jt"] == ((17, 62), 4)
    assert pf.rrtmgp_column_shapes(RunConfig(**_TINY)) == {}


# ---------------------------------------------------------------------------
# Estimates: itemization, shared counting, golden pins, d01 calibration
# ---------------------------------------------------------------------------

def test_estimate_domain_itemization_pins(exp1):
    est = pf.estimate_experiment(exp1)
    (d01,) = est.domains
    assert not est.uses_shared_scratch_arena
    assert not est.uses_shared_dycore_state_workspace
    assert est.scratch_arena_bytes == est.scratch_arena_saved_bytes == 0
    assert est.dycore_state_workspace_bytes == est.dycore_state_saved_bytes == 0
    by_cat = {c: d01.category_bytes(c) for c in
              ("state", "physics", "scratch", "lbc", "nest", "transient")}
    # Default-on CQ adds the three float32 faces for this moist domain:
    # 4 * (49*200*251 + 49*201*250 + 50*200*250) = 29,688,200 B.
    # Ring-guard note: the spec-zone microphysics exclusion adds its
    # mp_ring_save_* snapshot family to a specified mp=10 domain --
    # 17 volume slots (16 mutated fields + refl_10cm stash) x
    # 4*49*(2*250 + 2*198) = 175,616 B, plus 7 surface slots x
    # 4*(2*250 + 2*198) = 3,584 B: 2,985,472 + 25,088 = 3,010,560 B on
    # top of the previous 564,250,212-B scratch pin.
    # SFCLAY's USTM state (3669990e8c) adds one FP32 surface plane:
    # 4 * 200 * 250 = 200,000 B. Its actual producer/inventory is checked
    # by test_sfclay; this is distinct from OLR below.
    # Physics carries the domain's OLR publication buffer, one resident
    # (ny, nx) FP32 field: 4 * 200 * 250 = 200,000 B.
    # The EOS's base-thickness correction dphb_resid adds one (nz, ny, nx)
    # FP32 field to a terrain state, 4*49*200*250 = 9,800,000 B, and the
    # float64-differenced coefficient drops dc3f/dc4f a further
    # 2 * 4 * 49 = 392 B: 9,800,392 B on top of the previous
    # 563,557,756-B state pin.
    # RULINGS 1 (history CLDFRA, 2.8.7): physics holds the radiation cloud
    # fraction the history publishes, one (nz, ny, nx) FP32 buffer,
    # 4*49*200*250 = 9,800,000 B on top of the previous 276,106,760-B pin.
    # 2.9.1: the exact chain's 3-D PBL and cumulus tendency history rows are
    # strict-only (physics_inventory.history_tendency_rows): thirteen
    # 4*49*200*250 planes, 127,400,000 B, leave the default pin.
    # The six 0-d SCM forcing scalars (24 B) are strict-only too (scm_forcing_fields).
    # 2.9.2 (rrtmgp-rows): the default RTE+RRTMGP radiation carries WRF's
    # top/bottom fluxes, 16 instantaneous + 16 AC* planes, and COSZEN: 33
    # 4*200*250 planes, 6,600,000 B of physics.  Its call transients add the
    # (16, ncol) flux stack 3,200,000 B and one chunk's clear-sky solve:
    # three (3125, 50) flux outputs 1,875,000 B, the (3125, 256) LW incident
    # plane 3,200,000 B and the (3125, 49, 16) zero band cloud 9,800,000 B,
    # 18,075,000 B of transient.
    # Held-zero backing sharing removes 2*u + 2*mass = 39,278,400 B
    # (radiation's three extra backings plus KF's smaller momentum face).
    # 04e65faa5ff (cap-solar): the default radiation split history carries
    # SWDDIR, SWDDNI and SWDDIF, three 4*200*250 FP32 physics planes,
    # 600,000 B on top of the previous 274,428,360-B physics pin.
    # paper/conus-suite-exact (2.9.11): the stored MF_VX_INV state plane msfv_inv,
    # 4*(ny+1)*nx = 201,000 B on this 250x200 grid.
    assert by_cat == {'state': 573559148,
 'physics': 275028360,
 'scratch': 597187064,
 'lbc': 67091504,
 'nest': 0,
 'transient': 469137500}
    assert d01.resident_bytes == sum(
        v for c, v in by_cat.items() if c != "transient")
    assert d01.resident_bytes == 1512866076
    assert est.resident_bytes == d01.resident_bytes + est.k_tables_bytes
    assert d01.transient_bytes == 469137500


@requires_4dom_inputs
def test_estimate_experiment_shared_counting(est4, exp4):
    # k-distribution tables counted ONCE (lru_cache-shared,
    # rrtmgp.py:324/:436), while audited scratch is one per-slot maximum.
    # +13,584 B over the pre-RRTMGP-optimisation pin: GasTables.__post_init__
    # now derives the minor-absorber g-point CSR
    # (minor_gpt_start_/minor_gpt_list_, lower and upper) and declares it as
    # FIELDS, so to_device uploads it and this count picks it up.  LW
    # 1,028 + 3,840 + 1,028 + 2,176; SW 900 + 2,176 + 900 + 1,536, all int32.
    assert est4.k_tables_bytes == pf.k_distribution_bytes() == 23777296
    assert est4.uses_shared_scratch_arena
    assert est4.scratch_arena_request_bytes == sum(
        d.arena_scratch_bytes for d in est4.domains)
    assert est4.scratch_arena_bytes == pf.shared_scratch_arena_bytes(
        exp4.domains)
    assert est4.uses_shared_dycore_state_workspace
    assert est4.dycore_state_request_bytes == sum(
        d.rebuilt_state_bytes for d in est4.domains)
    assert est4.dycore_state_workspace_bytes == (
        pf.shared_dycore_state_workspace_bytes(exp4.domains))
    assert est4.resident_bytes == (
        sum(d.resident_bytes for d in est4.domains)
        - est4.scratch_arena_saved_bytes
        - est4.dycore_state_saved_bytes + est4.k_tables_bytes)
    # Step transients take the max over sequentially stepping domains.
    assert est4.transient_peak_bytes == max(
        d.transient_bytes for d in est4.domains)
    assert est4.transient_peak_bytes == est4.domains[-1].transient_bytes
    # ONE shared chunk workspace for all four domains (section E policy),
    # sized by the CASE's configured chunk and the CASE's model top, not
    # the library defaults.
    assert exp4.column_chunk == 6250 != pf.DEFAULT_COLUMN_CHUNK
    assert est4.workspace_bytes == pf._workspace_total_bytes(
        49, exp4.column_chunk, exp4.vertical.p_top)


@requires_4dom_inputs
def test_estimate_4dom_golden_pins(exp4, est4):
    per_domain = {d.grid_id: d.resident_bytes for d in est4.domains}
    # The production MP18 authority enables moist-CQ on every domain. Direct
    # MUDF storage still removes the obsolete acoustic_muprev mass plane.
    # Assembly merge (verification lineage): the ring-guard mp_ring_save_*
    # snapshot slots add exactly 3,010,560 / 6,034,560 / 6,720,000 /
    # 8,050,560 B on d01--d04 (sum 23,815,680 B, the ring lane's ledgered
    # 4-domain total) on top of the ports-branch pins; both components
    # byte-derived on their certified branches.
    # v1.1 hygiene merge: the co-located vertical-CFL reduction adds one
    # FP32 word to each of the 256 `integration_health_partial` blocks and
    # to `health_final`, so every domain gains exactly 4 * (256 + 1) =
    # 1,028 B over the ring-lane pins.  Constant per domain because the
    # health reduction's block count does not scale with the grid.  The
    # batched YSU validator adds one four-byte status word per domain.
    # OLR publication buffer: one resident (ny, nx) FP32 field per 4/4
    # domain, so each domain gains exactly 4*ny*nx B -- 200,000 /
    # 800,000 / 1,004,004 / 1,440,000 on d01--d04 (sum 3,444,004 B).
    # EOS correction 6b11e4c994 allocates dphb_resid (nz,ny,nx) and
    # dc3f/dc4f (nz each). SFCLAY 3669990e8c adds USTM (ny,nx).
    # These persisted after the old pins: per-domain increases are
    # 4*((nz+1)*ny*nx + 2*nz), totaling 172,201,768 B. Actual state
    # allocation shapes are independently checked against NumPy-backed
    # DomainState, and USTM is in the surface producer's inventory.
    # Two later additions, re-pinned here only in 2.9.0 because these cells
    # skip on every machine without the 1974 bundle (the single-domain pins
    # took both when they landed): the CLDFRA publication buffer
    # (cf1b03b17), one resident (nz, ny, nx) FP32 physics persistent per
    # domain -- 9,800,000 / 39,200,000 / 49,196,196 / 70,560,000 B -- and the
    # strict health rim table (a677ad4e1), integration_health_rim
    # (1024, 3) FP32 = 12,288 B per domain.  An item-by-item diff of the
    # estimate at 8e170fd66 (these pins, green) and at the 2.9.0 lane tip
    # shows exactly those two items and nothing else; the sum is
    # 168,805,348 B.
    assert per_domain == {1: 1523343280, 2: 5483131672,
                          3: 6900025776, 4: 9872912648}
    nest = {d.grid_id: d.category_bytes("nest") for d in est4.domains}
    assert nest == {1: 0, 2: 103165552, 3: 149390256, 4: 193084928}
    # The post-CQ request includes both simultaneously live nested-force
    # full-field slots.  The shared physical arena still aliases them to
    # distinct dead RK backings when those capacities fit.
    assert est4.scratch_arena_request_bytes == 9471818140
    # Every Smag path requires distinct horizontal face staging; x/m reuse z
    # while y remains independent.  The pre-RK Smag K pair then borrows the
    # later acoustic coefficient backings.  These remove 141,237,600 B and
    # 141,120,000 B of exact physical allocation.
    assert est4.scratch_arena_bytes == 3315315836
    # 9,471,818,140 requested - 3,315,315,836 physical = 6,156,502,304 B.
    assert est4.scratch_arena_saved_bytes == 6156502304
    # Omega'' retains its forced boundary column. Domain ownership replaces
    # the unsafe maximum backing with four allocations: sum172,200,200 B,
    # max72,000,000 B, so physical residency increases by100,200,200 B.
    omega_bytes = [4 * (dc.run.nz + 1) * dc.run.ny * dc.run.nx
                   for dc in exp4.domains]
    assert sum(omega_bytes) == 172_200_200
    assert max(omega_bytes) == 72_000_000
    assert est4.dycore_state_request_bytes == 4930458300 - sum(omega_bytes)
    assert est4.dycore_state_workspace_bytes == 2061345600 - max(omega_bytes)
    assert est4.dycore_state_saved_bytes == 2869112700 - 100_200_200
    # Ring snapshot slots are resident (arena-excluded): the ports-branch
    # residency plus the exact 23,815,680-B ring total, plus the
    # 3,444,004-B four-domain OLR publication total.
    # +13,584 B over the pre-RRTMGP-optimisation pin: the minor-absorber
    # g-point CSR the gas tables now upload (see
    # test_estimate_experiment_shared_counting).  Nothing else in residency
    # moved -- the optimisation is a workspace and kernel change.
    # +168,805,348 B: CLDFRA (cf1b03b17) and the rim table (a677ad4e1).
    assert est4.resident_bytes == 14708970520 + 168_805_348 == 14877775868
    # The case configures column_chunk = 6250 (byte-identical to 3125,
    # 33% faster per radiation call); the 3125 numbers stay pinned in the
    # ladder below, so the trade this bought is on the record both ways.
    #
    # 2,051,400,000 -> 719,650,000 B.  The maximum phase used to be lw_rte
    # at 328,224 B/column; the LW solver now derives the Planck sources in
    # registers, so lay_source/lev_source/sfc_source (76,800 B/column
    # together) exist in no phase, the finalize is fused into the solvers so
    # the finalized optics cubes are gone too, and lw_OPTICS at 115,144
    # B/column is what the workspace is now sized to.
    assert est4.workspace_bytes == 719650000
    # _prepare_atmosphere's independent rho allocation is one full-volume
    # FP32 plane on the peak domain. The CPU storage witness above checks
    # the real producer; the single-domain pin already included this plane.
    rho_bytes = max(4 * dc.run.nz * dc.run.ny * dc.run.nx
                    for dc in exp4.domains)
    assert rho_bytes == 70_560_000
    assert est4.transient_peak_bytes == 3182840000 + rho_bytes == 3253400000
    # +3,444,004 B: the four-domain OLR publication total.
    # -1,331,736,416 B against the pre-optimisation pin: the 1,331,750,000 B
    # of workspace the phase change removed, less the 13,584 B of CSR.
    # +168,805,348 B: CLDFRA (cf1b03b17) and the rim table (a677ad4e1).
    assert est4.subtotal_bytes == (
        18611460520 + rho_bytes + 168_805_348) == 18850825868
    assert est4.headroom == pf.forecast_pool_headroom(
        dc.run for dc in exp4.domains) == 1.13
    assert est4.alloc_estimate_bytes == math.ceil(
        est4.headroom * est4.subtotal_bytes) == 21301433231
    # Chunk ladder after arena sharing and physics-persistent reclamation.
    # The 1024-descriptor health-slot registration adds 49,168 B/domain to
    # the audited scratch; the pins below are computed on the merged tree.
    ladder = {chunk: pf.estimate_experiment(
        exp4, column_chunk=chunk).alloc_estimate_bytes
        for chunk in (6250, 3125, 1562, 256)}
    # Every rung carries the ring lane's ceil(1.15 x 23,815,680) =
    # 27,388,032 B on top of the ports-branch ladder.
    # Every rung also carries ceil(1.15 x 3,444,004) of OLR.
    # The ladder FLATTENED with the workspace: the whole rung-to-rung spread
    # is 1.13 x the workspace difference, and the workspace is now 2.85x
    # smaller, so 6250 -> 256 buys 824 MB where it used to buy 2,293 MB.
    # Every rung also carries ceil(1.13 x 168,805,348) = 190,750,043 B of
    # CLDFRA (cf1b03b17) and rim table (a677ad4e1), measured per rung.
    assert ladder == {6250: 21301433231, 3125: 20879123981,
                      1562: 20667901787, 256: 20491410305}


def test_d01_calibration_bounds_measured_fixture(exp1):
    """The pre-reclamation d01 measurement remains below the full estimate.

    Its persistent-used value is no longer a tight residency calibration:
    that run deliberately retained last_ysu, a composed stack, and copied
    diagnostics which this lane removes.
    """
    est = pf.estimate_experiment(exp1)
    measured = pf.CAL_D01_POOL_USED_PEAK_BYTES
    # Enforced bound: measured <= estimate.
    assert est.alloc_estimate_bytes >= measured
    # Current residency includes the EOS residual/coefficient arrays and
    # USTM and the three default-on CQ faces: 1,513,530,992 B, about 96%
    # of the historical 1.47-GiB peak; the held CLDFRA buffer (RULINGS 1,
    # 2.8.7) adds 9,800,000 B, 1,523,330,992 B or 96.5 %.
    # This comparison does not turn that old run into a new measurement.
    # The six 0-d SCM forcing scalars (24 B) are strict-only too (scm_forcing_fields).
    # The cap-solar SWDDIR/SWDDNI/SWDDIF history planes add 600,000 B.
    # paper/conus-suite-exact (2.9.11): the stored MF_VX_INV state plane msfv_inv,
    # 4*(ny+1)*nx = 201,000 B on this 250x200 grid.
    assert est.domains[0].resident_bytes == 1512866076
    # Compare the recorded pre-intake residency to its historical measurement.
    # New history carriers are separately priced above; this is no new pool measurement.
    ratio = 1523343280 / measured
    assert 0.95 <= ratio <= 0.97


def test_calibration_constants_pin_the_measurement_record():
    """CONSISTENCY pins, not validation (shadow F3 / review F5): these
    re-state the two controller measurement records -- the d01 run
    fixture (n0-preflight-baseline.log) and the N0 allocation probe
    (n0-alloc-probe-r2.json) -- so any silent constant edit is visible.
    The tier-2/3 model built on them is provisional reserve POLICY for
    controller ratification; nothing here can validate it against
    independent evidence."""
    assert pf.CAL_WDDM_FREE_BYTES == int(30.27 * GIB)
    assert pf.CAL_WDDM_TOTAL_BYTES == int(31.84 * GIB)
    assert pf.CAL_D01_POOL_USED_PEAK_BYTES == int(1.47 * GIB)
    assert pf.CAL_D01_POOL_HELD_BYTES == int(5.52 * GIB)
    assert pf.CAL_D01_DEVICE_FOOTPRINT_BYTES == int(11.24 * GIB)
    assert pf.CAL_FIXTURE_OVERHEAD_BYTES == int(11.24 * GIB) - int(5.52 * GIB)
    assert pf.CAL_D01_POOL_RETENTION_BYTES == (int(5.52 * GIB)
                                               - int(1.47 * GIB))
    assert pf.ALLOCATOR_HEADROOM == 1.15
    # N0 probe record: the fixture's 5.72 GiB memGetInfo gap was 12 h-run
    # drift -- a fresh allocation-only process measures 1.39 GiB, and
    # allocation-time pool retention is nil (16 MB).
    assert pf.PROBE_DEVICE_OVERHEAD_BYTES == 1489949696
    assert (pf.PROBE_POOL_HELD_BYTES
            - pf.PROBE_POOL_USED_PEAK_BYTES) == 16154112
    assert pf.PROBE_DEVICE_OVERHEAD_BYTES == (
        pf.PROBE_DEVICE_FOOTPRINT_BYTES - pf.PROBE_POOL_HELD_BYTES)


def test_tier_projection_algebra_is_consistent(exp1):
    """The tier-2/3 projections are labels over the estimate, pinned as
    ALGEBRAIC IDENTITIES (the old ">= fixture" assertions were
    tautologies -- the residual/overhead terms embed the same fixture
    numbers they were claimed to bound; shadow F3)."""
    est = pf.estimate_experiment(exp1)
    assert est.held_projection_bytes == (
        est.alloc_estimate_bytes + est.retention_residual_bytes)
    assert est.footprint_projection_bytes == (
        est.held_projection_bytes + est.device_overhead_bytes)
    # The two constants are the PLATFORM's: the Windows pool residual and
    # the 5090 zero-step probe overhead where the envelope family is
    # windows, zero on Linux, where neither showed up in any instrumented
    # run (platform_projection_constants).  Asserting the Windows numbers
    # by name made this identity a statement about the box it was written
    # on, and it was red on the Linux release node for that reason alone
    # (proof/node-reds-276).
    assert (est.retention_residual_bytes, est.device_overhead_bytes) == (
        pf.platform_projection_constants())
    if pf.envelope_platform() == "windows":
        assert est.device_overhead_bytes == pf.PROBE_DEVICE_OVERHEAD_BYTES
        assert est.retention_residual_bytes == \
            pf.pool_retention_residual_bytes()
    else:
        assert (est.retention_residual_bytes, est.device_overhead_bytes) == (0, 0)


# ---------------------------------------------------------------------------
# Reserve policy + the F11 enforced chain
# ---------------------------------------------------------------------------

def test_reserve_policy_split_proposals():
    """The two reserve proposals, split by gate (PENDING controller
    ratification at N0; instruction #5 of the fix round): the N0
    allocation gate carries only the probe-measured fresh-process
    overhead + external margin (alloc-time retention measured nil); the
    N5/N6 run gates add the fixture-calibrated run-churn residual."""
    n0 = pf.ReservePolicy.n0_alloc()
    assert n0.retention_residual_bytes == 0
    assert n0.device_overhead_bytes == pf.PROBE_DEVICE_OVERHEAD_BYTES
    assert n0.reserve_bytes == (pf.PROBE_DEVICE_OVERHEAD_BYTES
                                + pf.EXTERNAL_MARGIN_BYTES) == 2026820608

    run = pf.ReservePolicy.run_time()
    assert run.retention_residual_bytes == \
        pf.pool_retention_residual_bytes()
    assert run.reserve_bytes == n0.reserve_bytes + \
        pf.pool_retention_residual_bytes()
    assert run.reserve_bytes == 4959159922

    flat = pf.ReservePolicy.flat(2 * GIB)
    assert flat.reserve_bytes == 2 * GIB
    assert flat.budget_bytes(pf.PROBE_FREE_BYTES) == (
        pf.PROBE_FREE_BYTES - 2 * GIB)

    # The run-churn residual: fixture held minus the d01 alloc-estimate
    # basis (measured used + the workspace THAT FIXTURE RAN, with
    # headroom), clamped at zero -- calibration algebra, pinned.  The
    # workspace term is the pinned constant and not today's layout: see
    # CAL_D01_WORKSPACE_BYTES for why recomputing it punishes saving memory.
    basis = math.ceil(pf.ALLOCATOR_HEADROOM * (
        pf.CAL_D01_POOL_USED_PEAK_BYTES + pf.CAL_D01_WORKSPACE_BYTES))
    assert pf.pool_retention_residual_bytes() == max(
        0, pf.CAL_D01_POOL_HELD_BYTES - basis) == 2932339314


@requires_4dom_inputs
def test_n0_probe_projection_flags_stale_calibration_after_exact_aliases(
        exp4, est4):
    """Project retained bytes explicitly; never relabel them a fresh probe.

    The old receipt predates the EOS residual/coefficient arrays and USTM,
    and also the CLDFRA publication buffer (cf1b03b17) and the strict health
    rim table (a677ad4e1). Its algebra must add their physical allocation
    once, whereas the estimator applies the current measured forecast
    margin, 1.13. The rho array was already physically allocated by that
    probe and is now priced too, so it is not added twice to the receipt.
    The resulting margin is 316,805,219 B. PROBE_POOL_USED_PEAK_BYTES
    remains the original record; this constructed projection cannot certify
    a live measured-bound gate.
    """
    # ``PROBE_POOL_USED_PEAK_BYTES`` is a receipt taken at the LIBRARY default
    # chunk, so it must be projected against the default-chunk estimate.
    # Comparing it to the case's configured 6250 estimate would flip
    # ``alloc_measured_le_estimate`` to True on 0.36 GB of workspace the probe
    # never allocated -- concealing exactly the staleness this test exposes.
    est = pf.estimate_experiment(exp4, column_chunk=pf.DEFAULT_COLUMN_CHUNK)
    diff6_alias_saved = 141_237_600
    added_persistents = sum(
        4 * ((dc.run.nz + 1) * dc.run.ny * dc.run.nx + 2 * dc.run.nz)
        for dc in exp4.domains)
    assert added_persistents == 172_201_768
    # CLDFRA (nz, ny, nx) FP32 plus the (1024, 3) FP32 rim table, per domain.
    later_persistents = sum(
        4 * dc.run.nz * dc.run.ny * dc.run.nx + 4 * 1024 * 3
        for dc in exp4.domains)
    assert later_persistents == 168_805_348
    added_persistents += later_persistents
    projected_used = (pf.PROBE_POOL_USED_PEAK_BYTES + added_persistents
                      - est.dycore_state_saved_bytes
                      - (3035550000 - est.workspace_bytes)
                      - diff6_alias_saved - 141_120_000)
    assert projected_used - est.alloc_estimate_bytes == 316_805_219
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=projected_used,
        estimate_bytes=est.alloc_estimate_bytes,
        measured_free_bytes=pf.PROBE_FREE_BYTES,
        reserve=pf.ReservePolicy.flat(2 * GIB))
    # The measured-bound leg reads False BY DESIGN here: the projection of a
    # pre-optimisation receipt now exceeds the post-optimisation estimate,
    # which is this test's whole subject.  It is not a gate a live run trips
    # -- PROBE_POOL_USED_PEAK_BYTES is a retained calibration record, read
    # nowhere in preflight outside this file.
    assert legs == {"alloc_fits_wddm_budget": True,
                    "alloc_measured_le_estimate": False,
                    "alloc_estimate_le_wddm_budget": True}


def test_evaluate_alloc_gates_exact_chain():
    """F11 comparator semantics: measured_bound legs evaluate EXACTLY (no
    tolerance); a missing measurement can never pass (nest_gates F10)."""
    reserve = pf.ReservePolicy(retention_residual_bytes=0,
                               device_overhead_bytes=0,
                               external_margin_bytes=GIB)
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=10 * GIB, estimate_bytes=10 * GIB,
        measured_free_bytes=11 * GIB, reserve=reserve)
    assert legs == {"alloc_fits_wddm_budget": True,
                    "alloc_measured_le_estimate": True,
                    "alloc_estimate_le_wddm_budget": True}
    # One byte over the estimate is a FAILING GATE, not a note.
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=10 * GIB + 1, estimate_bytes=10 * GIB,
        measured_free_bytes=100 * GIB, reserve=reserve)
    assert legs["alloc_measured_le_estimate"] is False
    # estimate > budget fails independently (the leg the middle test
    # alone would miss: measured=20/estimate=40/budget=30).
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=20 * GIB, estimate_bytes=40 * GIB,
        measured_free_bytes=31 * GIB, reserve=reserve)
    assert legs["alloc_fits_wddm_budget"] is True
    assert legs["alloc_measured_le_estimate"] is True
    assert legs["alloc_estimate_le_wddm_budget"] is False
    # Missing measurements report None, never True.
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=None, estimate_bytes=GIB,
        measured_free_bytes=None, reserve=reserve)
    assert legs == {"alloc_fits_wddm_budget": None,
                    "alloc_measured_le_estimate": None,
                    "alloc_estimate_le_wddm_budget": None}


def test_gate_leg_names_match_the_n0_ledger():
    from woof.verify import nest_gates

    ledger = {g.metric for g in nest_gates.gates_for("N0")}
    assert set(pf.N0_GATE_METRICS) == ledger
    for metric in pf.N0_GATE_METRICS:
        assert nest_gates.gate("N0", metric).kind == "measured_bound"


@requires_4dom_inputs
def test_recommend_column_chunk_lever(exp4, est4):
    # Comfortably large budget: the CONFIGURED chunk already fits, so the
    # lever recommends it unchanged.
    assert pf.recommend_column_chunk(
        exp4, est4.alloc_estimate_bytes) == exp4.column_chunk == 6250
    # Budget between the 3125 and 6250 estimates: halving lands on 3125 --
    # a host with less VRAM still gets walked back to the library default.
    e3125 = pf.estimate_experiment(exp4, column_chunk=3125)
    e6250 = pf.estimate_experiment(exp4, column_chunk=6250)
    budget = e3125.alloc_estimate_bytes + (
        e6250.alloc_estimate_bytes - e3125.alloc_estimate_bytes) // 2
    assert pf.recommend_column_chunk(exp4, budget) == 3125
    # No halving can fit a tiny budget.
    assert pf.recommend_column_chunk(exp4, GIB) is None


# ---------------------------------------------------------------------------
# CLI registrar (`woof check` -- estimator mode is CPU-only)
# ---------------------------------------------------------------------------

def _run_check(argv):
    parser = argparse.ArgumentParser(prog="woof")
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(argv)
    args.explain = True  # This suite pins the detailed memory breakdown.
    # These tests isolate memory admission; GPU readiness has its own
    # compile-failure controls and must not contact a device in CPU tests.
    from unittest.mock import patch
    from woof.doctor import Check
    with patch("woof.doctor._cuda_headers_check", return_value=Check(
            "CUDA kernel headers", "verified", "fixture kernels ready")):
        return args.func(args)


@requires_grib1_bridge
@requires_4dom_inputs
def test_check_cli_estimator_json(capsys):
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["experiment"] == "real74_4dom"
    assert payload["column_chunk"] == 6250
    # Ring-guard mp_ring_save_* saves add 23,815,680 B of per-domain
    # (arena-excluded) scratch across the four domains; x1.15 headroom
    # lands 27,388,032 B above the ports-branch CLI pin; the OLR
    # publication buffers add a further 3,960,605 B of estimate.
    # -1,531,496,879 B on the RRTMGP optimisation: 1.15 x the
    # 1,331,750,000 B of chunk workspace the phase change removed at this
    # case's 6250 chunk, less 1.15 x the 13,584 B of gas-table CSR it added.
    # Current subtotal includes rho and uses the measured forecast margin.
    # +168,805,348 B of subtotal: the CLDFRA publication buffer (cf1b03b17)
    # and the strict health rim table (a677ad4e1) on all four domains; see
    # test_estimate_4dom_golden_pins.
    assert payload["alloc_estimate_bytes"] == math.ceil(
        1.13 * (18682020520 + 168805348)) == 21301433231
    # All requested moist-CQ slots are represented; the shared arena aliases
    # their lifetimes without changing the exact physical backing.
    assert payload["scratch_arena_saved_bytes"] == 6156502304
    assert payload["dycore_state_saved_bytes"] == 2869112700 - 100_200_200
    assert payload["domains"]["d04"]["by_category"]["nest"] == 193084928
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    # Estimator-only mode: the measured legs stay unevaluated.
    assert payload["gates"]["alloc_measured_le_estimate"] is None
    assert "alloc" not in payload and "abort" not in payload
    # The reserve split proposal is reported for the controller.  The
    # overhead leg is no longer the 2026-07-16 zero-step probe constant
    # (1.39 GiB): it is this configuration's own measured non-pool
    # residency -- the CUDA context plus the local-memory backing store the
    # widest frame it LAUNCHES reserves.
    #
    # That frame used to be `kf_column`'s unspecialized 24,064 B, worth
    # 6,016,204,800 B of driver reservation.  `kf.cu`'s bound now compiles
    # to this case's own nz = 49, measured 9,216 B, which drops it BELOW
    # `ysu`'s 9,232 B -- so this configuration's widest launched frame is
    # now YSU's, and the reservation is 8,208 * 1536 * 170 = 2,143,272,960
    # B.  3,872.9 MiB of the old reserve was a compile-time array bound.
    # retention_residual is 3% of the alloc estimate, which carries the
    # ring lane's 27,388,032 B: +821,641 B over the ports-branch pin,
    # and 3% of the OLR estimate is a further +118,818 B.
    # 2026-08-20 (task 206): +349,962,240 B on the pin, all of it the
    # CUDA-context term.  This is the ABSENT-card path (--budget-gib for
    # a card that is not in this machine), which prices the context from
    # the measured 2,304 B per resident thread plus the module-load
    # growth -- 766 MiB against the retired flat 432 MiB.  The flat
    # constant was one 2026-07-26 reading and it UNDER-charged this very
    # card by 215 MiB once it ran under Linux; an absent card is priced
    # above every card that has been measured, on purpose.
    # -45,944,906 B, entirely the retention_residual leg: it is 3% of the
    # alloc estimate, which the RRTMGP workspace change took down by
    # 1,531,496,879 B.  The other two legs are card properties and do not
    # move.  ``run_time_reserve_bytes`` below does NOT move either -- its
    # residual is the pinned CAL_D01_WORKSPACE_BYTES fixture basis.
    # 2026-08-21, the two column workspaces together: -308,469,760 B, and
    # the arithmetic is worth writing out because the two legs pull
    # opposite ways.  `kf` went 9,216 -> 512 B and `ysu` 9,232 -> 0, and
    # `ysu` was the widest frame this configuration LAUNCHED, so the
    # backing store fell from 2,143,272,960 B to `rrtmgp_rte`'s
    # 1,077,903,360 B -- a 1,065,369,600 B saving that neither cut could
    # have made alone, because whichever was left would have set the same
    # ceiling.  Against that, the two workspaces are charged for the
    # columns actually in flight: 443,555,840 B for KF (170 SMs x 8 blocks
    # x 32 lanes x 52 slots x 49 levels x 4 B) and 313,344,000 B for YSU.
    # EOS/USTM growth adds 5,940,961 B to the 3% retention term.
    # 2026-09-28, -8,355,840 B: `rrtmgp_rte`'s 5,152 B was a reading of
    # the source before the RRTMGP optimisation; every compile platform
    # re-read compiles it to 3,600 B, so the widest frame this
    # configuration launches is Morrison's 5,120 B and the backing store
    # is (5,120 - 1,024) x 1,536 x 170 = 1,069,547,520 B.
    # 2.9.0, +125,141,381 B, re-pinned only now because this cell skips on
    # every machine without the 1974 bundle: the WRF 4.6.1 line-for-line KF
    # port (26779da4e) keeps 66 column arrays in the workspace, not 52, so
    # KF's in-flight term is 170 x 256 x 66 x 49 x 4 = 562,974,720 B
    # (+119,418,880 B), and the 3% retention term follows the alloc
    # estimate's CLDFRA and rim-table growth (+5,722,501 B).
    reference = pf.card_local_memory_profile(None)
    assert payload["reserve_bytes"] == (
        math.ceil(0.03 * 21301433231) + reference.cuda_context_bytes
        + 1069547520 + 562974720 + 313344000 + 512 * (1 << 20)
    ) == 3924727221
    exp_4dom = load_experiment_case(CONFIG_4DOM)[0]
    assert payload["reserve_components"]["device_overhead_bytes"] == (
        reference.cuda_context_bytes + 1069547520
        + 562974720 + 313344000)
    assert payload["kernel_local_memory_bytes"] == 1069547520
    frames_4dom = pf.kernel_local_frame_bytes(exp_4dom)
    assert frames_4dom["rrtmgp_rte"] == 3600
    assert frames_4dom["morrison"] == max(frames_4dom.values()) == 5120
    assert "kf" in payload["kernel_modules"]
    assert frames_4dom["kf"] == 512
    from woof.core.kf import KFWS_SLOTS
    assert KFWS_SLOTS == 66
    assert pf.kf_column_workspace_bytes(
        exp_4dom, profile=reference) == 170 * 256 * 66 * 49 * 4 == 562974720
    # Same +349,962,240 B context shift as ``reserve_bytes`` above, the
    # same -308,469,760 B net from the two workspaces and the same
    # -8,355,840 B from the re-read `rrtmgp_rte` frame.  +119,418,880 B in
    # 2.9.0: the KF port's 66-array workspace (26779da4e); this leg's
    # residual is the pinned fixture basis, so the 3% term does not move it.
    assert payload["run_time_reserve_bytes"] == 6218023538
    assert payload["reserve_components"]["retention_residual_bytes"] == (
        math.ceil(0.03 * payload["alloc_estimate_bytes"]))


@requires_grib1_bridge
@requires_4dom_inputs
def test_check_cli_over_budget_fails_and_names_the_lever(capsys):
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "19.5"])
    out = capsys.readouterr().out
    assert rc == 1
    # The key is the key; the row is printed in the platform's spelling
    # (gate_display_name writes `_vram_` on Linux), and asserting the
    # Windows spelling made this test red on the Linux release node
    # (proof/node-reds-276).
    assert (pf.gate_display_name("alloc_estimate_le_wddm_budget")
            + ": FAIL") in out
    assert "OVER BUDGET" in out
    # The measured 1.13 margin and fully priced rho put 6250 above this
    # budget.3125 is20,688,373,938 B and is the first halving that fits.
    assert "--column-chunk 3125" in out


@requires_grib1_bridge
@requires_4dom_inputs
def test_check_over_budget_envelope_exits_nonzero(capsys, monkeypatch):
    """B-1: the report said "exceeds the WDDM budget" and exited 0.

    A a development machine pilot on virgin 1.0.1 read `woof check`'s own sentence --
    "observed peak envelope 12.98 GiB exceeds the WDDM budget 11.64 GiB"
    -- out of a command that exited 0, so every script wrapping it read
    green.  The prose and the exit code cannot disagree; the prose is the
    accurate one.  4, not 1: no gate failed, and the levers differ.
    """
    monkeypatch.setattr(pf, "host_platform", lambda: "win32")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, "a fitting envelope is still a clean pass"

    estimate_gib = payload["alloc_estimate_bytes"] / GIB
    tight = str(math.ceil(estimate_gib) + 1)
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert "WARNING: observed peak envelope" in out
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert rc != 0, "the sentence and the exit code must agree"
    # The warning names the code, so the reader of the text and the
    # reader of `echo $?` learn the same thing.
    assert f"exit code {pf._EXIT_ENVELOPE_OVER_BUDGET}" in out

    # A HARDER verdict outranks it: a failing gate is still 1, and a
    # fail-closed non-evaluable run is still 2.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "19.5"])
    assert capsys.readouterr().out.count("OVER BUDGET") == 1
    assert rc == 1
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = _run_check(["check", str(CONFIG_4DOM)])
    capsys.readouterr()
    assert rc == 2


@requires_grib1_bridge
@requires_4dom_inputs
def test_declared_free_is_capped_at_the_cards_physical_total(capsys):
    """B-2: `--card 16gb` declared 16.68 GiB free on a 16 GB card.

    The wizard states a budget and `check` adds the reserve back to
    recover a notional free.  That arithmetic never saw the card, so the
    16 GB tier bought the estimate about a gigabyte of budget the card
    does not physically have.  Free cannot exceed total, ever.
    """
    # The pure function first: declared size and measurement are both
    # ceilings, the tighter one binds, and neither ever widens.
    gib = int(GIB)
    assert pf.cap_free_to_physical(
        17 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=None) == (16 * gib, 16 * gib)
    # A measurement of the same card is tighter than its nameplate size
    # (a "16 GB" card has ~15.57 GiB usable), and it wins.
    assert pf.cap_free_to_physical(
        17 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=15 * gib) == (15 * gib, 15 * gib)
    # Already within capacity: untouched, and no cap is reported.
    assert pf.cap_free_to_physical(
        10 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=15 * gib) == (10 * gib, None)
    # No capacity statement at all imposes no ceiling -- a ceiling that
    # cannot be measured must never be invented.
    assert pf.cap_free_to_physical(
        99 * gib, card_total_bytes=None,
        measured_total_bytes=None) == (99 * gib, None)

    # And end to end through the CLI: a declared budget close enough to
    # the card that adding the reserve back overshoots its capacity.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "15",
                     "--vram-gib", "16", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["reserve_bytes"] > gib, "otherwise nothing to cap"
    assert payload["measured_free_bytes"] <= 16 * gib
    assert payload["free_bytes_capped_to_physical_bytes"] is not None
    assert "capped" in payload["free_bytes_source"]
    # The budget follows the capped free, so the gate is evaluated
    # against VRAM that exists.
    assert payload["budget_bytes"] == (
        payload["measured_free_bytes"] - payload["reserve_bytes"])
    assert rc != 0, "22.6 GiB of estimate does not fit a 16 GB card"

    # Text mode says so out loud rather than only in --json.
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "15",
                "--vram-gib", "16"])
    out = capsys.readouterr().out
    assert "CAPPED" in out
    assert "free VRAM cannot exceed the card" in out

    # Without a card size the declared figure stands: --budget-gib is
    # how you size for a machine that is not this one.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["free_bytes_capped_to_physical_bytes"] is None
    assert payload["measured_free_bytes"] > 100 * gib
    assert rc == 0


@requires_grib1_bridge
@requires_4dom_inputs
def test_check_cli_reports_observed_peak_envelope(capsys, monkeypatch):
    """The empirical envelope line: accurate, informational, budget-aware.

    On Windows the envelope is the MEASURED affine model (the 2026-08-19
    RTX 3080 calibration): estimate + itemized non-pool + unmodelled +
    the WDDM pool-slack fraction of the estimate (+ per-nest term).  The
    retired ``footprint x 1.75`` floor predicted 3.8x the measured peak
    on the calibration card and must be gone from the report -- WITHOUT
    changing any gate, because the enforced numbers remain the itemized
    estimate and the measured legs.

    The exit code is NOT informational, though: see
    ``test_check_over_budget_envelope_exits_nonzero``.  A a development machine pilot
    read this command's rc 0 out of a report whose own text said the
    configuration might not fit.
    """
    monkeypatch.setattr(pf, "host_platform", lambda: "win32")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["observed_peak_envelope_platform"] == "windows"
    # Zero for this configuration: the fraction is the LEGACY-RRTMG
    # lane's retained call-peak workspace, and this case is rte-rrtmgp.
    assert payload["envelope_pool_slack_fraction"] == 0.0
    assert payload["envelope_wddm_pool_slack_fraction"] == 0.0
    # The measured affine model, term for term.  This case runs the
    # rte-rrtmgp radiation lane, whose pool tracks the itemization at
    # 0.88-1.00x on every card instrumented, so it carries NO pool-slack
    # term -- that mechanism is the legacy engines' retained call-peak
    # workspace (task 206).  While the term was keyed to WDDM this same
    # configuration paid 20% of its estimate for it on Windows and
    # nothing for it on Linux, and neither figure described the run.
    assert payload["envelope_legacy_radiation"] is False
    assert payload["observed_peak_envelope_bytes"] == (
        payload["alloc_estimate_bytes"]
        + payload["non_pool_device_bytes"]
        + pf.ENVELOPE_UNMODELLED_BYTES
        + math.ceil(pf.ENVELOPE_PER_NEST_FRACTION
                    * (len(payload["domains"]) - 1)
                    * payload["alloc_estimate_bytes"]))
    # ...and strictly below what the retired multiplier would have said.
    assert payload["observed_peak_envelope_bytes"] < int(
        payload["footprint_projection_bytes"] * 1.75)
    # 100 GiB budget: envelope fits, no warning.
    assert payload["observed_peak_envelope_exceeds_budget"] is False
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    # This case is the rte-rrtmgp lane, so the slack term is absent from
    # the printed arithmetic entirely -- and the retired multiplicative
    # floor stays gone.
    assert "pool slack" not in out
    assert "WDDM floor" not in out
    assert "WARNING: observed peak envelope" not in out
    # The forecast is no longer the only phase this report prices, so the
    # historical line must say which phase it is, the preprocessing phase
    # must appear beside it, and one sentence must name the binding one.
    assert "FORECAST PEAK ENVELOPE" in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "INGEST (preprocessing, --source era5)" in out
    assert "BINDING PHASE:" in out
    assert "memory-binding phase" in out

    # A budget the ESTIMATE fits but the envelope exceeds: the estimate
    # gate still passes, and the warning names the accurate number.
    envelope_gib = payload["peak_envelope_bytes"] / GIB
    estimate_gib = payload["alloc_estimate_bytes"] / GIB
    tight = str(math.ceil(estimate_gib) + 1)
    assert float(tight) < envelope_gib
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight,
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    assert payload["observed_peak_envelope_exceeds_budget"] is True
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert "WARNING: observed peak envelope" in out
    assert "exceeds the WDDM budget" in out
    # The warned number is the LARGEST phase, and the report says which.
    assert "BINDING PHASE:" in out
    assert payload["binding_phase"] in ("forecast", "ingest")
    assert payload["peak_envelope_bytes"] >= payload[
        "observed_peak_envelope_bytes"]
    # Estimator-only mode (no budget, no GPU): nothing to compare, no
    # false alarm.  cupy is stubbed out exactly as the fails-closed test
    # does so this leg never queries a real device.
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = _run_check(["check", str(CONFIG_4DOM), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_bytes"] is None
    assert payload["observed_peak_envelope_exceeds_budget"] is None


def test_peak_envelope_factor_is_platform_conditional():
    """The retired multipliers stay recorded; the families stay platform.

    The factors are the HISTORICAL record (and `woof downscale`'s
    deliberately conservative child-fit bound); no gate reads them since
    the 3080 calibration replaced the WDDM multiplier with the measured
    affine slack term.  The family split itself remains, because the two
    platforms have different measured behaviour -- and it depends on the
    platform ONLY, never on card size (the #162 wizard/check split).
    """
    assert pf.PEAK_ENVELOPE_FACTORS == {"windows": 1.75, "linux": 1.45}
    assert pf.OBSERVED_PEAK_OVER_FOOTPRINT == 1.75

    for name in ("win32", "cygwin", "msys"):
        assert pf.envelope_platform(name) == "windows"
        assert pf.peak_envelope_factor(name) == 1.75
    # WSL and Linux containers report `linux` too, which is the point.
    for name in ("linux", "linux2"):
        assert pf.envelope_platform(name) == "linux"
        assert pf.peak_envelope_factor(name) == 1.45

    footprint = 11_310_000_000
    assert pf.observed_peak_envelope_bytes(
        footprint, platform="win32") == int(footprint * 1.75)
    assert pf.observed_peak_envelope_bytes(
        footprint, platform="linux") == int(footprint * 1.45)


def test_an_unmeasured_platform_takes_the_conservative_accounting():
    """v1.0.0 gave every non-Windows name the Linux (optimistic) numbers.

    Only two platforms have measurements: Windows/WDDM (with Cygwin and
    MSYS, the same driver under another shell) and Linux (which is also
    what WSL and Linux containers report).  Everything else -- Darwin,
    a BSD, a name that does not exist yet -- was silently priced with
    the envelope that omits 4.12 GiB of fixed constants, on no evidence
    at all.  Fail-open is the wrong direction here: the Linux numbers
    are three runs on two Linux cards, not a default.
    """

    for name in ("darwin", "freebsd13", "sunos5", "emscripten"):
        assert not pf.platform_is_measured(name)
        assert pf.envelope_platform(name) == "windows"
        assert pf.peak_envelope_factor(name) == 1.75
        assert pf.platform_projection_constants(name) == (
            pf.pool_retention_residual_bytes(), pf.PROBE_DEVICE_OVERHEAD_BYTES)
        # ...and the substitution is announced, naming the platform.
        note = pf.unknown_platform_note(name)
        assert note is not None and name in note
        assert "no VRAM measurements" in note

        # The small-card experiment is not extended to it: that tier is
        # an experiment about WDDM, and an unmeasured platform is not
        # the place to run a second experiment on top of the first.
        assert pf.envelope_platform(name, vram_gib=8.0) == "windows"

    for name in ("win32", "cygwin", "msys", "linux", "linux2"):
        assert pf.platform_is_measured(name)
        assert pf.unknown_platform_note(name) is None


def test_the_projection_constants_are_platform_conditional_too():
    """The 1.75 multiplier was only half of it.

    ``pool_retention_residual_bytes`` (2.73 GiB) and
    ``PROBE_DEVICE_OVERHEAD_BYTES`` (1.39 GiB) are grid-independent
    Windows-pool constants.  At the wizard's smallest layout they are
    4.12 GiB of a 5.38 GiB projection -- 77% -- so no smaller grid could
    ever fit a 12 GiB card, whose GPU then sat 66% idle.  None of the
    three instrumented Linux runs showed them.
    """
    windows = pf.platform_projection_constants("win32")
    assert windows == (pf.pool_retention_residual_bytes(),
                       pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert sum(windows) / GIB == pytest.approx(4.12, abs=0.02)
    assert pf.platform_projection_constants("linux") == (0, 0)

    # On Linux the projection is the itemized alloc estimate, and the
    # AFFINE envelope over it clears every instrumented run -- the three
    # 2026-07-30 pilots, each re-read with the non-pool term its own card
    # carries rather than the 5090's.
    pilots = ((7.20, 9.54, 128), (7.29, 8.99, 128), (3.51, 4.04, 46))
    for alloc_gib, measured_gib, sms in pilots:
        profile = pf.DeviceLocalMemoryProfile(
            name="pilot", multiprocessor_count=sms,
            max_threads_per_multiprocessor=1536)
        non_pool = pf.CUDA_CONTEXT_BYTES + profile.reservation_bytes(
            KF_AS_BUILT_FRAME.frame_bytes(49))
        envelope = pf.machine_peak_envelope_bytes(
            alloc_estimate_bytes=int(alloc_gib * GIB),
            non_pool_bytes=non_pool, family="linux")
        assert measured_gib * GIB < envelope, (alloc_gib, measured_gib)


@requires_grib1_bridge
@requires_4dom_inputs
def test_check_cli_prints_the_linux_envelope_factor_when_on_linux(
        capsys, monkeypatch):
    """`woof check` must say which platform factor it applied."""

    monkeypatch.setattr(pf.sys, "platform", "linux")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["observed_peak_envelope_platform"] == "linux"
    assert payload["envelope_wddm_pool_slack_fraction"] == 0.0
    # AFFINE on Linux: estimate + the itemized non-pool residency + the
    # measured unmodelled constant (+ a per-nest term).  Not a multiple
    # of the projection -- a multiple has no intercept, and a model with
    # no intercept changes the SIGN of its error with grid size.
    assert payload["observed_peak_envelope_bytes"] == (
        payload["alloc_estimate_bytes"]
        + payload["non_pool_device_bytes"]
        + pf.ENVELOPE_UNMODELLED_BYTES
        + math.ceil(pf.ENVELOPE_PER_NEST_FRACTION
                    * (len(payload["domains"]) - 1)
                    * payload["alloc_estimate_bytes"]))
    # And the projection itself dropped the two Windows-pool constants.
    assert payload["footprint_projection_bytes"] == payload[
        "alloc_estimate_bytes"]
    assert payload["reserve_components"]["retention_residual_bytes"] >= 0

    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FORECAST PEAK ENVELOPE (estimate" in out
    assert "affine, not a multiplier" in out
    assert "1.746x its footprint projection" not in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "BINDING PHASE:" in out


def test_check_cli_legacy_config_wraps(capsys):
    rc = _run_check(["check", str(CONFIG_D01), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert list(payload["domains"]) == ["d01"]
    # USTM adds 200,000 B to the previous d01 resident pin.
    # The 1024-descriptor health capacity adds 24,576 B and the ring-guard
    # saves 3,010,560 B to the pre-assembly pin, the OLR publication buffer
    # a further 200,000 B, and the EOS base-thickness correction plus the
    # coefficient drops 9,800,392 B (itemization-pin derivation). The
    # default-on CQ faces add 29,688,200 B, and the held CLDFRA buffer
    # 9,800,000 B (RULINGS 1). The strict health rim table (a677ad4e1) adds
    # 12,288 B.
    # The six 0-d SCM forcing scalars (24 B) are strict-only too (scm_forcing_fields).
    # 22aa4243869: check prices the forcing slots a chained `woof go` holds (one
    # streamed lateral slot for a one-domain, one-card config), not the whole
    # schedule: 8,899,840 B of boundary tables leave the d01 resident pin.
    # 04e65faa5ff (cap-solar): the SWDDIR/SWDDNI/SWDDIF split history
    # planes add 600,000 B.
    # paper/conus-suite-exact (2.9.11): the stored MF_VX_INV state plane msfv_inv,
    # 4*(ny+1)*nx = 201,000 B on this 250x200 grid.
    assert payload["domains"]["d01"]["resident_bytes"] == 1503966236


def test_check_cli_fails_closed_when_nothing_is_evaluable(capsys,
                                                          monkeypatch):
    """review F6 / shadow F5: estimator mode with no budget and no GPU
    verified NOTHING -- the exit code must say so (rc 2), never 0 via
    ``all([])``."""
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy fails
    rc = _run_check(["check", str(CONFIG_D01), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert all(v is None for v in payload["gates"].values())
    assert payload["budget_bytes"] is None


# ---------------------------------------------------------------------------
# Robust-5 failure paths (CPU, stubbed cupy -- no device touched)
# ---------------------------------------------------------------------------

class _StubPool:
    def __init__(self):
        self.freed = False

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def free_all_blocks(self):
        self.freed = True


def _stub_cupy(free_bytes, total_bytes=int(31.84 * GIB)):
    import types

    stub = types.ModuleType("cupy")

    class _OOM(Exception):
        pass

    stub.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            memGetInfo=lambda: (int(free_bytes), int(total_bytes)),
            deviceSynchronize=lambda: None),
        memory=types.SimpleNamespace(OutOfMemoryError=_OOM))
    pool = _StubPool()
    stub.get_default_memory_pool = lambda: pool
    return stub, pool


def test_headroom_abort_still_reports_and_exits_distinctly(capsys,
                                                           monkeypatch):
    """review F1 / shadow F5 fix: a headroom abort before measurement must
    still emit the structured report (estimate-side legs evaluated, abort
    reason recorded) with an exit code DISTINCT from a leg FAIL."""
    import sys
    stub, pool = _stub_cupy(free_bytes=2 * GIB)  # far short of the need
    monkeypatch.setitem(sys.modules, "cupy", stub)
    rc = _run_check(["check", str(CONFIG_D01), "--alloc", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3  # aborted-before-measurement, not a leg FAIL (1)
    assert payload["abort"]["error"] == "PreflightHeadroomError"
    assert payload["abort"]["phase"] == "domain d01 construction"
    assert payload["abort"]["free_bytes"] == 2 * GIB
    # Estimate-side legs evaluated from the abort's measured free; the
    # measured legs stay None and can never pass.
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert payload["gates"]["alloc_measured_le_estimate"] is None
    assert payload["gates"]["alloc_fits_wddm_budget"] is None
    assert payload["alloc_estimate_bytes"] > 0
    assert not pool.freed


def test_check_alloc_measures_on_the_recorded_sources_tables(
        tmp_path, capsys, monkeypatch):
    """``woof check --alloc`` asks its measurement for the source's tables.

    The recorded source publishes five hydrometeor masses, so the
    measurement is asked to build and price the tables that carry them
    (the measurement itself needs a card and is stood in for, aborting
    before it measures), and the estimate side the report then prints
    prices the same tables: its observed envelope is the peak envelope
    printed beside it.  Red before the A92 follow-up: the measurement was
    asked for water vapour alone and the abort's estimate priced none.
    """
    import sys

    from woof.boundary_fields import source_boundary_species
    from test_runplan_tiles import moist_specified_config

    stub, _pool = _stub_cupy(free_bytes=64 * GIB, total_bytes=64 * GIB)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    asked = []

    def measurement(exp, *, boundary_species=(), **kwargs):
        asked.append(tuple(boundary_species))
        raise pf.PreflightHeadroomError(
            "stand-in", phase="stand-in", free_bytes=64 * GIB,
            total_bytes=64 * GIB, reserve_bytes=0, remaining_bytes=0)

    monkeypatch.setattr(pf, "run_alloc_preflight", measurement)
    monkeypatch.setattr(pf, "_warn_unstaged_physics_tables", lambda *_: None)
    config = moist_specified_config(tmp_path)
    rc = _run_check(["check", str(config), "--alloc", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3
    species = source_boundary_species("hrrr")
    assert asked == [species] and species
    exp = pf._load_experiment_any(config)
    interval, count = pf.config_forcing_schedule(config, exp)
    lbc = {name: pf.estimate_experiment(
        exp, forcing_interval_seconds=interval, forcing_intervals=count,
        boundary_species=tables).domains[0].category_bytes("lbc")
        for name, tables in (("tables", species), ("vapour", ()))}
    assert lbc["tables"] > lbc["vapour"]
    assert payload["domains"]["d01"]["by_category"]["lbc"] == lbc["tables"]
    assert (payload["observed_peak_envelope_bytes"]
            == payload["peak_envelope_bytes"])


def test_headroom_error_carries_structured_fields():
    stub, _ = _stub_cupy(free_bytes=1 * GIB)
    reserve = pf.ReservePolicy.flat(GIB // 2)
    with pytest.raises(pf.PreflightHeadroomError) as err:
        pf._require_headroom(stub, 2 * GIB, reserve, "unit fixture")
    exc = err.value
    assert exc.phase == "unit fixture"
    assert exc.free_bytes == GIB
    assert exc.reserve_bytes == GIB // 2
    assert exc.remaining_bytes == 2 * GIB


def test_alloc_oom_terminates_without_freeing(exp1, monkeypatch):
    """Robust-5 OOM policy: diagnostics + termination, NEVER
    ``free_all_blocks()``-and-continue.  Stubbed cupy + a DomainState
    that raises the stub's OutOfMemoryError -- no device involved."""
    import sys

    import woof.core.state as state_mod

    stub, pool = _stub_cupy(free_bytes=64 * GIB, total_bytes=64 * GIB)
    monkeypatch.setitem(sys.modules, "cupy", stub)

    class _Boom:
        def __init__(self, cfg):
            raise stub.cuda.memory.OutOfMemoryError(
                "stub allocation failure")

    monkeypatch.setattr(state_mod, "DomainState", _Boom)
    with pytest.raises(pf.PreflightAllocError) as err:
        pf.run_alloc_preflight(exp1)
    assert err.value.phase == "domain d01 construction"
    assert "Terminating" in str(err.value)
    assert "column_chunk" in str(err.value)  # the first lever is named
    assert not pool.freed  # never free_all_blocks-and-continue


@pytest.mark.parametrize("retained_intervals", [None, 8])
def test_alloc_preflight_materializes_and_injects_shared_workspaces(
        monkeypatch, retained_intervals):
    """CPU/stubbed-CuPy proof that --alloc builds both shared workspaces."""
    import sys
    import types

    import woof.core.state as state_mod
    import woof.ingest.lateral_bc as lbc_mod

    # This is an allocation geometry control; no forcing values are read.
    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data", None)
    raw.pop("fetch", None)
    exp4 = build_experiment(raw, source=str(CONFIG_4DOM))

    # Keep the real four-domain geometry/registry but turn off physics so the
    # stub run has no unrelated CuPy allocation surface.
    domains = tuple(dataclasses.replace(
        dc, run=dataclasses.replace(
            dc.run, mp_physics=0, sf_sfclay_physics=0,
            sf_surface_physics=0, bl_pbl_physics=0,
            ra_physics=0, cu_physics=0)) for dc in exp4.domains)
    cpu_exp = dataclasses.replace(exp4, domains=domains)
    estimate = pf.estimate_experiment(cpu_exp, forcing_intervals=retained_intervals)
    assert estimate.uses_shared_scratch_arena

    stub, pool = _stub_cupy(free_bytes=64 * GIB, total_bytes=64 * GIB)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    scratch_sentinel = types.SimpleNamespace(
        nbytes=estimate.scratch_arena_bytes)
    dycore_sentinel = types.SimpleNamespace(
        nbytes=estimate.dycore_state_workspace_bytes)
    scratch_built = []
    dycore_built = []
    injected = []

    def build_scratch(domains_arg):
        scratch_built.append(tuple(domains_arg))
        return scratch_sentinel

    def build_dycore(domains_arg):
        dycore_built.append(tuple(domains_arg))
        return dycore_sentinel

    class _FakeState:
        def __init__(self, cfg, scratch_arena=None,
                     dycore_state_workspace=None):
            injected.append((scratch_arena, dycore_state_workspace))

        def scratch(self, shape, slot, dtype=None):
            return types.SimpleNamespace(shape=tuple(shape), nbytes=0)

    monkeypatch.setattr(
        state_mod, "build_shared_scratch_arena", build_scratch)
    monkeypatch.setattr(
        state_mod, "build_shared_dycore_state_workspace", build_dycore)
    monkeypatch.setattr(state_mod, "DomainState", _FakeState)
    attached = []
    monkeypatch.setattr(lbc_mod, "attach_lateral_boundaries",
                        lambda state, boundaries: attached.append(boundaries))

    report = pf.run_alloc_preflight(
        cpu_exp, reserve=pf.ReservePolicy.flat(0),
        forcing_intervals=retained_intervals)
    expected_intervals = (retained_intervals if retained_intervals is not None
                          else pf.lbc_intervals(cpu_exp.run_seconds, 21600))
    assert [len(boundaries.intervals) for boundaries in attached] == [expected_intervals]
    assert scratch_built == [cpu_exp.domains]
    assert dycore_built == [cpu_exp.domains]
    assert injected == [
        (scratch_sentinel, dycore_sentinel)] * len(cpu_exp.domains)
    assert report.estimate.scratch_arena_bytes == scratch_sentinel.nbytes
    assert (report.estimate.dycore_state_workspace_bytes
            == dycore_sentinel.nbytes)
    assert pool.freed


# ---------------------------------------------------------------------------
# N0 allocation runs (controller-run GPU; enforced gates)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_alloc_preflight_d01_measured_le_estimate(exp1):
    report = pf.run_alloc_preflight(exp1)
    assert report.pool_used_peak_bytes > 0
    # The enforced estimator contract: measured > estimate is a FAILING
    # GATE, not a recalibration note.
    assert report.gates["alloc_measured_le_estimate"] is True
    # Zero steps, freed at exit: the pool must actually release.
    assert report.free_after_release_bytes > report.free_at_peak_bytes


@pytest.mark.gpu
@requires_4dom_inputs
def test_alloc_preflight_n0_four_domain():
    """N0 (gates all wave-2 ARC-B merges): the full manifest-driven
    allocation.  The budget legs are recorded for the controller's ledger
    adjudication; the estimator-correctness leg is asserted here.

    The probe is DOCUMENTED as a fresh-process tool, so the test runs it
    exactly as shipped -- a subprocess of the CLI -- and asserts on its
    JSON.  Before spawning, the PARENT must surrender its own device
    residue: earlier gpu tests leave DomainState<->PhysicsDriver reference
    CYCLES whose arrays survive free_all_blocks until a cycle collection
    (diagnosed 2026-07-16: 1.48 GiB retained pre-gc, 22 MB post-gc)."""
    import gc
    import json
    import subprocess
    import sys
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    proc = subprocess.run(
        [sys.executable, "-m", "woof.cli", "check",
         str(CONFIG_4DOM),
         "--alloc", "--reserve-gib", "1.888", "--json"],
        capture_output=True, text=True, cwd=ROOT, timeout=1200)
    # exit 1 = a budget leg failed (recorded for controller adjudication);
    # exit 2 = nothing evaluable is a genuine failure.  exit 3 = headroom
    # abort: under the full suite, session-scoped gpu fixtures hold device
    # memory no gc can surrender, so the fresh-process probe legitimately
    # cannot fit -- assert the abort JSON is well-formed, then skip (the
    # binding N0 evidence is the controller's standalone probe).  exit 4 =
    # every measured leg passed but the observed peak envelope sits above
    # the budget, which on a four-domain config is the expected verdict on
    # most cards; the measured legs below are what this test is about.
    assert proc.returncode in (0, 1, 3, 4), proc.stderr[-2000:]
    report = json.loads(proc.stdout)
    if proc.returncode == 3:
        assert report.get("abort"), "exit 3 must carry a structured abort"
        pytest.skip("N0 probe headroom-aborted under suite residency: "
                    + str(report["abort"])[:200])
    print("N0", report["gates"], "alloc", report["alloc"])
    assert report["gates"]["alloc_measured_le_estimate"] is True
    assert set(report["gates"]) == set(pf.N0_GATE_METRICS)
    assert report["alloc"]["pool_used_peak_bytes"] > 0


def test_legacy_rrtmg_variant_prices_the_call_peak_envelope():
    """Variant-aware preflight (assembly item): under the legacy 4/4
    variant the RRTMGP per-domain column shapes disappear (the legacy
    transients are the shared call-peak envelope instead), and the
    envelope itself is positive, chunk-bounded, and grows with ncol up
    to the chunk bound."""
    import dataclasses
    import pathlib

    from woof.core.preflight import rrtmgp_column_shapes
    from woof.core.rrtmg_legacy import legacy_radiation_vram_bytes
    from woof.config import load_config

    legacy_cfg = load_config(
        pathlib.Path(__file__).resolve().parents[1]
        / "configs" / "real74_d01_rrtmg_legacy.toml")
    modern_cfg = dataclasses.replace(
        legacy_cfg, ra_rrtmg_variant="rte-rrtmgp",
        wrf_rrtmg_compatibility="none")
    assert rrtmgp_column_shapes(legacy_cfg, 10000.0) == {}
    assert rrtmgp_column_shapes(modern_cfg, 10000.0) != {}

    small = legacy_radiation_vram_bytes(
        ncol=1000, nz=legacy_cfg.nz, p_top=10000.0, column_chunk=None)
    big = legacy_radiation_vram_bytes(
        ncol=legacy_cfg.ny * legacy_cfg.nx, nz=legacy_cfg.nz,
        p_top=10000.0, column_chunk=None)
    assert 0 < small <= big
    # beyond the engine-default chunk bound the envelope must flatten
    # (chunking caps the transient, by construction of the engines)
    bigger = legacy_radiation_vram_bytes(
        ncol=4 * legacy_cfg.ny * legacy_cfg.nx, nz=legacy_cfg.nz,
        p_top=10000.0, column_chunk=None)
    assert bigger <= big * 2, (bigger, big)


def test_legacy_rrtmg_variant_prices_the_lw_chain_local_frame():
    """Variant-aware kernel-module selection (step-1 audit, major 1).

    ``ra_physics = 4`` is two implementations behind one selector value.
    Under ``ra_rrtmg_variant = 'rrtmg_legacy'`` the modern ``rrtmgp_*``
    kernels are never launched, and the legacy LW chain's widest kernel
    -- ``rlw_rtrn_march``, driver-measured 2,048 B/thread on sm_120
    (tests/test_rrtmg_lw_cuda.py ``LOCAL_FRAME_BOUNDS`` measurement
    record; docs/rrtmg_legacy_integration.md section 6) -- must be
    priced into the local-memory backing store instead of dropping out
    of the selector model fail-open.  The chained fragments themselves
    stay refused-without-measurement: they are covered by the measured
    composite translation units, never priced standalone.
    """
    legacy_cfg = load_config(
        ROOT / "configs" / "real74_d01_rrtmg_legacy.toml")
    start = datetime(1974, 4, 3, 12)
    exp = experiment_from_run_config(legacy_cfg, start)
    modules = pf.physics_kernel_modules(exp)
    # The legacy variant never launches the RTE+RRTMGP kernels ...
    assert not modules & {
        "rrtmgp_cloud", "rrtmgp_gas", "rrtmgp_mcica", "rrtmgp_rte"}
    # ... and does launch the device McICA twin plus the two chained TUs.
    assert {"rrtmg_mcica_wrf", "rrtmg_lw_legacy_chain",
            "rrtmg_sw_legacy"} <= modules

    frames = pf.kernel_local_frame_bytes(exp)
    assert frames["rrtmg_lw_legacy_chain"] == 2048
    assert frames["rrtmg_sw_legacy"] == 0  # post-spcvmc-restructure SW TU

    # Radiation alone (every masking selector stripped): the reservation
    # is exactly the LW chain's frame.  (2048 - 1024) x 1536 x 170 =
    # 267,386,880 B (~255 MiB) -- the incremental backing store the
    # fail-open selector omitted; the full ~510 MiB machine-wide store
    # at 2,048 B/thread includes the context's 1,024 B default-stack
    # half, which CUDA_CONTEXT_BYTES already carries.
    bare_run = dataclasses.replace(
        legacy_cfg, mp_physics=0, cu_physics=0, bl_pbl_physics=0,
        sf_sfclay_physics=0, sf_surface_physics=0)
    bare = experiment_from_run_config(bare_run, start)
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert (pf.kernel_local_memory_bytes(bare)
            == profile.reservation_bytes(2048) == 267386880)
    # The modern twin of the same bare selector set prices rrtmgp_rte's
    # module bound: 3,600 B on every compile platform read, so
    # (3600 - 1024) x 1536 x 170 = 672,645,120 B.  The ceiling carried
    # 5,152 B (a reading of the pre-optimisation source) until the sm_120
    # recordings were re-read, 405,258,240 B more for this same selection.
    modern_run = dataclasses.replace(
        bare_run, ra_rrtmg_variant="rte-rrtmgp",
        wrf_rrtmg_compatibility="none")
    assert (pf.kernel_local_memory_bytes(
                experiment_from_run_config(modern_run, start))
            == profile.reservation_bytes(3600) == 672645120)

    # Composite bookkeeping: the measured TU frames cover exactly the
    # fragments that cannot compile standalone, so selecting a fragment
    # without a measurement still refuses (fail-closed), while a legacy 4/4
    # request resolves to the composites and never trips it.
    #
    # "p3" joined this set with the P3 CUDA port (2026-08-29): p3.cu borrows
    # the tree's one glibc r_pow/r_exp/r_log from noahmp_leaves.cu instead
    # of carrying a second copy, so it is a fragment on exactly the
    # legacy-RRTMG footing and its composite unit is
    # woof/core/p3_device.p3_source().  The TABLE moved and this list is
    # the gate's own statement of what the table should hold, so it moves
    # with it -- the gate is not being loosened, and the two assertions
    # below (covered is a subset of the unmeasured set, and the legacy
    # module set never intersects it) still hold unchanged.
    #
    # "urban_bep_bem" joined 2026-09-30 on the same footing: BEP+BEM's
    # column kernel needs urban_bem.cuh and the glibc headers that
    # woof/core/urban_bem.py composes, and its unit is urban_bem_composed.
    # Two more joined on 2026-09-30, again with the table: the legacy
    # RRTMG speed lane's rrtmg_lw_chain_coalesced (668fa4c56) and
    # rrtmg_lw_zbatched (bfc177208) compile only inside the longwave chain
    # unit.  Each carries its sm_120 reading beside its row in preflight.
    # The megakernel lane's phy_column (257a5e729) sat here under its
    # surface_chain unit until A147 took the fragment and the unit out of
    # the tree together.
    # "noah_mosaic" joined the same way (2026-09-30): the Noah mosaic column
    # launches only as woof/core/noah_mosaic.py's own -fmad=false unit,
    # priced as noah_mosaic_unit.
    # "thompson_fork" joined 2026-10-07 the same way: the fork Thompson
    # column launches only as woof/core/thompson_fork.py's own unit,
    # priced as thompson_fork_unit.
    covered = frozenset().union(
        *(tu.covers for tu in pf.CHAINED_TRANSLATION_UNIT_FRAMES.values()))
    assert covered == {
        "rrtmg_sw", "rrtmg_lw_chain", "rrtmg_lw_taugb02_10_11_12",
        "rrtmg_lw_taugb03_05", "rrtmg_lw_taugb06_09",
        "rrtmg_lw_taugb13_16", "p3", "urban_bep_bem",
        "terrain_drag",
        "rrtmg_lw_chain_coalesced", "rrtmg_lw_zbatched", "noah_mosaic",
        "thompson_fork"}
    assert covered <= pf.UNMEASURED_KERNEL_MODULES
    assert not modules & pf.UNMEASURED_KERNEL_MODULES

    # An unknown variant is refused, not priced from either row.
    unknown_run = dataclasses.replace(
        bare_run, ra_rrtmg_variant="rrtmg_v3")
    with pytest.raises(ValueError, match="ra_rrtmg_variant"):
        pf.physics_kernel_modules(
            experiment_from_run_config(unknown_run, start))


# ---------------------------------------------------------------------------
# The non-pool residency the CuPy pool never reports (measured 2026-07-26)
# ---------------------------------------------------------------------------
#
# Every number asserted below was MEASURED on the run host, either by
# bracketing a kernel's first launch with ``cudaMemGetInfo`` inside a real
# forecast or by sampling NVML device-wide at 1 s for a whole run.  Nothing
# here is derived from another estimate.

#: ``kf_column``'s first launch, measured twice in two separate traced
#: three-domain forecasts: 5,738.0 MiB of device memory that the pool did
#: not allocate and never gets back.
MEASURED_KF_RESERVATION_MIB = 5738.0

#: The same instrument on a synthetic kernel at five local-frame widths,
#: one block of 32 threads each: cumulative device growth over a bare
#: context, in MiB, keyed by per-thread local frame in bytes.
MEASURED_SYNTHETIC_RESERVATION_MIB = {
    1024: 2, 4096: 766, 8192: 1786, 16384: 3827, 24064: 5742}

CONFIG_4DOM_MYNN_KF = ROOT / "configs" / "real74_4dom_mynn_norad.toml"
CONFIG_4DOM_MYNN_NOCU = (
    ROOT / "configs" / "real74_4dom_mynn_norad_nocu.toml")

#: The per-thread frame ``kf_column`` compiled to BEFORE its column
#: workspace landed (2026-08-21): 188 B per level, rounded up to the local
#: frame's 8-byte granularity, at whatever ``KF_KMAX`` the launcher chose.
#:
#: It lives here and not in ``preflight`` because ``preflight`` describes
#: the binary that runs NOW, and this one no longer exists: 52 of
#: ``kf_column``'s 54 column arrays moved off the stack, the frame stopped
#: following ``nz``, and the row is a flat 512 B.  Every historical
#: measurement in this file -- the two 2026-07-26 four-domain runs, the
#: 2026-07-30 pilots, the RTX 4080 fleet table -- was taken by a binary
#: whose widest frame was this, so it is what they are priced against.
#: Pricing them at 512 B would claim the old runs carried a reservation
#: they demonstrably did not.
KF_AS_BUILT_FRAME = pf.LevelSpecializedFrame("kf", "KF_KMAX", 128, 188)


def test_the_local_memory_law_reproduces_every_measurement():
    """One allocation, sized by the widest launched frame, over the whole
    resident-thread capacity, minus the default stack the context already
    carries.  Six independent measurements, 1% tolerance."""
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert profile.resident_thread_capacity == 1536 * 170
    for local_bytes, measured_mib in (
            MEASURED_SYNTHETIC_RESERVATION_MIB.items()):
        predicted_mib = profile.reservation_bytes(local_bytes) / 1024 ** 2
        assert abs(predicted_mib - measured_mib) <= max(
            4.0, 0.01 * measured_mib), (
                f"{local_bytes} B/thread: model {predicted_mib:.1f} MiB "
                f"vs measured {measured_mib} MiB")
    # MEASURED_KF_RESERVATION_MIB was read off `kf_column`'s first launch
    # at KF_KMAX = 128, so the law is checked against THAT frame and not
    # against today's row -- today's row is 512 B and reserves nothing,
    # which is the point of the workspace, not a counter-example to the law.
    kf_predicted = profile.reservation_bytes(
        KF_AS_BUILT_FRAME.frame_bytes(128)) / 1024 ** 2
    assert abs(kf_predicted - MEASURED_KF_RESERVATION_MIB) <= 1.0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["kf"] == 512
    assert profile.reservation_bytes(512) == 0


def test_a_frame_inside_the_default_stack_reserves_nothing():
    """The 1024 B/thread baseline store belongs to the context, not to a
    kernel; MYNN's own kernels declare no static local frame at all."""
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert profile.reservation_bytes(1024) == 0
    assert profile.reservation_bytes(0) == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["mynn_pbl"] == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["mynn_surface"] == 0


@requires_reference_bundle
def test_the_reservation_does_not_grow_with_domain_count():
    """The fingerprint that identified this term: it is a maximum over
    launched kernels, so three domains and four reserve the same bytes.

    The VALUE moved twice.  On 2026-07-26 `kf.cu`'s KF_KMAX started
    compiling to the configuration's nz: 24,064 B/thread at the
    unspecialized 128, 9,216 B at this case's 49.  On 2026-08-21 the column
    workspace took `kf` out of the running entirely (512 B, under the
    default stack), and `morrison`'s 5,120 B became the widest frame this
    configuration launches.  The invariant this test exists for --
    independence from domain count -- is unchanged through both.
    """
    exp4 = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    exp3 = dataclasses.replace(exp4, domains=exp4.domains[:3])
    assert {dc.run.nz for dc in exp4.domains} == {49}
    assert (pf.kernel_local_memory_bytes(exp3)
            == pf.kernel_local_memory_bytes(exp4)
            == pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(5120))
    # ... and it is no longer `kf` that sets it.
    assert pf.kernel_local_frame_bytes(exp4)["kf"] == 512


@requires_reference_bundle
def test_the_kf_reservation_stopped_growing_with_the_level_count():
    """`kf` left the level-specialized table, and this is what that means.

    Its frame used to be 188 B per level -- 9,216 B at 49, 18,424 B at 98 --
    which is why preflight priced it per domain.  The column workspace
    (2026-08-21) took 52 of its 54 column arrays off the stack, so the frame
    is now a flat 512 B at every level count and the module is compiled ONCE
    instead of once per nz.  A level-linear model would now UNDER-price
    nothing and OVER-price everything, so it is retired rather than refitted;
    `refl` still carries one and still moves.
    """
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    deeper = dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=dataclasses.replace(dc.run, nz=98))
        for dc in exp.domains))
    assert "kf" not in pf.LEVEL_SPECIALIZED_KERNEL_FRAMES
    assert pf.kernel_local_frame_bytes(exp)["kf"] == 512
    assert pf.kernel_local_frame_bytes(deeper)["kf"] == 512
    # The model that IS still level-specialized still grows with nz.
    refl = pf.LEVEL_SPECIALIZED_KERNEL_FRAMES["refl"]
    assert refl.frame_bytes(98) > refl.frame_bytes(49)
    # The ceiling is still a REFUSAL and not a silent truncation.  It moved
    # out of the frame model with the row, to the vertical contract that
    # every door already runs -- and that is where it names KF by name.
    from woof.physics_compat import (
        PhysicsVerticalPreflightError,
        validate_resolved_physics_vertical_levels)

    over = dataclasses.replace(exp.domains[0].run, nz=129)
    with pytest.raises(PhysicsVerticalPreflightError,
                       match=r"Kain-Fritsch cumulus requires 8 <= nz <= 128"):
        validate_resolved_physics_vertical_levels(over)


def test_the_level_specialized_frame_model_agrees_with_every_measurement():
    """Driver measurements on the RTX 5090 (2026-07-26), three level counts
    per module.  Each is `align8(bytes_per_level * n)`.

    `kf`'s three rows -- 24,064 / 9,216 / 5,640 -- used to sit here and are
    gone with the table entry: its column arrays moved into a global
    workspace on 2026-08-21 and its frame stopped following `nz`.  They
    survive as :data:`KF_AS_BUILT_FRAME`, which is what the historical
    measurements in this file are priced against.
    """
    assert KF_AS_BUILT_FRAME.frame_bytes(128) == 24064
    assert KF_AS_BUILT_FRAME.frame_bytes(49) == 9216
    assert KF_AS_BUILT_FRAME.frame_bytes(30) == 5640
    measured = {
        ("refl", 256): 18432, ("refl", 49): 3528, ("refl", 30): 2160,
    }
    for (module, levels), frame in measured.items():
        spec = pf.LEVEL_SPECIALIZED_KERNEL_FRAMES[module]
        assert spec.frame_bytes(levels) == frame, (module, levels)
    # The unspecialized row of each is exactly the driver-measured module
    # maximum, so the two tables cannot drift apart unnoticed.
    for module, spec in pf.LEVEL_SPECIALIZED_KERNEL_FRAMES.items():
        assert (spec.frame_bytes(spec.unspecialized_levels)
                == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[module])


def test_the_run_door_prices_wdm6_instead_of_refusing_mp_16():
    """The whole of mp=16's reachability, on CPU.

    ``woof.core.model.build_experiment``, ``domain_wizard``, ``runplan``
    and the prepared domain-tree forecast all call ``estimate_experiment``
    unconditionally, and ``estimate_experiment`` prices the local-memory
    reservation from ``kernel_local_frame_bytes``.  A ``wdm6`` row missing
    from ``KERNEL_MAX_LOCAL_SIZE_BYTES`` therefore did not degrade mp=16 --
    it made every one of those doors raise, so the ported scheme could not
    be run at all.  The refusal reproduced without a device, and so does
    the fix.
    """
    start = datetime(1974, 4, 3, 12)
    cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=16)
    exp = experiment_from_run_config(cfg, start)

    modules = pf.physics_kernel_modules(exp)
    assert "wdm6" in modules and "wsm6" not in modules
    frames = pf.kernel_local_frame_bytes(exp)
    # nz = 4 compiles the 64 tier, which is the recorded row.
    assert frames["wdm6"] == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6"] == 9264
    assert pf.kernel_local_memory_bytes(exp) > 0
    estimate = pf.estimate_experiment(exp)
    assert estimate.alloc_estimate_bytes > 0


def test_wdm6_is_priced_at_the_tier_its_launcher_compiles_not_at_nz():
    """The two rungs, and the two ways an nz-linear model would be wrong.

    ``woof/core/wdm6.py`` compiles ``WDM6_KMAX`` at 64 or 80, never at
    ``nz``, so a 49-level WDM6 run launches the 64-tier kernel and holds
    its 9,264 B frame.  Pricing 144 x 49 = 7,056 B there would under-price
    the reservation by 2,208 B/thread -- about 550 MiB of device memory the pool
    never reports -- which is the direction the header at the top of
    preflight.py says put a run 1,630 MiB over.
    """
    start = datetime(1974, 4, 3, 12)

    def frame(nz, mp_physics=16):
        cfg = RunConfig(**{**_TINY, "nz": nz}, moist=True, moist_cq=True,
                        mp_physics=mp_physics)
        return pf.kernel_local_frame_bytes(
            experiment_from_run_config(cfg, start)).get("wdm6")

    assert frame(4) == frame(49) == frame(64) == 9264
    assert frame(65) == frame(80) == 11568
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert (profile.reservation_bytes(9264) - profile.reservation_bytes(7056)
            ) > 500 * 1024 ** 2
    # Deeper than the deepest compiled tier is a refusal, not a guess.
    with pytest.raises(ValueError, match="WDM6 requires 2 <= nz <= 80"):
        frame(96)
    # ... and a scheme that does not launch WDM6 never reaches that bound.
    # Morrison, not WSM6: WSM6 became the SECOND conditionally tiered module
    # at 1.9 and carries its own 2 <= nz <= 80 ladder, so mp_physics=6 at
    # nz = 96 is refused on WSM6's own bound before this assertion can say
    # anything about WDM6.  The control needs a scheme that launches neither,
    # which is what the WSM6 half of this pair uses in the mirror direction.
    assert frame(96, mp_physics=10) is None


def test_the_tiered_frame_model_agrees_with_the_wdm6_measurements():
    """Both rungs, against the driver measurements they were taken from.

    ``tests/test_kernel_local_bounds.py`` re-reads them off the card; this
    is the CPU half, so a model edit fails somewhere even on a host with no
    device.
    """
    from woof.core.wdm6_constants import (WDM6_KERNEL_LEVEL_TIERS,
                                           wdm6_level_tier)

    assert pf.WDM6_TIER_FRAME.define == "WDM6_KMAX"
    assert WDM6_KERNEL_LEVEL_TIERS == (64, 80)
    for tier, measured in ((64, 9264), (80, 11568)):
        assert pf.WDM6_TIER_FRAME.frame_bytes(tier) == measured, tier
    assert (pf.WDM6_TIER_FRAME.frame_bytes(pf.WDM6_TIER_FRAME.shipped_tier)
            == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6"])
    assert wdm6_level_tier(64) == 64 and wdm6_level_tier(65) == 80


def test_the_rqi_budget_shapes_materialization_and_physics_name_one_set():
    """The agreement the ``--alloc`` comment asks for, enforced.

    ``physics_array_shapes`` (what the estimate prices),
    ``_materialize_physics`` (what the measurement constructs) and
    ``physics._pbl_optional_tendency_components`` (what the run allocates)
    have to admit the same schemes or ``--alloc`` stops covering true
    runtime residency.  They were three literal tuples and mp=16 moved only
    two of them, so the mp=16 + PBL run allocated a ``pbl_tendencies.rqi``
    the measurement never materialized.  One constant now, read by all
    three, and the reachable consequence is asserted below rather than the
    identity alone.
    """
    from woof.core.physics import (PBL_RQI_MICROPHYSICS,
                                    _pbl_optional_tendency_components)

    source = (ROOT / "woof/core/preflight.py").read_text(encoding="utf-8")
    assert source.count("cfg.mp_physics in PBL_RQI_MICROPHYSICS") == 2, (
        "both preflight rqi gates must read the named set, or they will "
        "drift apart again")
    for mp_physics in range(0, 60):
        cfg = RunConfig(**_TINY, moist=mp_physics != 0, moist_cq=True,
                        mp_physics=mp_physics, bl_pbl_physics=1,
                        sf_sfclay_physics=1)
        expects_rqi = mp_physics in PBL_RQI_MICROPHYSICS
        assert (_pbl_optional_tendency_components(cfg) == ("rqi",)
                ) is expects_rqi, mp_physics
        if mp_physics not in (0, 1, 6, 8, 10, 16, 18, 28):
            continue          # not an admitted selector; nothing to price
        shapes = pf.physics_array_shapes(cfg)
        assert ("pbl_tendencies/rqi" in shapes) is expects_rqi, mp_physics


def test_the_alloc_counter_advance_records_why_p3_is_out():
    """mp=50's absence from the counter advance is a DECISION, and checkable.

    ``_materialize_physics`` advances ``microphysics_updates`` for the
    selectors that reach a counter-gated allocation, so ``--alloc``
    measures the run's steady state rather than its construction state.
    The set was a bare ``(6, 10)`` literal with no statement of what it
    decides, so mp=50's absence read as an omission.  It is not one:
    ``ALLOC_COUNTER_INERT_MICROPHYSICS[50]`` records the reason, and every
    clause of that reason is asserted below.  The day the legacy adapter
    grows a first-call gate, or the RTE+RRTMGP p3 arm starts reading the
    counter, this fails instead of the measurement quietly understating a
    P3 run's radiation call.  (The tripwire's other arm -- "the day P3
    gains an RTE+RRTMGP cloud-optics row" -- fired at 2.6.1: the row
    landed, and mp=50 stays INERT because the p3 adapter arm copies its
    two radii unconditionally; P3 seeds valid radii at construction and
    has no first-call phase for a counter gate to express.)
    """
    from woof.config import validate_run_config
    from woof.core.physics_inventory import microphysics_scratch_slots
    from woof.core.rrtmgp import cloud_optics_scheme

    assert 50 not in pf.ALLOC_COUNTER_ADVANCED_MICROPHYSICS
    assert pf.ALLOC_COUNTER_INERT_MICROPHYSICS[50].strip()
    assert not (set(pf.ALLOC_COUNTER_ADVANCED_MICROPHYSICS)
                & set(pf.ALLOC_COUNTER_INERT_MICROPHYSICS))
    source = (ROOT / "woof/core/preflight.py").read_text(encoding="utf-8")
    assert source.count(
        "cfg.mp_physics in ALLOC_COUNTER_ADVANCED_MICROPHYSICS") == 1, (
        "the counter advance must read the named set, or the reason above "
        "it stops describing the code")

    p3 = dict(_TINY, moist=True, moist_cq=True, mp_physics=50)
    # Clause 1.  A P3 run DOES reach the RRTMGP adapter now (the
    # cloud-optics coupling landed at 2.6.1), and the p3 arm copies its
    # two radii UNCONDITIONALLY -- the counter's one allocation gate
    # stays inside the morrison arm, so advancing the counter for mp=50
    # would still move zero measured bytes.
    assert cloud_optics_scheme(50) == "p3"
    rte_p3 = validate_run_config(RunConfig(
        **p3, ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant="rte-rrtmgp"))
    rte_shapes = pf.rrtmgp_column_shapes(rte_p3, 10000.0, column_chunk=4)
    assert {"columns/effc", "columns/effi"} <= set(rte_shapes)
    assert "columns/effs" not in rte_shapes and \
        "columns/effr" not in rte_shapes
    # The pairings that price no RTE+RRTMGP columns still price none:
    # legacy RRTMG 4/4 and Dudhia 0/1 never construct the adapter.
    for lw, sw, variant in ((4, 4, "rrtmg_legacy"), (0, 1, "rte-rrtmgp")):
        cfg = validate_run_config(RunConfig(
            **p3, ra_lw_physics=lw, ra_sw_physics=sw,
            ra_rrtmg_variant=variant))
        assert pf.rrtmgp_column_shapes(cfg, 10000.0, column_chunk=4) == {}
    # The p3 adapter arm reads the counter nowhere: the one
    # microphysics_updates read in the RTE+RRTMGP adapter lives in the
    # morrison arm's first-call gate.
    rrtmgp_source = (ROOT / "woof/core/rrtmgp.py").read_text(
        encoding="utf-8")
    assert rrtmgp_source.count("microphysics_updates") == 1
    # ... and the set is not vacuous: mp=10 really does price the four
    # Morrison radii packs that rrtmgp.py:2535 withholds until the counter
    # has accepted one update.
    morrison = validate_run_config(RunConfig(
        **dict(_TINY, moist=True, moist_cq=True, mp_physics=10),
        ra_lw_physics=4, ra_sw_physics=4))
    assert {"columns/effc", "columns/effr", "columns/effi",
            "columns/effs"} <= set(
                pf.rrtmgp_column_shapes(morrison, 10000.0, column_chunk=4))

    # Clause 2.  The legacy RRTMG adapter is P3's only 4/4 pairing and it
    # has no counter read for anything to be gated on.
    assert "microphysics_updates" not in (
        ROOT / "woof/core/rrtmg_legacy.py").read_text(encoding="utf-8")

    # Clause 4.  P3's persistent set is already whole when
    # _materialize_physics runs: every slot it needs is in the registry
    # run_alloc_preflight prewarms, and the accumulator row is the FIVE
    # of CASE (P3_1CATEGORY), with no graupel.
    legacy_p3 = validate_run_config(RunConfig(
        **p3, ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant="rrtmg_legacy"))
    registry = pf.scratch_slot_registry(legacy_p3, n_lbc_intervals=0)
    slots = {slot for _component, slot in microphysics_scratch_slots(50)}
    assert len(slots) == 5 and "mp_graupelnc" not in slots
    assert slots <= set(registry)
    assert any(slot.startswith("p3_") for slot in registry)


def test_wsm6_is_priced_at_the_tier_its_launcher_compiles_not_at_a_flat_row():
    """The two rungs, and the way the flat row was wrong above 64 levels.

    ``woof/core/wsm6.py`` compiles ``WSM6_KMAX`` at 64 or 80, never at
    ``nz``, and until 1.8.9 preflight priced BOTH at the 64 row.  An
    nz = 72 WSM6 run -- the six shipped tornado-LES configs -- therefore
    reserved against 7,216 B while the kernel it launches holds 9,008 B:
    1,792 B/thread, +446.2 MiB of backing store the pool never reports,
    the direction the header at the top of preflight.py says put a run
    1,630 MiB over.
    """
    start = datetime(1974, 4, 3, 12)

    def frame(nz, mp_physics=6):
        cfg = RunConfig(**{**_TINY, "nz": nz}, moist=True, moist_cq=True,
                        mp_physics=mp_physics)
        return pf.kernel_local_frame_bytes(
            experiment_from_run_config(cfg, start)).get("wsm6")

    assert frame(4) == frame(49) == frame(64) == 7216
    assert frame(65) == frame(72) == frame(80) == 9008
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    moved = (profile.reservation_bytes(9008)
             - profile.reservation_bytes(7216))
    assert moved == 467927040                       # +446.2 MiB, measured
    # Deeper than the deepest compiled tier is a refusal, not a guess.
    with pytest.raises(ValueError, match="WSM6 requires 2 <= nz <= 80"):
        frame(96)
    # ... and a scheme that does not launch WSM6 never reaches that bound.
    assert frame(96, mp_physics=10) is None


def test_the_tiered_frame_model_agrees_with_the_wsm6_measurements():
    """Both rungs, against the driver measurements they were taken from.

    ``tests/test_kernel_local_bounds.py`` re-reads them off the card; this
    is the CPU half, so a model edit fails somewhere even on a host with
    no device.
    """
    from woof.core.wsm6_constants import (WSM6_KERNEL_LEVEL_TIERS,
                                           wsm6_level_tier)

    assert pf.WSM6_TIER_FRAME.define == "WSM6_KMAX"
    assert WSM6_KERNEL_LEVEL_TIERS == (64, 80)
    for tier, measured in ((64, 7216), (80, 9008)):
        assert pf.WSM6_TIER_FRAME.frame_bytes(tier) == measured, tier
    assert (pf.WSM6_TIER_FRAME.frame_bytes(pf.WSM6_TIER_FRAME.shipped_tier)
            == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wsm6"])
    assert wsm6_level_tier(64) == 64 and wsm6_level_tier(65) == 80
    # The launcher and the estimator read ONE ladder, so the adapter cannot
    # compile a tier the model has never heard of.
    from woof.core import wsm6 as wsm6_adapter
    assert wsm6_adapter._kernel_capacity is wsm6_level_tier

    # And the front-door bound agrees with it.  physics_vertical_contract
    # keeps its own literal deliberately -- it is a top-level module the
    # standalone preprocessing distribution imports without staging any
    # CUDA-backed component -- and its docstring says component tests bind
    # these values to their runtime counterparts.  For WSM6 nothing did,
    # which is how the deepest tier and the admitted maximum could part
    # company and let a front door accept an nz the launcher refuses.
    from woof.physics_vertical_contract import WSM6_VERTICAL_LEVEL_BOUNDS
    from woof.core.wsm6_constants import (
        WSM6_VERTICAL_LEVEL_BOUNDS as ladder_bounds)
    assert WSM6_VERTICAL_LEVEL_BOUNDS == ladder_bounds == (2, 80)
    assert WSM6_VERTICAL_LEVEL_BOUNDS[1] == WSM6_KERNEL_LEVEL_TIERS[-1]


def test_the_six_tornado_les_configs_are_the_ones_this_moves():
    """Reachability, named: which shipped configs the tier actually changes.

    All six are nz = 72 four-domain WSM6 trees, so all six were priced at
    the 64 rung.  Their local TOTAL used not to move, because ``ysu`` held
    9,232 B on these compositions -- 224 B wider than WSM6's 80-tier frame
    -- and the reservation is a MAX over the selected modules, so WSM6's
    under-pricing hid behind YSU's frame.

    It does not hide any more.  The YSU column workspace (2026-08-21) took
    that frame to 0, so WSM6's 9,008 B is what these six configurations
    now reserve against, and the tier ladder is what stands between them
    and a mispriced rail.  That is the reachability this test names.
    """
    from woof.experiment import load_experiment

    frames = []
    for path in sorted((ROOT / "configs").glob("les_tornado_100m_*.toml")):
        exp = load_experiment(path)
        f = pf.kernel_local_frame_bytes(exp)
        assert {int(d.run.nz) for d in exp.domains} == {72}, path.name
        frames.append((path.name, f["wsm6"], max(f.values())))
    assert len(frames) == 6, [name for name, _, _ in frames]
    assert all(wsm6 == 9008 and widest == 9008
               for _, wsm6, widest in frames), frames

    # WSM6 widest: the total moves by the frame's whole reservation delta.
    # 97e9a3ca061 (cap-hail): HAILCAST is automatic at dx <= 4 km and its
    # 7,392-B frame is wider than WSM6's 64-tier 7,216 B, so on a default
    # domain the 64 rung is hail-priced and the WSM6 tier step is hidden.
    # The tier is measured on hailcast_opt = 0 domains; the six shipped
    # configurations above keep HAILCAST on and are still WSM6-widest.
    start = datetime(2021, 12, 10, 3)
    totals = {}
    for nz in (64, 72):
        cfg = RunConfig(nx=400, ny=400, nz=nz, dx=100.0, dy=100.0,
                        ztop=18000.0, dt=0.5, run_seconds=60.0,
                        moist=True, moist_cq=True, mp_physics=6,
                        hailcast_opt=0)
        exp = experiment_from_run_config(cfg, start)
        assert max(pf.kernel_local_frame_bytes(exp),
                   key=pf.kernel_local_frame_bytes(exp).get) == "wsm6"
        totals[nz] = pf.kernel_local_memory_bytes(exp)
    assert totals[64] == 1616855040                 # 1542.0 MiB, unchanged
    assert totals[72] == 2084782080                 # 1988.2 MiB
    assert totals[72] - totals[64] == 467927040     # +446.2 MiB


@requires_reference_bundle
def test_the_widest_frame_is_no_longer_the_cumulus_kernel():
    """The cumulus kernel used to set this configuration's whole
    local-memory reservation, and now it sets none of it.

    History in one place: `kf_column` held 24,064 B at the unspecialized
    KF_KMAX = 128, then 9,216 B once the bound compiled to nz = 49
    (2026-07-26), and now 512 B with its column arrays in a global
    workspace (2026-08-21).  Under the 1,024 B default stack it reserves
    NOTHING, so the widest launched frame here is `morrison`'s 5,120 B --
    the runner-up that never moved.  Nothing MYNN owns has ever been in
    this contest; that is the other half of the claim and it still holds.
    """
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    frames = pf.kernel_local_frame_bytes(exp)
    widest = max(frames, key=lambda m: frames[m])
    assert widest == "morrison"
    assert frames["morrison"] == 5120
    assert frames["kf"] == 512
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["kf"] == 512
    assert pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(
        frames["kf"]) == 0
    assert max(f for m, f in frames.items() if m.startswith("mynn")) == 0


@requires_reference_bundle
def test_physics_kernel_modules_fails_closed_on_an_unpriced_selector():
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    d01 = exp.domains[0]
    bogus = dataclasses.replace(
        d01, run=dataclasses.replace(d01.run, mp_physics=55))
    broken = dataclasses.replace(exp, domains=(bogus,) + exp.domains[1:])
    with pytest.raises(ValueError,
                       match="no kernel-module row for mp_physics=55"):
        pf.physics_kernel_modules(broken)


@pytest.mark.parametrize("diff_opt", [1, 2])
def test_coordinate_diffusion_module_uses_its_measured_frame(diff_opt):
    """The selected coordinate path is visible in pricing and module reports."""
    cfg = RunConfig(**_TINY, diff_opt=diff_opt, km_opt=4, mp_physics=0,
                    bl_pbl_physics=0, sf_sfclay_physics=0,
                    sf_surface_physics=0, ra_physics=0)
    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    frames = pf.kernel_local_frame_bytes(
        exp, profile=pf.MEASURED_LOCAL_MEMORY_PROFILE)
    assert ("diff_opt1" in modules) == (diff_opt == 1)
    assert ("diff_opt1" in frames) == (diff_opt == 1)
    if diff_opt == 1:
        assert frames["diff_opt1"] == 0
        assert "diff_opt1" not in pf.assumed_bound_modules(modules)
    assert "bandwidth_glue" in modules
    assert frames["bandwidth_glue"] == 0


def test_noahmp_on_an_unread_card_is_priced_from_the_ceiling_and_says_so(monkeypatch):
    """Scheme 4 prices the composed units the model launches, from the row
    read on the card's own compile platform when there is one.  With no
    card read there is no platform to match, and the estimator prices the
    units from the ceiling over the recorded platforms -- the same rule
    every standalone kernel gets on an unrecorded platform -- and says so
    beside the number, naming the recorded platforms and the command that
    makes the price a reading of this card.

    CPU-only: no forcing bundle and no device.
    """
    from woof.core import noahmp_frame_provenance as prov

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    cfg = RunConfig(**(_TINY | dict(nx=40, ny=40, nz=40)), moist=True,
                    mp_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                    sf_surface_physics=4, ra_physics=90)
    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    assert {"noahmp_driver_composed", "noahmp_energy_composed",
            "noahmp_thermal_composed", "noahmp_glacier_composed",
            "noahmp_libm_slab_composed", "noahmp_vegeflux_runtime"} <= modules
    assert not modules & pf.UNMEASURED_KERNEL_MODULES
    frames = pf.kernel_local_frame_bytes(exp)
    ceiling = prov.composed_frame_ceiling(prov.usable_recordings())
    assert ceiling, "the shipped rows describe this tree"
    for key in ceiling:
        assert frames[key] == ceiling[key]
    reference = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert pf.kernel_local_memory_bytes(exp) == reference.reservation_bytes(max(frames.values()))
    basis = pf.non_pool_basis(reference, exp)
    assert "Noah-MP local frames priced from the ceiling over the recorded platforms" in basis
    for row in prov.usable_recordings():
        assert prov.platform_label(row) in basis
    assert "not measured on this card" in basis
    assert "measure_noahmp_frames.py measure" in basis


@requires_reference_bundle
def test_the_reflectivity_diagnostic_is_priced_only_when_it_can_fire():
    """``refl10cm_*`` is launched from the microphysics drivers'
    history-cadence branch alone.  The 60 s probes behind this model wrote
    their t=0 frames and never launched one."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    assert exp.run_seconds == 60.0
    assert not pf.refl_diagnostic_reachable(exp)
    assert "refl" not in pf.physics_kernel_modules(exp)
    production = dataclasses.replace(exp, run_seconds=43200.0)
    assert pf.refl_diagnostic_reachable(production)
    assert "refl" in pf.physics_kernel_modules(production)


@requires_reference_bundle
def test_the_reflectivity_time_bomb_no_longer_moves_the_reservation():
    """FAILING FORM FIRST.

    The four-domain config WITHOUT cumulus is the one the reflectivity
    reservation could detonate: its widest launched frame before the first
    history frame is Morrison's 5,120 B (64 MiB), and
    `refl10cm_morrison_column` at the unspecialized REFL_KMAX = 256 carries
    18,432 B (4,335 MiB).  Neither traced probe ran long enough to launch
    it, so an as-built production forecast would have taken that step
    MID-FLIGHT, past the gate that let it start.  The first block
    reproduces the jump; the second shows it gone.
    """
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    probe = load_experiment_case(CONFIG_4DOM_MYNN_NOCU)[0]
    production = dataclasses.replace(probe, run_seconds=43200.0)
    assert "refl" not in pf.physics_kernel_modules(probe)
    assert "refl" in pf.physics_kernel_modules(production)

    # As LAUNCHED, from the driver: Morrison's sedimentation kernel measures
    # 1,280 B (64 MiB reserved -- the traced no-cumulus run's whole
    # local-memory term) and refl10cm_morrison_column 18,432 B.
    as_launched_jump = (profile.reservation_bytes(18432)
                        - profile.reservation_bytes(1280))
    assert round(as_launched_jump / 1024 ** 2) == 4271
    # As PRICED, with the module maximum preflight carries for Morrison --
    # over-priced by design, and still a 3,315 MiB mid-flight step.
    as_priced_jump = (profile.reservation_bytes(18432)
                      - profile.reservation_bytes(5120))
    assert round(as_priced_jump / 1024 ** 2) == 3315

    # Specialized to nz = 49 the same kernel measures 3,528 B against
    # Morrison's unchanged 5,120 B, so the widest launched frame does not
    # move at all when the first history frame comes due.
    assert pf.kernel_local_frame_bytes(production)["refl"] == 3528
    assert (pf.kernel_local_memory_bytes(production)
            == pf.kernel_local_memory_bytes(probe)
            == profile.reservation_bytes(5120))


def _rail_gate(config, *, rail_mib, other_mib, overhead_bytes=None):
    """The `woof check` rail leg, with the card's occupancy supplied so the
    assertion does not depend on what the desktop happens to be holding."""
    exp = load_experiment_case(config)[0]
    estimate = pf.estimate_experiment(exp)
    reserve = pf.ReservePolicy.n0_alloc(
        exp, estimate_bytes=estimate.alloc_estimate_bytes)
    if overhead_bytes is not None:
        reserve = dataclasses.replace(
            reserve, device_overhead_bytes=overhead_bytes)
    free = pf.device_rail_free_bytes(
        rail_mib * 1024 ** 2, other_process_bytes=other_mib * 1024 ** 2)
    return pf.evaluate_alloc_gates(
        measured_used_bytes=None,
        estimate_bytes=estimate.alloc_estimate_bytes,
        measured_free_bytes=free, reserve=reserve)


@requires_reference_bundle
def test_the_old_overhead_constant_passes_the_run_that_breached_the_rail():
    """FAILING FORM FIRST.

    ``configs/real74_4dom_mynn_norad.toml`` was RUN: 31,130 MiB device-wide
    against a 29,500 MiB rail, 1,630 MiB over, with 3,381 MiB of desktop on
    the card.  Preflight reported ``alloc_estimate_le_wddm_budget: PASS``
    beforehand.  This reproduces that pass with the 2026-07-16 zero-step
    probe overhead in place, so the fixed gate is never merely assumed to
    work.
    """
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381,
                      overhead_bytes=pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert legs["alloc_estimate_le_wddm_budget"] is True


def _as_built_overhead(config):
    """Non-pool overhead of the binary that RAN on 2026-07-26: CUDA context
    plus the reservation of the widest module frame at its UNSPECIALIZED
    bound, which is how ``kf``/``refl`` compiled before the bounds were
    specialized.  Keeps the historical measurements priced against the code
    that produced them.

    ``kf`` is substituted explicitly because its LIVE row no longer
    describes any binary that ever ran these configurations: it is 512 B
    since the column workspace, and reading it here would price a
    2026-07-26 run as if it had carried no cumulus reservation at all --
    which is the one thing those runs are evidence AGAINST.  No workspace
    term is added for the same reason: the binary that ran had none."""
    exp = load_experiment_case(config)[0]
    modules = pf.physics_kernel_modules(exp)
    as_built = dict.fromkeys(modules, 0)
    as_built.update({m: pf.KERNEL_MAX_LOCAL_SIZE_BYTES[m] for m in modules})
    if "kf" in as_built:
        as_built["kf"] = KF_AS_BUILT_FRAME.frame_bytes(128)
    widest = max(as_built.values())
    return (pf.CUDA_CONTEXT_BYTES
            + pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(widest))


@requires_reference_bundle
def test_the_measured_overhead_refuses_the_run_that_breached_the_rail():
    """The as-built binary, priced with the measured local-memory law: the
    run that went 1,630 MiB over is refused before it starts."""
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381,
                      overhead_bytes=_as_built_overhead(CONFIG_4DOM_MYNN_KF))
    assert legs["alloc_estimate_le_wddm_budget"] is False


@requires_reference_bundle
def test_taking_kf_off_the_stack_is_what_lets_the_cumulus_config_through():
    """The same configuration, the same rail, the same desktop occupancy --
    what changed is where `kf.cu` keeps its column arrays.

    Measured, not projected.  `configs/real74_4dom_mynn_norad.toml` was RUN
    unchanged on 2026-07-26 after the bound was specialized to nz = 49:
    27,216 MiB device-wide peak (NVML, 1 Hz, 112 samples), status complete,
    2,284 MiB under the 29,500 MiB rail, against 31,130 MiB for the same
    file before.  See docs/kernel_local_memory_bounds.md.  The 2026-08-21
    workspace goes further in the same direction and adds a cost of its own,
    and both legs are priced separately below rather than netted silently.
    """
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381)
    assert legs["alloc_estimate_le_wddm_budget"] is True
    # The saving this test measures is the FRAME specialization, so both
    # sides are priced with the same CUDA context.  They were not, once
    # the context became a per-card term (task 206): the as-built figure
    # keeps the 2026-07 flat constant by construction, and subtracting a
    # reserve carrying the reference card's larger context turned a
    # compile-time-array saving into a saving mixed with an accounting
    # change.  Prices both against the same reference profile.
    reference = pf.card_local_memory_profile(None)
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    saved = (_as_built_overhead(CONFIG_4DOM_MYNN_KF)
             - pf.CUDA_CONTEXT_BYTES
             - (pf.ReservePolicy.n0_alloc(exp).device_overhead_bytes
                - reference.cuda_context_bytes))
    # Two legs, named.  The RESERVATION leg: 5,738 MiB at KF_KMAX = 128
    # against 1,020 MiB now, because `kf` is under the default stack and
    # `morrison`'s 5,120 B sets the reservation instead -- 4,718 MiB.  The
    # WORKSPACE leg costs 423 MiB back, which is what the tile of columns
    # actually in flight holds on this 170-SM reference card.  Net 4,294.
    workspace = pf.kf_column_workspace_bytes(exp, profile=reference)
    assert round(workspace / 1024 ** 2) == 423
    assert round((saved + workspace) / 1024 ** 2) == 4718
    assert round(saved / 1024 ** 2) == 4294


@requires_reference_bundle
def test_the_rail_gate_passes_the_four_domain_run_that_measured_under_it():
    """``configs/real74_4dom_mynn_norad_nocu.toml`` was RUN: 25,498 MiB
    device-wide, 4,002 MiB under the rail, same four domains, same
    861,001 columns."""
    legs = _rail_gate(CONFIG_4DOM_MYNN_NOCU, rail_mib=29500, other_mib=3464)
    assert legs["alloc_estimate_le_wddm_budget"] is True


@requires_reference_bundle
def test_the_rail_never_widens_the_budget():
    """A rail is an ADDITIONAL ceiling.  A rail below what the card would
    hand out must bind; it can never hand out more."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_NOCU)[0]
    estimate = pf.estimate_experiment(exp)
    reserve = pf.ReservePolicy.n0_alloc(
        exp, estimate_bytes=estimate.alloc_estimate_bytes)
    rail_free = pf.device_rail_free_bytes(
        29500 * 1024 ** 2, other_process_bytes=3464 * 1024 ** 2)
    card_free = 40 * GIB
    assert min(card_free, rail_free) == rail_free
    assert reserve.budget_bytes(rail_free) < reserve.budget_bytes(card_free)


@requires_reference_bundle
def test_the_non_pool_projection_brackets_both_measured_runs():
    """Estimate + non-pool residency, against the two device-wide peaks the
    runs actually reached (process share = device peak - desktop baseline).

    Both peaks were measured on 2026-07-26 by the AS-BUILT binary, before
    the ``kf``/``refl`` bounds were specialized, so the overhead leg is
    priced at the unspecialized frames those runs actually compiled to.
    """
    for config, other_mib, device_peak_mib in (
            (CONFIG_4DOM_MYNN_KF, 3381, 31130),
            (CONFIG_4DOM_MYNN_NOCU, 3464, 25498)):
        exp = load_experiment_case(config)[0]
        estimate = pf.estimate_experiment(exp)
        reserve = pf.ReservePolicy.n0_alloc(
            exp, estimate_bytes=estimate.alloc_estimate_bytes)
        projected = (estimate.alloc_estimate_bytes
                     + reserve.retention_residual_bytes
                     + _as_built_overhead(config))
        measured = (device_peak_mib - other_mib) * 1024 ** 2
        assert abs(projected - measured) / measured < 0.04, config.name
    # ... and the direction that mattered: the retired model under-projected
    # the run that breached, which is exactly how it passed it.
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    old = (pf.estimate_experiment(exp).alloc_estimate_bytes
           + pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert old < (31130 - 3381) * 1024 ** 2


@pytest.mark.gpu
def test_the_recorded_local_frames_match_the_driver():
    """Regenerate ``KERNEL_MAX_LOCAL_SIZE_BYTES`` from NVRTC + the driver.

    A kernel that grows its per-thread frame silently moves the whole
    process's device footprint by gigabytes, so the table is not allowed to
    go stale.

    The default-stack-limit leg is measured in a FRESH SUBPROCESS, and that
    is what it proves: ``cudaLimitStackSize`` is process state, not a device
    constant.  The CUDA runtime raises it for the life of the process the
    first time a fatter-framed kernel is loaded (kf/refl at 24,064 B against
    the 1,024 B fresh default -- nothing in this tree calls
    ``deviceSetLimit``), so reading it in the suite process measures which
    tests happened to run first, which made this test pass alone and fail
    after the kf/refl modules in a one-process full run.  The reservation
    law prices what a FRESH woof process reserves, so the fresh-process
    value is the only one the recorded constant may be compared against;
    the assertion is exact, the same production reader runs in the probe,
    and no launch order in this process can move the answer.

    WHAT IS ASSERTED AGAINST WHAT (rewritten 2026-08-20, task 201/202).
    A frame is what NVRTC emitted for one target architecture at one
    compiler build, not a property of the source, so this test used to
    judge every machine by one box's reading and went red on the desktop
    the moment its card changed.  It now asks two different questions:

    * on a compile platform this tree HAS a recording for, exact
      equality against THAT recording -- full strength, and stronger
      than before, because a box can no longer pass by matching somebody
      else's numbers;
    * on every platform, recorded or not, the invariant that actually
      protects a user: no module may compile WIDER than the shipped
      ceiling.  Wider means the reservation law under-charges by the
      frame delta times the whole resident-thread capacity, and a run
      admitted on the short number does not fail at the gate, it OOMs
      later with nothing pointing back here.

    That is not a skip: it is the assertion whose breakage can be named.
    """
    import re
    import subprocess
    import sys

    from woof.certify import compile_platform

    cp = pytest.importorskip("cupy")
    from woof.core.kernels import load_module, kernel_local_size_bytes

    kdir = ROOT / "woof" / "core" / "kernels"
    symbol = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)')
    probe = subprocess.run(
        [sys.executable, "-c",
         "import cupy as cp\n"
         "from woof.core import preflight as pf\n"
         "print(pf.local_memory_profile_from_device(cp)"
         ".default_stack_limit_bytes)"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert int(probe.stdout.strip().splitlines()[-1]) == (
        pf.MEASURED_LOCAL_MEMORY_PROFILE.default_stack_limit_bytes)

    observed = {}
    uncompilable = set()
    from woof.core.kernels import _DEFAULT_SOURCE_MODULES
    for path in sorted(kdir.glob("*.cu")):
        if path.stem in {f"{m}_default" for m in _DEFAULT_SOURCE_MODULES}:
            # A default process's text of <module>, compiled and priced
            # under <module>'s own name (kernels._unit_text).
            continue
        if path.stem in pf.PRODUCTION_LOADED_KERNEL_UNITS:
            # Compiled only through its own loader; measured below
            # through that loader (preflight.load_production_kernel_unit).
            continue
        if path.stem in ("noah_mosaic", "thompson_fork"):
            # Both mosaic launchers have dedicated --fmad=false composed
            # loaders and measured chained rows. The generic C++17 image
            # compiles, but no driver launches it and it has no frame row.
            # The fork Thompson unit is the same shape: it launches only
            # through woof/core/thompson_fork.py's own compile, priced as
            # the chained unit thompson_fork_unit and re-read by
            # tests/test_thompson_fork_column_gpu.py.
            uncompilable.add(path.stem)
            continue
        try:
            module = load_module(path.stem)
        except Exception:  # noqa: BLE001  -- recorded here, repaired elsewhere
            uncompilable.add(path.stem)
            continue
        widest = 0
        for name in sorted(set(symbol.findall(path.read_text()))):
            try:
                local_bytes = kernel_local_size_bytes(module.get_function(name))
            except Exception:  # noqa: BLE001
                continue
            widest = max(widest, local_bytes)
        observed[path.stem] = widest
    # Logical modules with no committed .cu of their own since slimming wave 3 (lanes 11-13), the same three
    # tests/test_kernel_frame_recordings.py adds to its source set: ysu and mynn_dmp_sibling compile retained
    # files under their own names (kernels._unit_text), so they are measured here like any unit;
    # rrtmg_lw._lw_coalesced_source generates rrtmg_lw_chain_coalesced inside the composed longwave unit, so it
    # cannot compile alone and stays declared unmeasured.
    from woof.core.kernels import module_source
    for logical in ("ysu", "mynn_dmp_sibling"):
        module = load_module(logical)
        widest = 0
        for name in sorted(set(symbol.findall(module_source(logical)))):
            try:
                local_bytes = kernel_local_size_bytes(module.get_function(name))
            except Exception:  # noqa: BLE001
                continue
            widest = max(widest, local_bytes)
        observed[logical] = widest
    uncompilable.add("rrtmg_lw_chain_coalesced")
    for unit in sorted(pf.PRODUCTION_LOADED_KERNEL_UNITS):
        module, source = pf.load_production_kernel_unit(unit)
        widest = 0
        for name in sorted(set(symbol.findall(source))):
            try:
                local_bytes = kernel_local_size_bytes(module.get_function(name))
            except Exception:  # noqa: BLE001
                continue
            widest = max(widest, local_bytes)
        observed[unit] = widest
    assert uncompilable == set(pf.UNMEASURED_KERNEL_MODULES)

    # The shipped table must know about every module that compiled, and
    # about nothing that does not exist.  This is what caught health_tile
    # -- a ``.cu`` that shipped with the out-of-core merge and had no row
    # in either table, so the regeneration gate could not enumerate it.
    assert set(observed) == set(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)

    fingerprint = compile_platform.compile_platform_fingerprint()
    recording = pf.kernel_frame_recording_for(fingerprint)
    if recording is not None:
        mine = {module: observed[module] for module in recording.frames
                if module in observed}
        assert mine == dict(recording.frames), (
            f"this box IS {recording.box} "
            f"(sm_{recording.compute_capability}, NVRTC "
            f"{recording.nvrtc_build}) and its own recording has gone "
            "stale; re-measure it with "
            "`python tools/vram_reserve_probe.py frames` and move the row "
            "in woof/core/kernel_frame_recordings.py")
        if recording.complete:
            assert set(recording.frames) == set(observed)

    profile = pf.local_memory_profile_from_device(cp)
    over = pf.under_priced_kernel_frames(observed, profile=profile)
    assert not over, (
        "these modules compile WIDER on this platform (sm_"
        f"{fingerprint['device_compute_capability']}, NVRTC "
        f"{fingerprint['nvrtc_build']}) than the shipped ceiling, so the "
        "local-memory reservation under-charges and a run this card "
        "cannot hold would be admitted: "
        + "; ".join(
            f"{row.module} {row.observed_bytes} B against "
            f"{row.shipped_bytes} B = "
            f"{row.unpriced_device_bytes / 1024 ** 3:.3f} GiB unpriced"
            for row in sorted(over.values(),
                              key=lambda r: -r.unpriced_device_bytes))
        + ".  Remedy: add this platform as a KernelFrameRecording in "
        "woof/core/kernel_frame_recordings.py (the ceiling is the "
        "element-wise maximum over the recordings, so adding one raises "
        "every row it needs to raise and nothing else)")

# ---------------------------------------------------------------------------
# Noah-MP land-surface transients (noahmp_lsm_transient_shapes)
# ---------------------------------------------------------------------------
# These live here, not in tests/test_noahmp_column_slab.py, because that
# module's helpers import cupy and conftest._cupy_scope therefore marks the
# whole file gpu -- and a pricing gate that only runs where a device happens
# to be present is the exact blindness the allocation ratchet was moved out
# of test_mynn_pbl_scratch.py to escape.  Nothing below opens a device.

@pytest.mark.parametrize("land_surface,nsoil", [(3, 9), (4, 4)])
def test_mynn_lsm_pairings_price_the_union_of_both_components(
        land_surface, nsoil):
    """The newly reachable tuples cannot shed either side's persistent state."""
    from woof.core.mynn_pbl_runtime import MYNN_PBL_STATE_3D
    from woof.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS

    cfg = RunConfig(
        nx=8, ny=6, nz=12, dx=1000.0, dy=1000.0, ztop=12000.0,
        dt=5.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=land_surface, num_soil_layers=nsoil)
    shapes = pf.physics_array_shapes(cfg)
    for name in MYNN_SURFACE_OUTPUTS:
        assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
    for name in MYNN_PBL_STATE_3D:
        assert shapes[f"fields/{name}"] == (cfg.nz, cfg.ny, cfg.nx)

    if land_surface == 3:
        from woof.core.ruc_runtime import RUC_STATE_2D, RUC_STATE_3D
        for name in MYNN_SURFACE_OUTPUTS:
            assert shapes[f"fields/{name}_sea"] == (cfg.ny, cfg.nx)
        for name in RUC_STATE_2D:
            assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
        for name in RUC_STATE_3D:
            assert shapes[f"fields/{name}"] == (nsoil, cfg.ny, cfg.nx)
    else:
        from woof.core.noahmp_runtime import (
            NOAHMP_STATE_2D, NOAHMP_STATE_SNOWSOIL_3D,
            NOAHMP_STATE_SNOW_3D, NSNOW,
        )
        for name in NOAHMP_STATE_2D:
            assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
        for name in NOAHMP_STATE_SNOW_3D:
            assert shapes[f"fields/{name}"] == (NSNOW, cfg.ny, cfg.nx)
        for name in NOAHMP_STATE_SNOWSOIL_3D:
            assert shapes[f"fields/{name}"] == (
                NSNOW + nsoil, cfg.ny, cfg.nx)


def test_preflight_prices_the_bound_the_runtime_launches_with():
    """The transient term reads the runtime's own constants, by name.

    ``SLAB_COLUMN_CHUNK`` is the explicit column-chunk bound the slab
    modules' allocation-inventory rows demanded; this is the assertion that
    the number preflight prices IS that bound and not a copy that can drift.
    """
    from woof.config import RunConfig
    from woof.core.noahmp_runtime import (
        COLUMN_BATCH, SLAB_COLUMN_CHUNK, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN,
        SLAB_TRANSIENT_BYTES_PER_COLUMN)
    from woof.core.preflight import noahmp_lsm_transient_shapes

    base = dict(nx=600, ny=600, nz=40, dx=1000.0 / 3.0, dy=1000.0 / 3.0,
                ztop=16000.0, dt=5.0 / 3.0, run_seconds=0.0,
                time_step_sound=4, moist=True, mp_physics=6,
                sf_sfclay_physics=1, bl_pbl_physics=1, bldt=0.0)
    shapes = noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=4, **base))
    assert shapes["noahmp_lsm/slab_chunk_transients"] == (
        min(SLAB_COLUMN_CHUNK, 360000), SLAB_TRANSIENT_BYTES_PER_COLUMN)
    assert shapes["noahmp_lsm/slab_grid_transients"] == (
        360000, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN)
    assert shapes["noahmp_lsm/staged_leaf_batches"] == (
        min(COLUMN_BATCH, 360000), 620)
    # A domain smaller than the chunk prices its own width, not the bound.
    small = noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=4, **{**base, "nx": 8, "ny": 6}))
    assert small["noahmp_lsm/slab_chunk_transients"] == (
        48, SLAB_TRANSIENT_BYTES_PER_COLUMN)
    # And a run without Noah-MP pays nothing.
    assert noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=2, **base)) == {}


def test_estimate_domain_carries_the_noahmp_transient_items():
    """The term is in the estimate a launcher actually reads, not only in a
    helper a launcher could forget to call."""
    from woof.config import RunConfig
    from woof.core.preflight import estimate_domain
    from woof.experiment import DomainConfig

    cfg = RunConfig(nx=64, ny=64, nz=40, dx=1000.0 / 3.0, dy=1000.0 / 3.0,
                    ztop=16000.0, dt=5.0 / 3.0, run_seconds=0.0,
                    time_step_sound=4, moist=True, mp_physics=6,
                    sf_sfclay_physics=1, sf_surface_physics=4,
                    bl_pbl_physics=1, bldt=0.0)
    estimate = estimate_domain(DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=3600.0, run=cfg, time_step=1))
    names = {item.name: item for item in estimate.items
             if item.name.startswith("noahmp_lsm/")}
    assert set(names) == {"noahmp_lsm/slab_chunk_transients",
                          "noahmp_lsm/slab_grid_transients",
                          "noahmp_lsm/staged_leaf_batches",
                          "noahmp_lsm/slab_caches"}
    for name, item in names.items():
        assert item.category == ("physics" if name == "noahmp_lsm/slab_caches"
                                 else "transient")
        assert item.itemsize == 1


def test_noahmp_slab_caches_are_priced_resident_per_grid_column():
    """The layout and parameter caches outlive a call, so they are priced
    as resident physics for every column, 522 B each today."""
    from woof.core.noahmp_column_slab import PARAMETER_INTS, PARAMETER_WIDTHS
    from woof.core.noahmp_runtime import (
        SLAB_LAYOUT_CACHE_BYTES_PER_COLUMN, slab_cache_bytes_per_column)
    from woof.core.preflight import noahmp_lsm_cache_shapes

    per_column = slab_cache_bytes_per_column()
    assert per_column == SLAB_LAYOUT_CACHE_BYTES_PER_COLUMN + 4 * (
        sum(PARAMETER_WIDTHS.values()) + len(PARAMETER_INTS)) == 522
    base = dict(nz=40, dx=1000.0, dy=1000.0, ztop=16000.0, dt=5.0,
                run_seconds=0.0, time_step_sound=4, moist=True, mp_physics=6,
                sf_sfclay_physics=1, bl_pbl_physics=1)
    assert noahmp_lsm_cache_shapes(RunConfig(
        nx=600, ny=600, sf_surface_physics=4, **base)) == {
            "noahmp_lsm/slab_caches": (360000, per_column)}
    assert noahmp_lsm_cache_shapes(RunConfig(
        nx=600, ny=600, sf_surface_physics=2, **base)) == {}


# ---------------------------------------------------------------------------
# Preprocessing (ingest) phase -- the phase nobody used to price
# ---------------------------------------------------------------------------


def test_ingest_analysis_is_sized_by_source_levels_and_target_grid(exp1):
    """Horizontal interpolation lands source LEVELS on the MODEL grid."""
    run = exp1.root.run
    shapes = pf.ingest_analysis_shapes(run, source="gfs")
    levels = pf.SOURCE_ANALYSIS_LEVELS["gfs"]
    mass = [s for n, s in shapes.items() if n.startswith("analysis_mass")]
    assert mass and all(s == (levels, run.ny, run.nx) for s in mass)
    assert shapes["analysis_u_level_0"] == (levels, run.ny, run.nx + 1)
    assert shapes["analysis_v_level_0"] == (levels, run.ny + 1, run.nx)
    surface = [s for n, s in shapes.items() if n.startswith("analysis_surf")]
    assert len(surface) == pf.SOURCE_ANALYSIS_SURFACE_FIELDS["gfs"]
    assert all(s == (run.ny, run.nx) for s in surface)
    # ERA5 carries more levels, so its analysis is strictly larger.
    era5 = pf.ingest_analysis_shapes(run, source="era5")
    assert (math.prod(era5["analysis_mass_level_0"])
            > math.prod(shapes["analysis_mass_level_0"]))
    with pytest.raises(ValueError, match="no forcing-analysis level"):
        pf.ingest_analysis_shapes(run, source="not-a-product")


def test_ingest_state_term_is_the_real_domain_state_inventory(exp1):
    """Ingest builds a full DomainState per time -- the same one priced
    for the forecast, not a smaller setup-only object."""
    est = pf.estimate_ingest(exp1, source="gfs")
    expected = sum(4 * math.prod(shape) for shape
                   in pf.state_array_shapes(exp1.root.run).values())
    assert est.category_bytes("state") == expected


def test_ingest_holds_one_forcing_time_not_all_of_them(exp1):
    """The defect in one assertion.

    Ingest used to keep every forcing time's analysis AND state resident,
    which is why preprocessing a 24 h GFS case peaked at roughly twice the
    forecast.  Streaming cut that to two, and the start-last reordering
    (woof/ingest/lateral_bc.py:start_last_forcing_order) cut it to ONE:
    nothing reads the start time until the boundaries are complete, so
    building it first meant holding it for the whole loop for no reason.
    Both the streamed number and the all-at-once one are reported so a
    user can see what the phase would have cost.
    """
    est = pf.estimate_ingest(
        exp1, source="gfs", forcing_interval_seconds=10800.0)
    assert est.resident_times == pf.INGEST_RESIDENT_FORCING_TIMES == 1
    assert est.n_forcing_times > est.resident_times
    assert (est.resident_bytes
            == est.per_time_bytes + est.forcing_table_bytes)
    assert (est.unstreamed_resident_bytes
            == est.n_forcing_times * est.per_time_bytes
            + est.forcing_table_bytes)
    # Every time beyond the one resident time is pure saving.
    assert (est.unstreamed_resident_bytes - est.resident_bytes
            == (est.n_forcing_times - est.resident_times)
            * est.per_time_bytes)
    # The retained perimeter frames are what replaced those states, and
    # all of them together stay far below one of them.
    assert est.boundary_frame_bytes < est.per_time_bytes


def test_ingest_envelope_is_conservative_against_the_measured_case():
    """Every end of the CONUS 12 km measurement, re-derived here.

    Measured on an RTX 5090 (process-attributed peak, 432 MiB CUDA
    context included): 15,288 MiB with every forcing time resident and
    4,672 MiB with the streamed TWO, same case, byte-identical outputs.
    The estimate must bound both -- a sizing number that lands under a
    measured peak is the failure mode this whole phase estimate exists to
    prevent.

    The shipped estimate now prices ONE resident time, which no device
    has been watched doing.  So the third bound here is a derivation
    rather than a measurement, and it is written as one: the reordering
    removes a RESIDENT term and no transient one, so the peak it should
    produce is the measured two-resident peak minus exactly one forcing
    time, and the estimate has to stay above THAT.  If a device ever
    measures the one-resident form, this is the assertion to replace with
    the number.
    """
    run = RunConfig(nx=414, ny=330, nz=49, dx=12000.0, dy=12000.0,
                    ztop=20000.0, dt=60.0, run_seconds=86400.0,
                    mp_physics=10, moist=True, terrain_opt=1,
                    spec_bdy_width=5, specified=True)
    exp = experiment_from_run_config(run, datetime(2026, 7, 30))
    est = dataclasses.replace(
        pf.estimate_ingest(exp, source="gfs",
                           forcing_interval_seconds=10800.0),
        device_overhead_bytes=0)  # the measured node is Linux
    assert est.n_forcing_times == 9
    assert est.resident_times == 1

    # The form that WAS measured at 4,672 MiB: identical itemization,
    # one more resident forcing time.
    two_resident = dataclasses.replace(est, resident_times=2)
    measured_two_resident = 4672 * 1024 ** 2
    assert two_resident.peak_envelope_bytes >= measured_two_resident
    assert two_resident.peak_envelope_bytes <= 1.30 * measured_two_resident

    # The reordering is worth exactly one forcing time of residency and
    # nothing else -- the transient is charged per CALL, and the same
    # calls happen in either order.
    assert (two_resident.resident_bytes - est.resident_bytes
            == est.per_time_bytes)
    assert two_resident.transient_bytes == est.transient_bytes

    derived_one_resident = measured_two_resident - est.per_time_bytes
    assert est.peak_envelope_bytes >= derived_one_resident
    assert est.peak_envelope_bytes <= 1.30 * derived_one_resident

    measured_before = 15288 * 1024 ** 2
    before = (math.ceil(est.headroom * (est.unstreamed_resident_bytes
                                        + est.transient_bytes))
              + est.context_bytes)
    assert before >= measured_before
    assert before <= 1.30 * measured_before


def test_phase_estimate_names_the_binding_phase_and_the_number(exp1):
    phases = pf.estimate_phases(exp1, source="gfs")
    assert phases.binding_phase in ("forecast", "ingest")
    assert phases.peak_envelope_bytes == max(
        phases.forecast_envelope_bytes, phases.ingest_envelope_bytes)
    budget = phases.peak_envelope_bytes - 1
    assert not phases.fits(budget)
    verdict = phases.verdict(budget)
    assert "EXCEEDS" in verdict
    assert f"{phases.peak_envelope_bytes / GIB:.2f}" in verdict
    assert "forecast" in verdict and "ingest" in verdict
    assert phases.fits(phases.peak_envelope_bytes)
    assert "fits" in phases.verdict(phases.peak_envelope_bytes)


def test_config_forcing_source_refuses_to_guess(tmp_path):
    """A config whose source cannot be priced says so; it never lets the
    forecast number stand in for the whole run."""
    # A plain RunConfig TOML is not an experiment TOML at all, so it
    # records no forcing product and must not be guessed at.
    plain = tmp_path / "plain.toml"
    plain.write_text(CONFIG_D01.read_text(encoding="utf-8"),
                     encoding="utf-8")
    assert pf.config_forcing_source(plain) is None
    note = pf.unpriced_ingest_note(plain)
    assert "NOT PRICED" in note and "FORECAST only" in note
    assert "no forcing product at all" in note
    named = pf.unpriced_ingest_note(plain, "hrrr")
    assert "--source hrrr" in named and "does not model" in named


def test_an_unpriced_source_is_said_out_loud_not_scored_zero(exp1):
    """HRRR's ingest is a different lane and nothing here measured it.

    Returning zero for a phase you did not model reads exactly like
    "this phase is free", which is the failure this estimate exists to
    end.  So it is reported absent, the forecast stands alone, and the
    verdict says which of the two happened.
    """
    priced = pf.estimate_phases(exp1, source="gfs")
    unpriced = pf.estimate_phases(exp1, source="hrrr")
    assert priced.ingest_priced and priced.ingest is not None
    assert not unpriced.ingest_priced
    assert unpriced.ingest is None
    assert unpriced.ingest_envelope_bytes is None
    assert unpriced.binding_phase == "forecast"
    assert (unpriced.peak_envelope_bytes
            == unpriced.forecast_envelope_bytes)
    assert "NOT PRICED" in unpriced.verdict(None)
    assert "hrrr" in unpriced.verdict(None)
    assert pf.estimate_phases(exp1, source=None).ingest is None


# ---------------------------------------------------------------------------
# 2026-08-01 sizing calibration: the affine envelope and the tree ingest
# ---------------------------------------------------------------------------

#: Every whole-forecast run instrumented on the 16 GiB fleet node (RTX
#: 4080, Linux, driver 595.58.03, machine-wide ``nvidia-smi`` at 250 ms,
#: GPU otherwise idle), as ``(label, domains, itemized alloc estimate GiB,
#: measured machine peak GiB)``.  The 4080 carries 76 SMs, which is what
#: sizes the non-pool term for every row.
FLEET_4080_FORECAST_RUNS = (
    ("s07      170x136",              1,  2.0652,  3.6494),
    ("small8   224x180",              1,  2.7500,  4.1436),
    ("small8   224x180 (go route)",   1,  2.7500,  4.3818),
    ("s11      340x272",              1,  4.8242,  5.9541),
    ("edge15   448x360 (go route)",   1,  7.5576,  8.7529),
    ("L12      474x378 (go route)",   1,  8.2683,  9.2510),
    ("over22   594x476 (go route)",   1, 12.3809, 12.5889),
    ("big24    630x504 (go route)",   1, 13.7616, 13.8799),
    ("n10      2-domain tree",        2,  4.1204,  5.8330),
    ("n2_16    2-domain tree",        2,  8.2162, 10.0928),
    ("c07      4-domain tree",        4,  2.0612,  3.8564),
)

#: SM count of the card every row above was measured on.
FLEET_4080_MULTIPROCESSORS = 76

#: ...and the radiation lane every row above ran, which is what decides
#: the pool-slack term (:data:`woof.core.preflight.POOL_SLACK_FRACTION`).
#: These are the default morrison-kf-RTE-RRTMGP suite, whose pool tracked
#: the itemization at 0.94-1.00x -- the lane that does NOT retain a
#: call-peak workspace between radiation calls.
FLEET_4080_LEGACY_RADIATION = False


def _fleet_4080_non_pool_bytes(nz: int = 49) -> int:
    """The non-pool term a 4080 carries for the default suite at ``nz``."""

    profile = pf.DeviceLocalMemoryProfile(
        name="RTX 4080", multiprocessor_count=FLEET_4080_MULTIPROCESSORS,
        max_threads_per_multiprocessor=1536)
    # AS BUILT, not as it compiles today: every row in the table above was
    # measured on a binary whose widest frame was kf_column's 188 B/level.
    widest = KF_AS_BUILT_FRAME.frame_bytes(nz)
    return pf.CUDA_CONTEXT_BYTES + profile.reservation_bytes(widest)


def test_the_envelope_bounds_every_instrumented_run():
    """An envelope must never land under a measured peak.

    The x1.45 multiplier did, because it had no intercept: the smallest
    run in this table was declared 3.99 GiB and peaked at 4.38.  The same
    model over-predicted the largest by 30%.  One model, no intercept,
    read at two grid sizes -- which is also why a 5090 datapoint saying
    "19% under" and this card saying "25-30% over" were never in
    conflict.
    """
    non_pool = _fleet_4080_non_pool_bytes()
    worst_over = 0.0
    for label, domains, estimate_gib, measured_gib in (
            FLEET_4080_FORECAST_RUNS):
        # The rows recorded the estimate at the plan's 1.15; A163 prices
        # the same itemized subtotal at the measured margin, so each row
        # is re-based onto the margin that ships.
        subtotal = estimate_gib * GIB / pf.ALLOCATOR_HEADROOM
        envelope = pf.machine_peak_envelope_bytes(
            alloc_estimate_bytes=math.ceil(
                pf.FORECAST_POOL_HEADROOM * subtotal),
            non_pool_bytes=non_pool, domains=domains, family="linux",
            legacy_radiation=FLEET_4080_LEGACY_RADIATION)
        assert envelope >= measured_gib * GIB, label
        over = envelope / (measured_gib * GIB) - 1.0
        worst_over = max(worst_over, over)
    # Conservative, but not absurdly so: the old model reached +30% at
    # the top of this table while being optimistic at the bottom.
    assert worst_over < 0.30


def test_the_old_multiplier_is_optimistic_where_the_affine_form_is_not():
    """The negative control for the test above.

    This is the defect, executed: the retired multiplicative envelope
    lands UNDER the measured peak of the smallest run in the table, and
    the affine one does not.  If this ever stops failing for the old
    model, the evidence changed and the calibration needs re-deriving.
    """
    label, domains, estimate_gib, measured_gib = FLEET_4080_FORECAST_RUNS[1]
    footprint = int(estimate_gib * GIB)  # Linux: projection == estimate
    old = pf.observed_peak_envelope_bytes(footprint, platform="linux")
    assert old < measured_gib * GIB, (
        "the retired x1.45 envelope must still be the optimistic one "
        "this calibration replaced")
    new = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=footprint,
        non_pool_bytes=_fleet_4080_non_pool_bytes(), domains=domains,
        family="linux")
    assert new >= measured_gib * GIB


def test_the_envelope_has_an_intercept_that_does_not_scale_with_the_grid():
    """The structural property, not a number: doubling the estimate must
    NOT double the envelope, because part of it is a device constant."""

    non_pool = _fleet_4080_non_pool_bytes()
    small = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=2 * GIB, non_pool_bytes=non_pool,
        family="linux", legacy_radiation=FLEET_4080_LEGACY_RADIATION)
    large = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=4 * GIB, non_pool_bytes=non_pool,
        family="linux", legacy_radiation=FLEET_4080_LEGACY_RADIATION)
    assert large - small == 2 * GIB, "the pool side is 1:1"
    assert large < 2 * small, "an intercept is not a multiplier"
    # And the intercept is the thing this module already itemizes.
    assert small - 2 * GIB == non_pool + pf.ENVELOPE_UNMODELLED_BYTES


def test_the_pool_slack_term_is_the_measured_slack_not_the_multiplier():
    """The affine form + the measured pool-slack fraction.

    The retired multiplicative floor turned a 5.66 GiB footprint into a
    9.91 GiB envelope on the 3080 walk while the run measured 2.6 GiB;
    the calibrated term is proportional to the ESTIMATE (worst measured
    +0.30x, legacy-RRTMG pool retention) and the footprint projection is
    display-only.

    2026-08-20 (task 206): the term is no longer keyed to WDDM.  It was,
    and the Linux envelope consequently under-predicted every one of
    fifteen instrumented legacy-RRTMG forecasts on two Linux cards.  The
    boundary it really splits on is the RADIATION LANE -- measured on
    both driver models -- so the two families now price the same
    configuration identically and the lane is what moves the number.
    """

    legacy = dict(alloc_estimate_bytes=8 * GIB, non_pool_bytes=GIB,
                  legacy_radiation=True)
    modern = dict(alloc_estimate_bytes=8 * GIB, non_pool_bytes=GIB,
                  legacy_radiation=False)
    # A163 (2026-09-30) retired the slack term: it was a second margin on
    # the headroom the estimate's own margin prices, and the measured
    # margin (FORECAST_POOL_HEADROOM) bounds the legacy lane's battery
    # rows on its own (tests/test_memory_gate_a163.py).  The two lanes
    # now price one estimate identically on both driver models.
    for family in ("linux", "windows"):
        assert (pf.machine_peak_envelope_bytes(**legacy, family=family)
                == pf.machine_peak_envelope_bytes(**modern, family=family))
    assert (pf.machine_peak_envelope_bytes(**legacy, family="linux")
            == pf.machine_peak_envelope_bytes(**legacy, family="windows"))
    # The footprint projection no longer moves the envelope at all.
    assert pf.machine_peak_envelope_bytes(
        **legacy, footprint_projection_bytes=20 * GIB,
        family="windows") == pf.machine_peak_envelope_bytes(
        **legacy, footprint_projection_bytes=200 * GIB, family="windows")


def test_the_absent_card_profile_is_the_conservative_measured_reference():
    """Sizing a card that is NOT in this machine never discounts the
    intercept below what a real device measures.

    The 4090 stress run (2026-08-03) falsified the per-class SM discount:
    an absent-card sizing priced the non-pool intercept at 1.45 GiB (the
    12 GiB class's 70-SM row) where the same code on the real RTX 4090
    measured 2.30 GiB -- a config certified "fits with 0.27 GiB to
    spare" landed 0.015 GiB from the budget, a margin 18x smaller than
    advertised.  The class table was a market survey, not a measurement
    (a 12 GiB RTX 3080 Ti ships 80 SMs against the row's 70), so the
    absent-card path now prices against the conservative measured
    reference profile -- the max of known-device intercepts.
    """
    for gib in (12.0, 16.0, 24.0, 32.0, None, 999.0):
        assert (pf.card_local_memory_profile(gib)
                is pf.MEASURED_LOCAL_MEMORY_PROFILE), gib


@requires_4dom_inputs
def test_absent_card_sizing_is_never_more_optimistic_than_a_present_card(
        exp4):
    """THE stress-run inequality: for the same config, the absent-card
    estimate must be >= the present-card measurement, for every device
    this project has measured the law on."""
    known_devices = (
        pf.MEASURED_LOCAL_MEMORY_PROFILE,  # RTX 5090, 170 SMs
        pf.DeviceLocalMemoryProfile(       # the stress run's RTX 4090
            name="NVIDIA GeForce RTX 4090", multiprocessor_count=128,
            max_threads_per_multiprocessor=1536),
        pf.DeviceLocalMemoryProfile(       # the fleet's RTX 4080
            name="NVIDIA GeForce RTX 4080", multiprocessor_count=76,
            max_threads_per_multiprocessor=1536),
    )
    for gib in (12.0, 16.0, 24.0, 32.0):
        absent = pf.non_pool_device_bytes(
            exp4, profile=pf.card_local_memory_profile(gib))
        for device in known_devices:
            present = pf.non_pool_device_bytes(exp4, profile=device)
            assert absent >= present, (
                f"sizing an absent {gib:g} GiB card priced the non-pool "
                f"intercept at {absent / GIB:.2f} GiB, below the "
                f"{present / GIB:.2f} GiB the same config prices on a "
                f"present {device.name}")


@requires_grib1_bridge
@requires_4dom_inputs
def test_declared_budget_sizing_says_it_is_an_estimate_for_absent_hardware(
        capsys):
    """--budget-gib is the sizing-for-a-card-you-intend-to-buy path; its
    report must say the numbers are estimates for hardware not present."""
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "HARDWARE NOT PRESENT" in out
    assert "declared, not measured" in out

    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["sized_for_hardware_not_present"] is True
    assert payload["local_memory_profile"] == (
        pf.MEASURED_LOCAL_MEMORY_PROFILE.name)


@requires_4dom_inputs
def test_ingest_prices_every_domain_in_the_tree(exp1, exp4):
    """v1.4.0 priced this phase on the ROOT alone.

    So the prediction FELL as nests were added -- 5.46 -> 1.89 -> 1.30
    GiB across one, two and four domains -- while the machine measured it
    flat at 4.0-4.8, under by 3.4x at four domains and in the unsafe
    direction, on the very number the before-the-fetch gate half-relies
    on.  A deeper ladder has a smaller root; only the root was priced;
    so adding domains made the answer shrink.
    """
    one = pf.estimate_ingest(exp1, source="gfs")
    four = pf.estimate_ingest(exp4, source="gfs")

    assert one.nest_state_bytes == 0, "a single domain has no nests"
    assert one.nest_state_items == ()
    assert four.nest_state_bytes > 0
    assert len(four.nest_state_items) == len(exp4.domains) - 1

    # Every nest carries one complete initial state, and they are all
    # resident for the single export transaction.
    assert four.resident_bytes == (
        four.resident_times * four.per_time_bytes
        + four.forcing_table_bytes + four.nest_state_bytes)

    # The transient is charged against the WIDEST domain in the tree, not
    # the root: on a real ladder the widest domain is usually a nest.
    assert four.transient_basis_bytes >= four.per_time_bytes

    # THE DEFECT, as an inequality: the tree's ingest estimate must not
    # be reachable by pricing the root alone.  The setup term is the
    # tree's itemized one (A65), the widest build's, which bounds the
    # root's own from above, so this is the stronger form of the check.
    root_only = (math.ceil(four.headroom * (
        four.resident_times * four.per_time_bytes
        + four.forcing_table_bytes
        + four.transient_bytes))
        + four.context_bytes + four.device_overhead_bytes)
    assert four.peak_envelope_bytes > root_only


def test_adding_a_nest_never_lowers_the_ingest_estimate():
    """The falsifiable form of the same defect.

    Same root, one nest added: the answer must go UP.  Under the root-only
    model a two-domain tree whose root was smaller than the single-domain
    layout priced LOWER than the single domain, which is how the four-
    domain ladder came to declare 1.30 GiB and measure 4.39.
    """
    root = RunConfig(nx=240, ny=192, nz=49, dx=12000.0, dy=12000.0,
                     ztop=20000.0, dt=60.0, run_seconds=7200.0,
                     mp_physics=8, moist=True, terrain_opt=1,
                     spec_bdy_width=5, specified=True)
    alone = experiment_from_run_config(root, datetime(2026, 7, 30))
    one = pf.estimate_ingest(alone, source="gfs")

    child = dataclasses.replace(root, grid_id=2, nx=480, ny=384,
                                dx=3000.0, dy=3000.0, dt=15.0)
    from woof.experiment import DomainConfig, ExperimentConfig
    tree = dataclasses.replace(
        alone, domains=alone.domains + (DomainConfig(
            grid_id=2, parent_id=1, i_parent_start=31, j_parent_start=25,
            parent_grid_ratio=4, parent_time_step_ratio=4,
            history_interval_s=3600.0, run=child),))
    two = pf.estimate_ingest(tree, source="gfs")

    assert two.peak_envelope_bytes > one.peak_envelope_bytes
    # And the nest that was 4.9x the root's cells is visible by name.
    assert [grid for grid, _ in two.nest_state_items] == [2]
    assert two.nest_state_bytes > one.per_time_bytes


@requires_grib1_bridge
@requires_4dom_inputs
def test_a_negative_budget_is_clamped_and_explained(capsys):
    """A reserve larger than free VRAM leaves NO budget.

    ``budget = free - reserve`` is unbounded below, and a 4000x4000
    config drove it to -7.15 GiB, which the report then printed as a
    capacity to compare an envelope against.
    """
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "0.001",
                     "--vram-gib", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_bytes"] >= 0, "a capacity is never negative"
    assert payload["budget_underwater_bytes"] > 0
    assert rc != 0

    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "0.001",
                "--vram-gib", "1"])
    out = capsys.readouterr().out
    assert "NO BUDGET AT ALL" in out
    assert "the reserve alone is" in out
    assert " -" not in out.split("NO BUDGET AT ALL")[1].split("\n")[0]


@requires_grib1_bridge
@requires_4dom_inputs
def test_the_over_budget_remedy_is_an_action_not_a_design_pointer(capsys):
    """It used to end "staged residency (DESIGN REOPEN) per section E".

    No pip user has a section E, and the sentence names nothing to do.
    The actionable form already existed one layer up, in `woof go`'s
    refusal, and is reused here.
    """
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "OVER BUDGET" in out
    assert "DESIGN REOPEN" not in out
    assert "section E" not in out
    # And the action is REACHABLE: the 3080 walk followed the previous
    # `woof domain --vram-gib <free>` remedy and was refused at every
    # grid size, because the flag names a card and the number fed to it
    # was a free-VRAM figure.  The bare wizard measures the card itself.
    assert "remedy: re-size against this machine -- woof domain" in out
    assert "--vram-gib" not in out.split("remedy:")[1].split("\n")[0]


@requires_grib1_bridge
@requires_4dom_inputs
def test_the_printed_exit_code_is_the_one_the_process_returns(capsys):
    """The WARNING used to assert "(exit code 4: gates passed)" even when
    a gate had just failed and the process therefore exited 1."""

    # Envelope over, gates over too: the harder verdict, announced.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                     "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "exit code 1: a gate above FAILED as well" in out
    assert "exit code 4: gates passed" not in out

    # Envelope over, every gate passed: 4, and it says 4.
    payload_rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib",
                             "100", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload_rc == 0
    tight = f"{payload['alloc_estimate_bytes'] / GIB + 0.5:.2f}"
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert rc == 4, out
    assert "exit code 4: gates passed, envelope did not" in out


@requires_grib1_bridge
@requires_4dom_inputs
def test_the_budget_word_follows_the_platform(capsys, monkeypatch):
    """"WDDM budget" on a Linux box, in the same report that has just
    finished explaining there is no WDDM here."""

    monkeypatch.setattr(pf, "host_platform", lambda: "linux")
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "WARNING: observed peak envelope" in out
    assert "exceeds the WDDM budget" not in out

    monkeypatch.setattr(pf, "host_platform", lambda: "win32")
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "exceeds the WDDM budget" in out


@requires_4dom_inputs
def test_the_small_windows_tier_is_retired_into_the_measured_model():
    """The experimental tier's own advisory asked for exactly one thing:
    a measured peak from a small Windows card.  The 2026-08-19 3080
    calibration delivered it, so the tier is gone -- every Windows card
    takes the ONE measured model, and the envelope intercept is the
    itemized non-pool residency on every platform.  Card size changes
    the numbers (through the profile and the estimate), never the
    formula: that split is how the wizard and `woof check` returned
    opposite verdicts on the same bytes (task #162).
    """
    exp = load_experiment_case(CONFIG_4DOM)[0]
    small = pf.estimate_experiment(exp, vram_gib=12.0)
    large = pf.estimate_experiment(exp, vram_gib=32.0)
    unsized = pf.estimate_experiment(exp)
    assert small.envelope_family == large.envelope_family \
        == unsized.envelope_family
    for est in (small, large, unsized):
        assert est.envelope_intercept_bytes == est.non_pool_device_bytes
    # Same profile pricing => same envelope, sized or not: one envelope.
    assert small.peak_envelope_bytes == large.peak_envelope_bytes \
        == unsized.peak_envelope_bytes


def test_the_gate_names_printed_on_linux_do_not_say_wddm(monkeypatch):
    """A-6.  WDDM is a Windows display driver model, and these three gate
    names are the first `woof check` output a new user reads.  The prose
    beside them has been platform-correct since envelope_platform was
    introduced; only the names were left behind."""
    from woof.core import preflight

    monkeypatch.setattr(preflight.sys, "platform", "linux")
    shown = [preflight.gate_display_name(m)
             for m in preflight.N0_GATE_METRICS]
    assert shown == ["alloc_fits_vram_budget", "alloc_measured_le_estimate",
                     "alloc_estimate_le_vram_budget"]
    assert not any("wddm" in name for name in shown)

    monkeypatch.setattr(preflight.sys, "platform", "win32")
    assert [preflight.gate_display_name(m)
            for m in preflight.N0_GATE_METRICS] == list(
                preflight.N0_GATE_METRICS)


def test_the_gate_display_name_never_moves_the_receipt_key(monkeypatch):
    """The negative control, and the reason this is a DISPLAY label.

    These strings are pre-registered N0 ledger record names read by
    woof/verify/nest_gates.py and written into certification receipts.
    Renaming them per host would break every receipt written on one
    platform and read on another, so the tuple itself must not move.
    """
    from woof.core import preflight
    from woof.verify import nest_gates

    for platform in ("linux", "win32", "darwin", "freebsd13"):
        monkeypatch.setattr(preflight.sys, "platform", platform)
        assert preflight.N0_GATE_METRICS == (
            "alloc_fits_wddm_budget", "alloc_measured_le_estimate",
            "alloc_estimate_le_wddm_budget")
        # and the evaluated gate dict is still keyed by the record names
        gates = preflight.evaluate_alloc_gates(
            estimate_bytes=1, measured_used_bytes=1,
            measured_free_bytes=1 << 40,
            reserve=preflight.ReservePolicy(
                retention_residual_bytes=0, device_overhead_bytes=0))
        assert tuple(gates) == preflight.N0_GATE_METRICS

    # the verifier reads the keys, not the labels
    source = Path(nest_gates.__file__).read_text(encoding="utf-8")
    assert "alloc_fits_wddm_budget" in source
    assert "alloc_fits_vram_budget" not in source


# ---------------------------------------------------------------------------
# The subprocess device probe: the memory gate's only device access
# ---------------------------------------------------------------------------

class _CompletedProbe:
    """The slice of CompletedProcess the probe reads."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def test_the_device_probe_asks_everything_in_a_new_interpreter(monkeypatch):
    """``memGetInfo``/``deviceGetLimit`` stand up a CUDA primary context
    wherever they are asked, so the probe must ask them in a NEW
    interpreter whose context dies with it -- a long-lived caller (the
    ``woof go`` orchestrator, which outlives its gate as a progress
    printer) then holds no device memory at all."""
    import sys as _sys

    # The never-touch-the-local-device switch stops the probe before the
    # run seam (tests/test_no_local_gpu_contract.py owns that contract);
    # this test is about what the spawned interpreter asks.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    seen = {}

    def _runner(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        payload = {"free_bytes": 7 * GIB, "total_bytes": 8 * GIB,
                   "profile": {"name": "probe card",
                               "multiprocessor_count": 84,
                               "max_threads_per_multiprocessor": 1536,
                               "default_stack_limit_bytes": 1024}}
        return _CompletedProbe(stdout=json.dumps(payload) + "\n")

    payload = pf.device_memory_probe_subprocess(run=_runner)
    assert seen["command"][0] == _sys.executable
    assert seen["command"][1] == "-c"
    source = seen["command"][2]
    for question in ("memGetInfo", "getDeviceProperties", "deviceGetLimit"):
        assert question in source
    compile(source, "<device probe>", "exec")  # the source must parse
    assert (seen["kwargs"]["timeout"]
            == pf.DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS)
    assert payload["free_bytes"] == 7 * GIB

    profile = pf.profile_from_device_probe(payload)
    assert profile is not None
    assert profile.name == "probe card"
    assert profile.multiprocessor_count == 84
    assert profile.max_threads_per_multiprocessor == 1536
    assert profile.default_stack_limit_bytes == 1024
    assert profile.resident_thread_capacity == 84 * 1536


def test_a_probe_that_cannot_answer_reads_as_no_device():
    """Every way the probe can fail is 'no card here', never a throw:
    the gate must never refuse on a card it could not see."""
    import subprocess

    for outcome in (
            _CompletedProbe(returncode=3),           # no cupy / no device
            _CompletedProbe(stdout="not json"),
            _CompletedProbe(stdout=""),
            _CompletedProbe(stdout=json.dumps({"free_bytes": "many"})),
            _CompletedProbe(stdout=json.dumps({"free_bytes": True})),
            _CompletedProbe(stdout=json.dumps(["free_bytes"]))):
        assert pf.device_memory_probe_subprocess(
            run=lambda *_a, _o=outcome, **_k: _o) is None

    def _timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

    assert pf.device_memory_probe_subprocess(run=_timeout) is None

    def _unlaunchable(command, **kwargs):
        raise OSError("no interpreter")

    assert pf.device_memory_probe_subprocess(run=_unlaunchable) is None

    # ...and the profile half is as defensive as the free half.
    assert pf.profile_from_device_probe(None) is None
    assert pf.profile_from_device_probe({"free_bytes": 1}) is None
    assert pf.profile_from_device_probe({"profile": "a 5090"}) is None
    assert pf.profile_from_device_probe(
        {"profile": {"name": "half a card"}}) is None


#: The stand-in cgroup, meminfo and membership files every host-memory
#: reader is held to: these and ``rw_host_memory`` (the renderer's
#: ``rusty_weather::host_memory`` and the MPAS static builder's limit).
_HOST_MEMORY_CASES = json.loads(
    (ROOT / "tools" / "rustwx" / "crates" / "rw-host-memory" / "src"
     / "host_memory_cgroup_cases.json").read_text(encoding="utf-8"))["cases"]


def _host_memory_case(tmp_path, monkeypatch, case):
    """Stage one shared host-memory table row for either reader."""
    import sys

    from tilestream import autoplan

    root = tmp_path / "cgroup"
    root.mkdir()
    for relative, text in case["cgroup"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    meminfo = tmp_path / "meminfo"
    if case["meminfo"] is not None:
        meminfo.write_bytes(case["meminfo"].encode("ascii"))
    membership = tmp_path / "proc-self-cgroup"
    if case["proc_self_cgroup"] is not None:
        membership.write_bytes(case["proc_self_cgroup"].encode("ascii"))
    monkeypatch.setattr(autoplan, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(autoplan, "_PROC_SELF_CGROUP", str(membership))
    monkeypatch.setattr(autoplan, "_PROC_MEMINFO", str(meminfo))
    # Exercise the procfs route used inside a container.
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.mark.parametrize("case", _HOST_MEMORY_CASES,
                         ids=[case["name"] for case in _HOST_MEMORY_CASES])
def test_host_available_bytes_is_capped_by_the_memory_cgroup_it_runs_in(
        tmp_path, monkeypatch, case):
    """THE BREAKAGE: MemAvailable inside a container is the host's, and the
    cap read only the mount root's limit, never what the process had used
    under it nor a limit on its own scope or slice.  A render planned
    against that figure ran past the limit and was killed.  The renderer's
    reader answers the same table (cargo test -p rusty-weather)."""

    from tilestream import autoplan

    _host_memory_case(tmp_path, monkeypatch, case)

    assert autoplan._cgroup_memory_headroom() == case["headroom"]
    assert pf.host_available_bytes() == case["available"]


@pytest.mark.parametrize("case", _HOST_MEMORY_CASES,
                         ids=[case["name"] for case in _HOST_MEMORY_CASES])
def test_the_host_total_is_the_smallest_limit_on_the_process_cgroup_path(
        tmp_path, monkeypatch, case):
    """THE BREAKAGE: the planner's limit read only the cgroup mount root's
    ``memory.max``.  In a systemd scope with ``MemoryMax=2G`` on a 30 GiB
    worker the root carries no limit, so ``Machine.detect``'s host RAM and
    ``streaming._host_total_bytes`` read the whole host's 32.8 GB, the
    pinned host store was sized to it, and the kernel killed the run.  The
    limit is the smallest on the path from the process's own cgroup up to
    the mount, the table the renderer's and the MPAS builder's readers
    answer (cargo test -p rw-host-memory, -p rw-mpas)."""
    import re

    from woof.core import streaming
    from tilestream import autoplan

    assert "limit" in case, "the shared table's case names no limit"
    _host_memory_case(tmp_path, monkeypatch, case)

    assert autoplan._cgroup_memory_limit() == case["limit"]
    memtotal = int(re.search(r"^MemTotal:\s+(\d+) kB$", case["meminfo"],
                             re.MULTILINE).group(1)) * 1024
    expected = memtotal if case["limit"] is None else min(case["limit"], memtotal)
    assert streaming._host_total_bytes() == expected


def test_shinhong_workspace_pricing_tracks_the_tile_and_levels():
    """The workspace must be charged when scheme 11 is selected."""
    profile = pf.DeviceLocalMemoryProfile(
        name="workspace-test", multiprocessor_count=2,
        max_threads_per_multiprocessor=1536)
    cfg = RunConfig(**dict(_TINY, nx=257, ny=10, nz=50), bl_pbl_physics=11)
    exp = experiment_from_run_config(cfg, datetime(2000, 1, 1))
    assert pf.shinhong_column_workspace_bytes(exp, profile=profile) == (
        2 * 16 * 32 * 50 * 52 * 4)
    assert pf.column_workspace_bytes(exp, profile=profile) == (
        pf.gf_column_workspace_bytes(exp, profile=profile)
        + pf.kf_column_workspace_bytes(exp, profile=profile)
        + pf.ntiedtke_column_workspace_bytes(exp, profile=profile)
        + 2 * 16 * 32 * 50 * 52 * 4)
    exp = experiment_from_run_config(
        dataclasses.replace(cfg, bl_pbl_physics=1), datetime(2000, 1, 1))
    assert pf.shinhong_column_workspace_bytes(exp, profile=profile) == 0

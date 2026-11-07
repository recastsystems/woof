"""CPU-only WDM6 (mp_physics=16) integration and coefficient tests.

The device half is ``tests/test_wdm6_column.py``.  The comparison against
WRF's own ``module_mp_wdm6.F`` lives in ``tests/test_wdm6_wrf461_oracle.py``
(the column oracle tools/wdm6_wrf461_oracle).
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig, load_config, validate_run_config
from woof.core import preflight
from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
from woof.core.physics import (microphysics_scheme_sr_available,
                                microphysics_scratch_slots)
from woof.core.wdm6_constants import (WDM6_NUMBER_SPECIES, coefficients,
                                       rimed_ice_constants)
from woof.io.restart import MICROPHYSICS_ALGORITHM_IDENTITIES


_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)
_KERNEL = Path(__file__).resolve().parents[1] / "woof/core/kernels/wdm6.cu"


def _cfg(**changes):
    return RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=16,
                     **changes)


# --------------------------------------------------------------------------
# Admission and refusal
# --------------------------------------------------------------------------

def test_config_accepts_wdm6_and_validates_its_two_knobs(tmp_path):
    path = tmp_path / "wdm6.toml"
    path.write_text(
        "[grid]\n"
        "nx=8\nny=6\nnz=4\ndx=1000.0\ndy=1000.0\nztop=10000.0\n"
        "[dynamics]\ndt=1.0\nmoist=true\nmoist_cq=true\nmp_physics=16\n"
        "wdm6_hail_opt=1\nwdm6_ccn_conc=5.0e8\n"
        "[run]\nrun_seconds=10.0\n",
        encoding="utf-8")
    cfg = load_config(path)
    assert cfg.mp_physics == 16
    assert cfg.wdm6_hail_opt == 1
    assert cfg.wdm6_ccn_conc == 5.0e8
    with pytest.raises(ValueError, match="wdm6_hail_opt"):
        validate_run_config(replace(cfg, wdm6_hail_opt=2))
    # WDM6's own clamp (module_mp_wdm6.F:585) would discard anything
    # outside [1e8, 2e10] on the first minor loop, so config refuses it.
    with pytest.raises(ValueError, match="1e8, 2e10"):
        validate_run_config(replace(cfg, wdm6_ccn_conc=1.0e6))
    with pytest.raises(ValueError, match="1e8, 2e10"):
        validate_run_config(replace(cfg, wdm6_ccn_conc=1.0e11))


@pytest.mark.parametrize("value", [14, 26])
def test_wdm_siblings_have_their_own_admitted_selector(value):
    validate_run_config(RunConfig(**_TINY, moist=True, moist_cq=True,
                                  mp_physics=value))


def test_wdm6_requires_moisture_like_every_other_scheme():
    with pytest.raises(ValueError, match="requires moist=true"):
        validate_run_config(RunConfig(**_TINY, moist=False, mp_physics=16))


@pytest.mark.parametrize(("field", "value"),
                         [("wdm6_hail_opt", 1), ("wdm6_ccn_conc", 5.0e8)])
def test_a_wdm6_knob_under_another_scheme_is_refused_not_ignored(field, value):
    """Nothing outside the mp=16 path reads either field, so a set value
    under another scheme is a request woof would silently drop."""
    cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=6,
                    **{field: value})
    with pytest.raises(ValueError, match=f"{field} is a WDM6 selector"):
        validate_run_config(cfg)


def test_the_wdm6_knobs_leave_the_identity_of_every_non_wdm6_run():
    """Adding the pair moved no pre-WDM6 fingerprint, and that is enforced.

    ``restart_identity_payload`` drops a scheme-scoped field from the
    identity of every domain that does not select its scheme -- the
    absent-stays-absent convention ``perturbation`` and per-domain
    ``spawn`` already use.  Two things are checked here, because only the
    pair of them is the property: the keys are ABSENT for a non-WDM6
    domain (so every experiment written before this lane hashes exactly as
    it did), and they are PRESENT AND BOUND for a WDM6 one (so a WDM6
    checkpoint cannot resume under a different CCN initial condition or a
    different rimed-ice mode).
    """
    from woof.core.model import (SCHEME_SCOPED_RUN_FIELDS,
                                  restart_identity_payload)
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    assert SCHEME_SCOPED_RUN_FIELDS[16] == ("wdm6_hail_opt", "wdm6_ccn_conc")
    scaffold = load_scaffold()

    def payload(run):
        exp = replace(scaffold, domains=(
            replace(scaffold.domains[0], run=run),))
        return restart_identity_payload(exp)["domains"][0]["run"]

    wsm6 = payload(RunConfig(**_TINY, moist=True, moist_cq=True,
                             mp_physics=6))
    for name in SCHEME_SCOPED_RUN_FIELDS[16]:
        assert name not in wsm6, name

    wdm6 = payload(_cfg(wdm6_hail_opt=1, wdm6_ccn_conc=5.0e8))
    assert wdm6["wdm6_hail_opt"] == 1
    assert wdm6["wdm6_ccn_conc"] == 5.0e8


# --------------------------------------------------------------------------
# wdm6init coefficients, and the FP32 literals the kernel bakes from them
# --------------------------------------------------------------------------

def test_wdm6init_reuses_wsm6_rimed_ice_rather_than_copying_it():
    """wdm6init's hail_opt arm (:2096-2108) sets the same five constants
    mp_wsm6_init does, so the port SHARES the host function."""
    for hail_opt in (0, 1):
        assert coefficients(hail_opt).rimed == rimed_ice_constants(hail_opt)
    assert coefficients(0).rimed.n0g == 4.0e6
    assert coefficients(0).rimed.deng == 500.0
    assert coefficients(1).rimed.n0g == 4.0e4
    assert coefficients(1).rimed.deng == 700.0


def test_wdm6_double_moment_coefficients_follow_the_fortran():
    c = coefficients(0)
    pi = 4.0 * math.atan(1.0)
    # :2143 pidnr = 4*pi*denr -- the prognostic rain intercept's cube-root
    # denominator, replacing WSM6's fixed-N0r fourth root.
    assert c.pidnr == pytest.approx(4.0 * pi * 1000.0, rel=1e-15)
    # :2116 pidnc = pi*denr/6, the cloud droplet slope.
    assert c.pidnc == pytest.approx(pi * 1000.0 / 6.0, rel=1e-15)
    # :2136/:2137 the mass- and number-weighted rain fall speeds.  They are
    # DIFFERENT numbers; a port that used one for both would sediment rain
    # number at the mass speed and drift the mean drop size every step.
    #
    # The reference is WRF's OWN rgmma, not math.gamma.  rgmma truncates the
    # Weierstrass product at 10000 terms, whose error is about x**2/(2N) --
    # 0.17 per cent at x = 5.8, which is far above any float32 tolerance.
    # Comparing against the true Gamma here would fail, and "fixing" the
    # port to match the true Gamma would be a divergence from WRF, so the
    # truncation is pinned rather than papered over.
    from woof.core.wsm6_constants import _rgmma

    assert c.pvtr != c.pvtrn
    assert c.pvtr == pytest.approx(841.9 * _rgmma(5.8) / 24.0, rel=1e-13)
    assert c.pvtrn == pytest.approx(841.9 * _rgmma(2.8), rel=1e-13)
    assert abs(c.pvtr / (841.9 * math.gamma(5.8) / 24.0) - 1.0) < 3.0e-3
    # :2113/:2114 the maritime and continental autoconversion thresholds
    # differ by exactly the xncr0/xncr1 ratio.
    assert c.qc1 / c.qc0 == pytest.approx(10.0, rel=1e-12)
    # :2140-2141 rain evaporation carries NO n0r factor; nr multiplies at
    # the call site.
    assert c.precr1 == pytest.approx(2.0 * pi * 1.56, rel=1e-15)


def test_the_kernel_bakes_wrf_s_words_not_the_float64_products():
    """The kernel's coefficients are wdm6init's own float32 words
    (woof.core.wdm6_constants.WDM6_WRF_WORDS); the float64 derivation is
    close to them but is NOT them, and baking it would move every fall
    speed off WRF's word (tests/test_wdm6_wrf461_oracle.py)."""
    from woof.core.wdm6_constants import WDM6_WRF_WORDS, wrf_word

    source = _KERNEL.read_text(encoding="utf-8")
    c = coefficients(0)
    for name in ("pvtr", "pvtrn", "pvts", "pidnr", "pidnc", "precr1",
                 "precr2", "precs2", "pacrc", "g4pbr", "g7pbr", "cdm2"):
        word = np.float32(wrf_word(WDM6_WRF_WORDS[name]))
        assert word == pytest.approx(getattr(c, name), rel=1.0e-5), name
        assert f"#define WDM6_{name.upper()} " in source, name
    # At least the rgmma-derived fall speeds really differ from float64.
    assert np.float32(wrf_word(WDM6_WRF_WORDS["pvtr"])) != np.float32(c.pvtr)


def test_the_kernel_source_names_its_oracle():
    """ACCURACY PIN.  The header names the column oracle that grades it,
    and no longer claims there is none."""
    source = _KERNEL.read_text(encoding="utf-8")
    assert "tools/wdm6_wrf461_oracle" in source
    assert "no oracle comparison" not in source


# --------------------------------------------------------------------------
# State, preflight and driver seams
# --------------------------------------------------------------------------

def test_wdm6_state_carries_three_number_moments_and_the_ccn_fill(
        monkeypatch):
    import woof.core.state as state_module

    monkeypatch.setattr(state_module, "cp", np)
    cfg = _cfg(wdm6_ccn_conc=3.0e8)
    state = state_module.DomainState(cfg)
    for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0"):
        assert getattr(state, name).shape == (4, 6, 8)
    for name in WDM6_NUMBER_SPECIES:
        assert getattr(state, name).shape == (4, 6, 8)
        assert getattr(state, name + "0").shape == (4, 6, 8)
    # module_mp_wdm6.F:220-227: the CCN reservoir starts at ccn_conc, the
    # droplet and rain numbers at WRF's allocator zero.
    np.testing.assert_array_equal(state.nn, np.float32(3.0e8))
    np.testing.assert_array_equal(state.nc, np.float32(0.0))
    np.testing.assert_array_equal(state.nr, np.float32(0.0))
    # Not Morrison's set, and not Thompson's.
    for name in ("ni", "ns", "ng", "effr", "nwfa", "nifa", "qnn"):
        assert getattr(state, name, None) is None
    np.testing.assert_array_equal(state.effc, np.float32(2.49))
    np.testing.assert_array_equal(state.effi, np.float32(4.99))
    np.testing.assert_array_equal(state.effs, np.float32(9.99))


def test_wdm6_preflight_prices_its_state_scratch_and_kernel():
    cfg = _cfg(km_opt=4, bl_pbl_physics=1)
    shapes = preflight.state_array_shapes(cfg)
    assert {"qi", "qs", "qg", "effc", "effi", "effs"} <= shapes.keys()
    assert set(WDM6_NUMBER_SPECIES) <= shapes.keys()
    assert {n + "0" for n in WDM6_NUMBER_SPECIES} <= shapes.keys()

    scratch = preflight.scratch_slot_registry(cfg)
    for name in ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
                 "wdm6_z8w", "mp_rainnc", "mp_snownc", "mp_graupelnc",
                 "mp_sr", "refl_t", "refl_10cm"):
        assert name in scratch, name
    # mp=16 never loads the WSM6 translation unit.
    assert "wsm6_theta" not in scratch
    assert preflight._MICROPHYSICS_KERNEL_MODULES[16] == (
        "wdm6", "microphysics_validation")
    assert 16 in preflight._REFLECTIVITY_MICROPHYSICS


def test_wdm6_reflectivity_is_priced_from_its_own_translation_unit():
    """refl.cu is byte-frozen, so WDM6's reflectivity kernel lives in
    wdm6_refl.cu -- and the rail must reserve THAT frame, not refl.cu's."""
    from woof.experiment import ExperimentConfig  # noqa: F401  (shape doc)

    spec = preflight.LEVEL_SPECIALIZED_KERNEL_FRAMES["wdm6_refl"]
    assert spec.define == "REFL_KMAX"
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6_refl"] == 16128
    # Driver-measured rows (RTX 5090); tests/test_wdm6_column.py re-measures
    # them against the real kernel.
    for levels, frame in ((256, 16128), (128, 8064), (64, 4032),
                          (49, 3088), (30, 1904), (10, 640)):
        assert spec.frame_bytes(levels) == frame, levels


def test_the_wdm6_refl_prologue_is_byte_identical_to_refl_cu_s():
    """wdm6_refl.cu copies refl.cu's device prologue because refl.cu may
    not be appended to.  Copied code that nobody checks is code that
    drifts, so the copy is pinned to the original here."""
    root = Path(__file__).resolve().parents[1] / "woof/core/kernels"
    refl = (root / "refl.cu").read_text(encoding="utf-8")
    wdm6_refl = (root / "wdm6_refl.cu").read_text(encoding="utf-8")
    start = refl.index("// Compile-time bound")
    end = refl.index('extern "C" __global__ void refl10cm_morrison_column')
    prologue = refl[start:end].rstrip()
    assert prologue in wdm6_refl, (
        "wdm6_refl.cu's copy of refl.cu's device prologue has drifted; the "
        "two translation units must agree on the Blahak backscatter and "
        "the radar constants")


def test_wdm6_scratch_lifetime_audit_is_closed_world():
    slots = preflight.scratch_slot_registry(
        _cfg(km_opt=4, diff_6th_opt=2, specified=True), n_lbc_intervals=2)
    assert slots
    assert {slot for slot in slots
            if preflight.scratch_slot_lifetime(slot) is None} == set()
    for slot in ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
                 "wdm6_z8w"):
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "write_before_read"
        assert preflight.scratch_slot_uses_arena(slot)
    for slot in ("mp_rainnc", "mp_sr", "mp_graupelncv"):
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "carrying"
        assert not preflight.scratch_slot_uses_arena(slot)


def test_wdm6_driver_seams_treat_it_as_a_full_ice_scheme():
    assert microphysics_scheme_sr_available(16) is True
    assert dict(microphysics_scratch_slots(16)) == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv",
        "graupelnc": "mp_graupelnc", "graupelncv": "mp_graupelncv",
    }
    assert kf_phase_mode_for_microphysics(16) == \
        KFPhaseMode.SEPARATE_ICE_SNOW
    # v4 (A144): the rain condensation cap keeps the rate's sign, so rain
    # with no number no longer evaporates; checkpoints from v3 must refuse.
    # v5 (2.8.8): the shared finish clamps at WRF's REAL product
    # mp_tend_lim*dt.  v6: WRF's own coefficient words and operation order
    # (the column oracle); checkpoints from v5 must refuse.
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[16].startswith(
        "wdm6-double-moment-warm-rain-wrf-v4.6.1-v6")
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[16].endswith(
        "conservative-rain-interface-flux-bounded-transport-time-"
        "zero-rate-rain-no-evaporation-real-clamp-product-wrf-words")


def test_the_ring_guard_captures_wdm6_s_number_moments():
    """Rain that fully evaporates returns its number to the CCN reservoir
    (:1249-1252), so nn is WRITTEN and an unguarded specified-zone ring
    would keep the change."""
    from woof.core.microphysics import spec_zone_ring_save_slots

    slots = spec_zone_ring_save_slots(_cfg(specified=True, spec_zone=2))
    for name in WDM6_NUMBER_SPECIES:
        assert any(key.startswith(f"mp_ring_save_{name}_")
                   for key in slots), name


def test_wdm6_microphysics_dispatch_is_lazy_and_forwards_diagflag(
        monkeypatch):
    import woof.core.microphysics as dispatcher
    import woof.core.wdm6 as wdm6

    sentinel = object()

    def fake_apply(state, cfg, dt, *, refl_10cm_due=False):
        assert cfg.mp_physics == 16
        assert dt == 7.5
        assert refl_10cm_due is True
        return sentinel

    monkeypatch.setattr(wdm6, "apply", fake_apply)
    state = SimpleNamespace(qv=object())
    assert dispatcher.apply(state, _cfg(), 7.5,
                            refl_10cm_due=True) is sentinel


def test_wdm6_vertical_kernel_tiers_cover_80_without_enlarging_z49():
    from woof.core.wdm6 import _kernel_capacity

    assert _kernel_capacity(49) == 64
    assert _kernel_capacity(64) == 64
    assert _kernel_capacity(65) == 80
    assert _kernel_capacity(80) == 80
    with pytest.raises(ValueError, match="2 <= nz <= 80"):
        _kernel_capacity(81)


def test_the_adapter_refuses_a_state_with_no_land_mask():
    from woof.core.wdm6 import column_land_mask

    for driver in (None, SimpleNamespace(fields={}),
                   SimpleNamespace(fields={"xland": None})):
        with pytest.raises(ValueError, match="XLAND"):
            column_land_mask(SimpleNamespace(physics=driver))


# --------------------------------------------------------------------------
# Reflectivity host products
# --------------------------------------------------------------------------

def test_radar_init_wdm6_moves_only_the_rain_moments_to_mu_1():
    """wdm6init sets xmu_r = 1 and leaves snow/graupel at 0 (:2198-2206)."""
    from woof.core.refl import radar_init, radar_init_wdm6

    base = radar_init(0)
    for hail_opt in (0, 1):
        rc = radar_init_wdm6(hail_opt)
        assert rc.xcre == (4.0, 2.0, 5.0, 8.0)
        assert rc.xcrg[2] == pytest.approx(24.0, rel=1e-13)
        assert rc.xcrg[3] == pytest.approx(5040.0, rel=1e-13)
        assert rc.xorg2 == pytest.approx(1.0, rel=1e-13)
        # Snow and graupel keep the mu = 0 tuples verbatim.
        assert rc.xcse == base.xcse
        assert rc.xcge == base.xcge
        # Rain density is WSM6's 1000, not Morrison's 997.
        assert rc.xam_r == pytest.approx(1000.0 * math.pi / 6.0, rel=1e-13)
        assert rc.xam_g == pytest.approx(
            rimed_ice_constants(hail_opt).deng * math.pi / 6.0, rel=1e-13)


# --------------------------------------------------------------------------
# Namelist importer and registry
# --------------------------------------------------------------------------

def test_a_namelist_naming_wdm6_imports_natively(tmp_path):
    """CROSS-LANE FIXTURE: the WRF namelist pair comes from the importer's
    own suite, so this test cannot pass against a namelist WDM6 invented
    for itself."""
    from test_namelist_import import _generic_hierarchy_pair

    from woof.namelist_import import import_namelists

    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    text = inp.read_text(encoding="utf-8")
    assert " mp_physics = 6,\n" in text
    text = text.replace(
        " mp_physics = 6,\n",
        " mp_physics = 16,\n hail_opt = 1,\n ccn_conc = 5.0E8,\n")
    inp.write_text(text, encoding="utf-8")

    toml_text, report = import_namelists(wps, inp, name="wdm6-import")
    assert "mp_physics = 16" in toml_text
    assert "wdm6_hail_opt = 1" in toml_text
    assert "wdm6_ccn_conc = 500000000" in toml_text.replace(".0", "")
    # A TRANSLATION, not a substitution: nothing was swapped for WDM6.
    assert not any(sub.key == "mp_physics"
                   for sub in report.substitutions)


def test_the_registry_row_states_the_oracle_and_no_more():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "woof/physics_registry_v2.json").read_text(
            encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"]["wdm6-mp16"]
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    # RETIRED with the defect it guarded (audit R-067): this pinned
    # ``component-override``, i.e. "the only way here is a hand-written
    # tuple".  WDM6 has a named suite now and the state is COMPUTED from
    # the declared template lists, so the assertion is the new fact.
    assert option["reachability"]["state"] == "template"
    assert option["selectors"] == {"mp_physics": 16}
    assert option["scientific_evidence"] == "none"
    first = option["warnings"][0]
    assert first.startswith(
        "WDM6's distance from WRF v4.6.1 is MEASURED by a column oracle")
    assert "tools/wdm6_wrf461_oracle" in first
    assert "tests/test_wdm6_wrf461_oracle.py" in first
    # Conformance is not validation: the row still says what is missing.
    assert "no woof/WRF forecast trajectory comparison exists" in first
    for forbidden in ("model-validated", "validation-candidate",
                      "wrf-matched-run"):
        assert forbidden not in first, forbidden
    # RETIRED with the same defect, and replaced by what the absence was
    # standing in for.  A template DOES register mp=16 now -- that is the
    # front door R-067 minted -- and the property that still matters is
    # that a suite reaching an unjudged scheme cannot outrank it and
    # cannot become anybody's default.
    suites = [
        template_id
        for template_id, template in registry["templates"].items()
        if template.get("components", {}).get("microphysics") == "wdm6-mp16"
    ]
    assert suites, "an implemented scheme with no named suite has no door"
    for template_id in suites:
        assert registry["templates"][template_id]["maturity"] == (
            "implemented-unverified"), template_id
    for route in registry["runner_routes"].values():
        for declared in (route.get("source_template_ids") or {}).values():
            assert not declared or declared[0] not in suites, declared


# --------------------------------------------------------------------------
# The mixed nest edge, and its offline mirror
# --------------------------------------------------------------------------

def test_a_mixed_nest_edge_touching_wdm6_seeds_wdm6_and_not_thompson():
    """The edge mp=16 advertises must actually run, with WDM6's own values.

    History: 16 was added to ``UNVALIDATED_MIXED_EDGE_SELECTORS`` and to no
    other table, so the resolver raised a bare ``KeyError(16)`` while two
    other surfaces advertised a named refusal.  That refusal is now retired
    (audit R-004) -- its premise was that seeding the CCN reservoir from the
    domain's ``ccn_conc`` was unmeasured, while the cold start and the
    lateral inflow face do exactly that on every WDM6 run -- so what this
    test protects is the same property one level up: a WDM6 edge is closed
    with WDM6's own values and nobody else's, and the receipt says so.
    """
    from woof.core import microphysics_transition as mt

    for source, target in ((16, 6), (6, 16), (16, 28), (28, 16)):
        parent = SimpleNamespace(mp_physics=source, moist=True, moist_cq=True,
                                 wdm6_hail_opt=0, wsm6_hail_opt=0,
                                 morr_rimed_ice=1, wdm6_ccn_conc=1.0e8)
        child = SimpleNamespace(
            mp_physics=target, moist=True, moist_cq=True,
            wdm6_hail_opt=0, wsm6_hail_opt=0, morr_rimed_ice=1,
            wdm6_ccn_conc=1.0e8,
            nest_microphysics_transition=mt.EDGE_MATRIX_POLICY)
        contract = mt.resolve_microphysics_transition(parent, child)
        assert contract.mixed is True
        note = mt.mixed_edge_entry_note(contract)
        if target not in (16, 28):
            # Leaving WDM6 needs no closure: the target's moments come
            # from target mass and the reservoir is dropped, so plan
            # review has nothing to name.
            assert note is None
        else:
            assert len(note.split(". ")) <= 2
        rows = {row["target_field"]: row
                for row in contract.species_actions()
                if row["action"] == "diagnosed"}
        if target == 16:
            # WDM6's own triple, out of WDM6's own table.
            assert rows["nn"]["seeded_value"] == 1.0e8
            assert rows["nc"]["seeded_value"] == 0.0
            assert rows["nr"]["seeded_value"] == 0.0
            assert "wdm6" in rows["nn"]["reason"]
            assert "thompson" not in rows["nn"]["reason"]
            assert "wdm6_ccn_conc" in note
            assert contract.target_wdm6_ccn_conc == 1.0e8
        if target == 28:
            # And Thompson-aerosol's own, out of Thompson's.
            assert rows["nwfa"]["seeded_value"] == 11.1e6
            assert "thompson" in rows["nwfa"]["reason"]
            assert "wdm6" not in rows["nwfa"]["reason"]
            assert "non-aerosol-aware" in note


def test_same_scheme_wdm6_nesting_is_unaffected_by_the_refusal():
    """The refusal is about MIXED edges only."""
    from woof.core import microphysics_transition as mt

    cfg = SimpleNamespace(
        mp_physics=16,
        nest_microphysics_transition=mt.SAME_SCHEME_POLICY)
    contract = mt.resolve_microphysics_transition(cfg, cfg)
    assert contract.mixed is False
    assert contract.source_mp_physics == contract.target_mp_physics == 16


def test_a_refused_selector_cannot_join_without_its_own_sentence():
    """Import-time guard, exercised rather than trusted.

    The bare KeyError was reachable because the three tables describing a
    refused selector are independent.  They are now cross-checked at import;
    this drives that check with a selector that has no row.
    """
    from woof.core import microphysics_transition as mt

    source = Path(mt.__file__).read_text(encoding="utf-8")
    import ast
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id == "UNVALIDATED_MIXED_EDGE_SELECTORS")
    before = ast.get_source_segment(source, node.value)
    values = ast.literal_eval(node.value)
    assert 99 not in values
    patched = source.replace(before, repr((*values, 99)), 1)
    assert patched != source, "the selector tuple moved; re-point this test"
    namespace = {"__name__": "mt_probe", "__file__": mt.__file__}
    with pytest.raises(RuntimeError, match=r"\[99\]"):
        exec(compile(patched, mt.__file__, "exec"), namespace)


def test_the_offline_lane_admits_wdm6_at_both_ends_of_an_edge():
    """mp=16 was held out of the offline lane for want of a QNCCN row and
    a conversion leg.  The row exists and the leg is the online edge
    itself, so a WDM6 parent is read and a WDM6 child is seeded by the
    same closure a WDM6 nest is."""
    from woof import offline_child as oc
    from woof.core import microphysics_transition as mt

    assert 16 in mt.PORTED_MP_PHYSICS
    assert 16 in oc.OFFLINE_CHILD_MP_PHYSICS
    assert 16 in oc.PARENT_SCHEME_CONTRACT
    assert oc.offline_cross_scheme_refusal(16, 18) is None
    assert oc.offline_cross_scheme_refusal(10, 16) is None
    contract = oc._offline_transition_contract(
        10, 16, morr_rimed_ice=0, hail_opt=None, child_cfg=None)
    assert mt.transition_target_fields(contract) == (
        "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr")


# --------------------------------------------------------------------------
# The reflectivity constants have ONE statement
# --------------------------------------------------------------------------

def test_the_wdm6_refl_kernel_takes_its_radar_words_from_the_host():
    """No second, silent copy of the radar_init products inside the kernel:
    every one arrives in the launcher's word tables, which are WRF's words
    (woof.core.refl.wdm6_radar_words, graded against the oracle's dump by
    tests/test_wdm6_wrf461_oracle.py)."""
    from woof.core.refl import radar_init_wdm6, wdm6_radar_words

    source = (Path(__file__).resolve().parents[1]
              / "woof/core/kernels/wdm6_refl.cu").read_text(encoding="utf-8")
    kernel = source.split("extern \"C\" __global__ void refl10cm_wdm6_column")[1]
    for literal in ("24.0", "5040.0", "RPI * 1000.0 / 6.0", "RGAMMA_7",
                    "RGAMMA_4", "RPI5", "RLAMDA4"):
        assert literal not in kernel, literal
    assert "const float* __restrict__ rw, const double* __restrict__ rd" in (
        kernel)
    # The words are WRF's REAL values, which are near -- not equal to --
    # the float64 derivation.
    for hail_opt in (0, 1):
        rc = radar_init_wdm6(hail_opt)
        fw, dw = wdm6_radar_words(hail_opt)
        assert float(fw[3]) == pytest.approx(rc.xcrg[3], rel=1e-6)
        assert float(fw[3]) != rc.xcrg[3]          # WGAMMA(8.) = 5040.002
        assert float(fw[1]) == pytest.approx(rc.xam_r, rel=1e-6)
        assert float(dw[1]) == pytest.approx(rc.pi5, rel=1e-6)
        assert float(dw[0]) == pytest.approx(rc.k_w, rel=1e-6)

    # Morrison's rain density (RHOW = 997) must not be reachable here.
    assert "#undef RXAM_R" in source and "#undef RXAM_S" in source


def test_the_wdm6_refl_launcher_refuses_an_unknown_hail_opt():
    from woof.core.refl import wdm6_radar_words

    with pytest.raises(ValueError, match="hail_opt"):
        wdm6_radar_words(2)


def test_the_plm_remap_clamp_divergence_is_documented_at_the_site():
    """The PLM difference keeps its source citation beside the implementation."""
    source = _KERNEL.read_text(encoding="utf-8")
    index = source.index("kt = (kt > 0) ? kt - 1 : 0;\n        if (kt == kb)")
    preamble = source[max(0, index - 1800):index]
    assert "DELIBERATE, DOCUMENTED PLM DIVERGENCE" in preamble
    for citation in ("nislfv_rain_plmr:2629", "nislfv_rain_plm6:2891"):
        assert citation in preamble, citation


def test_the_oracle_known_deltas_note_exists_and_the_registry_cites_it():
    """The inherited deltas land somewhere the campaign will look."""
    root = Path(__file__).resolve().parents[1]
    note = root / "docs" / "wdm6_oracle_known_deltas.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    for topic in ("_rgmma", "-35 dBZ", "module_mp_wdm6.F:607-614"):
        assert topic in text, topic

    registry = json.loads(
        (root / "woof" / "physics_registry_v2.json").read_text(
            encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"]["wdm6-mp16"]
    assert any("docs/wdm6_oracle_known_deltas.md" in warning
               for warning in option["warnings"])


@pytest.mark.parametrize("restore_module", ["woof.verify.cases.real74_n5s", "woof.ingest.wrfinput"])
def test_the_wrfinput_restore_refloods_the_ccn_reservoir_wrf_discards(restore_module):
    """PAR-MP-WDM6-10: WRF throws wrfinput's QNCCN away; so must the restore.

    ``WDM6_NUMBER_WRFINPUT`` makes QNCCN a REQUIRED member of an mp=16
    wrfinput, and ``_restore_active_moisture`` copies every mapped field in
    unconditionally -- over the ``cfg.wdm6_ccn_conc`` fill ``DomainState``
    performed at allocation, with no later refill anywhere in the tree
    (``state.nn``'s only other reader is the kernel launch in
    ``woof/core/wdm6.py``, and ``apply()`` has no first-step branch).

    WRF does the opposite, twice.  ``dyn_em/start_em.F:1750-1774`` writes
    ``grid%ccn_conc`` into ``scalar(...,p_qnn)`` whenever the field arrives
    empty -- its WDM6 arm is an explicit NO OP, so ``ccn_conc`` keeps the
    namelist value -- and ``module_mp_wdm6.F:220-227`` then floods the whole
    ``nn`` memory window with ``ccn0`` on ``itimestep == 1`` regardless of
    what the file held.  ``real.exe`` never fills QNCCN
    (``dyn_em/module_initialize_real.F`` mentions neither ``qnn`` nor
    ``ccn``), so on the ingest path this test's zeros are the realistic
    case, and a zero reservoir shuts off ``pcact``/``ncact`` activation --
    the source term the whole double-moment warm rain is built on.

    The nonzero half of the fixture is the essential one: it fails on a
    "fill only if the file is empty" repair as well as on no repair at all,
    which is the mutation this test exists to catch, because WRF's flood is
    unconditional in the file's value.
    """
    from importlib import import_module
    n5s = import_module(restore_module)

    cfg = _cfg()
    shape = (cfg.nz, cfg.ny, cfg.nx)
    raw = {name: np.zeros(shape, dtype=np.float32) for name in
           ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
            "QNRAIN", "QNCLOUD", "QNCCN")}
    raw["QNCLOUD"] = np.full(shape, 3.5e8, dtype=np.float32)
    state = SimpleNamespace(**{
        name: np.zeros(shape, dtype=np.float32)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "nc", "nn")})
    state.nn0 = np.full(shape, -99., dtype=np.float32)

    n5s._restore_active_moisture(state, raw, cfg, np)
    np.testing.assert_array_equal(
        state.nn, np.full(shape, np.float32(cfg.wdm6_ccn_conc)))
    np.testing.assert_array_equal(state.nn0, state.nn)
    # Every other mapped field still comes from the file: this reflood is
    # QNCCN alone, not a blanket discard of the restored moisture.
    np.testing.assert_array_equal(state.nc, raw["QNCLOUD"])
    np.testing.assert_array_equal(state.nr, raw["QNRAIN"])

    # A wrfinput that already carries ccn_conc everywhere -- and one that
    # carries something else -- both land on ccn_conc, as WRF's flood does.
    raw["QNCCN"] = np.full(shape, 7.5e9, dtype=np.float32)
    n5s._restore_active_moisture(state, raw, cfg, np)
    np.testing.assert_array_equal(
        state.nn, np.full(shape, np.float32(cfg.wdm6_ccn_conc)))

    # mp=10 has no ``nn`` and must not acquire one from this arm.
    morrison = replace(cfg, mp_physics=10)
    morr_raw = {name: np.zeros(shape, dtype=np.float32) for name in
                ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                 "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL")}
    morr_state = SimpleNamespace(**{
        name: np.zeros(shape, dtype=np.float32)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
                     "ns", "ng", "nc")})
    n5s._restore_active_moisture(morr_state, morr_raw, morrison, np)
    assert not hasattr(morr_state, "nn")

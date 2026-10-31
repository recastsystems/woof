"""The chem configuration door (woof.config.validate_chem_config).

CPU only.  Each refusal names the breakage it prevents; the default is
inert: ``chem_sets = ""`` reads no table, whatever the other chem keys say.
"""

from __future__ import annotations

import dataclasses

import pytest

from woof.config import (RunConfig, load_config, validate_chem_config,
                          validate_run_config)


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=12, ny=10, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0,
                  dt=18.0, run_seconds=60.0)
    values.update(overrides)
    return RunConfig(**values)


def test_chem_is_off_by_default_and_its_keys_are_inert(monkeypatch):
    import woof.chem_table as chem_table

    def no_table(*_a, **_k):
        raise AssertionError("a chem-off config read the chem table")

    monkeypatch.setattr(chem_table, "load_sets", no_table)
    monkeypatch.setattr(chem_table, "catalog", no_table)
    cfg = _cfg()
    assert cfg.chem_sets == ""
    validate_run_config(cfg)
    # WRF reads none of &chem with chem_opt = 0; neither does ArWen.
    validate_chem_config(_cfg(dust_opt=4, wetscav_onoff=1, chem_adv_opt=3,
                              dmsemis_opt=1, cu_physics=1))


def test_tracer_set_is_admitted():
    validate_run_config(_cfg(chem_sets="tracer_test"))


@pytest.mark.parametrize("overrides, match", [
    (dict(chem_sets=["tracer_test"]), "comma-separated string"),
    (dict(chem_sets="Tracer Test"), "not a chem table name"),
    (dict(chem_sets="no_such_set"), "unknown set"),
    (dict(chem_sets="tracer_test", chem_sources="no_such_source"),
     "unknown source"),
    (dict(chem_sets="tracer_test", chem_adv_opt=3), "not transcribed"),
    (dict(chem_sets="tracer_test", cu_physics=1), "under the cloud base"),
    (dict(chem_sets="tracer_test", wetscav_onoff=1), "mechanism"),
    (dict(chem_sets="tracer_test", dust_opt=4), "no dust would be"),
    (dict(chem_sets="tracer_test", dmsemis_opt=1), "absent field"),
    (dict(chem_sets="tracer_test", mynn_chem_vertmx=True),
     "not in this build"),
    (dict(chem_sets="tracer_test", fire_emission_mode="guess"),
     "must be one of"),
    (dict(chem_sets="tracer_test", aerosol_mp_coupling="diagnose"),
     "mp_physics=28"),
    (dict(chem_sets="tracer_test", aer_ra_feedback=1),
     "optics nothing computes"),
], ids=["list", "bad-name", "unknown-set", "unknown-source",
        "weno", "conv-tr", "aqueous", "uoc-dust", "dms", "mynn-chem",
        "fire-mode", "mp-coupling", "ra-feedback"])
def test_each_refusal_names_its_breakage(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_chem_config(_cfg(**overrides))


@pytest.mark.parametrize("overrides", [
    dict(km_opt=1, khdif=10.0),
    dict(km_opt=1, kvdif=5.0, bl_pbl_physics=0),
], ids=["khdif", "kvdif-pbl-off"])
def test_constant_k_mixes_chem_like_wrf(overrides):
    """km_opt=1 runs WRF's isotropic_km package (3641a45f7), whose
    horizontal and vertical drivers mix the chem rows, so chem with
    constant K is admitted; the refusal existed only for the retired
    dry-only Laplacian.  tests/test_chem_transport.py holds the mixing on
    the card (a tracer started as qv stays qv under constant K)."""
    from woof.config import constant_k_mixing_active
    cfg = _cfg(chem_sets="tracer_test", **overrides)
    assert constant_k_mixing_active(cfg)
    validate_chem_config(cfg)


@pytest.mark.parametrize("overrides, match", [
    (dict(aer_op_opt=2), "only WRF-Chem's volume approximation"),
], ids=["aer-op-opt"])
def test_unwired_aerosol_switches_are_refused(overrides, match):
    """aer_op_opt 2-5 would silently take the volume rule.
    (The GOCART rows' wetdep.ls is in the build since the smoke set merged,
    so the packaged table reaches these refusals as it is.)"""
    base = dict(chem_sets="gocart_primary", bl_pbl_physics=1,
                sf_sfclay_physics=1, sf_surface_physics=2)
    validate_chem_config(_cfg(**base))
    with pytest.raises(ValueError, match=match):
        validate_chem_config(_cfg(**base, **overrides))


_MP28 = dict(mp_physics=28, bl_pbl_physics=2, sf_sfclay_physics=2,
             sf_surface_physics=2)


def test_diagnose_refuses_coupling_rows_that_start_at_zero():
    """get_niwfa overwrites nwfa/nifa from the rows that feed it every chem
    step; with no data-store source filling those rows they start at zero,
    and an aerosol-aware Thompson forecast of the Phoenix haboob lost 99.99 %
    of its surface water-friendly aerosol in its first hour.  A source that
    fills the rows (cams-global) admits it."""
    with pytest.raises(ValueError, match="start at zero"):
        validate_chem_config(_cfg(chem_sets="gocart_primary",
                                  aerosol_mp_coupling="diagnose", **_MP28))
    validate_chem_config(_cfg(chem_sets="gocart_primary",
                              chem_sources="cams-global",
                              aerosol_mp_coupling="diagnose", **_MP28))


def test_diagnose_refuses_a_table_with_no_coupling_row():
    """With no row feeding get_niwfa the flag would be a silent no-op."""
    with pytest.raises(ValueError, match="no active row feeds get_niwfa"):
        validate_chem_config(_cfg(chem_sets="tracer_test",
                                  aerosol_mp_coupling="diagnose", **_MP28))


def test_the_monotonic_option_is_admitted_now_its_kernel_is_in_the_build():
    validate_chem_config(_cfg(chem_sets="tracer_test", chem_adv_opt=2))


def test_an_option_whose_module_is_absent_is_refused(monkeypatch):
    """The door refuses rather than letting the positive-definite stage
    stand in for an operator whose module a build lacks."""
    import woof.config as config

    monkeypatch.setitem(config.CHEM_ADV_OPT_MODULES, 2,
                        "woof.core.no_such_chem_advect_module")
    with pytest.raises(ValueError, match="not in this build"):
        validate_chem_config(_cfg(chem_sets="tracer_test", chem_adv_opt=2))


def test_chem_conv_tr_zero_admits_a_cumulus_scheme():
    """WRF's own chem_conv_tr = 0 with a cumulus scheme is admitted."""
    validate_chem_config(_cfg(chem_sets="tracer_test", chem_conv_tr=0,
                              cu_physics=1))


def test_toml_arrays_become_the_comma_string(tmp_path):
    path = tmp_path / "chem.toml"
    path.write_text(
        "[grid]\nnx = 12\nny = 10\nnz = 6\ndx = 3000.0\ndy = 3000.0\n"
        "ztop = 12000.0\n[run]\ndt = 18.0\nrun_seconds = 60.0\n"
        "chem_sets = [\"tracer_test\"]\n", encoding="utf-8")
    try:
        cfg = load_config(path)
    except ValueError as error:
        if "unknown key" in str(error):
            pytest.skip(f"[run] does not take chem keys here: {error}")
        raise
    assert cfg.chem_sets == "tracer_test"
    hash(cfg)


def test_the_chem_block_is_appended_after_every_older_field():
    from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS

    names = [f.name for f in dataclasses.fields(RunConfig)]
    # Appended last after the campaign, keyword-only: sea ice's real.exe
    # threshold (4c82704f8f) and the HRRR v4 output controls
    # (lane/hrrr-v4-output: e55b68accb, bd4a1c2090, f80ac46aeb).
    later = ["seaice_threshold", "prec_acc_dt", "prec_acc_dt1",
             "gsd_diagnostics", "reset_interval1", "history_interval2_s",
             "history_interval_change_s", "smoke_off_reason",
             # 2.9.7, keyword-only: the HRRR v4 climate statistics
             # (lane/hrrr-v4-clwrf) and tc/nest's fixed-frame archive.
             "output_diagnostics", "diag_int", "wind_int",
             "nest_accumulation_archive",
             # 2.9.10, keyword-only: WRF's ghg_input (fix/wrf461-2dom-exact,
             # placed last by d1e35bf2da).
             "ghg_input", "umove", "vmove",
             # 2.9.11, keyword-only: the Fitch wind-farm controls
             # (labs/windfarm 73cb37dd11), folded after ds-umove's frame.
             "wind_farm_scheme", "wind_farms", "wind_farm_table",
             "windfarm_tke_factor", "wind_farm_max_dx_m",
             # fix/hailcast, keyword-only: the HAILCAST generation.
             "hailcast_version"]
    assert names[-len(later):] == later
    for field in dataclasses.fields(RunConfig):
        if field.name in ("umove", "vmove"):
            assert field.kw_only and field.default == 0.0
    names = names[:-len(later)]
    # The 2.9.4 capabilities campaign's keyword-only keys follow the fire
    # block: lane/cap-waves-couple, cap-hail, cap-seaice, cap-eclipse and
    # cap-lake, in merge order.
    campaign = ["ocean_config", "hailcast_opt", "haildt",
                "seaice_prognostic", "seaice_initial_thickness",
                "seaice_new_ice_thickness", "seaice_ocean_heat_flux",
                "ra_sw_eclipse", "lake_state_source", "lake_state_path",
                "lake_state_accept_licence"]
    assert names[-len(campaign):] == campaign
    names = names[:-len(campaign)]
    # The keyword-only fire block follows the chem block.
    names = names[:-len(FIRE_RUN_FIELDS)]
    assert names[-len(CHEM_RUN_FIELDS):] == list(CHEM_RUN_FIELDS)


def _experiment(**overrides):
    from datetime import datetime

    from woof.experiment import experiment_from_run_config

    values = dict(nz=59, bl_pbl_physics=1)
    values.update(overrides)
    return experiment_from_run_config(_cfg(**values), datetime(2026, 8, 6, 18))


def test_a_chem_domain_prices_the_kernels_it_launches():
    """The local-frame price of a chem run names every chem unit, and
    vertmx's is its level-specialized frame (8 B a level: 472 B at 59
    levels, under the default stack), not its 256-level ceiling."""
    from woof.core import preflight as pf

    plain = pf.kernel_local_frame_bytes(_experiment())
    assert not {m for m in plain if m.startswith("chem_")} | (
        {"mono_advection"} & set(plain))
    frames = pf.kernel_local_frame_bytes(_experiment(
        chem_sets="tracer_test,tracer_mixing", chem_adv_opt=2))
    for module in ("chem_bdy", "chem_ledger", "chem_outputs", "chem_prep",
                   "mono_advection"):
        assert frames[module] == 0, module
    assert frames["chem_vertmx"] == 472
    assert pf.kernel_local_frame_bytes(_experiment(
        chem_sets="tracer_test"))["chem_bdy"] == 0
    assert "chem_vertmx" not in pf.kernel_local_frame_bytes(_experiment(
        chem_sets="tracer_test"))


def test_a_process_that_names_no_kernels_is_refused_by_the_price(monkeypatch):
    import woof.core.chem_vertmx as vertmx
    from woof.core import preflight as pf

    monkeypatch.delattr(vertmx, "KERNEL_MODULES")
    with pytest.raises(ValueError, match="declares no KERNEL_MODULES"):
        pf.kernel_local_frame_bytes(_experiment(
            chem_sets="tracer_test,tracer_mixing"))


def test_a_chem_domain_prices_its_process_and_prep_arrays():
    """The device inventory carries every array a process-running chem
    domain allocates once: process ALLOCATES, chem_prep's met fields and
    the deposition-velocity plane, beside the arena and the ledger."""
    from woof.core.device_inventory import chem_state_array_shapes

    cfg = _cfg(nz=59, bl_pbl_physics=1, chem_sets="tracer_test,tracer_mixing")
    shapes = chem_state_array_shapes(cfg)
    assert shapes["chemdiag_vertmx_deposited"] == (1, 10, 12)
    assert shapes["chemprep_z_at_w"] == (60, 10, 12)
    assert shapes["chemprep_rho"] == (59, 10, 12)
    assert shapes["chemprep_ddvel"] == (3, 10, 12)
    passive = chem_state_array_shapes(_cfg(chem_sets="tracer_test"))
    assert not [name for name in passive if name.startswith("chemprep_")]


def test_an_emission_source_no_frame_provider_serves_is_refused_at_the_door():
    """EDGAR's annual flux densities have no fetch route and no remap into
    emission frames in this build; enabling them passed the door and stopped
    the run at its first chem step after a whole preparation."""
    base = dict(chem_sets="gocart_primary", bl_pbl_physics=1,
                sf_sfclay_physics=1, sf_surface_physics=2)
    validate_chem_config(_cfg(**base))
    with pytest.raises(ValueError, match="no emission frame provider"):
        validate_chem_config(_cfg(**base, chem_sources="edgar-v81"))
    validate_chem_config(_cfg(chem_sets="smoke", chem_sources="rave-3km",
                              bl_pbl_physics=1, sf_sfclay_physics=1,
                              sf_surface_physics=2))

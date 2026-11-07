"""UDM's mass and number inventory must survive every public handoff."""
from dataclasses import replace
import pytest
from woof.config import RunConfig, validate_run_config


def configuration():
    return RunConfig(nx=8, ny=6, nz=40, dx=1000., dy=1000., ztop=10000.,
                     dt=1., run_seconds=10., moist=True, moist_cq=True,
                     mp_physics=27, ra_lw_physics=4, ra_sw_physics=4,
                     ra_rrtmg_variant="rrtmg_legacy", wdm6_ccn_conc=5e8)


def test_rcon_waits_for_the_measured_speed_gap_to_be_fixed():
    from woof.config import ConfigurationRefused
    with pytest.raises(ConfigurationRefused, match="1.885431x MP28"):
        validate_run_config(replace(configuration(), mp_physics=29))


def test_udm_prognostics_cross_ingest_transport_and_restart_doors():
    cfg = validate_run_config(configuration())
    from woof.core.device_inventory import state_array_shapes
    shapes = state_array_shapes(cfg)
    for name in ("qi", "qs", "qg", "qh", "nn", "nc", "nr"):
        assert shapes[name] == (40, 6, 8)
        assert shapes[name + "0"] == (40, 6, 8)
    from woof.ingest.wrfinput import SUPPORTED_MICROPHYSICS, UNSUPPORTED_MICROPHYSICS, active_moisture_inventory
    assert 27 in SUPPORTED_MICROPHYSICS and 27 not in UNSUPPORTED_MICROPHYSICS
    required, allowed = active_moisture_inventory(cfg)
    assert {"QHAIL", "QNCCN", "QNCLOUD", "QNRAIN"} <= required <= allowed
    from woof.core.nest_fields import nest_field_kinds
    assert nest_field_kinds(cfg) == ("u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "nn", "nc", "nr")
    from woof.ingest.microphysics_cold_start import source_absent_microphysics
    fields, values = source_absent_microphysics(cfg)
    assert fields == ("QHAIL", "QNCLOUD", "QNRAIN", "QNCCN")
    assert values == dict(qh=0., nc=0., nr=0., nn=5e8)
    from woof.ingest.real import HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS, HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE
    assert 27 in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
    assert HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[27]["retained"] == ("QC", "QR", "QI", "QS", "QG")
    from woof.wrf_physics_inventory import stock_wrf_physics_inventory
    row = stock_wrf_physics_inventory(27)
    assert row.registry_package == "udmscheme"
    assert {f.netcdf_name for f in row.wrfinput_fields} >= set(fields)
    from woof.checkpoint_identity import MICROPHYSICS_ALGORITHM_IDENTITIES
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[27] != MICROPHYSICS_ALGORITHM_IDENTITIES[26]
    from woof.core.rrtmg_legacy import legacy_scheme_declares_radii
    assert legacy_scheme_declares_radii(27, 1) is False
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
    assert kf_phase_mode_for_microphysics(27) == KFPhaseMode.SEPARATE_ICE_SNOW
    with pytest.raises(ValueError, match="den1d"):
        validate_run_config(replace(cfg, ra_rrtmg_variant="rte-rrtmgp"))


def test_udm_native_selector_and_ccn_survive_namelist_import(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    inp.write_text(inp.read_text().replace(" mp_physics = 6,\n",
        " mp_physics = 27,\n ccn_conc = 5.0E8,\n"))
    translated, report = import_namelists(wps, inp, name="udm-import")
    assert "mp_physics = 27" in translated
    assert "wdm6_ccn_conc = 500000000" in translated.replace(".0", "")
    assert not any(row.key == "ccn_conc" for row in report.dropped)
    assert not any(row.key == "mp_physics" for row in report.substitutions)


def test_udm_analysis_history_keeps_native_initial_radar_words():
    import numpy as np
    from types import SimpleNamespace
    from woof.core.refl import analysis_refl_10cm
    qv = np.full((3, 2, 4), 0.01, dtype=np.float32)
    state = SimpleNamespace(qv=qv)
    radar = analysis_refl_10cm(state, configuration(), array_module=np)
    assert radar.shape == qv.shape
    assert np.count_nonzero(radar.view(np.uint32)) == 0
    assert np.all(qv == np.float32(0.01))


def test_udm_acoustic_loading_uses_seven_masses_and_no_number_moments(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from woof.core import acoustic
    cfg = configuration()
    names = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "nn", "nc", "nr")
    fields = {n: np.full((cfg.nz, cfg.ny, cfg.nx), i + 1, np.float32)
              for i, n in enumerate(names)}
    state = SimpleNamespace(**fields, scratch=lambda shape, key: np.empty(shape, np.float32))
    launches = []
    monkeypatch.setattr(acoustic, "get_kernel", lambda *a: lambda grid, block, args: launches.append(args))
    assert acoustic.prepare_moist_cq(state, cfg)[3]
    args, = launches
    assert int(args[10]) == 7
    assert all(args[i] is fields[n] for i, n in enumerate(names[:7]))
    from woof.verify.npref import np_calc_cq
    full = np_calc_cq(fields, 27)
    only_mass = np_calc_cq({n: fields[n] for n in names[:7]}, 26)
    for actual, expected in zip(full, only_mass, strict=True):
        np.testing.assert_array_equal(actual, expected)

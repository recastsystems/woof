from argparse import Namespace
from woof.hex.init_physics import init_selection
from woof.hex.init_door import run_declarations


def test_suite_selects_ruc_and_complete_tempo_species():
    selected = init_selection("rrfs_v2_tempo_candidate", None)
    assert selected["land_surface"] == "ruc"
    assert selected["requires_climatology"] is True
    assert selected["scalar_names"] == (
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "volg")


def test_noahmp_changes_geometry_and_keeps_tempo_species():
    a = init_selection("rrfs_v2_tempo_candidate", "noahmp")
    b = init_selection("rrfs_v2_tempo_candidate", None)
    assert a["land_surface"] == "noahmp"
    assert a["scalar_names"] == b["scalar_names"]


def test_legacy_noah_init_keeps_four_levels_and_no_added_species():
    assert init_selection(None, None) == {"land_surface": "noah", "scalar_names": ()}
    args = Namespace(start_time="2026-10-07_12:00:00", nfglevels=51,
        nfgsoillevels=4, use_spechumd="yes", frac_seaice="yes",
        extrap_airtemp="constant", tsk_seaice_threshold=100,
        landuse_table="MODIFIED_IGBP_MODIS_NOAH")
    assert run_declarations(args)["config_nsoillevels"] == 4
    args.land_surface = "ruc"
    assert run_declarations(args)["config_nsoillevels"] == 9


def test_tempo_refuses_when_neither_analysis_nor_climatology_resolves(tmp_path, monkeypatch):
    """A row that requires climatology never falls through to the synthetic profile."""
    import stat, sys, types
    import pytest
    from woof.hex import init_physics
    engine = tmp_path / "fake_engine"
    engine.write_text("#!" + sys.executable + "\nimport json,sys\n"
        "print(json.dumps({'analysis_pair': False, 'soil_category_mismatches': 0}))\n"
        "sys.exit(0 if '--aerosol-status' in sys.argv else 9)\n")
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    import woof.ingest.wif_climatology as wif
    import woof.table_assets as assets
    unresolved = types.SimpleNamespace(resolved=False, path=None)
    monkeypatch.setattr(wif, "resolve_wif_climatology", lambda *a, **k: unresolved)
    monkeypatch.setattr(wif, "describe_wif_source", lambda r: {"resolved": False})
    monkeypatch.setattr(assets, "stage_wif_dataset", lambda *a, **k: 0)
    selection = dict(init_selection("rrfs_v2_tempo_candidate", None))
    with pytest.raises(ValueError, match="requires analyzed aerosols or the monthly WIF"):
        init_physics.initialize_species(engine, tmp_path / "init.nc", selection,
            Namespace(start_time="2026-10-07_12:00:00", wif_climatology=None))


def test_a_suite_the_init_door_does_not_initialize_is_refused_by_name():
    # woof hex init --physics-suite <unknown> stopped with a bare KeyError (hex 0.4.0 recut, 2026-10-16).
    # hrrrv5 is no longer one of them: the RRFS producer row initializes it (tests/test_rrfs_init_door.py).
    import pytest
    from woof.hex.errors import MpasPortError

    assert init_selection("hrrrv5", None)["land_surface"] == "ruc"
    with pytest.raises(MpasPortError, match="hrrrv5.*rrfs_v2_dev"):
        init_selection("no_such_suite", None)


def test_an_mpas_v842_suite_is_initialized_by_the_plain_init_and_named():
    # hex parity run group 7, #28: `init --physics-suite convection_permitting` refused ("drop the flag"),
    # though the suite runs from the init written with no suite.
    import pytest
    from woof.hex.errors import MpasPortError

    for suite in ("convection_permitting", "mesoscale_reference"):
        selection = init_selection(suite, None)
        assert selection["land_surface"] == "noah" and selection["scalar_names"] == ()
        assert selection["physics_initialization"]["suite"] == suite
    assert init_selection("convection_permitting", None)["scalar_names"] == init_selection(None, None)["scalar_names"]
    with pytest.raises(MpasPortError, match="four soil levels"):
        init_selection("convection_permitting", "ruc")

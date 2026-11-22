# SPDX-License-Identifier: Apache-2.0
"""MODEL-AUDIT D13: ``--physics-suite hrrrv5`` with no namelist runs NOAA's deck (rrfs-workflow 570c81fe
parm/hrrrv5/namelist.atmosphere and config.meshdep), not the fork Registry's defaults.

THE BREAKAGE THIS PREVENTS: every adjoint recording ran the Registry defaults (no CAM filter, no Rayleigh u
damping, w damping from 22 km, mynn_ess=1, mixnumcon=1, tempo_ml_for_bl_nc false, gvf_update true), so the
adjoint linearised a configuration NOAA does not run and fwdmatch's bit-for-bit proof did not cover it."""
import argparse

import pytest

from woof.hex.mpasphys.forecast_v842 import native_route, namelist_option


def _route(mesh, **given):
    values = dict(physics_namelist=None, physics_suite="hrrrv5", physics=None, physics_mode="exact",
                  physics_correct_skip=None, physics_profile=None, physics_backend=None, physics_pin_set=None,
                  physics_default_backend=None, init=None, mesh=mesh, repo=None, dt=None, acoustic_substeps=None)
    values.update(given)
    return native_route(argparse.Namespace(**values), default_mode="exact")


@pytest.mark.parametrize("pin", [None, "rrfs"])
def test_a_plain_hrrrv5_run_is_noaas_deck(pin):
    config, text = _route("conus12km", physics_pin_set=pin)
    assert config.options["config_mynn_ess"] == 0
    assert config.options["config_mynn_mixnumcon"] == 0
    assert config.options["config_tempo_ml_for_bl_nc"] is True
    assert config.options["config_gvf_update"] is False
    assert config.options["config_radtlw_interval"] == "00:30:00"
    assert namelist_option(text, "damping", "config_mpas_cam_coef") == 2.0
    assert namelist_option(text, "damping", "config_number_cam_damping_levels") == 8
    assert namelist_option(text, "damping", "config_rayleigh_damp_u") is True
    assert namelist_option(text, "damping", "config_zd") == 16000.0
    assert namelist_option(text, "nhyd_model", "config_dt") == 60.0
    assert namelist_option(text, "nhyd_model", "config_number_of_sub_steps") == 2


def test_the_3km_mesh_takes_the_3km_meshdep_row_and_an_explicit_step_stays():
    config, text = _route("conus3km", dt="15")
    assert config.options["config_radtlw_interval"] == "00:15:00"
    assert namelist_option(text, "nhyd_model", "config_number_of_sub_steps") == 4
    assert namelist_option(text, "nhyd_model", "config_dt") is None


def test_what_the_run_names_decides():
    config, text = _route("conus12km", acoustic_substeps=4, physics=["config_mynn_ess=1"])
    assert config.options["config_mynn_ess"] == 1
    assert namelist_option(text, "nhyd_model", "config_number_of_sub_steps") is None


@pytest.mark.parametrize("mesh", ["r0.8.102.56619", "v6.75.112676", "u60.163632", "x1.40962"])
def test_a_mesh_far_from_noaas_keeps_the_doors_own_step(mesh):
    # 800 m took NOAA's 1.25 km step (12 s, the door's is 4 s) and the 120 km globe its 12 km step (60 s, the door's 600 s)
    config, text = _route(mesh)
    assert namelist_option(text, "nhyd_model", "config_dt") is None
    assert namelist_option(text, "nhyd_model", "config_number_of_sub_steps") is None
    assert config.options["config_mynn_ess"] == 0
    assert namelist_option(text, "damping", "config_mpas_cam_coef") == 2.0


def test_a_run_with_only_an_init_file_takes_its_meshes_row(tmp_path):
    # fwdmatch 15 Nov: --init of the 12 km cut q12.96.2890 with no --mesh ran 72 s (the door's own), not NOAA's 60 s
    netCDF4 = pytest.importorskip("netCDF4")
    init = tmp_path / "init.nc"
    with netCDF4.Dataset(init, "w") as ds:
        ds.createDimension("nEdges", 3)
        ds.createVariable("dcEdge", "f8", ("nEdges",))[:] = [12000.0, 13500.0, 30000.0]
    config, text = _route(None, init=str(init))
    assert namelist_option(text, "nhyd_model", "config_dt") == 60.0
    assert namelist_option(text, "nhyd_model", "config_number_of_sub_steps") == 2
    assert config.options["config_radtlw_interval"] == "00:30:00"


@pytest.mark.parametrize("pin", [None, "rrfs"])
def test_noaas_12km_meshes_run_its_cumulus(pin):
    # job cumulus: exrrfs_fcst.sh:62-67 adds cu_ntiedtke and config_gfl_sub3d=1 on na12km and conus12km
    config, text = _route("conus12km", physics_pin_set=pin)
    assert config.options["config_convection_scheme"] == "cu_ntiedtke"
    assert namelist_option(text, "physics", "config_convection_scheme") == "cu_ntiedtke"
    assert namelist_option(text, "physics", "config_gfl_sub3d") == 1


@pytest.mark.parametrize("mesh", ["conus3km", "r0.8.102.56619", "x1.40962"])
def test_other_meshes_stay_convection_off(mesh):
    config, text = _route(mesh)
    assert namelist_option(text, "physics", "config_convection_scheme") is None
    assert str(config.options.get("config_convection_scheme", "off")).strip() in ("off", "", "None")


def test_a_named_convection_scheme_decides_at_12km():
    config, text = _route("conus12km", physics=["config_convection_scheme=off"])
    assert str(config.options["config_convection_scheme"]).strip() == "off"

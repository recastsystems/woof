"""Selected leaf code must not change the candidate's default CUDA image."""
from __future__ import annotations

import pytest

from woof.core.noahmp_kernel_sources import runtime_unit


@pytest.mark.parametrize("name,added", [
    ("noahmp_bareflux", {"noahmp_bare_flux_stc3", "noahmp_bare_flux_masked", "noahmp_bare_flux_stc2", "noahmp_bare_flux_masked_stc2", "noahmp_bare_flux_sfc2", "noahmp_bare_flux_masked_sfc2"}),
    ("noahmp_glacier", {"noahmp_glacier_column_gla2",
                         "noahmp_glacier_column_gla2_bats", "noahmp_glacier_column_masked",
                         "noahmp_glacier_driver_resident", "noahmp_glacier_column_options"}),
    ("noahmp_radiation", {"noahmp_radiation_batch_bats",
                           "noahmp_rad_snowalb_bats", "noahmp_radiation_batch_dense", "noahmp_radiation_batch_crown", "noahmp_radiation_batch_snow_papers"}),
    ("noahmp_vegeflux", {"k_vegeflux_masked_stc3", "k_vegeflux_stc2", "k_vegeflux_masked_stc2", "k_vegeflux_crs2", "k_vegeflux_masked_crs2", "k_vegeflux_sfc2", "k_vegeflux_masked_sfc2"}),
    ("noahmp_leaves", {"noahmp_leaf_atm_selected", "noahmp_pedotransfer_sr2006", "noahmp_depth_soil_parameters", "noahmp_carbon_flux", "noahmp_crop_flux"}),
    ("noahmp_vegprecip", {"noahmp_phenology_input_lai", "noahmp_phenology_crop"}),
    ("noahmp_water", {"k_water_selected", "k_irrigation_prefix", "k_groundwater_heads", "k_groundwater_lateral", "k_groundwater_update", "k_groundwater_cold_adjust", "k_groundwater_cold_river", "k_groundwater_cold_soil"}),
    ("noahmp_thermal", {"noahmp_thermal_phasechange_frz2", "noahmp_thermal_tsnosoi_selected", "noahmp_energy_stc2_snow", "noahmp_tsnosoi_snow_papers"}),
])
def test_selected_images_reuse_and_seal_the_default_and_extension(name, added):
    default = runtime_unit(name)
    selected = runtime_unit(name, selected_options=True)
    assert selected.source.startswith(default.source)
    if name == "noahmp_leaves":
        assert selected.parts[:-4] == default.parts
        assert tuple(p[0] for p in selected.parts[-4:]) == ("noahmp_carbon_options.cuh","noahmp_crop_options.cuh","noahmp_soil_options.cuh","noahmp_leaves_options.cuh")
    elif name == "noahmp_radiation":
        assert selected.parts[:-2] == default.parts
        assert selected.parts[-2][0] == ({"noahmp_radiation":"glibc_trig_flt32.cuh", "noahmp_leaves":"noahmp_soil_options.cuh"}[name])
    elif name == "noahmp_water":
        assert selected.parts[:-2] == default.parts
        assert selected.parts[-1][0] == "noahmp_groundwater_options.cuh"
    else:
        assert selected.parts[:-1] == default.parts
    assert selected.parts[-2 if name == "noahmp_water" else -1][0] == name + "_options.cuh"
    assert set(selected.exports) - set(default.exports) == added
    assert selected.options == default.options
    assert selected.identity() != default.identity()
    assert selected.identity()["parts"][-2 if name == "noahmp_water" else -1]["file"] == name + "_options.cuh"


def test_selected_vege_retains_the_explicit_native_libm_profile():
    default = runtime_unit("noahmp_vegeflux", atan_profile="glibc_2_43")
    selected = runtime_unit("noahmp_vegeflux", atan_profile="glibc_2_43",
                            selected_options=True)
    assert selected.source.startswith(default.source)
    assert selected.parts[:-1] == default.parts
    assert "glibc_trig_flt32.cuh" == selected.parts[0][0]


def test_unsupported_selected_unit_refuses_before_cuda_import():
    with pytest.raises(ValueError, match="no selected option unit"):
        runtime_unit("noahmp_driver", selected_options=True)


def test_native_option_referee_rejects_unpinned_source(tmp_path):
    from tools.noahmp_options_wrf461_oracle.build import build
    source = tmp_path / "untrusted.F"
    source.write_bytes(b"not the pinned WRF source\n")
    with pytest.raises(ValueError, match="refusing unpinned"):
        build(source, tmp_path / "build")
    assert not (tmp_path / "build").exists()


def test_selected_parameter_trajectories_have_distinct_restart_seals():
    from woof.core.noahmp_runtime import NoahmpRuntimeParameters
    from woof.io.restart import _packed_parameters_identity

    default = NoahmpRuntimeParameters()
    explicit = NoahmpRuntimeParameters(opt_alb=2, opt_stc=1, opt_rsf=1, opt_btr=1)

    def seal(parameters):
        return _packed_parameters_identity(parameters, label="noahmp")

    baseline = seal(default)
    assert baseline == seal(explicit)
    assert "land_options" not in baseline["declared_identity"]
    hashes = {baseline["sha256"]}
    for selection in (dict(opt_alb=1), dict(opt_stc=3), dict(opt_rsf=2),
                      dict(opt_rsf=3), dict(opt_rsf=4), dict(opt_btr=2),
                      dict(opt_btr=3), *(dict(opt_snf=v) for v in (2, 3, 4, 5)),
                      *(dict(dveg=v) for v in (1, 3)), dict(opt_inf=2),
                      dict(opt_run=4), dict(opt_inf=2,opt_run=4), dict(opt_run=6), dict(opt_inf=2,opt_run=6), dict(opt_run=7), dict(opt_inf=2,opt_run=7), dict(opt_rad=2), dict(opt_rad=2,opt_alb=1), dict(opt_rad=1), dict(opt_rad=1,opt_alb=1), dict(opt_frz=2), dict(opt_frz=2,opt_inf=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3,opt_tbot=1), dict(opt_stc=2,opt_tbot=1), dict(opt_crs=2), dict(opt_crs=2,opt_stc=2), dict(opt_sfc=2), dict(opt_sfc=2,opt_crs=2,opt_stc=2), *(dict(dveg=v) for v in (7,8,9)), *(dict(opt_run=v) for v in (1,2)), *(dict(opt_run=v,opt_inf=2) for v in (1,2)), dict(soiltstep=180.0), dict(opt_tdrn=1), dict(opt_tdrn=2), dict(opt_stc=3,tanh_profile="glibc_2_43"), dict(opt_stc=3,atan_profile="glibc_2_43"), *(dict(opt_run=8,opt_infdv=v) for v in (1,2,3))):
        selected = NoahmpRuntimeParameters(bundle=default.bundle, **selection)
        record = seal(selected)
        assert record["declared_identity"]["tables"] == baseline["declared_identity"]["tables"]
        assert record["declared_identity"]["land_options"].items() >= selection.items()
        assert record["sha256"] not in hashes
        hashes.add(record["sha256"])
        with pytest.raises(TypeError):
            selected.land_options["opt_btr"] = 1


@pytest.mark.parametrize("selection", [dict(opt_crop=2)])
def test_native_block_checks_do_not_admit_unqualified_forecast_selectors(selection):
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config

    with pytest.raises(ConfigurationRefused, match=next(iter(selection))):
        validate_run_config(RunConfig(nx=8, ny=8, nz=10, dx=1000., dy=1000.,
                                      ztop=10000., dt=1., run_seconds=60., **selection))


def test_native_log10_profile_seals_only_the_thermal_override():
    default=runtime_unit("noahmp_thermal")
    native=runtime_unit("noahmp_thermal",log10_profile="glibc_2_43")
    assert native.identity()!=default.identity()
    assert native.parts[0][0]=="noahmp_log10_profile.cuh"
    assert "AD(noahmp_native_log10(satratio), 1.0f)" in native.source
    assert "AD(r_log10(satratio), 1.0f)" in default.source
    assert runtime_unit("noahmp_thermal")==default

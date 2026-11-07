"""Admission must price the operators and column bounds the run launches."""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from woof.config import RunConfig
from woof.core import preflight as pf
from woof.experiment import experiment_from_run_config


def config(**updates):
    values = dict(nx=8, ny=8, nz=50, dx=9000., dy=9000., dt=10.,
                  ztop=10000., run_seconds=60., ra_physics=0,
                  hailcast_opt=0)
    return RunConfig(**(values | updates))


def modules(cfg):
    return pf.domain_kernel_modules(SimpleNamespace(run=cfg, grid_id=1),
                                    prices_refl=False)


@pytest.mark.parametrize("scheme", ["auto", "fitch", "jensen", "gad"])
def test_active_wind_farm_is_in_the_launch_projection(scheme):
    cfg = config(wind_farm_scheme=scheme)
    assert "wind_farm" in modules(cfg)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    assert pf.kernel_local_frame_bytes(exp)["wind_farm"] == 56


def test_wind_farm_off_does_not_add_a_module():
    assert "wind_farm" not in modules(config())


@pytest.mark.parametrize("lw,sw,expected", [
    (0, 0, set()),
    (0, 1, {"dudhia_sw"}),
    (1, 0, {"noahmp_libm_slab_composed"}),
    (1, 1, {"noahmp_libm_slab_composed", "dudhia_sw"}),
    (5, 5, {"goddard"}),
    (5, 1, {"goddard", "dudhia_sw"}),
    (0, 5, {"goddard"}),
    (5, 0, {"goddard"}),
    (90, 90, set()),
])
def test_radiation_projection_resolves_each_spectrum(lw, sw, expected):
    selected = modules(config(ra_lw_physics=lw, ra_sw_physics=sw))
    radiation = {"goddard", "dudhia_sw", "noahmp_libm_slab_composed"}
    assert selected & radiation == expected
    assert not selected & set(pf._RADIATION_KERNEL_MODULES[4])


@pytest.mark.parametrize("selector", [0, 1, 4, 5, 90])
def test_aggregate_and_explicit_pair_price_the_same_modules(selector):
    assert modules(config(ra_physics=selector)) == modules(config(
        ra_lw_physics=selector, ra_sw_physics=selector))


@pytest.mark.parametrize("variant", ["rrtmg_legacy", "rte-rrtmgp"])
@pytest.mark.parametrize("lw,sw", [(4, 0), (0, 4), (4, 5), (5, 4)])
def test_mixed_rrtmg_preserves_its_implementation_row(variant, lw, sw):
    cfg = config(ra_lw_physics=lw, ra_sw_physics=sw,
                 ra_rrtmg_variant=variant)
    assert set(pf._radiation_44_kernel_modules(cfg)) <= modules(cfg)
    assert ("goddard" in modules(cfg)) == (5 in (lw, sw))


def test_conflicting_radiation_spellings_are_still_refused():
    with pytest.raises(ValueError, match="two radiation spellings contradict"):
        modules(config(ra_physics=4, ra_lw_physics=5, ra_sw_physics=5))


@pytest.mark.parametrize("spectrum", ["ra_lw_physics", "ra_sw_physics"])
def test_unknown_spectrum_cannot_disappear_from_the_budget(spectrum):
    cfg = config(ra_lw_physics=0, ra_sw_physics=0)
    cfg = replace(cfg, **{spectrum: 12345})
    with pytest.raises(ValueError, match=f"{spectrum}=12345"):
        modules(cfg)


@pytest.mark.parametrize("levels", [16, 32, 50, 64, 96, 128, 192])
def test_goddard_budget_covers_the_recovered_depth_measurements(levels):
    cfg = config(nz=levels, ra_lw_physics=5, ra_sw_physics=5)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    frames = pf.kernel_local_frame_bytes(exp)
    # Recovered 12.8.93 readings are a lower bound on the provisional price,
    # not a claim that this source/compiler has been measured again.
    assert frames["goddard"] >= 704 + 1544 * levels
    assert frames["goddard"] >= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["goddard"]


def test_deeper_non_goddard_domain_does_not_inflate_its_frame():
    cfg = config(nz=32, ra_lw_physics=5, ra_sw_physics=5)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    first = pf.kernel_local_frame_bytes(exp)["goddard"]
    other = replace(exp.domains[0], grid_id=2, run=config(nz=192))
    assert pf.kernel_local_frame_bytes(replace(
        exp, domains=exp.domains + (other,)))["goddard"] == first


def test_deepest_selecting_domain_owns_the_goddard_frame():
    cfg = config(nz=32, ra_lw_physics=5, ra_sw_physics=5)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    other = replace(exp.domains[0], grid_id=2, run=replace(cfg, nz=50))
    assert pf.kernel_local_frame_bytes(replace(
        exp, domains=exp.domains + (other,)))["goddard"] == 77904


def test_goddard_fit_report_exposes_the_provisional_basis():
    exp = experiment_from_run_config(
        config(nz=50, ra_lw_physics=5, ra_sw_physics=5), datetime(2026, 1, 1))
    basis = pf.non_pool_basis(pf.MEASURED_LOCAL_MEMORY_PROFILE, exp)
    assert "Goddard local frame 77904 B per thread" in basis
    assert "provisional depth envelope" in basis
    assert "compiler-specific depth verification required" in basis


def test_classic_lw_exposes_its_composed_unit_frame_provenance():
    exp = experiment_from_run_config(
        config(ra_lw_physics=1, ra_sw_physics=0), datetime(2026, 1, 1))
    assert not pf.selects_noahmp(exp)
    assert "Noah-MP local frames" in pf.non_pool_basis(
        pf.MEASURED_LOCAL_MEMORY_PROFILE, exp)


@pytest.mark.gpu
@pytest.mark.parametrize("levels", [16, 32, 50, 64, 96, 128, 192])
def test_current_goddard_exports_fit_the_provisional_depth_envelope(levels):
    pytest.importorskip("cupy")
    from woof.core.kernels import get_kernel_int_defines, kernel_local_size_bytes

    cfg = config(nz=levels, ra_lw_physics=5, ra_sw_physics=5)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    price = pf.kernel_local_frame_bytes(exp)["goddard"]
    # Include preparation and publication: the widest export owns residency.
    for name in ("goddard_interface_temperature", "goddard_prepare_columns",
                 "goddard_cloud_columns", "goddard_bands_columns",
                 "goddard_publish_columns"):
        kernel = get_kernel_int_defines(
            "goddard", name, (("GDD_MAX_LEVELS", levels),))
        assert kernel_local_size_bytes(kernel) <= price, name

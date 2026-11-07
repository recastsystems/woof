"""Keep WRF's two scalar paths distinct and new MYNN choices reachable."""

from dataclasses import asdict

import pytest

from woof.config import (RunConfig, mynn_mixscalars_active,
                          mynn_mixscalars_driver_value,
                          mynn_mixscalars_unported_species,
                          validate_run_config)
from woof.io.restart import (
    RestartMismatchError, _drop_default_off_run_keys, _require_config_match,
)


def _config(**overrides):
    values = dict(nx=8, ny=8, nz=50, dx=3000.0, dy=3000.0,
                  dt=15.0, run_seconds=300.0, ztop=20000.0,
                  moist=True, mp_physics=28, bl_pbl_physics=5,
                  sf_sfclay_physics=5, sf_surface_physics=3,
                  num_soil_layers=9, bldt=0.0)
    values.update(overrides)
    return RunConfig(**values)


@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("scalar", [0, 1])
def test_supported_mynn_options_reach_config(length, scalar):
    cfg = _config(bl_mynn_mixlength=length, scalar_pblmix=scalar)
    validate_run_config(cfg)
    assert cfg.bl_mynn_mixlength == length
    assert cfg.scalar_pblmix == scalar
    assert cfg.bl_mynn_mixscalars == 0


@pytest.mark.parametrize("value", [-1, 3, True, 2.0])
def test_unimplemented_mixing_length_is_never_substituted(value):
    with pytest.raises(ValueError, match="bl_mynn_mixlength"):
        validate_run_config(_config(bl_mynn_mixlength=value))


@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_scalar_selector_is_an_integer_switch(value):
    with pytest.raises(ValueError, match="scalar_pblmix"):
        validate_run_config(_config(scalar_pblmix=value))


@pytest.mark.parametrize("overrides, reason", [
    ({"bl_mynn_mixscalars": 1}, "cannot be combined"),
    ({"mp_physics": 30}, "SCALAR_PBLMIX_SPECIES has no row"),
    ({"bldt": 1.0}, "restart"),
])
def test_scalar_mixing_refuses_missing_carriers_or_dropped_rates(overrides, reason):
    with pytest.raises((ValueError, NotImplementedError), match=reason):
        validate_run_config(_config(scalar_pblmix=1, **overrides))


@pytest.mark.parametrize("mp_physics, species", [
    (8, ("ni",)), (10, ("ni",)), (18, ("qndrop", "qni", "qnn", "qvolh")), (50, ("ni", "qir", "qib")), (6, ()),
])
def test_scalar_mixing_admits_every_tabled_microphysics(mp_physics, species):
    """U-E4: WRF's diff4d mixes each scheme's own scalars, so scalar_pblmix=1
    with any tabled microphysics is admitted (it refused all but mp=28)."""
    from woof.config import scalar_pblmix_species
    cfg = _config(scalar_pblmix=1, mp_physics=mp_physics)
    validate_run_config(cfg)
    assert scalar_pblmix_species(cfg) == species


def test_mixscalars_default_is_the_ported_generation_registry_default():
    # WRF v4.6.1 Registry.EM_COMMON:2479 reads 0 for the MYNN generation
    # woof ports; HRRR's V3.9 line has no such key.  Only v4.7.x (the
    # MYNN-EDMF submodule woof does not carry) flipped it to 1.
    assert RunConfig.__dataclass_fields__["bl_mynn_mixscalars"].default == 0
    assert _config().bl_mynn_mixscalars == 0


@pytest.mark.parametrize("mp_physics", [6, 1])
def test_mixscalars_without_number_species_is_admitted_and_inert(mp_physics):
    # A moist-only scheme gives WRF no number species: every scalar solve
    # is gated on a false flag and WRF runs the key as a no-op.  The former
    # "requires mp_physics=28" refusal named no breakage here.
    cfg = _config(mp_physics=mp_physics, bl_mynn_mixscalars=1)
    validate_run_config(cfg)
    assert cfg.bl_mynn_mixscalars == 1
    assert not mynn_mixscalars_active(cfg)
    assert mynn_mixscalars_unported_species(cfg) == ()
    assert mynn_mixscalars_driver_value(cfg) == 0
    # The every-step restart invariant and the gsd_41 plume-water refusal
    # bind the active case only: an inert key holds no qn tendencies and
    # mixes no plume water.
    validate_run_config(_config(mp_physics=mp_physics, bl_mynn_mixscalars=1,
                                bldt=60.0))
    validate_run_config(_config(mp_physics=mp_physics, bl_mynn_mixscalars=1,
                                bl_mynn_version="gsd_41"))


@pytest.mark.parametrize("mp_physics, species", [
    (8, "qni"), (9, "qnc, qni"), (10, "qni"), (16, "qnc"), (18, "qni"),
    (50, "qni"),
])
def test_mixscalars_under_a_partly_mixed_scheme_selects_wrf471(
        mp_physics, species):
    cfg = _config(mp_physics=mp_physics, bl_mynn_mixscalars=1)
    assert cfg.bl_mynn_version == "wrf_471"
    validate_run_config(cfg)
    assert mynn_mixscalars_unported_species(cfg) == ()
    assert mynn_mixscalars_driver_value(cfg) == 1
    assert _config(mp_physics=mp_physics).bl_mynn_version == "wrf_461"


def test_mixscalars_flag_species_table_covers_every_accepted_scheme():
    from woof.config import MP_PHYSICS_ACCEPTED, MYNN_QN_FLAG_SPECIES
    assert set(MYNN_QN_FLAG_SPECIES) == set(MP_PHYSICS_ACCEPTED)
    assert MYNN_QN_FLAG_SPECIES[28] == ("qnc", "qni", "qnwfa", "qnifa", "qnbca")


def test_mixscalars_with_qn_family_is_active_and_keeps_its_guards():
    cfg = _config(bl_mynn_mixscalars=1)
    validate_run_config(cfg)
    assert mynn_mixscalars_active(cfg)
    assert mynn_mixscalars_driver_value(cfg) == 1
    with pytest.raises(NotImplementedError, match="restart"):
        validate_run_config(_config(bl_mynn_mixscalars=1, bldt=60.0))
    with pytest.raises(ValueError, match="gsd_41"):
        validate_run_config(_config(bl_mynn_mixscalars=1,
                                    bl_mynn_version="gsd_41"))


def test_mixscalars_storage_is_priced_for_the_active_key_only():
    from woof.core.preflight import mynn_mixscalars_memory_items
    assert mynn_mixscalars_memory_items(
        _config(mp_physics=6, bl_mynn_mixscalars=1)) == ()
    assert mynn_mixscalars_memory_items(_config(bl_mynn_mixscalars=0)) == ()
    assert mynn_mixscalars_memory_items(_config(bl_mynn_mixscalars=1))


def test_default_off_keeps_old_restart_identity_and_on_is_bound():
    old = asdict(_config())
    old.pop("scalar_pblmix")
    off = asdict(_config(scalar_pblmix=0))
    on = asdict(_config(scalar_pblmix=1))
    for values in (old, off, on):
        _drop_default_off_run_keys(values)
    assert old == off
    assert on.pop("scalar_pblmix") == 1
    assert on == off
    # The reader must apply the same absence rule as the writer. Otherwise
    # even a checkpoint written by this build cannot restore with option 0.
    _require_config_match(old, _config(), "off-checkpoint")
    with pytest.raises(RestartMismatchError, match="scalar_pblmix"):
        _require_config_match(old, _config(scalar_pblmix=1), "off-checkpoint")
    _require_config_match(asdict(_config(scalar_pblmix=1)),
                          _config(scalar_pblmix=1), "on-checkpoint")

# Release census replacements (reviewed against the merged sources).
# Census: test_mixscalars_under_a_partly_mixed_scheme_is_refused_by_name is covered by test_mixscalars_under_a_partly_mixed_scheme_selects_wrf471.

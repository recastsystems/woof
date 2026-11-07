"""MYNN you can run overnight: the radiation-bearing MYNN family.

Until 1.8.7 every shipped MYNN suite ran shortwave with longwave OFF.
That is a DAYTIME validation configuration -- ``PHYSICS.md``'s nocturnal
validity table listed all three MYNN rows under "nocturnally valid: no",
and ``woof.experiment.build_experiment`` refuses an undeclared one for
any window containing local night -- so a user who chose MYNN from a
menu landed in the nocturnally-invalid class with no MYNN row to move
to.  Composing MYNN with radiation by hand was always accepted; what did
not exist was a NAMED suite, and every menu, ``--physics-profile`` choice
list and route declaration is keyed by name.

Three rows close it, one per land surface the no-radiation family
covers.  This file pins what they are FOR:

* each loads through the front door every runner shares and resolves to
  MYNN 5/5 with both radiation streams on;
* each is nocturnally valid, proved against the shipped guard on a real
  48 h window that contains local night -- and its no-radiation sibling
  is refused on the SAME window, which is what makes the first half a
  measurement rather than a guard that cannot fire;
* each differs from that sibling in the radiation block and nothing
  else, so a paired run isolates radiation;
* the MYNN option identity is untouched by all of it.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from woof.config import (MYNN_PBL_OPTION_IDENTITY, RunConfig,
                          validate_run_config)
from woof.domain_wizard import experiment_from_text, render_config
from woof.experiment import build_experiment
from woof.physics_compat import (
    ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    RRTMG_VARIANT_RTE_RRTMGP,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    WRF_RRTMG_TO_RTE_RRTMGP,
    first_local_night_time,
    identify_single_domain_profile,
    single_domain_runtime_switches,
    validate_physics_capabilities,
)

#: (radiation-bearing row, the no-radiation row it mirrors).
MIRRORED_PAIRS = (
    (MYNN_RTE_RRTMGP_PROFILE_ID, MYNN_PROFILE_ID),
    (MYNN_RUC_RTE_RRTMGP_PROFILE_ID, MYNN_RUC_PROFILE_ID),
    (MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID),
)
RADIATION_BEARING_MYNN = tuple(new for new, _old in MIRRORED_PAIRS)

#: The reporting user's geometry, reused verbatim from
#: ``tests/test_nocturnal_radiation_guard.py``: an Alabama reference point
#: whose April evening puts local night about 12 h into a 12Z window.
_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 33.8, "ref_lon": -87.29,
    "truelat1": 23.8, "truelat2": 43.8, "stand_lon": -87.29,
}
_NIGHT_START = datetime(2011, 4, 26, 12)
_NIGHT_HOURS = 48


def _emitted(profile, *, start=_NIGHT_START, hours=_NIGHT_HOURS):
    return render_config(
        name="mynnradcase", start_time=start, hours=hours,
        projection=dict(_PROJECTION), dims=[(120, 100)], ratios=(),
        fetch_hints={"source": "era5"}, case_data=None, profile=profile)


def _raw(text):
    import tomllib

    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


def _cfg(**overrides) -> RunConfig:
    # nz=5 is MYNN's own floor (MYNN_VERTICAL_LEVEL_BOUNDS); a four-level
    # column is refused by the vertical preflight before the identity
    # check is ever reached.
    values = dict(nx=4, ny=3, nz=5, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# The window this whole family exists for really does contain night.
# ---------------------------------------------------------------------------

def test_the_test_window_contains_local_night():
    """The instrument, before anything is measured with it.

    Every nocturnal assertion below is vacuous if this window is all
    daylight, so it is checked rather than assumed -- and checked
    against the shipped scanner, not a hand-computed sunset.
    """
    night = first_local_night_time(
        _NIGHT_START, _NIGHT_HOURS * 3600.0,
        ref_lat=_PROJECTION["ref_lat"], ref_lon=_PROJECTION["ref_lon"])
    assert night is not None
    assert datetime(2011, 4, 26, 22) < night < datetime(2011, 4, 27, 3)


# ---------------------------------------------------------------------------
# Each row loads, and is MYNN with both streams on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", RADIATION_BEARING_MYNN)
def test_the_profile_is_shipped_and_registered(profile):
    from woof.physics_registry import physics_registry

    assert profile in SINGLE_DOMAIN_PHYSICS_PROFILES
    template = physics_registry()["templates"][profile]
    assert template["components"]["pbl"] == "mynn"
    assert template["components"]["surface_layer"] == "mynn"
    assert template["components"]["radiation"] == "rte-rrtmgp"


@pytest.mark.parametrize("profile", RADIATION_BEARING_MYNN)
def test_the_profile_resolves_to_mynn_with_longwave_on(profile):
    """The three facts a menu entry has to be able to promise."""

    switches = single_domain_runtime_switches(profile)
    assert int(switches["bl_pbl_physics"]) == 5
    assert int(switches["sf_sfclay_physics"]) == 5
    assert int(switches["ra_lw_physics"]) != 0
    assert int(switches["ra_sw_physics"]) != 0
    # And which 4/4 implementation, stated rather than left to a default:
    # the selector integer alone does not name the radiation code that runs.
    assert switches["wrf_rrtmg_compatibility"] == WRF_RRTMG_TO_RTE_RRTMGP
    assert switches["ra_rrtmg_variant"] == RRTMG_VARIANT_RTE_RRTMGP
    # Resolved through the registry, not read off the table above.
    components = validate_physics_capabilities(switches)
    assert components["pbl"] == "mynn"
    assert components["surface_layer"] == "mynn"
    assert components["radiation"] == "rte-rrtmgp"


@pytest.mark.parametrize("profile", RADIATION_BEARING_MYNN)
def test_the_profile_loads_through_the_shared_front_door(profile):
    """An emitted real case builds, with every switch preserved.

    ``experiment_from_text`` is the loader ``woof run``, ``woof go``,
    ``woof check``, both prepared runners and the DA drivers all reach
    :func:`woof.experiment.build_experiment` through, so this is the
    artifact and not a mock of it.
    """

    experiment = experiment_from_text(
        _emitted(profile), source="<mynn-radiation>")
    run = experiment.root.run
    expected = single_domain_runtime_switches(profile)
    observed = {name: getattr(run, name) for name in expected}
    assert observed == expected
    assert identify_single_domain_profile(run) == profile


# ---------------------------------------------------------------------------
# Nocturnal validity, both directions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", RADIATION_BEARING_MYNN)
def test_a_night_window_loads_with_no_declaration(profile):
    raw = _raw(_emitted(profile))
    raw["experiment"].pop("acknowledgements", None)
    experiment = build_experiment(raw, source="<mynn-radiation-night>")
    assert experiment.acknowledgements == ()
    assert experiment.root.run.ra_lw_physics == 4


@pytest.mark.parametrize("profile,sibling", MIRRORED_PAIRS)
def test_the_no_radiation_sibling_is_refused_on_the_same_window(
        profile, sibling):
    """The control that makes the row above a measurement.

    Same window, same reference point, same emitter: the only thing that
    changed is the profile, and the guard fires on the sibling and not on
    the row that replaced it.
    """

    raw = _raw(_emitted(sibling))
    raw["experiment"].pop("acknowledgements", None)
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<mynn-no-radiation-night>")
    message = str(caught.value)
    assert "local night" in message
    assert "ra_lw_physics 0" in message
    assert sibling in message
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in message


@pytest.mark.parametrize("profile", RADIATION_BEARING_MYNN)
def test_the_wizard_writes_the_nocturnally_valid_header(profile):
    text = _emitted(profile)
    assert "# NOCTURNALLY VALID" in text
    assert "NOT NOCTURNALLY VALID" not in text
    # A valid suite must not be handed the declaration for an invalid one.
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in text


def test_at_least_one_mynn_row_the_wizard_offers_is_nocturnally_valid():
    """The defect, stated as the property that closes it.

    Before this family, every ``bl_pbl_physics = 5`` entry in the
    wizard's own list ran shortwave with longwave off.
    """
    from woof.domain_wizard import WIZARD_PHYSICS_PROFILES

    mynn_offered = [
        profile for profile in WIZARD_PHYSICS_PROFILES
        if int(single_domain_runtime_switches(profile)["bl_pbl_physics"]) == 5
    ]
    assert mynn_offered, "the wizard offers no MYNN suite at all"
    valid = [
        profile for profile in mynn_offered
        if int(single_domain_runtime_switches(profile)["ra_lw_physics"]) != 0
    ]
    assert valid, (
        "every MYNN suite the wizard offers still runs longwave OFF: "
        f"{mynn_offered}")


# ---------------------------------------------------------------------------
# Radiation is the ONLY difference.
# ---------------------------------------------------------------------------

#: The switches a radiation change is allowed to move.  Anything else
#: differing between a twin and its sibling makes a paired run a
#: composition comparison instead of a radiation one.
_RADIATION_SWITCHES = frozenset({
    "ra_lw_physics", "ra_sw_physics", "radt",
    "wrf_rrtmg_compatibility", "ra_rrtmg_variant",
})


@pytest.mark.parametrize("profile,sibling", MIRRORED_PAIRS)
def test_the_pair_differs_in_the_radiation_block_and_nothing_else(
        profile, sibling):
    new = single_domain_runtime_switches(profile)
    old = single_domain_runtime_switches(sibling)
    moved = {
        name for name in set(new) | set(old)
        if new.get(name) != old.get(name)
    }
    assert moved <= _RADIATION_SWITCHES, (
        f"{profile} moves non-radiation switches away from {sibling}: "
        f"{sorted(moved - _RADIATION_SWITCHES)}")
    # Non-vacuous: the radiation block really did move.
    assert {"ra_lw_physics", "ra_sw_physics"} <= moved
    assert new["radt"] == 12.0 and old["radt"] == 1.0


# ---------------------------------------------------------------------------
# The option identity is not loosened by any of this.
# ---------------------------------------------------------------------------

def test_the_mynn_option_identity_is_still_pinned_under_radiation():
    """Adding suites must not add knobs.

    The identity refusal is what keeps ``bl_pbl_physics = 5`` meaning one
    implemented closure, and it has to keep firing with radiation on --
    otherwise a user reads "MYNN now has radiation" as "MYNN now takes
    options".
    """

    switches = single_domain_runtime_switches(MYNN_RTE_RRTMGP_PROFILE_ID)
    base = {name: switches[name] for name in (
        "moist", "mp_physics", "ra_physics", "ra_lw_physics",
        "ra_sw_physics", "radt", "wrf_rrtmg_compatibility",
        "ra_rrtmg_variant", "sf_sfclay_physics", "sf_surface_physics",
        "bl_pbl_physics", "num_soil_layers")}
    validate_run_config(_cfg(**base))

    # Both WRF mixing lengths are implemented; unported branches still refuse.
    assert "bl_mynn_mixlength" not in MYNN_PBL_OPTION_IDENTITY
    validate_run_config(_cfg(**base, bl_mynn_mixlength=2))
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(**base, bl_mynn_mixlength=3))
    message = str(caught.value)
    assert "bl_mynn_mixlength" in message

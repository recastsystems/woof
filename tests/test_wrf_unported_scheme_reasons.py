"""Every WRF physics selector value is either run or refused with its reason.

Named breakage (2.9.0, reproduced from the PyPI wheel on NCAR's CONUS 12 km
benchmark namelist): about 29 WRF option values -- BMJ, KSAS and MSKF
cumulus, the GFS, QNSE, ACM2 and MRF PBLs, the slab, CLM4, Pleim-Xiu and
SSiB land surfaces, the Goddard, CAM, SBU-YLin, Thompson graupel-hail,
Morrison-aerosol and NTU microphysics -- were refused with only a menu of
numbers ("cu_physics = 2: no woof mapping (implemented: [...])"), and the
ocean and wind-farm selectors were refused as an "unmapped key" even when
written OFF.  A user could not tell an unported scheme from a typing error,
and the release notes' "refused by name" was untrue for them.
"""
import re

import pytest

from woof import config as config_mod
from woof import namelist_import as ni
from woof.namelist_import import import_namelists

from test_namelist_import import INPUT_TEXT, _pair

#: WRF v4.6.1, v4.7.1 and v4.8.0 Registry.EM_COMMON package selectors
#: (cu 98, bl 98, mp 27 and 29 are v4.7 packages).
WRF_VALUES = {
    "cu_physics": (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 98, 99),
    "bl_pbl_physics": (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 98, 99),
    "sf_sfclay_physics": (0, 1, 2, 3, 4, 5, 7, 10, 89, 91),
    "sf_surface_physics": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "mp_physics": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   18, 24, 26, 27, 28, 29, 30, 32, 38, 40, 50, 51, 52, 53, 55, 56, 95,
                   96, 97, 98, 99, 106),
    "ra_lw_physics": (0, 1, 3, 4, 5, 7, 14, 24, 31, 99),
    "ra_sw_physics": (0, 1, 2, 3, 4, 5, 7, 14, 24, 99),
}

#: What the importer admits for each selector.
ADMITTED = {
    "cu_physics": set(ni._CU_ALLOWED),
    "bl_pbl_physics": set(ni._BL_MAP),
    "sf_sfclay_physics": set(ni._SFCLAY_ALLOWED),
    "sf_surface_physics": set(ni._SFSFC_ALLOWED),
    "mp_physics": set(ni._MP_MAP),
    "ra_lw_physics": set(config_mod.RA_LW_PHYSICS_ACCEPTED),
    "ra_sw_physics": set(config_mod.RA_SW_PHYSICS_ACCEPTED),
}


@pytest.mark.parametrize("key", sorted(WRF_VALUES))
def test_every_wrf_value_is_admitted_or_has_a_reason(key):
    reasons = config_mod.unported_scheme_reasons(key)
    for value in WRF_VALUES[key]:
        if value in ADMITTED[key]:
            assert value not in reasons, (key, value)
        else:
            assert len(reasons.get(value, "")) > 40, (key, value)
    # No stale row: a row names a value WRF defines and woof refuses
    # (bl_pbl_physics = 6 is the one row for a value WRF itself dropped).
    extra = set(reasons) - set(WRF_VALUES[key]) - {6 if key == "bl_pbl_physics" else -1}
    assert not extra, (key, extra)


def _with(**values) -> str:
    text = INPUT_TEXT
    for key, value in values.items():
        text, count = re.subn(rf"(?m)^ {key} = .*$", f" {key} = {value}, {value},",
                              text)
        assert count == 1, key
    return text


UNPORTED = [(key, value) for key in ("cu_physics", "bl_pbl_physics",
                                      "sf_sfclay_physics", "sf_surface_physics")
            for value in WRF_VALUES[key] if value not in ADMITTED[key]]
UNPORTED.append(("bl_pbl_physics", 6))


@pytest.mark.parametrize("key,value", UNPORTED)
def test_the_importer_prints_the_reason(tmp_path, key, value):
    reason = config_mod.wrf_unported_reason(key, value)
    with pytest.raises(ValueError) as caught:
        import_namelists(*_pair(tmp_path, inp=_with(**{key: value})))
    message = str(caught.value)
    assert reason[1:] in message, message
    assert "Nothing is substituted for it" in message


@pytest.mark.parametrize("value", [v for v in WRF_VALUES["mp_physics"]
                                   if v not in ADMITTED["mp_physics"]])
def test_every_unported_microphysics_is_refused_by_name(tmp_path, value):
    with pytest.raises(ValueError) as caught:
        import_namelists(*_pair(tmp_path, inp=_with(mp_physics=value)))
    message = str(caught.value)
    assert "no ratified woof mapping" not in message
    assert f"mp_physics={value} (" in message and "is not ported" in message


@pytest.mark.parametrize("key,value", [("cu_physics", 2), ("bl_pbl_physics", 7),
                                       ("sf_sfclay_physics", 7),
                                       ("sf_surface_physics", 1)])
def test_the_config_loader_prints_the_reason(key, value):
    from test_checkpoint_scheme_identity_gate import _cfg
    with pytest.raises(ValueError) as caught:
        config_mod.validate_run_config(
            _cfg(moist=True, mp_physics=6, **{key: value}))
    assert config_mod.wrf_unported_reason(key, value)[1:] in str(caught.value)


def _extra(lines: str) -> str:
    return INPUT_TEXT.replace(" cudt = 5, 0,", " cudt = 5, 0,\n" + lines)


@pytest.mark.parametrize("lines", [
    " sf_ocean_physics = 0,\n oml_hml0 = 50,\n omdt = 1,",
    " windfarm_opt = 0, 0,\n windfarm_ij = 0,",
    " tmn_update = 0,\n lagday = 150,",
])
def test_ocean_windfarm_and_lagday_off_import_as_if_absent(tmp_path, lines):
    plain, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT), name="x")
    written, report = import_namelists(*_pair(tmp_path, inp=_extra(lines)),
                                       name="x")
    # An omitted ocean selector now resolves to the default selection (PWP
    # on multi-day forecasts, lane/cap-slab-ocean a382c235f2b), so the
    # importer keeps an explicit off as its own line; nothing else moves.
    kept = "sf_ocean_physics = 0\n" if "sf_ocean_physics" in lines else ""
    assert kept in written
    assert written.replace(kept, "", 1) == plain
    reported = ({entry.key for entry in report.dropped}
                | {entry.key for entry in report.fixed})
    for key in ("oml_hml0", "omdt", "windfarm_ij", "lagday"):
        if key in lines:
            assert key in reported


def test_the_3d_pwp_ocean_imports(tmp_path):
    """WRF's 3-D PWP ocean is ported (lane/cap-slab-ocean, aa6bdeed308)."""
    written, _ = import_namelists(
        *_pair(tmp_path, inp=_extra(" sf_ocean_physics = 2,")), name="x")
    assert "sf_ocean_physics = 2\n" in written


@pytest.mark.parametrize("lines,needle", [
    (" windfarm_opt = 1, 1,\n windfarm_ij = 1,", "Fitch"),
    (" windfarm_opt = 2, 2,", "MAV"),
])
def test_ocean_and_windfarm_on_are_refused_by_name(tmp_path, lines, needle):
    with pytest.raises(ValueError) as caught:
        import_namelists(*_pair(tmp_path, inp=_extra(lines)))
    message = str(caught.value)
    assert needle in message
    assert "unmapped key" not in message


def test_the_namelist_contract_publishes_the_reasons():
    from woof.namelist_contract import _value_rules
    rules = _value_rules()
    assert "Betts-Miller-Janjic" in rules[("physics", "cu_physics")]["unported"]["2"]
    assert "Pleim-Xiu" in rules[("physics", "sf_surface_physics")]["unported"]["7"]
    assert "WDM7" in rules[("physics", "mp_physics")]["unported"]["26"]


def test_the_3d_ocean_profile_in_domains_drops_as_inert(tmp_path):
    plain, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT), name="x")
    inp = INPUT_TEXT.replace(" smooth_option = 0,",
                             " smooth_option = 0,\n ocean_levels = 30,")
    written, report = import_namelists(*_pair(tmp_path, inp=inp), name="x")
    assert written == plain
    assert "ocean_levels" in {entry.key for entry in report.dropped}

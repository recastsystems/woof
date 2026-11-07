"""A stock diagnostic request must produce its statistics or be refused."""
import pytest


@pytest.mark.parametrize("key", ["mean_diag", "diurnal_diag"])
def test_rasm_request_is_not_silently_discarded(tmp_path, key):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    original = inp.read_text()
    baseline, _ = import_namelists(wps, inp, name="rasm-admission")
    inp.write_text(original.replace("&time_control", f"&time_control\n {key}=0,"))
    disabled, _ = import_namelists(wps, inp, name="rasm-admission")
    assert disabled == baseline
    inp.write_text(original.replace("&time_control", f"&time_control\n {key}=1, mean_diag_interval_m=60,"))
    enabled, _ = import_namelists(wps, inp, name="rasm-admission")
    assert f"{key} = 1" in enabled


def test_solar_admission_and_traditional_native_failure(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    original = inp.read_text()
    baseline, _ = import_namelists(wps, inp, name="solar-admission")
    inp.write_text(original+"\n&diags\n solar_diagnostics=0,\n/\n")
    disabled, _ = import_namelists(wps, inp, name="solar-admission")
    assert baseline == disabled
    inp.write_text(original+"\n&diags\n solar_diagnostics=1,\n/\n")
    enabled, _ = import_namelists(wps, inp, name="solar-admission")
    assert "solar_diagnostics = 1" in enabled
    inp.write_text(original+"\n&diags\n diag_nwp2=1,\n/\n")
    with pytest.raises(ValueError, match="dz8w"):
        import_namelists(wps, inp, name="traditional-admission")


def test_rasm_calendar_crosses_year_and_leap_day():
    from datetime import datetime
    from woof.core.rasm import alarm
    assert alarm(0,12,1,monthly=True,now=datetime(2024,12,31,23,59,48)) == (False,True)
    assert alarm(12,12,1,monthly=True,now=datetime(2025,1,1)) == (True,False)
    assert alarm(0,12,1,monthly=True,now=datetime(2024,2,28,23,59,48)) == (False,False)
    assert alarm(0,12,1,monthly=True,now=datetime(2024,2,29,23,59,48)) == (False,True)

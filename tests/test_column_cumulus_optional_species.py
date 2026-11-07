from types import SimpleNamespace
from woof.core.column_cumulus import _phase_presence

def test_warm_rain_state_without_ice_attributes():
    assert _phase_presence(SimpleNamespace(qr=object())) == (True, False, False)

def test_mixed_phase_state_retains_all_allocated_species():
    assert _phase_presence(SimpleNamespace(qr=object(), qi=object(), qs=object())) == (True, True, True)

def test_explicitly_absent_species_are_not_present():
    assert _phase_presence(SimpleNamespace(qr=None, qi=None, qs=None)) == (False, False, False)

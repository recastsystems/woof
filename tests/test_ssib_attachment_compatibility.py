"""Surface precipitation bookkeeping preserves direct legacy attachment."""
import numpy as np
from test_sr_expression_family import _cpu_diagnostics_driver


def test_attachment_without_surface_mapping_keeps_rain_and_sr(monkeypatch):
    _module, driver, Diagnostics, shape = _cpu_diagnostics_driver(monkeypatch, 6)
    assert not hasattr(driver, "fields")
    rain = np.full(shape, 0.25, np.float32)
    sr = np.full(shape, 0.5, np.float32)
    def packet():
        return Diagnostics(rainnc=rain.copy(), rainncv=rain.copy(), sr=sr.copy())
    driver.accept_microphysics(packet())
    np.testing.assert_array_equal(driver._pending_rainbl, rain)
    np.testing.assert_array_equal(driver.microphysics.sr, sr)
    assert driver.microphysics_updates == 1
    driver.fields = {name: np.zeros(shape, np.float32) for name in
                     ("surface_rainncv", "surface_snowncv", "surface_graupelncv", "surface_hailncv")}
    driver.accept_microphysics(packet())
    np.testing.assert_array_equal(driver._pending_rainbl, rain * np.float32(2))
    np.testing.assert_array_equal(driver.fields["surface_rainncv"], rain)
    assert driver.microphysics_updates == 2

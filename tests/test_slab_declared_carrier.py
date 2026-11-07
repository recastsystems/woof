from types import SimpleNamespace
import numpy as np
from woof.core.landuse import load_landuse_table
from tilestream.harness import declared_carrier_kwargs

def test_seeded_slab_receives_table_thermal_inertia(monkeypatch):
    import tilestream.harness as h
    monkeypatch.setattr(h, "declared_glw_kwargs", lambda cfg: {})
    monkeypatch.setattr(h, "declared_swdown_kwargs", lambda cfg: {})
    cfg = SimpleNamespace(sf_surface_physics=1)
    value = declared_carrier_kwargs(cfg)["thermal_inertia"]
    table = load_landuse_table("MODIFIED_IGBP_MODIS_NOAH")
    expected = np.float32(table.values[0, 9, 4]) / np.float32(100.)
    assert value == expected and value > 0
    assert declared_carrier_kwargs(SimpleNamespace(sf_surface_physics=3)) == {}

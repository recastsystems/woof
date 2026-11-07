"""A restart between radiation calls must retain the split-flux producer.

The real ACM2/PX restart lost three history fields with unchanged checkpoint
arrays: restoring only SWDDIR/SWDDIF/SWDDNI without their publication latch
made the resumed writer omit them until radiation ran again.
"""
import pytest


@pytest.mark.parametrize("published", [False, True])
def test_split_flux_publication_survives_resident_and_streamed_restart(
        monkeypatch, tmp_path, published):
    from woof.io import restart
    from tilestream.checkpoint import driver_restart_header, restart_scalars
    from tilestream.physics_inventory import carrier_scalars, set_carrier_scalars
    from test_restart import _cfg, _fill_setup, _shim_driver_state

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1, bldt=0.0)
    source, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    driver._swdd_from_scheme = published
    path = restart.write_restart(tmp_path / "split-flux.npz", source, cfg)
    resumed, fresh = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(resumed)
    fresh._swdd_from_scheme = not published
    restart.restore_restart(path, resumed, cfg)
    assert fresh._swdd_from_scheme is published

    scalars = carrier_scalars(source)
    header = driver_restart_header(scalars)
    assert header["swdd_from_scheme"] is published
    fresh._swdd_from_scheme = not published
    set_carrier_scalars(resumed, restart_scalars({
        "elapsed_seconds": source.elapsed_seconds, "driver": header}))
    assert fresh._swdd_from_scheme is published

"""The feedback door admits only radiation that consumes live row optics."""
import pytest
from woof.config import RunConfig, validate_chem_config
from woof.chem_table import load


def config(**kwargs):
    values = dict(nx=12, ny=10, nz=20, dx=3000., dy=3000., ztop=20000.,
                  dt=18., run_seconds=3600., moist=True,
                  chem_sets="gocart_primary", bl_pbl_physics=1,
                  sf_sfclay_physics=1, sf_surface_physics=2,
                  ra_lw_physics=4, ra_sw_physics=4, aer_ra_feedback=1)
    values.update(kwargs)
    return RunConfig(**values)


@pytest.mark.parametrize("variant", ["rrtmg_legacy", "rte-rrtmgp"])
@pytest.mark.parametrize("chem_set", ["gocart_primary", "dust", "smoke"])
def test_feedback_is_reachable(variant, chem_set):
    cfg = config(chem_sets=chem_set, ra_rrtmg_variant=variant,
                 chem_sources="rave-3km" if chem_set == "smoke" else "")
    validate_chem_config(cfg)
    assert load(cfg).rows_for("optics.gocart")


@pytest.mark.parametrize("values,reason", [
    (dict(ra_lw_physics=0, ra_sw_physics=0), "another solver would discard"),
    (dict(ra_lw_physics=1), "another solver would discard"),
    (dict(aer_opt=2), "double count"),
    (dict(rrtmg_smoke_manifest="forcing.json"), "double count"),
])
def test_unsupported_or_double_counted_optics_refuse(values, reason):
    with pytest.raises(ValueError, match=reason):
        validate_chem_config(config(**values))


def test_bulk_smoke_optics_preserve_particulate_mass():
    row = load(config(chem_sets="smoke")).rows_for("optics.gocart")[0]
    assert row.optics["mass_multiplier"] == 1.
    assert row.optics["merra_mass_scale"] == 1.e-9
    assert "Unresolved soot" in row.optics["from"]


def test_optics_memory_prices_retained_and_modern_chunk_storage():
    from woof.core.chem_radiation import memory_shapes, feedback_row_count
    cfg = config(ra_rrtmg_variant="rte-rrtmgp")
    held, work = memory_shapes(cfg, 5000., 64)
    assert held["chem_optics/tau4"] == (4, cfg.nz, cfg.ny, cfg.nx)
    assert work["chem_optics/finalized_gpoints"][1] == 64
    assert feedback_row_count(cfg) > 0
    smaller = memory_shapes(cfg, 5000., 16)[1]
    assert smaller["chem_optics/finalized_gpoints"][1] == 16
    disabled = memory_shapes(config(aer_ra_feedback=0), 5000., 64)[1]
    assert "chem_optics/finalized_gpoints" not in disabled
    assert feedback_row_count(config(aer_ra_feedback=0)) == 0


def test_modern_feedback_refuses_omitted_optical_row(monkeypatch):
    from dataclasses import replace
    import woof.chem_table as chem_table
    cfg = config(ra_rrtmg_variant="rte-rrtmgp")
    table = load(cfg)
    row = table.rows_for("optics.gocart")[0]
    optics = dict(row.optics)
    optics.pop("merra_type")
    altered = replace(row, optics=optics)
    incomplete = replace(table, rows=tuple(altered if r.name == row.name else r
                                           for r in table.rows), _by_name=None)
    monkeypatch.setattr(chem_table, "load_sets", lambda *args, **kwargs: incomplete)
    with pytest.raises(ValueError, match="masses would otherwise be omitted"):
        validate_chem_config(cfg)


def test_legacy_cache_memory_accounts_for_overlapping_refresh():
    from woof.core.chem_radiation import memory_shapes
    cfg = config(ra_rrtmg_variant="rrtmg_legacy")
    held, work = memory_shapes(cfg, 5000., 64)
    assert held["chem_optics/radiation_bands"] == (24, cfg.nz, cfg.ny, cfg.nx)
    assert work["chem_optics/radiation_refresh"] == (28, cfg.nz, cfg.ny, cfg.nx)
    off_held, off_work = memory_shapes(
        config(ra_rrtmg_variant="rrtmg_legacy", aer_ra_feedback=0), 5000., 64)
    assert "chem_optics/radiation_bands" not in off_held
    assert "chem_optics/radiation_refresh" not in off_work


def test_disabled_feedback_does_not_read_state_or_import_device_work(monkeypatch):
    import builtins
    from types import SimpleNamespace
    from woof.core.chem_radiation import prepare
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("woof.core.chem_"):
            raise AssertionError("disabled feedback imported device work: " + name)
        return original(name, *args, **kwargs)

    class UnreadableState:
        def __getattribute__(self, name):
            raise AssertionError("disabled feedback read state: " + name)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert prepare(UnreadableState(), SimpleNamespace(aer_ra_feedback=0)) is None

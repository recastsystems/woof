"""The GFS row's dated surface forecast, and the S2S door reading it from the table (hex 0.4.15 fit16-rows).

THE BREAKAGE THESE PIN (2026-11-06): a GFS-forecast SST series for the 60 km
globe stopped with "rw_sfc_series: source has no surface forecast row", and the S2S door asked any source but
CFS for analyses at every valid time, which do not exist past the latest cycle.
"""
from woof.hex import s2s_door
from woof.hex import s2s_energy as energy
from woof.hex.cli import build_parser
from woof.hex.init_sources import load_table


def _row(source):
    return next(r for r in load_table()["sources"] if r["id"] == source)


def test_gfs_row_carries_a_dated_surface_forecast_block():
    forecast = _row("gfs")["surface_series"]["forecast"]
    assert forecast["interval_hours"] == 6 and forecast["max_hours"] == 384
    assert forecast["decode"] == _row("gfs")["surface_series"]["decode"]
    records = forecast["products"][0]["index"]["select"]["records"]
    assert records == [["^(TMP|ICEC|LAND|SNOD)$", "^surface$"]]


def test_gfs_forecast_objects_come_from_one_cycle_with_growing_leads():
    surface = {"source": "gfs", "forecast_cycle": "2026-10-13T00", "start": "2026-10-15_00:00:00",
               "end": "2026-10-22_00:00:00"}
    objects = energy.forcing_objects(surface)
    assert len(objects) == 29
    assert {o["source_cycle"] for o in objects} == {"2026-10-13_00:00:00"}
    assert [objects[0]["lead_hours"], objects[-1]["lead_hours"]] == [48, 216]
    assert objects[-1]["urls"][0].endswith("/gfs.20261013/00/atmos/gfs.t00z.pgrb2.0p25.f216")


def test_door_takes_forecast_forcing_from_any_row_with_the_block(tmp_path):
    def plan(source):
        args = build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs", "--cycle", "2026-10-15T00",
                                          "--members", "1", "--days", "7", "--work-dir", str(tmp_path),
                                          "--ocean-mode", "prescribed", "--surface-source", source])
        return s2s_door.plan(args)["surface"]
    assert plan("gfs")["forecast_cycle"] == "2026-10-13T00"
    assert plan("cfs")["forecast_cycle"] == "2026-10-13T00"
    assert plan("era5")["forecast_cycle"] is None
    assert s2s_door.surface_forecast_row("gfs") and not s2s_door.surface_forecast_row("era5")
